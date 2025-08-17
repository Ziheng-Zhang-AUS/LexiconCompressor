# sft_lexicon_mt.py
import os, json, random
import torch
from typing import List, Optional
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, Qwen3ForCausalLM
from peft import LoraConfig, get_peft_model
from tokenization_lexicon import LexiconTokenizer
from modeling_lexicon_compressor import LexiconCompressorModel
from row_column_attention_stack import RowColumnAttentionStack
from row_column_attention import RowColumnAttention


MODEL_NAME        = "Qwen/Qwen3-0.6B"
CSV_PATH          = "cleaned_lexicon_tiny.csv"
LEX_COLUMNS       = ["lexical_unit", "pos", "gloss", "variant"]
TRAIN_JSONL       = "data/train.jsonl"
OUTPUT_DIR        = "ckpt_sft"
NUM_EPOCHS        = 10
BATCH_SIZE        = 1                   
GRAD_ACCUM_STEPS  = 16                   
LR_QWEN           = 2e-5
LR_LCM            = 3e-4
NUM_LAYERS_LCM    = 2                   
ROWS_PER_SAMPLE   = None                
MAX_SRC_TOKENS    = 256
MAX_TGT_TOKENS    = 256

LORA_R            = 32
LORA_ALPHA        = 64
LORA_DROPOUT      = 0.1

SEED              = 42
DEVICE            = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE             = (torch.bfloat16 if (DEVICE.startswith("cuda") and torch.cuda.is_bf16_supported())
                    else (torch.float16 if DEVICE.startswith("cuda") else torch.float32))
os.environ["TOKENIZERS_PARALLELISM"] = "false"
random.seed(SEED); torch.manual_seed(SEED)

USE_PEFT = True
print("PEFT library available, using LoRA")

def full_vis_mask(L: int, device):
    return torch.zeros(1, 1, L, L, device=device, dtype=torch.float32)

def build_row_rope(rotary, L: int, H: int, device):
    dummy  = torch.zeros(1, L, H, device=device)
    pos    = torch.arange(L, device=device).unsqueeze(0)
    return rotary(dummy, pos)  

def build_col_identity_rope(rotary, N: int, H: int, device):
    dummy  = torch.zeros(1, N, H, device=device)
    pos0   = torch.zeros(1, N, dtype=torch.long, device=device)
    return rotary(dummy, pos0) 
def choose_rows(all_ids: List[List[int]], rows_per_sample: Optional[int]) -> List[List[int]]:
    """None -> 全部；int -> 随机采样该数量的行"""
    if rows_per_sample is None or rows_per_sample >= len(all_ids):
        return all_ids
    return random.sample(all_ids, rows_per_sample)

def autocast_ctx():
    if DEVICE.startswith("cuda") and DTYPE in (torch.float16, torch.bfloat16):
        return torch.autocast(device_type="cuda", dtype=DTYPE)
    return torch.cuda.amp.autocast(enabled=False)

def get_qwen_rotary_emb(m):
    # m 可以是 Qwen3ForCausalLM 或 PeftModelForCausalLM
    # 1) PEFT 包裹的情况
    if hasattr(m, "base_model") and hasattr(m.base_model, "model"):
        base = m.base_model.model  # 这通常是 Qwen3ForCausalLM
        if hasattr(base, "model") and hasattr(base.model, "rotary_emb"):
            return base.model.rotary_emb
        if hasattr(base, "rotary_emb"):
            return base.rotary_emb
    # 2) 非 PEFT
    if hasattr(m, "model") and hasattr(m.model, "rotary_emb"):
        return m.model.rotary_emb
    raise AttributeError("Cannot locate rotary_emb in given model.")


class MTJsonlDataset(Dataset):
    def __init__(self, path: str):
        self.samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                obj = json.loads(line)
                if "text" in obj and "translation" in obj:
                    self.samples.append({"src": obj["text"], "tgt": obj["translation"]})
        if not self.samples:
            raise RuntimeError(f"No data in {path}")
    def __len__(self): return len(self.samples)
    def __getitem__(self, i): return self.samples[i]

def main():
    print(f"Device={DEVICE}, dtype={DTYPE}")
    
    if USE_PEFT:
        print(f"Using PEFT LoRA with r={LORA_R}, alpha={LORA_ALPHA}")
    else:
        print("Using full fine-tuning (no LoRA)")

    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tok.add_tokens(["[COMP]"], special_tokens=False)
    comp_id = tok.convert_tokens_to_ids("[COMP]")

    qwen = Qwen3ForCausalLM.from_pretrained(
        MODEL_NAME, 
        trust_remote_code=True,
        torch_dtype=DTYPE
    ).to(DEVICE)
    qwen.resize_token_embeddings(len(tok))

    if USE_PEFT:
        lora_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules=["q_proj", "v_proj"],  
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM"
        )
        

        qwen = get_peft_model(qwen, lora_config)
        qwen.print_trainable_parameters()
    
    qwen.train()

    cfg    = qwen.config
    rotary = get_qwen_rotary_emb(qwen)
    H      = cfg.hidden_size

    lex_tok = LexiconTokenizer(
        csv_file_path=CSV_PATH,
        tokenizer=tok,
        column_names=LEX_COLUMNS,
        compress_token_id=comp_id,
        delimiter=",",
        add_special_tokens=False
    )
    all_row_ids = lex_tok.process_lexicon()
    print(f"[Lexicon] total rows = {len(all_row_ids)}")
    lcm = LexiconCompressorModel(config=cfg, num_attention_layers=NUM_LAYERS_LCM).to(device=DEVICE, dtype=DTYPE)
    
    emb_weights = qwen.get_input_embeddings().state_dict()
    
    attn_weights = []
    original_qwen = Qwen3ForCausalLM.from_pretrained(
        MODEL_NAME, 
        trust_remote_code=True,
        torch_dtype=torch.float32
    ).to("cpu")
    for i in range(NUM_LAYERS_LCM):
        row_sd = original_qwen.model.layers[2*i].state_dict()
        col_sd = original_qwen.model.layers[2*i+1].state_dict()
        attn_weights.append((row_sd, col_sd))
    original_qwen = None  # 释放内存
    torch.cuda.empty_cache()

    lcm.load_embeddings_weights(emb_weights)
    if attn_weights:
        lcm.load_attention_weights(attn_weights)
    lcm.train()

    ds = MTJsonlDataset(TRAIN_JSONL)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

    params = [
        {"params": qwen.parameters(), "lr": LR_QWEN},
        {"params": lcm.parameters(),  "lr": LR_LCM},
    ]
    optim = torch.optim.AdamW(params, weight_decay=0.01)
    use_scaler = (DEVICE.startswith("cuda") and DTYPE == torch.float16)
    scaler = torch.amp.GradScaler(enabled=use_scaler)

    global_step = 0
    optim.zero_grad(set_to_none=True)

    for epoch in range(1, NUM_EPOCHS+1):
        for batch_idx, batch in enumerate(dl, start=1):
            row_ids_list = choose_rows(all_row_ids, ROWS_PER_SAMPLE)

            row_pos_layers, row_msk_layers = [], []
            for _ in range(NUM_LAYERS_LCM):
                row_pos = [build_row_rope(rotary, len(ids), H, DEVICE) for ids in row_ids_list]
                row_msk = [full_vis_mask(len(ids), DEVICE) for ids in row_ids_list]
                row_pos_layers.append(row_pos)
                row_msk_layers.append(row_msk)
            N = len(row_ids_list)
            col_pos_layers = [build_col_identity_rope(rotary, N, H, DEVICE) for _ in range(NUM_LAYERS_LCM)]
            col_msk_layers = [full_vis_mask(N, DEVICE) for _ in range(NUM_LAYERS_LCM)]

            with autocast_ctx():
                out_rows = lcm(
                    token_ids_list=row_ids_list,
                    attention_weights=None,            
                    embeddings_weights=None,
                    row_attention_masks=row_msk_layers,
                    column_attention_masks=col_msk_layers,
                    row_position_embeddings=row_pos_layers,
                    column_position_embeddings=col_pos_layers
                )
                prefix = torch.stack([r[0] for r in out_rows], dim=0)  
            R = prefix.size(0)

            src_texts = batch["src"]
            tgt_texts = batch["tgt"]

            inputs_embeds_list, labels_list, attn_mask_list, pos_ids_list = [], [], [], []
            emb = qwen.get_input_embeddings()

            for src, tgt in zip(src_texts, tgt_texts):
                src_enc = tok(src, return_tensors="pt", truncation=True, max_length=MAX_SRC_TOKENS)
                tgt_enc = tok(tgt, return_tensors="pt", truncation=True, max_length=MAX_TGT_TOKENS)
                src_ids = src_enc.input_ids.to(DEVICE)
                tgt_ids = tgt_enc.input_ids.to(DEVICE)

                src_emb = emb(src_ids) 
                sep_ids  = tok("\n### Translation:\n", return_tensors="pt").input_ids.to(DEVICE)
                sep_emb  = emb(sep_ids)
                tgt_emb  = emb(tgt_ids)

                prefix_emb = prefix.unsqueeze(0)                 
                in_embeds  = torch.cat([prefix_emb, src_emb, sep_emb, tgt_emb], dim=1)   

                Ts, S, Tt = src_emb.size(1), sep_emb.size(1), tgt_emb.size(1)
                total_len = R + Ts + S + Tt

                labels = torch.full((1, total_len), fill_value=-100, device=DEVICE, dtype=torch.long)
                labels[:, R+Ts+S:] = tgt_ids

                attn_mask = torch.ones(1, total_len, device=DEVICE, dtype=torch.long)
                pos_ids   = torch.arange(total_len, device=DEVICE).unsqueeze(0)

                inputs_embeds_list.append(in_embeds)
                labels_list.append(labels)
                attn_mask_list.append(attn_mask)
                pos_ids_list.append(pos_ids)

            inputs_embeds = torch.cat(inputs_embeds_list, dim=0)  
            labels        = torch.cat(labels_list, dim=0)         
            attention_mask= torch.cat(attn_mask_list, dim=0)      
            position_ids  = torch.cat(pos_ids_list, dim=0)        

            with autocast_ctx():
                out  = qwen(inputs_embeds=inputs_embeds,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            labels=labels)
                loss = out.loss / GRAD_ACCUM_STEPS  

            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (batch_idx % GRAD_ACCUM_STEPS) == 0:
                if use_scaler:
                    scaler.step(optim)
                    scaler.update()
                else:
                    optim.step()
                optim.zero_grad(set_to_none=True)

                global_step += 1
                print(f"[epoch {epoch}] step {global_step} | loss {out.loss.item():.4f} | R={R}")

        print(f"Epoch {epoch}/{NUM_EPOCHS} done.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if USE_PEFT:
        qwen.save_pretrained(os.path.join(OUTPUT_DIR, "qwen_lora"))
    else:
        torch.save(qwen.state_dict(), os.path.join(OUTPUT_DIR, "qwen_sft.pt"))
    
    torch.save(lcm.state_dict(), os.path.join(OUTPUT_DIR, "lcm.pt"))
    tok.save_pretrained(OUTPUT_DIR)
    print(f"Saved to {OUTPUT_DIR}/ (rows_per_sample={ROWS_PER_SAMPLE or len(all_row_ids)})")

if __name__ == "__main__":
    main()