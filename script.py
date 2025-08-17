# sft_lexicon_mt.py
import os, json, random
import torch
from typing import List, Optional
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, Qwen3ForCausalLM

# ---- 你的模块（确保这些文件在同目录或可导入路径）----
from tokenization_lexicon import LexiconTokenizer
from modeling_lexicon_compressor import LexiconCompressorModel
from row_column_attention_stack import RowColumnAttentionStack
from row_column_attention import RowColumnAttention

# -------------------- 配置 --------------------
MODEL_NAME        = "Qwen/Qwen3-0.6B"
CSV_PATH          = "cleaned_lexicon_tiny.csv"
LEX_COLUMNS       = ["lexical_unit", "pos", "gloss", "variant"]
TRAIN_JSONL       = "data/train.jsonl"
OUTPUT_DIR        = "ckpt_sft"

NUM_EPOCHS        = 2
BATCH_SIZE        = 1                   # 建议先用 1
GRAD_ACCUM_STEPS  = 8                   # ✅ 新增：梯度累计
LR_QWEN           = 2e-5
LR_LCM            = 3e-4
NUM_LAYERS_LCM    = 2                   # 注意：8层较重，OOM 就改小
ROWS_PER_SAMPLE   = None                # ✅ None=使用全部行；否则随机采样该数量的行
MAX_SRC_TOKENS    = 256
MAX_TGT_TOKENS    = 256

SEED              = 42
DEVICE            = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE             = (torch.bfloat16 if (DEVICE.startswith("cuda") and torch.cuda.is_bf16_supported())
                    else (torch.float16 if DEVICE.startswith("cuda") else torch.float32))
os.environ["TOKENIZERS_PARALLELISM"] = "false"
random.seed(SEED); torch.manual_seed(SEED)

# -------------------- 工具 --------------------
def full_vis_mask(L: int, device):
    return torch.zeros(1, 1, L, L, device=device, dtype=torch.float32)

def build_row_rope(rotary, L: int, H: int, device):
    dummy  = torch.zeros(1, L, H, device=device)
    pos    = torch.arange(L, device=device).unsqueeze(0)
    return rotary(dummy, pos)  # (cos, sin)

def build_col_identity_rope(rotary, N: int, H: int, device):
    dummy  = torch.zeros(1, N, H, device=device)
    pos0   = torch.zeros(1, N, dtype=torch.long, device=device)
    return rotary(dummy, pos0) # identity

def choose_rows(all_ids: List[List[int]], rows_per_sample: Optional[int]) -> List[List[int]]:
    """None -> 全部；int -> 随机采样该数量的行"""
    if rows_per_sample is None or rows_per_sample >= len(all_ids):
        return all_ids
    return random.sample(all_ids, rows_per_sample)

def autocast_ctx():
    if DEVICE.startswith("cuda") and DTYPE in (torch.float16, torch.bfloat16):
        return torch.autocast(device_type="cuda", dtype=DTYPE)
    # 其它情况不开 autocast
    return torch.cuda.amp.autocast(enabled=False)

# -------------------- 数据集 --------------------
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

# -------------------- 主流程 --------------------
def main():
    print(f"Device={DEVICE}, dtype={DTYPE}")

    # 1) tokenizer & Qwen
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tok.add_tokens(["[COMP]"], special_tokens=False)
    comp_id = tok.convert_tokens_to_ids("[COMP]")

    qwen = Qwen3ForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
    qwen.resize_token_embeddings(len(tok))
    qwen.to(device=DEVICE, dtype=DTYPE)
    qwen.train()

    cfg    = qwen.config
    rotary = qwen.model.rotary_emb
    H      = cfg.hidden_size

    # 2) 词典 → ids（前置 [COMP]）
    lex_tok = LexiconTokenizer(
        csv_file_path=CSV_PATH,
        tokenizer=tok,
        column_names=LEX_COLUMNS,
        compress_token_id=comp_id,
        delimiter=",",
        add_special_tokens=False
    )
    all_row_ids = lex_tok.process_lexicon()
    if len(all_row_ids) == 0:
        raise RuntimeError("Lexicon rows are empty.")
    print(f"[Lexicon] total rows = {len(all_row_ids)}")

    # 3) 压缩器，用 Qwen 权重初始化
    lcm = LexiconCompressorModel(config=cfg, num_attention_layers=NUM_LAYERS_LCM).to(device=DEVICE, dtype=DTYPE)
    emb_weights = qwen.get_input_embeddings().state_dict()
    attn_weights = []
    for i in range(NUM_LAYERS_LCM):
        row_sd = qwen.model.layers[2*i].state_dict()
        col_sd = qwen.model.layers[2*i+1].state_dict()
        attn_weights.append((row_sd, col_sd))
    lcm.load_embeddings_weights(emb_weights)
    lcm.load_attention_weights(attn_weights)
    lcm.train()

    # 4) 数据
    ds = MTJsonlDataset(TRAIN_JSONL)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

    # 5) 优化器 & 可选 scaler（仅 fp16 需要）
    params = [
        {"params": qwen.parameters(), "lr": LR_QWEN},
        {"params": lcm.parameters(),  "lr": LR_LCM},
    ]
    optim = torch.optim.AdamW(params, weight_decay=0.01)
    use_scaler = (DEVICE.startswith("cuda") and DTYPE == torch.float16)
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    # 6) 训练循环（支持梯度累计）
    global_step = 0
    optim.zero_grad(set_to_none=True)

    for epoch in range(1, NUM_EPOCHS+1):
        for batch_idx, batch in enumerate(dl, start=1):

            # ---- (A) 取词典行（全部或采样）并构建 RoPE / mask ----
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

            # ---- (B) 过 LCM 得到每行容器（行首 [COMP]）----
            with autocast_ctx():
                out_rows = lcm(
                    token_ids_list=row_ids_list,
                    attention_weights=None,              # 已预加载
                    embeddings_weights=None,
                    row_attention_masks=row_msk_layers,
                    column_attention_masks=col_msk_layers,
                    row_position_embeddings=row_pos_layers,
                    column_position_embeddings=col_pos_layers
                )
                prefix = torch.stack([r[0] for r in out_rows], dim=0)  # [R,H]
            R = prefix.size(0)

            # ---- (C) 准备源/目标并拼接到 Qwen ----
            src_texts = batch["src"]
            tgt_texts = batch["tgt"]

            inputs_embeds_list, labels_list, attn_mask_list, pos_ids_list = [], [], [], []
            emb = qwen.get_input_embeddings()

            for src, tgt in zip(src_texts, tgt_texts):
                src_enc = tok(src, return_tensors="pt", truncation=True, max_length=MAX_SRC_TOKENS)
                tgt_enc = tok(tgt, return_tensors="pt", truncation=True, max_length=MAX_TGT_TOKENS)
                src_ids = src_enc.input_ids.to(DEVICE)
                tgt_ids = tgt_enc.input_ids.to(DEVICE)

                src_emb = emb(src_ids)  # [1, Ts, H]
                sep_ids  = tok("\n### Translation:\n", return_tensors="pt").input_ids.to(DEVICE)
                sep_emb  = emb(sep_ids)
                tgt_emb  = emb(tgt_ids)

                prefix_emb = prefix.unsqueeze(0)                  # [1,R,H]
                in_embeds  = torch.cat([prefix_emb, src_emb, sep_emb, tgt_emb], dim=1)   # [1, R+Ts+S+Tt, H]

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

            inputs_embeds = torch.cat(inputs_embeds_list, dim=0)  # [B, L, H]
            labels        = torch.cat(labels_list, dim=0)         # [B, L]
            attention_mask= torch.cat(attn_mask_list, dim=0)      # [B, L]
            position_ids  = torch.cat(pos_ids_list, dim=0)        # [B, L]

            # ---- (D) 前向 & 反传（梯度累计）----
            with autocast_ctx():
                out  = qwen(inputs_embeds=inputs_embeds,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            labels=labels)
                loss = out.loss / GRAD_ACCUM_STEPS  # ✅ 累计梯度需要除以 steps

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

    # 7) 保存
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch.save(qwen.state_dict(), os.path.join(OUTPUT_DIR, "qwen_sft.pt"))
    torch.save(lcm.state_dict(),  os.path.join(OUTPUT_DIR, "lcm.pt"))
    tok.save_pretrained(OUTPUT_DIR)
    print(f"Saved to {OUTPUT_DIR}/ (rows_per_sample={ROWS_PER_SAMPLE or len(all_row_ids)})")

if __name__ == "__main__":
    main()