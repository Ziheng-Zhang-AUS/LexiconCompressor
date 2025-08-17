# inference_lcm_vanilla_dtype_safe.py
import os
import random
from typing import List, Optional

import torch
from transformers import AutoTokenizer, Qwen3ForCausalLM

from modeling_lexicon_compressor import LexiconCompressorModel
from tokenization_lexicon import LexiconTokenizer

# ---------------- 配置 ----------------
MODEL_NAME        = "Qwen/Qwen3-0.6B"
CSV_PATH          = "cleaned_lexicon_tiny.csv"
LEX_COLUMNS       = ["lexical_unit", "pos", "gloss", "variant"]

NUM_LAYERS_LCM    = 2           # 与 LCM 实现一致
ROWS_PER_SAMPLE   = None        # None=全部行；也可设 64/128/256 以提速
MAX_NEW_TOKENS    = 120
SEED              = 0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# DTYPE 只是加载 Qwen 的建议精度；实际以 base_dtype（从 Qwen embedding 权重读取）为准
DTYPE  = torch.float16 if DEVICE.startswith("cuda") else torch.float32

os.environ["TOKENIZERS_PARALLELISM"] = "false"
random.seed(SEED); torch.manual_seed(SEED)

# ---------------- 小工具（注意 dtype 全链路一致） ----------------
def full_vis_mask(L: int, device, dtype):
    """加性 mask：全可见 = 0；注意 dtype 必须与 q/k/v 一致（base_dtype），否则 SDPA 报错。"""
    return torch.zeros(1, 1, L, L, device=device, dtype=dtype)

def build_row_rope(rotary, L: int, H: int, device, dtype):
    """行 RoPE：dummy 用 base_dtype，pos 用 long；返回与 dummy 同 dtype。"""
    dummy = torch.zeros(1, L, H, device=device, dtype=dtype)
    pos   = torch.arange(L, device=device, dtype=torch.long).unsqueeze(0)
    return rotary(dummy, pos)

def build_col_identity_rope(rotary, N: int, H: int, device, dtype):
    """列 RoPE：同相位（全 0 索引），保持置换等变；返回与 dummy 同 dtype。"""
    dummy = torch.zeros(1, N, H, device=device, dtype=dtype)
    pos0  = torch.zeros(1, N, dtype=torch.long, device=device)
    return rotary(dummy, pos0)

def choose_rows(all_ids: List[List[int]], rows_per_sample: Optional[int]) -> List[List[int]]:
    if rows_per_sample is None or rows_per_sample >= len(all_ids):
        return all_ids
    return random.sample(all_ids, rows_per_sample)

# ---------------- 加载 tokenizer ----------------
print("[Load] tokenizer...")
tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tok.add_tokens(["[COMP]"], special_tokens=False)
comp_id = tok.convert_tokens_to_ids("[COMP]")

# ---------------- 加载 Qwen 基座（无 LoRA、无 ckpt） ----------------
print("[Load] Qwen3 base...")
qwen = Qwen3ForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=DTYPE
).to(DEVICE)
qwen.resize_token_embeddings(len(tok))
qwen.eval()

cfg        = qwen.config
H          = cfg.hidden_size
rotary     = qwen.model.rotary_emb
base_dtype = qwen.get_input_embeddings().weight.dtype  # 权威 dtype（通常 torch.float16/bfloat16）
print(f"[Info] base_dtype = {base_dtype}")

# ---------------- 构建 & 注入 LCM 权重（直接来自基座 Qwen） ----------------
print("[Load] LCM from base Qwen weights (no ckpt)...")
lcm = LexiconCompressorModel(config=cfg, num_attention_layers=NUM_LAYERS_LCM).to(DEVICE, dtype=base_dtype)

# 1) 嵌入权重：建议你的 load_embeddings_weights 使用 nn.Embedding(..., dtype=w.dtype)
emb_weights = qwen.get_input_embeddings().state_dict()  # {'weight': tensor(..., dtype=base_dtype)}
lcm.load_embeddings_weights(emb_weights)

# 2) 注意力权重：偶数层 row、奇数层 col（按你的 Row/Col 约定）
attn_weights = []
for i in range(NUM_LAYERS_LCM):
    row_sd = qwen.model.layers[2*i].state_dict()
    col_sd = qwen.model.layers[2*i + 1].state_dict()
    attn_weights.append((row_sd, col_sd))
lcm.load_attention_weights(attn_weights)

# 3) 再次对齐设备与 dtype（保险）
lcm.to(device=DEVICE, dtype=base_dtype)
lcm.eval()

# ---------------- 词典 → token ids → 前缀向量 ----------------
print("[Lexicon] tokenize + compress...")
lex_tok = LexiconTokenizer(
    csv_file_path=CSV_PATH,
    tokenizer=tok,
    column_names=LEX_COLUMNS,
    compress_token_id=comp_id,
    delimiter=",",
    add_special_tokens=False
)
all_row_ids = lex_tok.process_lexicon()
row_ids_list = choose_rows(all_row_ids, ROWS_PER_SAMPLE)
print(f"[Lexicon] rows used = {len(row_ids_list)}")

# 为每一层准备 row/col 的 RoPE 和 mask（RoPE/Mask 都用 base_dtype）
row_pos_layers, row_msk_layers = [], []
for _ in range(NUM_LAYERS_LCM):
    row_pos = [build_row_rope(rotary, len(ids), H, DEVICE, base_dtype) for ids in row_ids_list]
    row_msk = [full_vis_mask(len(ids), DEVICE, base_dtype) for ids in row_ids_list]
    row_pos_layers.append(row_pos)
    row_msk_layers.append(row_msk)

N = len(row_ids_list)
col_pos_layers = [build_col_identity_rope(rotary, N, H, DEVICE, base_dtype) for _ in range(NUM_LAYERS_LCM)]
col_msk_layers = [full_vis_mask(N, DEVICE, base_dtype) for _ in range(NUM_LAYERS_LCM)]

with torch.inference_mode():
    out_rows = lcm(
        token_ids_list=row_ids_list,
        attention_weights=None,             # 已通过 load_* 注入
        embeddings_weights=None,
        row_attention_masks=row_msk_layers,
        column_attention_masks=col_msk_layers,
        row_position_embeddings=row_pos_layers,
        column_position_embeddings=col_pos_layers
    )
    # 前缀 = 每行第 0 个 token（[COMP]）向量，堆叠为 [R, H]；保持 base_dtype
    prefix = torch.stack([r[0] for r in out_rows], dim=0).to(DEVICE, dtype=base_dtype)   # [R, H]
R = prefix.size(0)
print(f"[Prefix] shape = {tuple(prefix.shape)}  (= [R, H])")

# ---------------- dtype 自检（可选调试） ----------------
def _print_dtypes():
    print("[DTypeCheck] prefix:", prefix.dtype)
    print("[DTypeCheck] qwen emb weight:", qwen.get_input_embeddings().weight.dtype)
_print_dtypes()

# ---------------- 构造带前缀的生成 ----------------
def build_inputs_with_prefix(src_text: str):
    """
    返回：inputs_embeds, attention_mask, position_ids
    拼接：[ prefix(R,H) ] + token_embeds(prompt)
    prompt = src + "\\n### Translation:\\n"
    """
    with torch.inference_mode():
        emb = qwen.get_input_embeddings()
        sep = "\n### Translation:\n"
        prompt = src_text + sep

        # tokenizer 输出的 input_ids -> emb 取出的 tok_emb 与 base_dtype 一致
        enc = tok(prompt, return_tensors="pt", truncation=True, max_length=1024).to(DEVICE)
        tok_emb = emb(enc.input_ids)                           # [1, T, H]，dtype=base_dtype
        prefix_emb = prefix.unsqueeze(0).to(dtype=tok_emb.dtype)  # [1, R, H] 对齐 dtype
        inputs_embeds = torch.cat([prefix_emb, tok_emb], dim=1)   # [1, R+T, H]，dtype=base_dtype

        total_len = inputs_embeds.size(1)
        # 注意：这里的 attention_mask 仍按 HF 的惯例用 long/bool；不会作为 SDPA 的加性 bias 参与 matmul
        attention_mask = torch.ones(1, total_len, device=DEVICE, dtype=torch.long)
        position_ids = torch.arange(total_len, device=DEVICE, dtype=torch.long).unsqueeze(0)

        return inputs_embeds, attention_mask, position_ids

@torch.inference_mode()
def generate_translation(src_text: str, max_new_tokens: int = MAX_NEW_TOKENS):
    inputs_embeds, attention_mask, position_ids = build_inputs_with_prefix(src_text)
    eos_id = tok.eos_token_id
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else eos_id

    out_ids = qwen.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,      # 贪心，稳定复现；要更自然可 do_sample=True + temperature/top_p
        eos_token_id=eos_id,
        pad_token_id=pad_id
    )
    # 解码整条序列（包含提示）
    return tok.decode(out_ids[0], skip_special_tokens=True)

# ---------------- Demo 样本 ----------------
samples = [
    "Please translate this sentence into English.",
    "yilgbayi yininya nanawunyinyilgbayi yininya nanawunyin gabarri yidugalyi gunganginyi waray gandiya bla naughty waray gandi yilgbayiwan ngayana yinggi yinggangala yanimayi dularriya gungangin yidugal nana na waray gandiya yibiyanyi waray yanggunburrgan yanima yimbalyarriwan yimbanay",
    "`That’s OK in relation to you’ . Her son-in-law led her astray . He led her astray for `naughty’ . It ok I’m telling you, yinggangala skin . yanimayi dularriya gungan.gin yidugal. That he led her astray. Men lead them astray like that. yimbalyarri-wan yimbanay."
]

print("\n========== Zero-shot + LCM（来自基座权重）推理结果 ==========\n")
for i, s in enumerate(samples, 1):
    out = generate_translation(s, MAX_NEW_TOKENS)
    print(f"[{i}] SRC: {s}\n    OUT: {out}\n")



