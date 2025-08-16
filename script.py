import os
import random
import torch
import torch.nn as nn
from typing import List
from transformers import AutoTokenizer, Qwen3ForCausalLM

# === 你的类：请确保同目录可导入，或直接粘贴类定义 ===
from modeling_lexicon_compressor import LexiconCompressorModel
from row_column_attention_stack import RowColumnAttentionStack  # 仅为依赖注册
from row_column_attention import RowColumnAttention             # 仅为依赖注册
from tokenization_lexicon import LexiconTokenizer                  # 如果你前面存成独立文件

# ------------------------ 配置 ------------------------
MODEL_NAME       = "Qwen/Qwen3-0.6B"
CSV_PATH         = "lexicon_demo.csv"
COLUMNS          = ["lexical_unit", "pos", "gloss", "variant"]
NUM_LAYERS       = 2                  # 行/列注意力层数
ROWS_PER_SAMPLE  = 3                  # 每个训练样本包含多少行词典
BATCH_SIZE       = 1
EPOCHS           = 1
LR               = 5e-5
MAX_NEW_TOKENS   = 0                  # 仅计算LM loss，不做生成
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"
SEED             = 42
random.seed(SEED); torch.manual_seed(SEED)

# ------------------------ 小工具 ------------------------
def full_vis_mask(L: int, device):
    # 4D 全 0：形状 [B,1,Q,K]，非因果；如有 padding 再把 key 端置 -inf
    return torch.zeros(1, 1, L, L, device=device, dtype=torch.float32)

def build_row_rope(rotary, L: int, H: int, device):
    dummy  = torch.zeros(1, L, H, device=device)
    pos_ids= torch.arange(L, device=device).unsqueeze(0)
    return rotary(dummy, pos_ids)  # (cos, sin)

def build_col_identity_rope(rotary, N: int, H: int, device):
    dummy  = torch.zeros(1, N, H, device=device)
    pos0   = torch.zeros(1, N, dtype=torch.long, device=device)
    return rotary(dummy, pos0)     # identity: cos=1, sin=0

def chunk_list(lst, n):
    # 把 lst 切成若干块，每块 n 个元素（最后不足则丢弃，简化）
    return [lst[i:i+n] for i in range(0, len(lst) - len(lst)%n, n)]

# ------------------------ 数据准备 ------------------------
def build_dataset(token_ids_all: List[List[int]]):
    # 每样本 = ROADS_PER_SAMPLE 行（list of list[int]）
    samples = chunk_list(token_ids_all, ROWS_PER_SAMPLE)
    return samples

# ------------------------ 训练 ------------------------
def main():
    print(f"Device: {DEVICE}")
    # 1) tokenizer + [COMP]
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tok.add_tokens(["[COMP]"], special_tokens=False)
    comp_id = tok.convert_tokens_to_ids("[COMP]")

    # 2) 读 CSV → 每行 token ids（前置 [COMP]）
    lt = LexiconTokenizer(
        csv_file_path=CSV_PATH,
        tokenizer=tok,
        column_names=COLUMNS,
        compress_token_id=comp_id,
        delimiter=",",
        add_special_tokens=False
    )
    token_ids_per_row = lt.process_lexicon()
    if len(token_ids_per_row) < ROWS_PER_SAMPLE:
        raise RuntimeError("CSV 行数太少，至少需要满足 ROWS_PER_SAMPLE。")
    samples = build_dataset(token_ids_per_row)
    print(f"Total samples: {len(samples)} (each has {ROWS_PER_SAMPLE} rows)")

    # 3) 加载 Qwen + 权重
    qwen  = Qwen3ForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True).to(DEVICE)
    qwen.resize_token_embeddings(len(tok))  # 适配新增 [COMP]
    qwen.train()
    cfg    = qwen.config
    rotary = qwen.model.rotary_emb
    H      = cfg.hidden_size

    # 4) 构建 LexiconCompressorModel 并加载 Qwen 的 embedding/attention 权重
    lcm = LexiconCompressorModel(config=cfg, num_attention_layers=NUM_LAYERS).to(DEVICE)
    emb_weights = qwen.get_input_embeddings().state_dict()  # {"weight": ...}
    attn_weights = []
    for i in range(NUM_LAYERS):
        row_sd = qwen.model.layers[2*i].state_dict()
        col_sd = qwen.model.layers[2*i+1].state_dict()
        attn_weights.append((row_sd, col_sd))
    # 先一次性喂给 lcm（你 forward 也会接收，但我们提早装载）
    lcm.load_embeddings_weights(emb_weights)
    lcm.load_attention_weights(attn_weights)
    lcm.train()

    # 5) 优化器（只演示全参微调；你也可以只训 lcm 或配 LoRA）
    optimizer = torch.optim.AdamW([
        {"params": qwen.parameters(), "lr": LR},
        {"params": lcm.parameters(),  "lr": LR},
    ], weight_decay=0.01)

    # 6) 训练循环（最小化）
    global_step = 0
    for epoch in range(EPOCHS):
        random.shuffle(samples)
        for si in range(0, len(samples), BATCH_SIZE):
            batch = samples[si:si+BATCH_SIZE]
            if len(batch) < BATCH_SIZE:
                continue

            optimizer.zero_grad()

            # ---- (A) LexiconCompressor 前向：得到每个样本的“行首容器向量”前缀 ----
            # 这里简化成逐样本处理（B=1时最简单）；你可改成批处理
            prefix_list = []      # 每个样本的 prefix embeds: [R,H]
            row_pos_all  = []     # 每层的行 RoPE 列表（按样本、按行）
            row_mask_all = []     # 每层的行 4D 全可见 mask
            col_pos_all  = []     # 每层的列 identity RoPE
            col_mask_all = []     # 每层的列 4D 全可见 mask

            for sample in batch:
                # sample: List[List[int]]  (R 行)
                R = len(sample)
                # 为 lcm 构建行/列 RoPE & mask（每层都需要）
                row_pos_layers = []
                row_msk_layers = []
                for _ in range(NUM_LAYERS):
                    row_pos = []
                    row_msk = []
                    for ids in sample:
                        L = len(ids)
                        row_pos.append(build_row_rope(rotary, L, H, DEVICE))
                        row_msk.append(full_vis_mask(L, DEVICE))
                    row_pos_layers.append(row_pos)
                    row_msk_layers.append(row_msk)
                N = R
                col_pos_layers = [build_col_identity_rope(rotary, N, H, DEVICE) for _ in range(NUM_LAYERS)]
                col_msk_layers = [full_vis_mask(N, DEVICE) for _ in range(NUM_LAYERS)]

                # lcm 前向：得到 List[Tensor]（每行 [L,H]）
                out_rows = lcm(
                    token_ids_list=sample,
                    attention_weights=attn_weights,        # 第一次会装载
                    embeddings_weights=emb_weights,        # 第一次会装载
                    row_attention_masks=row_msk_layers,
                    column_attention_masks=col_msk_layers,
                    row_position_embeddings=row_pos_layers,
                    column_position_embeddings=col_pos_layers
                )
                # 取各行的 [COMP] 容器向量，拼成 [R,H]
                prefix = torch.stack([r[0] for r in out_rows], dim=0)  # [R,H]
                prefix_list.append(prefix)

                row_pos_all.append(row_pos_layers)
                row_mask_all.append(row_msk_layers)
                col_pos_all.append(col_pos_layers)
                col_mask_all.append(col_msk_layers)

            # ---- (B) 构造 Qwen 侧的输入（inputs_embeds = prefix + text_embeds）----
            # 这里的“任务”很简单：让模型复写 prompt_text 本身，作为最小训练目标
            # 你可以改成任意你想要的监督形式
            prompt_texts = []
            for sample in batch:
                # 例如把每行抽取文本拼起来当 prompt
                lines = []
                for ids in sample:
                    text = tok.decode(ids, skip_special_tokens=True)
                    lines.append(text)
                prompt_texts.append(" | ".join(lines))

            inputs_embeds_list = []
            labels_list        = []
            attn_mask_list     = []
            position_ids_list  = []

            for prefix, ptxt in zip(prefix_list, prompt_texts):
                R, Hc = prefix.shape
                # 正文 ids & embeds
                input_ids = tok(ptxt, return_tensors="pt").input_ids.to(DEVICE)   # [1,T]
                text_emb  = qwen.get_input_embeddings()(input_ids)               # [1,T,H]
                # 拼接
                prefix_emb = prefix.unsqueeze(0)                                  # [1,R,H]
                inputs_emb = torch.cat([prefix_emb, text_emb], dim=1)             # [1,R+T,H]
                inputs_embeds_list.append(inputs_emb)

                # labels：前缀部分 -100（不计损失），正文部分= input_ids
                T = input_ids.size(1)
                labels = torch.full((1, R+T), fill_value=-100, device=DEVICE, dtype=torch.long)
                labels[:, R:] = input_ids
                labels_list.append(labels)

                # mask & position_ids
                attn_mask = torch.ones(1, R+T, device=DEVICE, dtype=torch.long)
                pos_ids   = torch.arange(R+T, device=DEVICE).unsqueeze(0)
                attn_mask_list.append(attn_mask)
                position_ids_list.append(pos_ids)

            # 拼 batch 维
            inputs_embeds = torch.cat(inputs_embeds_list, dim=0)  # [B, R+T, H]
            labels        = torch.cat(labels_list, dim=0)         # [B, R+T]
            attention_mask= torch.cat(attn_mask_list, dim=0)      # [B, R+T]
            position_ids  = torch.cat(position_ids_list, dim=0)   # [B, R+T]

            # ---- (C) 前向 & 反传 ----
            out = qwen(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                labels=labels
            )
            loss = out.loss
            loss.backward()
            optimizer.step()

            global_step += 1
            if global_step % 10 == 0:
                print(f"step {global_step} | loss {loss.item():.4f}")

        print(f"Epoch {epoch+1}/{EPOCHS} done.")

    # 保存
    os.makedirs("ckpt_min", exist_ok=True)
    torch.save(qwen.state_dict(), "ckpt_min/qwen.pt")
    torch.save(lcm.state_dict(),  "ckpt_min/lcm.pt")
    print("Saved to ckpt_min/")

if __name__ == "__main__":
    main()