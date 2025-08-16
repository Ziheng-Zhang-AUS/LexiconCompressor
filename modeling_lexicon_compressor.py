# lexicon_compressor_model.py
import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Dict, Any
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config
from row_column_attention_stack import RowColumnAttentionStack

class LexiconCompressorModel(nn.Module):
    def __init__(self, config: Qwen3Config, num_attention_layers: int = 2):
        """
        词汇表压缩器模型
        
        Args:
            config: Qwen3配置
            num_attention_layers: 行列注意力层数量
        """
        super().__init__()
        self.config = config
        self.num_attention_layers = num_attention_layers
        
        # 行列注意力堆栈
        self.attention_stack = RowColumnAttentionStack(config, num_attention_layers)
        
        # embedding层参数占位符（实际权重通过load_embeddings_weights加载）
        self.embeddings = None
        self._embeddings_loaded = False
        
    def load_embeddings_weights(self, embedding_weights: Dict[str, torch.Tensor]):
        """
        加载embedding层权重
        
        Args:
            embedding_weights: embedding层的权重字典
        """
        # 创建embedding层并加载权重
        vocab_size = embedding_weights['weight'].shape[0]
        hidden_size = embedding_weights['weight'].shape[1]
        
        self.embeddings = nn.Embedding(vocab_size, hidden_size)
        self.embeddings.load_state_dict(embedding_weights)
        self.embeddings.to(torch.device("cuda")) 
        self._embeddings_loaded = True
    
    def load_attention_weights(self, attention_weights_list: List[Tuple[Dict, Dict]]):
        """
        加载所有注意力层的权重
        
        Args:
            attention_weights_list: 注意力层权重列表，每个元素是(row_weights, col_weights)元组
        """
        self.attention_stack.load_weights_once(attention_weights_list)
    
    def tokenize_and_embed(self, token_ids_list: List[List[int]]) -> List[torch.Tensor]:
        """
        将token IDs转换为嵌入向量
        
        Args:
            token_ids_list: List[List[int]] 每行的token IDs
            
        Returns:
            List[torch.Tensor]: 每行的嵌入向量列表
        """
        if not self._embeddings_loaded:
            raise ValueError("Embeddings weights must be loaded first")
        
        embedded_rows = []
        for token_ids in token_ids_list:
            # 转换为tensor
            token_tensor = torch.tensor(token_ids, dtype=torch.long, device=self.embeddings.weight.device)
            # 获取嵌入
            embedded = self.embeddings(token_tensor)  # shape: [seq_len, hidden_size]
            embedded_rows.append(embedded)
        
        return embedded_rows
    
    def forward(
        self,
        token_ids_list: List[List[int]],  # 输入的token IDs列表
        attention_weights: Optional[List[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]]] = None,
        embeddings_weights: Optional[Dict[str, torch.Tensor]] = None,
        row_attention_masks: Optional[List[List[torch.Tensor]]] = None,
        column_attention_masks: Optional[List[torch.Tensor]] = None,
        row_position_embeddings: Optional[List[List[Tuple[torch.Tensor, torch.Tensor]]]] = None,
        column_position_embeddings: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ):
        """
        前向传播 - 词汇表压缩
        
        Args:
            token_ids_list: List[List[int]] 每行的token IDs
            attention_weights: 注意力层权重列表
            embeddings_weights: embedding层权重
            其他注意力相关参数...
        """
        
        # 加载权重（如果提供了的话）
        if embeddings_weights is not None and not self._embeddings_loaded:
            self.load_embeddings_weights(embeddings_weights)
        
        if attention_weights is not None:
            self.load_attention_weights(attention_weights)
        
        # 1. Tokenize and embed
        embedded_rows = self.tokenize_and_embed(token_ids_list)
        
        # 2. 通过行列注意力堆栈
        compressed_rows = self.attention_stack(
            embedded_rows=embedded_rows,
            row_attention_masks=row_attention_masks,
            column_attention_masks=column_attention_masks,
            row_position_embeddings=row_position_embeddings,
            column_position_embeddings=column_position_embeddings
        )
        
        return compressed_rows



def main():
    import torch
    from transformers import AutoTokenizer, Qwen3ForCausalLM
    from tokenization_lexicon import LexiconTokenizer

    MODEL_NAME = "Qwen/Qwen3-0.6B"
    CSV_PATH   = "cleaned_lexicon_tiny.csv"
    COLUMNS    = ["lexical_unit", "pos", "gloss", "variant"]
    NUM_LAYERS = 6
    SHOW_N     = 100

    torch.set_grad_enabled(False)

    # 1) tokenizer + [COMP]
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tok.add_tokens(["[COMP]"], special_tokens=False)
    comp_id = tok.convert_tokens_to_ids("[COMP]")

    # 2) CSV → token ids（每行前置 [COMP]）
    lt = LexiconTokenizer(
        csv_file_path=CSV_PATH,
        tokenizer=tok,
        column_names=COLUMNS,
        compress_token_id=comp_id,
        delimiter=",",
        add_special_tokens=False
    )
    token_ids_list = lt.process_lexicon()[:SHOW_N]
    assert token_ids_list, "No entries parsed from CSV."

    # 3) Qwen 权重（embedding / 层 / RoPE）
    qwen   = Qwen3ForCausalLM.from_pretrained(MODEL_NAME)
    cfg    = qwen.config
    rotary = qwen.model.rotary_emb
    H      = cfg.hidden_size

    lcm = LexiconCompressorModel(config=cfg, num_attention_layers=NUM_LAYERS)
    emb_weights = qwen.get_input_embeddings().state_dict()
    attn_weights = []
    for i in range(NUM_LAYERS):
        row_sd = qwen.model.layers[2*i].state_dict()
        col_sd = qwen.model.layers[2*i+1].state_dict()
        attn_weights.append((row_sd, col_sd))

    # --------- helpers: 构造 RoPE / mask，并前向一次 ----------
    def full_vis_mask(L: int):
        return torch.zeros(1, 1, L, L, dtype=torch.float32)

    def row_rope(L: int):
        dummy = torch.zeros(1, L, H)
        pos   = torch.arange(L).unsqueeze(0)
        return rotary(dummy, pos)  # 有序 RoPE

    def col_identity_rope(N: int):
        dummy = torch.zeros(1, N, H)
        pos0  = torch.zeros(1, N, dtype=torch.long)
        return rotary(dummy, pos0)  # identity（无序）

    def build_inputs(tids):
        # 行：每层都要一份（简单复制）
        row_pos = []; row_msk = []
        for _ in range(NUM_LAYERS):
            row_pos.append([row_rope(len(x)) for x in tids])
            row_msk.append([full_vis_mask(len(x)) for x in tids])
        # 列：每层一份
        N = len(tids)
        col_pos = [col_identity_rope(N) for _ in range(NUM_LAYERS)]
        col_msk = [full_vis_mask(N)     for _ in range(NUM_LAYERS)]
        return row_pos, row_msk, col_pos, col_msk

    def fwd_once(tids):
        row_pos, row_msk, col_pos, col_msk = build_inputs(tids)
        out_rows = lcm(
            token_ids_list=tids,
            attention_weights=attn_weights,
            embeddings_weights=emb_weights,
            row_attention_masks=row_msk,
            column_attention_masks=col_msk,
            row_position_embeddings=row_pos,
            column_position_embeddings=col_pos
        )
        # 取每行容器（行首 [COMP]）向量
        heads = torch.stack([r[0].detach().cpu() for r in out_rows], dim=0)
        return out_rows, heads
    # ------------------------------------------------------------

    print("Forward once (baseline)...")
    out_rows, heads = fwd_once(token_ids_list)
    for i, (ids, row) in enumerate(zip(token_ids_list, out_rows)):
        print(f"Row {i}: ids_len={len(ids)} -> out_shape={tuple(row.shape)}; head_norm={row[0].norm().item():.4f}")

    # ============== 测试 1：列无序（列应对行顺序不敏感） ==============
    import random
    perm = list(range(len(token_ids_list)))
    random.shuffle(perm)
    perm_tids = [token_ids_list[i] for i in perm]
    _, heads_perm = fwd_once(perm_tids)

    assert torch.allclose(heads_perm, heads[perm], atol=1e-5, rtol=1e-5), \
        "Column invariance FAILED: shuffling rows changed container vectors beyond permutation."
    print("Test#1 Column invariance: OK ✅")

    # ============ 测试 2：行有序（行内交换 token 应改变输出） ============
    # 选第一行，尝试交换两个非 [COMP] 的位置（确保长度≥3）
    swapped_tids = [x[:] for x in token_ids_list]
    if len(swapped_tids[0]) >= 3:
        i0, j0 = 1, 2  # 交换第1/2个真实 token（0 是 [COMP]）
        swapped_tids[0][i0], swapped_tids[0][j0] = swapped_tids[0][j0], swapped_tids[0][i0]
        _, heads_swapped = fwd_once(swapped_tids)
        assert not torch.allclose(heads_swapped[0], heads[0], atol=1e-6), \
            "Row order FAILED: swapping tokens did not change row container."
        print("Test#2 Row order sensitivity: OK ✅")
    else:
        print("Test#2 Row order sensitivity: SKIP (row too short)")

    # ====== 测试 3：非因果（右侧信息能影响左侧容器 [COMP]） ======
    # 在第一行，把靠右的一个 token 替换成随机向量对应的“影子 token id”（这里简单用重复交换来模拟变化）
    # 更直接的做法是替换一个右侧 token（i>=2）为另一个 id
    noncausal_tids = [x[:] for x in token_ids_list]
    if len(noncausal_tids[0]) >= 4:
        # 交换更靠右的位置，查看 [COMP] 是否受影响
        i1, j1 = 2, 3
        noncausal_tids[0][i1], noncausal_tids[0][j1] = noncausal_tids[0][j1], noncausal_tids[0][i1]
        _, heads_noncausal = fwd_once(noncausal_tids)
        assert not torch.allclose(heads_noncausal[0], heads[0], atol=1e-6), \
            "Non-causal FAILED: changing right-side tokens did not affect [COMP]."
        print("Test#3 Non-causal (full visibility): OK ✅")
    else:
        print("Test#3 Non-causal (full visibility): SKIP (row too short)")

    print("All tests passed 🎉")

if __name__ == "__main__":
    main()