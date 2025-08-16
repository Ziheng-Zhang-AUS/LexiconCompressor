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

    # 固定配置
    MODEL_NAME = "Qwen/Qwen3-0.6B"
    CSV_PATH   = "lexicon_demo.csv"
    COLUMNS    = ["lexical_unit", "pos", "gloss", "variant"]
    NUM_LAYERS = 2       # Row/Column 注意力堆栈层数
    SHOW_N     = 3       # 取前N行做演示，防止太慢

    torch.set_grad_enabled(False)

    # 1) 加载分词器并添加 [COMP]
    print("Loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tok.add_tokens(["[COMP]"], special_tokens=False)
    comp_id = tok.convert_tokens_to_ids("[COMP]")
    print(f"[COMP] id = {comp_id}")

    # 2) 用你的 LexiconTokenizer 解析 CSV，得到每行 ids（前置 [COMP]）
    lt = LexiconTokenizer(
        csv_file_path=CSV_PATH,
        tokenizer=tok,
        column_names=COLUMNS,
        compress_token_id=comp_id,
        delimiter=",",
        add_special_tokens=False  # 词典行一般不自动加 special tokens
    )
    token_ids_list = lt.process_lexicon()
    if not token_ids_list:
        print("No entries parsed from CSV.")
        return
    token_ids_list = token_ids_list[:SHOW_N]
    print(f"Prepared {len(token_ids_list)} entries from CSV.")

    # 3) 加载 Qwen 模型（只拿权重与 RoPE，不做前向）
    print("Loading Qwen weights...")
    qwen = Qwen3ForCausalLM.from_pretrained(MODEL_NAME)
    cfg = qwen.config
    rotary = qwen.model.rotary_emb  # 用于生成 RoPE (cos,sin)

    # 4) 组装 LexiconCompressorModel，并加载 embedding/attention 权重
    lcm = LexiconCompressorModel(config=cfg, num_attention_layers=NUM_LAYERS)

    # embedding 权重来自 Qwen 的词嵌入
    emb_weights = qwen.get_input_embeddings().state_dict()  # {"weight": tensor}
    # 每层 (row_weights, col_weights)；这里用 Qwen 的第 0..(2*NUM_LAYERS-1) 层作为初始化
    attn_weights = []
    for i in range(NUM_LAYERS):
        row_sd = qwen.model.layers[2*i].state_dict()
        col_sd = qwen.model.layers[2*i + 1].state_dict()
        attn_weights.append((row_sd, col_sd))

    # 5) 构造 RoPE 与全可见 mask
    def full_vis_mask(L: int):
        # 4D 全 0：形状 [B, 1, Q, K]，非因果，全可见
        return torch.zeros(1, 1, L, L, dtype=torch.float32)

    def row_rope_for_len(L: int, H: int):
        dummy = torch.zeros(1, L, H)
        pos_ids = torch.arange(L).unsqueeze(0)
        return rotary(dummy, pos_ids)  # (cos, sin)

    def col_identity_rope(N: int, H: int):
        dummy = torch.zeros(1, N, H)
        pos_ids0 = torch.zeros(1, N, dtype=torch.long)
        return rotary(dummy, pos_ids0)  # cos=1, sin=0

    H = cfg.hidden_size
    # 行：每层都需要为“每一行”准备 (cos,sin) 与 mask
    row_pos_embs_per_layer = []
    row_masks_per_layer = []
    for _ in range(NUM_LAYERS):
        row_pos_embs = []
        row_masks = []
        for ids in token_ids_list:
            L = len(ids)
            row_pos_embs.append(row_rope_for_len(L, H))
            row_masks.append(full_vis_mask(L))
        row_pos_embs_per_layer.append(row_pos_embs)
        row_masks_per_layer.append(row_masks)

    # 列：每层一个 (cos,sin) 与 mask（长度为行数）
    N = len(token_ids_list)
    col_pos_embs_per_layer = [col_identity_rope(N, H) for _ in range(NUM_LAYERS)]
    col_masks_per_layer    = [full_vis_mask(N) for _ in range(NUM_LAYERS)]

    # 6) 前向：把 token ids、权重、RoPE、mask 一次性喂给压缩器
    print("Forward...")
    out_rows = lcm(
        token_ids_list=token_ids_list,
        attention_weights=attn_weights,
        embeddings_weights=emb_weights,
        row_attention_masks=row_masks_per_layer,
        column_attention_masks=col_masks_per_layer,
        row_position_embeddings=row_pos_embs_per_layer,
        column_position_embeddings=col_pos_embs_per_layer
    )

    # 7) 打印结果（形状与行首 token 范数变化）
    print("Results:")
    # 先手动生成一次“embedding 前”的行首向量范数做对比
    with torch.no_grad():
        emb_layer = lcm.embeddings
        in_head_norms = []
        for ids in token_ids_list:
            x = emb_layer(torch.tensor(ids, dtype=torch.long))  # [L,H]
            in_head_norms.append(x[0].norm().item())

    for i, (ids, row, in_norm) in enumerate(zip(token_ids_list, out_rows, in_head_norms)):
        print(f"Row {i}: ids_len={len(ids)} -> out_shape={tuple(row.shape)}; "
              f"head-norm before={in_norm:.4f}, after={row[0].norm().item():.4f}")

    print("Done.")

if __name__ == "__main__":
    main()