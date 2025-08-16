# row_column_attention_stack.py
import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Dict, Any
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config
from row_column_attention import RowColumnAttention

class RowColumnAttentionStack(nn.Module):
    def __init__(self, config: Qwen3Config, num_layers: int):
        """
        初始化RowColumnAttention堆栈
        
        Args:
            config: Qwen3配置
            num_layers: 堆栈层数
        """
        super().__init__()
        self.config = config
        self.num_layers = num_layers
        
        # 创建多个RowColumnAttention层
        self.attention_layers = nn.ModuleList([
            RowColumnAttention(config) for _ in range(num_layers)
        ])
        
        self._weights_loaded = False
    
    def load_weights_once(self, layer_weights: List[Tuple[Dict, Dict]]):
        """
        为所有层加载权重
        
        Args:
            layer_weights: 每个元素是一个元组 (row_weights, col_weights)
        """
        if len(layer_weights) != self.num_layers:
            raise ValueError(f"Expected {self.num_layers} weight pairs, got {len(layer_weights)}")
        
        for i, (row_weights, col_weights) in enumerate(layer_weights):
            self.attention_layers[i].load_weights_once(row_weights, col_weights)
        
        self._weights_loaded = True
    
    def forward(
        self,
        embedded_rows: List[torch.Tensor],
        row_attention_masks: Optional[List[List[torch.Tensor]]] = None,
        column_attention_masks: Optional[List[torch.Tensor]] = None,
        row_position_embeddings: Optional[List[List[Tuple[torch.Tensor, torch.Tensor]]]] = None,
        column_position_embeddings: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        layer_weights: Optional[List[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]]] = None
    ):
        """
        前向传播
        
        Args:
            embedded_rows: 输入的行嵌入列表
            row_attention_masks: 每层的行注意力掩码列表
            column_attention_masks: 每层的列注意力掩码列表
            row_position_embeddings: 每层的行位置嵌入列表
            column_position_embeddings: 每层的列位置嵌入列表
            layer_weights: 每层的权重参数列表
        """
        
        # 如果还没有加载权重且提供了权重参数
        if not self._weights_loaded:
            if layer_weights is None:
                raise ValueError("Weights should be provided in param 'layer_weights'")
            self.load_weights_once(layer_weights)
        
        # 初始化参数
        if row_attention_masks is None:
            row_attention_masks = [None] * self.num_layers
        if column_attention_masks is None:
            column_attention_masks = [None] * self.num_layers
        if row_position_embeddings is None:
            row_position_embeddings = [None] * self.num_layers
        if column_position_embeddings is None:
            column_position_embeddings = [None] * self.num_layers
        
        # 验证参数长度
        if len(row_attention_masks) != self.num_layers:
            raise ValueError(f"Expected {self.num_layers} row_attention_masks, got {len(row_attention_masks)}")
        if len(column_attention_masks) != self.num_layers:
            raise ValueError(f"Expected {self.num_layers} column_attention_masks, got {len(column_attention_masks)}")
        if len(row_position_embeddings) != self.num_layers:
            raise ValueError(f"Expected {self.num_layers} row_position_embeddings, got {len(row_position_embeddings)}")
        if len(column_position_embeddings) != self.num_layers:
            raise ValueError(f"Expected {self.num_layers} column_position_embeddings, got {len(column_position_embeddings)}")
        
        # 逐层处理
        current_rows = embedded_rows
        
        for i in range(self.num_layers):
            current_rows = self.attention_layers[i](
                embedded_rows=current_rows,
                row_attention_mask=row_attention_masks[i],
                column_attention_mask=column_attention_masks[i],
                row_position_embeddings=row_position_embeddings[i],
                column_position_embeddings=column_position_embeddings[i]
            )
        
        return current_rows


if __name__ == "__main__":
    import torch
    from transformers import Qwen3ForCausalLM
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
    from row_column_attention_stack import RowColumnAttentionStack  # 与当前文件同目录
    from row_column_attention import RowColumnAttention

    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) 加载 Qwen 模型与配置/rotary
    model = Qwen3ForCausalLM.from_pretrained("Qwen/Qwen3-0.6B").to(device)
    config = model.config
    rotary = model.model.rotary_emb  # Qwen 自带 RoPE

    # 2) 准备假输入（3 行，长度不同），形如 [L_i, H]
    H = config.hidden_size
    rows = [
        torch.randn(4, H, device=device),
        torch.randn(6, H, device=device),
        torch.randn(5, H, device=device),
    ]

    # 3) 工具函数
    def full_vis_mask(L: int):
        # [B, 1, Q, K] 全 0（非因果全可见）
        return torch.zeros(1, 1, L, L, device=device, dtype=torch.float32)

    def rope_for_row(L: int):
        dummy = torch.zeros(1, L, H, device=device)
        pos_ids = torch.arange(L, device=device).unsqueeze(0)
        return rotary(dummy, pos_ids)  # (cos, sin)

    def rope_identity(L: int):
        dummy = torch.zeros(1, L, H, device=device)
        pos_ids0 = torch.zeros(1, L, dtype=torch.long, device=device)
        return rotary(dummy, pos_ids0)  # cos=1, sin=0

    # 4) 构造 2 层 Stack，并准备各层权重（row/col 对应 Qwen 的不同行）
    num_layers = 2
    stack = RowColumnAttentionStack(config=config, num_layers=num_layers).to(device)

    # 用 Qwen 的连续层作初始化：第 i 层用 (2*i, 2*i+1)
    layer_weights = []
    for i in range(num_layers):
        row_sd = model.model.layers[2*i].state_dict()
        col_sd = model.model.layers[2*i + 1].state_dict()
        layer_weights.append((row_sd, col_sd))

    # 5) 为每层准备位置编码与 mask
    # 行：每层需要一个“每行一个 (cos,sin)”的列表
    row_pos_embs_per_layer = []
    row_masks_per_layer = []
    for _ in range(num_layers):
        row_pos_embs = [rope_for_row(r.size(0)) for r in rows]
        row_masks = [full_vis_mask(r.size(0)) for r in rows]
        row_pos_embs_per_layer.append(row_pos_embs)
        row_masks_per_layer.append(row_masks)

    # 列：每层一个 (cos,sin) 与一个 mask
    col_pos_embs_per_layer = []
    col_masks_per_layer = []
    N = len(rows)
    for _ in range(num_layers):
        col_pos_embs_per_layer.append(rope_identity(N))  # 无序 → identity RoPE
        col_masks_per_layer.append(full_vis_mask(N))

    # 6) 前向：打印每层前后行首 token 的范数变化以示生效
    print(">>> Forward through RowColumnAttentionStack")
    out_rows = stack(
        embedded_rows=rows,
        row_attention_masks=row_masks_per_layer,
        column_attention_masks=col_masks_per_layer,
        row_position_embeddings=row_pos_embs_per_layer,
        column_position_embeddings=col_pos_embs_per_layer,
        layer_weights=layer_weights
    )

    # 打印形状与行首向量范数
    for i, (inp, out) in enumerate(zip(rows, out_rows)):
        print(f"Row {i}: {tuple(inp.shape)} -> {tuple(out.shape)}; "
              f"head-norm before={inp[0].norm().item():.4f}, after={out[0].norm().item():.4f}")

    print("Done.")