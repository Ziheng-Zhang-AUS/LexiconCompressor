# row_column_attention_stack.py
import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Dict, Any
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config
from row_column_attention import RowColumnAttention

class RowColumnAttentionStack(nn.Module):
    def __init__(self, config: Qwen3Config, num_layers: int):
        """
        Initialize row_column_attention_stack
        
        Args:
            config: Qwen3Config
            num_layers: number of row_column_attention layers
        """
        super().__init__()
        self.config = config
        self.num_layers = num_layers
        
        self.attention_layers = nn.ModuleList([
            RowColumnAttention(config) for _ in range(num_layers)
        ])
        
        self._weights_loaded = False
    
    def load_weights_once(self, layer_weights: List[Tuple[Dict, Dict]]):
        """
        Load weigths for all the row_column_attention layers once.
        
        Args:
            layer_weights: Weights to be loaded. Each element is a tuple of weight(Dict), for the row attention layer and column attention layer.
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
        Forward Propagation

        Args:
            embedded_rows: List of input row embeddings

            row_attention_masks: List of row attention masks for each layer

            column_attention_masks: List of column attention masks for each layer

            row_position_embeddings: List of row position embeddings for each layer

            column_position_embeddings: List of column position embeddings for each layer

            layer_weights: List of weight parameters for each layer
        """
        
        if not self._weights_loaded:
            if layer_weights is None:
                raise ValueError("Weights should be provided in param 'layer_weights'")
            self.load_weights_once(layer_weights)
        
        if row_attention_masks is None:
            row_attention_masks = [None] * self.num_layers
        if column_attention_masks is None:
            column_attention_masks = [None] * self.num_layers
        if row_position_embeddings is None:
            row_position_embeddings = [None] * self.num_layers
        if column_position_embeddings is None:
            column_position_embeddings = [None] * self.num_layers
        
        if len(row_attention_masks) != self.num_layers:
            raise ValueError(f"Expected {self.num_layers} row_attention_masks, got {len(row_attention_masks)}")
        if len(column_attention_masks) != self.num_layers:
            raise ValueError(f"Expected {self.num_layers} column_attention_masks, got {len(column_attention_masks)}")
        if len(row_position_embeddings) != self.num_layers:
            raise ValueError(f"Expected {self.num_layers} row_position_embeddings, got {len(row_position_embeddings)}")
        if len(column_position_embeddings) != self.num_layers:
            raise ValueError(f"Expected {self.num_layers} column_position_embeddings, got {len(column_position_embeddings)}")
        
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
    from row_column_attention_stack import RowColumnAttentionStack  
    from row_column_attention import RowColumnAttention

    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Load the Qwen model and configure /rotary
    model = Qwen3ForCausalLM.from_pretrained("Qwen/Qwen3-0.6B").to(device)
    config = model.config
    rotary = model.model.rotary_emb 

    # 2) Fake input
    H = config.hidden_size
    rows = [
        torch.randn(4, H, device=device),
        torch.randn(6, H, device=device),
        torch.randn(5, H, device=device),
    ]

    # 3) utils
    def full_vis_mask(L: int):
        return torch.zeros(1, 1, L, L, device=device, dtype=torch.float32)

    def rope_for_row(L: int):
        dummy = torch.zeros(1, L, H, device=device)
        pos_ids = torch.arange(L, device=device).unsqueeze(0)
        return rotary(dummy, pos_ids) 

    def rope_identity(L: int):
        dummy = torch.zeros(1, L, H, device=device)
        pos_ids0 = torch.zeros(1, L, dtype=torch.long, device=device)
        return rotary(dummy, pos_ids0) 

    # 4) 2-layers-stack
    num_layers = 2
    stack = RowColumnAttentionStack(config=config, num_layers=num_layers).to(device)
    layer_weights = []
    for i in range(num_layers):
        row_sd = model.model.layers[2*i].state_dict()
        col_sd = model.model.layers[2*i + 1].state_dict()
        layer_weights.append((row_sd, col_sd))

    # 5) RoPE and Mask
    row_pos_embs_per_layer = []
    row_masks_per_layer = []
    for _ in range(num_layers):
        row_pos_embs = [rope_for_row(r.size(0)) for r in rows]
        row_masks = [full_vis_mask(r.size(0)) for r in rows]
        row_pos_embs_per_layer.append(row_pos_embs)
        row_masks_per_layer.append(row_masks)

    col_pos_embs_per_layer = []
    col_masks_per_layer = []
    N = len(rows)
    for _ in range(num_layers):
        col_pos_embs_per_layer.append(rope_identity(N))  # Identity since column should not be order sensitive
        col_masks_per_layer.append(full_vis_mask(N))

    # 6) forward
    print(">>> Forward through RowColumnAttentionStack")
    out_rows = stack(
        embedded_rows=rows,
        row_attention_masks=row_masks_per_layer,
        column_attention_masks=col_masks_per_layer,
        row_position_embeddings=row_pos_embs_per_layer,
        column_position_embeddings=col_pos_embs_per_layer,
        layer_weights=layer_weights
    )

    # check shape and norm
    for i, (inp, out) in enumerate(zip(rows, out_rows)):
        print(f"Row {i}: {tuple(inp.shape)} -> {tuple(out.shape)}; "
              f"head-norm before={inp[0].norm().item():.4f}, after={out[0].norm().item():.4f}")

    print("Done.")