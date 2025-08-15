# row_column_attention: Use @Qwen
import torch
import torch.nn as nn
from typing import List, Optional, Tuple
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3Config, Qwen3RotaryEmbedding

class RowColumnAttention(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config
        self.row_attention_layer = Qwen3DecoderLayer(config=config, layer_idx=0)
        self.column_attention_layer = Qwen3DecoderLayer(config=config, layer_idx=1)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)

    def forward(
        self,
        embedded_rows: List[torch.Tensor],
        row_attention_mask: Optional[List[torch.Tensor]] = None,
        column_attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ):
        if row_attention_mask is None:
            row_attention_mask = [None] * len(embedded_rows)
        
        
        processed_rows = []
        container_tokens = []

        for row_embed, row_mask in zip(embedded_rows, row_attention_mask):
            if row_embed.dim() == 2:
                row_input = row_embed.unsqueeze(0) # shape -> [1, seq_len, hidden_size]
            else:
                row_input = row_embed

            seq_len = row_input.shape[1]
            position_ids = torch.arange(seq_len, device=row_input.device).unsqueeze(0)

            if position_embeddings is None:
                pos_emb = self.rotary_emb(row_input, position_ids)
            else:
                pos_emb = position_embeddings
            # pos_emb = self.rotary_emb(row_input, position_ids)

            row_output = self.row_attention_layer(
                hidden_states=row_input,
                attention_mask=row_mask,
                position_ids=position_ids,
                position_embeddings=pos_emb
            )

            row_output = row_output[0] # QwenDecoderLayer returns a tuple, (hidden_state, attention_weight)
            processed_row = row_output.squeeze(0) # shape -> [seq_len, hidden_size]
            processed_rows.append(processed_row)

            container_token = processed_row[0:1] # shape -> [1, hidden_size]
            container_tokens.append(container_token)
        
        container_tensor = torch.cat(container_tokens, dim=0)
        container_input = container_tensor.unsqueeze(0) # shape -> [1, num_rows, hidden_size]
        num_rows = container_input.shape[1]
        col_position_ids = torch.arange(num_rows, device=container_input.device).unsqueeze(0)

        if position_embeddings is None:
            col_pos_emb = self.rotary_emb(container_input, col_position_ids)
        else:
            col_pos_emb = position_embeddings
        # col_pos_emb = self.rotary_emb(container_input, col_position_ids)

        col_output = self.column_attention_layer(
            hidden_states=container_input,
            attention_mask=column_attention_mask,
            position_ids=col_position_ids,
            position_embeddings=col_pos_emb
        )
        col_output = col_output[0]
        final_containers = col_output.squeeze(0) # shape -> [num_rows, hidden_size]

        updated_rows = []
        for i, original_row in enumerate(processed_rows):
            updated_row = original_row.clone()
            updated_row[0] = final_containers[i]
            updated_rows.append(updated_row)
        
        return updated_rows

if __name__ == "__main__":
    from transformers import Qwen3Config
    
    config = Qwen3Config(
        vocab_size=32000,
        hidden_size=512,
        num_attention_heads=8,
        num_key_value_heads=4,
        num_hidden_layers=32,  
        intermediate_size=1024,
        max_position_embeddings=2048
    )
    
    processor = RowColumnAttention(config)
    
    # rows with different seq_len
    embedded_rows = [
        torch.randn(5, 512),  
        torch.randn(3, 512),  
        torch.randn(7, 512),  
    ]
    
    print("input shape: ")
    for i, row in enumerate(embedded_rows):
        print(f"  row{i}: {row.shape}")
    
    try:
        output_rows = processor(embedded_rows=embedded_rows)
        
        print("\noutput shape:")
        for i, row in enumerate(output_rows):
            print(f"  row{i}: {row.shape}")
            
        print("\nSuccess!")
        
    except Exception as e:
        print(f"Fail: {e}")
        import traceback
        traceback.print_exc()