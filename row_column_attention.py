# row_column_attention: Use @Qwen
import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Dict
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3Config, Qwen3RotaryEmbedding

class RowColumnAttention(nn.Module):
    def __init__(self, config: Qwen3Config):
        """
        Initialize row and column attention module.

        Args:
            config: Qwen3Config
        """
        super().__init__()
        self.config = config
        self.row_attention_layer = Qwen3DecoderLayer(config=config, layer_idx=0)
        self.column_attention_layer = Qwen3DecoderLayer(config=config, layer_idx=1)
        self._weights_loaded = False

    def load_weights_once(self, row_weights: Dict, col_weights: Dict):
        """
        Load weights for the decoder layer(Class: Qwen3DecoderLayer) of row attention and column attention.

        Args:
            row_weights: decoder layer weights of row attention part
            column_weights: decoder layer weights of column attention part
        """
        self.row_attention_layer.load_state_dict(row_weights)
        self.column_attention_layer.load_state_dict(col_weights)
        self._weights_loaded = True

    def forward(
        self,
        embedded_rows: List[torch.Tensor],
        row_attention_mask: Optional[List[torch.Tensor]] = None,
        column_attention_mask: Optional[torch.Tensor] = None,
        row_position_embeddings: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        column_position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        row_layer_weights: Optional[Dict[str, torch.Tensor]] = None,
        col_layer_weights: Optional[Dict[str, torch.Tensor]] = None
    ):
        """
        Args:
            embedded_rows: Hidden state of the dictionary. Each element representing a row of the dictionary, with shape of (seq_len, hidden_size). The first element should be compress token instead.

            row_attention_mask: Attention mask of each row, which should be ones tensor. 

            column_attention_mask: Attention mask of the first column(compress token).

            row_position_embeddings: RoPE of each row, which should have position info, because the tokens within a row are order sensitive.

            column_position_embeddings: RoPE of the first column, which should not have position info, because the compress tokens are not order sensitive.

            row_layer_weights: Weights should be loaded for the decoder layer of row attention.

            column_layer_weights: Weights should be loaded for the decoder layer of column attention.

        """
        
        # for k, row_embed in enumerate(embedded_rows):
        # print(f">>> row[{k}].device =", row_embed.device)
        # print(">>> row_layernorm.weight.device =", self.row_attention_layer.input_layernorm.weight.device)

        if not self._weights_loaded:
            if row_layer_weights is None:
                raise ValueError("Qwen3DecoderLayer weights should be provided for row_attention_layer in param 'row_layer_weights'")
            if col_layer_weights is None:
                raise ValueError("Qwen3DecoderLayer weights should be provided for column_attention_layer in param 'col_layer_weights'")
            self.load_weights_once(row_layer_weights, col_layer_weights)

        if row_attention_mask is None:
            row_attention_mask = [None] * len(embedded_rows)
        
        if row_position_embeddings is None or column_position_embeddings is None:
            raise ValueError("row_position_embeddings and column_position_embeddings must be provided")
        
        if len(row_position_embeddings) != len(embedded_rows):
            raise ValueError("The number of row_position_embeddings must be the same as the number of embedded_rows")
        
        
        processed_rows = []
        container_tokens = []

        for row_embed, row_mask, row_position_embedding in zip(embedded_rows, row_attention_mask, row_position_embeddings):
            if row_embed.dim() == 2:
                row_input = row_embed.unsqueeze(0) # shape -> [1, seq_len, hidden_size]
            else:
                row_input = row_embed

            seq_len = row_input.shape[1]
            position_ids = torch.arange(seq_len, device=row_input.device).unsqueeze(0)

            row_output = self.row_attention_layer(
                hidden_states=row_input,
                attention_mask=row_mask,
                position_ids=position_ids,
                position_embeddings=row_position_embedding
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

        col_output = self.column_attention_layer(
            hidden_states=container_input,
            attention_mask=column_attention_mask,
            position_ids=col_position_ids,
            position_embeddings=column_position_embeddings
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
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
    import torch
    
    print("Testing RowColumnAttention...")
    try:
        print("Load Qwen3 Model and Config...")
        try:
            qwen_model = Qwen3ForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
            config = qwen_model.config
            qwen_rotary_emb = qwen_model.model.rotary_emb
            print("Success: Load Qwen3 Model and Config")
            
        except Exception as e:
            print(f"Failed to Load Qwen3 Model and Config: {e}")
            print("Use Fake Config instead...")
            config = Qwen3Config(
                vocab_size=32000,
                hidden_size=1024,
                num_attention_heads=16,
                num_key_value_heads=8,
                num_hidden_layers=24,
                intermediate_size=2048,
                max_position_embeddings=32768,
                rope_theta=10000.0
            )
            qwen_rotary_emb = None
        
        processor = RowColumnAttention(config)
        print("RowColumnAttention Created...")
        
        embedded_rows = [
            torch.randn(4, config.hidden_size),  
            torch.randn(3, config.hidden_size),  
            torch.randn(5, config.hidden_size),  
        ]
        
        device = embedded_rows[0].device
        print(f"Test Data Prepared (Device: {device})")
        print(f"   - Row Length: {[row.shape[0] for row in embedded_rows]}")
        
        print("Start getting the weights...")
        try:
            if qwen_rotary_emb is not None:
                row_weights = qwen_model.model.layers[0].state_dict()
                col_weights = qwen_model.model.layers[1].state_dict()
                print("Use Real Qwen3 Weights")
            else:
                raise Exception("No real model")
        except:
            print("Failed to load Qwen3 Weights, Use fake Weights instead...")
            row_weights = {name: torch.randn_like(param) for name, param in processor.row_attention_layer.named_parameters()}
            col_weights = {name: torch.randn_like(param) for name, param in processor.column_attention_layer.named_parameters()}
        
        print("Creating Row and Column RoPE...")
        
        row_position_embeddings = []
        for row in embedded_rows:
            seq_len = row.shape[0]
            dummy_input = torch.randn(1, seq_len, config.hidden_size, device=device)
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
            if qwen_rotary_emb is not None:
                pos_emb = qwen_rotary_emb(dummy_input, position_ids)
            else:
                rotary_emb = Qwen3RotaryEmbedding(config=config)
                pos_emb = rotary_emb(dummy_input, position_ids)
            row_position_embeddings.append(pos_emb)
            print(f"   - Row Rope (Length{seq_len}): {[p.shape for p in pos_emb]}")
        
        num_rows = len(embedded_rows)
        col_dummy_input = torch.randn(1, num_rows, config.hidden_size, device=device)
        col_position_ids = torch.arange(num_rows, device=device).unsqueeze(0)
        if qwen_rotary_emb is not None:
            column_position_embeddings = qwen_rotary_emb(col_dummy_input, col_position_ids)
        else:
            rotary_emb = Qwen3RotaryEmbedding(config=config)
            column_position_embeddings = rotary_emb(col_dummy_input, col_position_ids)
        
        print(f"   - Column RoPE (Length{num_rows}): {[p.shape for p in column_position_embeddings]}")
        
        print("Testing Forward Propagation...")
        try:
            output_rows = processor(
                embedded_rows=embedded_rows,
                row_layer_weights=row_weights,
                col_layer_weights=col_weights,
                row_position_embeddings=row_position_embeddings,
                column_position_embeddings=column_position_embeddings
            )
            
            print("Success: Forward Propagation!")
            print(f"   - Input Row number: {len(embedded_rows)}")
            print(f"   - Output Row number: {len(output_rows)}")
            
            for i, (inp, out) in enumerate(zip(embedded_rows, output_rows)):
                print(f"   - Row{i}: {inp.shape} → {out.shape}")
            
            print("Success: Testing")
            
        except Exception as e:
            print(f"Failed to Forward Propagation: {e}")
            import traceback
            traceback.print_exc()
        
    except Exception as e:
        print(f"Failed to Test: {e}")
        import traceback
        traceback.print_exc()