# # row_column_attention: Use @Qwen
# import torch
# import torch.nn as nn
# from typing import List, Optional, Tuple, Dict
# from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3Config, Qwen3RotaryEmbedding

# class RowColumnAttention(nn.Module):
#     def __init__(self, config: Qwen3Config):
#         """
#         Initialize row and column attention module.

#         Args:
#             config: Qwen3Config
#         """
#         super().__init__()
#         self.config = config
#         self.row_attention_layer = Qwen3DecoderLayer(config=config, layer_idx=0)
#         self.column_attention_layer = Qwen3DecoderLayer(config=config, layer_idx=1)
#         self._weights_loaded = False

#     def load_weights_once(self, row_weights: Dict, col_weights: Dict):
#         """
#         Load weights for the decoder layer(Class: Qwen3DecoderLayer) of row attention and column attention.

#         Args:
#             row_weights: decoder layer weights of row attention part
#             column_weights: decoder layer weights of column attention part
#         """
#         self.row_attention_layer.load_state_dict(row_weights)
#         self.column_attention_layer.load_state_dict(col_weights)
#         self._weights_loaded = True

#     def forward(
#         self,
#         embedded_rows: List[torch.Tensor],
#         row_attention_mask: Optional[List[torch.Tensor]] = None,
#         column_attention_mask: Optional[torch.Tensor] = None,
#         row_position_embeddings: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
#         column_position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
#         row_layer_weights: Optional[Dict[str, torch.Tensor]] = None,
#         col_layer_weights: Optional[Dict[str, torch.Tensor]] = None
#     ):
#         """
#         Args:
#             embedded_rows: Hidden state of the dictionary. Each element representing a row of the dictionary, with shape of (seq_len, hidden_size). The first element should be compress token instead.

#             row_attention_mask: Attention mask of each row, which should be ones tensor. 

#             column_attention_mask: Attention mask of the first column(compress token).

#             row_position_embeddings: RoPE of each row, which should have position info, because the tokens within a row are order sensitive.

#             column_position_embeddings: RoPE of the first column, which should not have position info, because the compress tokens are not order sensitive.

#             row_layer_weights: Weights should be loaded for the decoder layer of row attention.

#             column_layer_weights: Weights should be loaded for the decoder layer of column attention.

#         """
        
#         # for k, row_embed in enumerate(embedded_rows):
#         # print(f">>> row[{k}].device =", row_embed.device)
#         # print(">>> row_layernorm.weight.device =", self.row_attention_layer.input_layernorm.weight.device)

#         if not self._weights_loaded:
#             if row_layer_weights is None:
#                 raise ValueError("Qwen3DecoderLayer weights should be provided for row_attention_layer in param 'row_layer_weights'")
#             if col_layer_weights is None:
#                 raise ValueError("Qwen3DecoderLayer weights should be provided for column_attention_layer in param 'col_layer_weights'")
#             self.load_weights_once(row_layer_weights, col_layer_weights)

#         if row_attention_mask is None:
#             row_attention_mask = [None] * len(embedded_rows)
        
#         if row_position_embeddings is None or column_position_embeddings is None:
#             raise ValueError("row_position_embeddings and column_position_embeddings must be provided")
        
#         if len(row_position_embeddings) != len(embedded_rows):
#             raise ValueError("The number of row_position_embeddings must be the same as the number of embedded_rows")
        
        
#         processed_rows = []
#         container_tokens = []

#         for row_embed, row_mask, row_position_embedding in zip(embedded_rows, row_attention_mask, row_position_embeddings):
#             if row_embed.dim() == 2:
#                 row_input = row_embed.unsqueeze(0) # shape -> [1, seq_len, hidden_size]
#             else:
#                 row_input = row_embed

#             seq_len = row_input.shape[1]
#             position_ids = torch.arange(seq_len, device=row_input.device).unsqueeze(0)

#             row_output = self.row_attention_layer(
#                 hidden_states=row_input,
#                 attention_mask=row_mask,
#                 position_ids=position_ids,
#                 position_embeddings=row_position_embedding
#             )

#             row_output = row_output[0] # QwenDecoderLayer returns a tuple, (hidden_state, attention_weight)
#             processed_row = row_output.squeeze(0) # shape -> [seq_len, hidden_size]
#             processed_rows.append(processed_row)

#             container_token = processed_row[0:1] # shape -> [1, hidden_size]
#             container_tokens.append(container_token)
        
#         container_tensor = torch.cat(container_tokens, dim=0)
#         container_input = container_tensor.unsqueeze(0) # shape -> [1, num_rows, hidden_size]
#         num_rows = container_input.shape[1]
#         col_position_ids = torch.arange(num_rows, device=container_input.device).unsqueeze(0)

#         col_output = self.column_attention_layer(
#             hidden_states=container_input,
#             attention_mask=column_attention_mask,
#             position_ids=col_position_ids,
#             position_embeddings=column_position_embeddings
#         )
#         col_output = col_output[0]
#         final_containers = col_output.squeeze(0) # shape -> [num_rows, hidden_size]

#         updated_rows = []
#         for i, original_row in enumerate(processed_rows):
#             updated_row = original_row.clone()
#             updated_row[0] = final_containers[i]
#             updated_rows.append(updated_row)
        
#         return updated_rows


# row_column_attention: Use @Qwen
import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Dict
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer
from torch.nn.utils.rnn import pad_sequence
from configuration_lexicon_compressor import LexiconCompressorConfig

class RowColumnAttention(nn.Module):
    def __init__(self, config: LexiconCompressorConfig):
        """
        Initialize row and column attention module.
        """
        super().__init__()
        self.config = config
        self.row_attention_layer = Qwen3DecoderLayer(config=config.qwen_config, layer_idx=0)
        self.column_attention_layer = Qwen3DecoderLayer(config=config.qwen_config, layer_idx=1)

    def load_weights_once(self, row_weights: Dict, col_weights: Dict):
        """
        Load weights for the decoder layers.
        """
        if not self._weights_loaded:
            if row_weights is None:
                raise ValueError("row_layer_weights required")
            if col_weights is None:
                raise ValueError("col_layer_weights required")
        self.row_attention_layer.load_state_dict(row_weights)
        self.column_attention_layer.load_state_dict(col_weights)

    def forward(
    self,
    learned_tokens: torch.Tensor,        # (B, T, H)
    dict_tokens: List[torch.Tensor],     # len=B, each (Li, H)
    row_attention_mask: Optional[torch.Tensor] = None,
    column_attention_mask: Optional[torch.Tensor] = None,
    row_position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    column_position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
):

        B, T, H = learned_tokens.shape
        device = learned_tokens.device

        dict_lens = torch.tensor([seq.size(0) for seq in dict_tokens],
                                device=device, dtype=torch.long)  # (B,)

        dict_padded = pad_sequence(dict_tokens, batch_first=True, padding_value=0.0)  # (B, Ld_max, H)

        if self.config.learned_tokens_prepend:
            padded_sequences = torch.cat([learned_tokens, dict_padded], dim=1)  # (B, T+Ld_max, H)
        else:
            padded_sequences = torch.cat([dict_padded, learned_tokens], dim=1)  # (B, Ld_max+T, H)

        Lmax = padded_sequences.size(1)

        if row_attention_mask is None:
            lengths = dict_lens + T  # (B,)
            row_attention_mask = (
                torch.arange(Lmax, device=device).unsqueeze(0).expand(B, -1)
                < lengths.unsqueeze(1)
            ).to(padded_sequences.dtype)  # (B, Lmax)

        position_ids = torch.arange(Lmax, device=device).unsqueeze(0).expand(B, -1)  # (B, Lmax)

        row_outputs = self.row_attention_layer(
            hidden_states=padded_sequences,
            attention_mask=row_attention_mask,
            position_ids=position_ids,
            position_embeddings=row_position_embeddings
        )
        if isinstance(row_outputs, torch.Tensor):
            row_processed = row_outputs  # (B, Lmax, H)
        elif isinstance(row_outputs, (tuple, list)):
            row_processed = row_outputs[0]
        else:
            row_processed = getattr(row_outputs, "last_hidden_state", None)

        if self.config.learned_tokens_prepend:
            extracted_learned_tokens = row_processed[:, :T, :]  # (B, T, H)
            start_for_dict = T
        else:
            pos = dict_lens.unsqueeze(1) + torch.arange(T, device=device).unsqueeze(0)  # (B, T)
            idx = pos.unsqueeze(-1).expand(B, T, H)                                     # (B, T, H)
            extracted_learned_tokens = torch.gather(row_processed, dim=1, index=idx)   # (B, T, H)
            start_for_dict = 0

        offs = torch.arange(Lmax, device=device).unsqueeze(0).expand(B, -1)             # (B, Lmax)
        valid = (offs >= start_for_dict) & (offs < start_for_dict + dict_lens.unsqueeze(1))  # (B, Lmax)

        flat = row_processed[valid]                                                     # (sum(dict_lens), H)
        extracted_dict_tokens = list(torch.split(flat, dict_lens.tolist(), dim=0))      # len=B, each (Li, H)

        column_input = extracted_learned_tokens  # (B, T, H)

        col_position_ids = torch.arange(T, device=column_input.device).unsqueeze(0)     # (1, T)

        if column_attention_mask is None:
            column_attention_mask = torch.ones(B, T, device=column_input.device, dtype=column_input.dtype)  # (B, T)

        col_outputs = self.column_attention_layer(
            hidden_states=column_input,
            attention_mask=column_attention_mask,
            position_ids=col_position_ids,
            position_embeddings=column_position_embeddings
        )
        if isinstance(col_outputs, torch.Tensor):
            col_output = col_outputs  # (B, T, H)
        elif isinstance(col_outputs, (tuple, list)):
            col_output = col_outputs[0]
        else:
            col_output = getattr(col_outputs, "last_hidden_state", None)

        return col_output, extracted_dict_tokens


if __name__ == "__main__":
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
    from configuration_lexicon_compressor import LexiconCompressorConfig

    USE_CUSTOM_ROPE = True  # True: explicitly pass RoPE (B,L,Dh); False: rely on position_ids

    def build_batched_rope(rotary, position_ids: torch.LongTensor, head_dim: int, device, dtype):
        # position_ids: (B,L) -> cos,sin: (B,L,head_dim)
        B, L = position_ids.shape
        dummy = torch.ones(B, L, head_dim, device=device, dtype=dtype)
        cos, sin = rotary(dummy, position_ids)
        return cos.to(dtype), sin.to(dtype)

    print("Testing RowColumnAttention...")

    # ---- Load Qwen3 config ----
    try:
        qwen_model = Qwen3ForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
        qwen_config = qwen_model.config
        hidden_size = qwen_config.hidden_size
        num_heads   = qwen_config.num_attention_heads
        head_dim    = hidden_size // num_heads
        rotary_emb  = qwen_model.model.rotary_emb
        print("Loaded pretrained Qwen3 config")
    except Exception as e:
        print(f"Warning: failed to load pretrained config: {e}")
        qwen_config = Qwen3Config(
            vocab_size=32000,
            hidden_size=512,
            num_attention_heads=8,
            num_key_value_heads=4,
            num_hidden_layers=4,
            intermediate_size=2048,
            max_position_embeddings=2048,
            rope_theta=10000.0
        )
        hidden_size = qwen_config.hidden_size
        num_heads   = qwen_config.num_attention_heads
        head_dim    = hidden_size // num_heads
        rotary_emb  = Qwen3RotaryEmbedding(qwen_config)

    # ---- Instantiate our module ----
    config = LexiconCompressorConfig(qwen_config=qwen_config)
    config.learned_tokens_prepend = True   # test prepend=True case
    processor = RowColumnAttention(config)
    print("RowColumnAttention created")

    # ---- Prepare test data ----
    B = 2
    N = 3   # num learned tokens
    dict_lengths = [4, 7]
    learned_tokens = torch.randn(B, N, hidden_size)
    dict_tokens = [torch.randn(Li, hidden_size) for Li in dict_lengths]
    device = learned_tokens.device
    dtype  = learned_tokens.dtype

    print(f"learned_tokens: {learned_tokens.shape}")
    print(f"dict tokens lens: {[t.shape for t in dict_tokens]}")

    # ---- Row dimension setup ----
    L_row = max(l + N for l in dict_lengths)
    row_position_ids = torch.arange(L_row, device=device).unsqueeze(0).expand(B, -1)  # (B,L_row)

    if USE_CUSTOM_ROPE:
        row_cos, row_sin = build_batched_rope(rotary_emb, row_position_ids, head_dim, device, dtype)
        row_position_embeddings = (row_cos, row_sin)  # (B,L_row,Dh)
    else:
        row_position_embeddings = None

    # row mask: (B,1,L,S) with 1=keep, 0=mask
    row_attention_mask = torch.zeros(B, 1, L_row, L_row, device=device, dtype=dtype)
    for i, Ld in enumerate(dict_lengths):
        seq_len = Ld + N
        row_attention_mask[i, 0, :, :seq_len] = 1.0

    # ---- Column dimension setup ----
    col_position_ids = torch.arange(N, device=device).unsqueeze(0)  # (1,N)
    if USE_CUSTOM_ROPE:
        col_cos, col_sin = build_batched_rope(rotary_emb, col_position_ids, head_dim, device, dtype)
        column_position_embeddings = (col_cos, col_sin)  # (B,N,Dh)
    else:
        column_position_embeddings = None

    column_attention_mask = torch.ones(B, 1, N, N, device=device, dtype=dtype)  # (B,1,N,N)

    # ---- Forward ----
    print("Running forward...")
    out_learned, out_dict_list = processor(
        learned_tokens=learned_tokens,
        dict_tokens=dict_tokens,
        row_attention_mask=row_attention_mask,
        column_attention_mask=column_attention_mask,
        row_position_embeddings=row_position_embeddings,
        column_position_embeddings=column_position_embeddings
    )
    if isinstance(out_learned, (tuple, list)):
        out_learned = out_learned[0]

    # ---- Check results ----
    print(f"Output learned: {out_learned.shape}")         # (B,N,H)
    for i, (inp, out) in enumerate(zip(dict_tokens, out_dict_list)):
        print(f"dict{i}: {inp.shape} -> {out.shape}")