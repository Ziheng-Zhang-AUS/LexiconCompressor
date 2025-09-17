from __future__ import annotations
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
from configuration_lexicon_compressor import LexiconCompressorConfig


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat key/value tensors for multi-head attention."""
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def rms_norm_functional(hidden_states: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Functional RMS normalization."""
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    return weight * hidden_states.to(input_dtype)


def mlp_functional(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    gate_bias: Optional[torch.Tensor] = None,
    up_bias: Optional[torch.Tensor] = None,
    down_bias: Optional[torch.Tensor] = None,
    activation: str = "silu"
) -> torch.Tensor:
    """Functional MLP computation."""
    gate_out = F.linear(x, gate_weight, gate_bias)
    up_out = F.linear(x, up_weight, up_bias)
    
    if activation == "silu":
        gate_out = F.silu(gate_out)
    elif activation == "gelu":
        gate_out = F.gelu(gate_out)
    else:
        raise ValueError(f"Unsupported activation: {activation}")
    
    intermediate = gate_out * up_out
    return F.linear(intermediate, down_weight, down_bias)


def attention_functional_pytorch(
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    o_weight: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    q_bias: Optional[torch.Tensor] = None,
    k_bias: Optional[torch.Tensor] = None,
    v_bias: Optional[torch.Tensor] = None,
    o_bias: Optional[torch.Tensor] = None,
    num_attention_heads: int = 32,
    num_key_value_heads: int = 32,
    head_dim: int = 128,
    attention_dropout: float = 0.0,
    rms_norm_eps: float = 1e-6,
    training: bool = True,
) -> torch.Tensor:
    """Hybrid approach: Qwen3-specific preprocessing + PyTorch MHA core."""
    from torch.nn.functional import multi_head_attention_forward
    
    batch_size, seq_len, hidden_size = hidden_states.shape
    
    # Step 1: Qwen3-specific preprocessing
    query_states = F.linear(hidden_states, q_weight, q_bias)
    key_states = F.linear(hidden_states, k_weight, k_bias)
    value_states = F.linear(hidden_states, v_weight, v_bias)
    
    # Reshape for multi-head attention
    query_states = query_states.view(batch_size, seq_len, num_attention_heads, head_dim).transpose(1, 2)
    key_states = key_states.view(batch_size, seq_len, num_key_value_heads, head_dim).transpose(1, 2)
    value_states = value_states.view(batch_size, seq_len, num_key_value_heads, head_dim).transpose(1, 2)
    
    # Apply q_norm and k_norm (Qwen3 specific)
    query_states = rms_norm_functional(query_states, q_norm_weight, rms_norm_eps)
    key_states = rms_norm_functional(key_states, k_norm_weight, rms_norm_eps)
    
    # Apply RoPE (Qwen3 specific)
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    
    # Handle GQA if needed
    if num_key_value_heads != num_attention_heads:
        num_key_value_groups = num_attention_heads // num_key_value_heads
        key_states = repeat_kv(key_states, num_key_value_groups)
        value_states = repeat_kv(value_states, num_key_value_groups)
    
    # Step 2: Convert to PyTorch MHA format and call
    # Reshape: (batch, heads, seq, head_dim) -> (seq, batch*heads, head_dim)
    q_mha = query_states.transpose(0, 2).contiguous().view(seq_len, batch_size * num_attention_heads, head_dim)
    k_mha = key_states.transpose(0, 2).contiguous().view(seq_len, batch_size * num_attention_heads, head_dim) 
    v_mha = value_states.transpose(0, 2).contiguous().view(seq_len, batch_size * num_attention_heads, head_dim)
    
    # Convert attention mask format if needed
    attn_mask_mha = None
    if attention_mask is not None:
        # Convert from (B, 1, seq, seq) to (B*heads, seq, seq) format expected by PyTorch
        attn_mask_mha = attention_mask.expand(batch_size, num_attention_heads, seq_len, seq_len)
        attn_mask_mha = attn_mask_mha.reshape(batch_size * num_attention_heads, seq_len, seq_len)
    
    # Call PyTorch's optimized attention
    attn_output, _ = multi_head_attention_forward(
        query=q_mha,
        key=k_mha, 
        value=v_mha,
        embed_dim_to_check=hidden_size,
        num_heads=num_attention_heads,
        in_proj_weight=None,  # We already projected
        in_proj_bias=None,
        bias_k=None,
        bias_v=None,
        add_zero_attn=False,
        dropout_p=attention_dropout if training else 0.0,
        out_proj_weight=o_weight,
        out_proj_bias=o_bias,
        training=training,
        key_padding_mask=None,
        need_weights=False,
        attn_mask=attn_mask_mha,
        use_separate_proj_weight=False,  # We handle projection ourselves
    )
    
    # Reshape back: (seq, batch, hidden) -> (batch, seq, hidden)
    attn_output = attn_output.transpose(0, 1).contiguous()
    
    return attn_output


def attention_functional(
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    o_weight: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    q_bias: Optional[torch.Tensor] = None,
    k_bias: Optional[torch.Tensor] = None,
    v_bias: Optional[torch.Tensor] = None,
    o_bias: Optional[torch.Tensor] = None,
    num_attention_heads: int = 32,
    num_key_value_heads: int = 32,
    head_dim: int = 128,
    attention_dropout: float = 0.0,
    rms_norm_eps: float = 1e-6,
    training: bool = True,
    use_pytorch_mha: bool = False,
) -> torch.Tensor:
    """Functional attention computation following Qwen3Attention."""
    if use_pytorch_mha:
        return attention_functional_pytorch(
            hidden_states, attention_mask, position_embeddings,
            q_weight, k_weight, v_weight, o_weight, q_norm_weight, k_norm_weight,
            q_bias, k_bias, v_bias, o_bias, num_attention_heads, num_key_value_heads,
            head_dim, attention_dropout, rms_norm_eps, training
        )
    
    # Original implementation
    batch_size, seq_len, hidden_size = hidden_states.shape
    
    # Project to q, k, v
    query_states = F.linear(hidden_states, q_weight, q_bias)
    key_states = F.linear(hidden_states, k_weight, k_bias)
    value_states = F.linear(hidden_states, v_weight, v_bias)
    
    # Reshape for multi-head attention
    query_states = query_states.view(batch_size, seq_len, num_attention_heads, head_dim).transpose(1, 2)
    key_states = key_states.view(batch_size, seq_len, num_key_value_heads, head_dim).transpose(1, 2)
    value_states = value_states.view(batch_size, seq_len, num_key_value_heads, head_dim).transpose(1, 2)
    
    # Apply q_norm and k_norm
    query_states = rms_norm_functional(query_states, q_norm_weight, rms_norm_eps)
    key_states = rms_norm_functional(key_states, k_norm_weight, rms_norm_eps)
    
    # Apply rotary position embedding
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    
    # Repeat k/v for GQA
    num_key_value_groups = num_attention_heads // num_key_value_heads
    key_states = repeat_kv(key_states, num_key_value_groups)
    value_states = repeat_kv(value_states, num_key_value_groups)
    
    # Compute attention
    scaling = head_dim ** -0.5
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * scaling
    
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask
    
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_weights = F.dropout(attn_weights, p=attention_dropout, training=training)
    attn_output = torch.matmul(attn_weights, value_states)
    
    # Reshape and project output
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(batch_size, seq_len, hidden_size)
    attn_output = F.linear(attn_output, o_weight, o_bias)
    
    return attn_output


def qwen3_decoder_layer_functional(
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    weights_dict: Dict[str, torch.Tensor],
    config_dict: Dict[str, Any],
) -> torch.Tensor:
    """
    Functional implementation of Qwen3DecoderLayer.forward().
    
    Args:
        hidden_states: Input tensor (batch_size, seq_len, hidden_size)
        attention_mask: Attention mask tensor
        position_embeddings: Tuple of (cos, sin) for RoPE
        weights_dict: Dictionary containing all layer weights with keys like:
            - "input_layernorm.weight"
            - "self_attn.q_proj.weight", "self_attn.q_proj.bias" (optional)
            - "self_attn.k_proj.weight", "self_attn.k_proj.bias" (optional)
            - "self_attn.v_proj.weight", "self_attn.v_proj.bias" (optional)
            - "self_attn.o_proj.weight", "self_attn.o_proj.bias" (optional)
            - "self_attn.q_norm.weight"
            - "self_attn.k_norm.weight"
            - "post_attention_layernorm.weight"
            - "mlp.gate_proj.weight", "mlp.gate_proj.bias" (optional)
            - "mlp.up_proj.weight", "mlp.up_proj.bias" (optional)
            - "mlp.down_proj.weight", "mlp.down_proj.bias" (optional)
        config_dict: Configuration dictionary with keys like:
            - "num_attention_heads", "num_key_value_heads", "head_dim"
            - "hidden_act", "attention_dropout", "rms_norm_eps"
            - "training" (boolean)
    
    Returns:
        Output hidden states with same shape as input
    """
    
    # Extract config
    num_attention_heads = config_dict["num_attention_heads"]
    num_key_value_heads = config_dict["num_key_value_heads"]
    head_dim = config_dict["head_dim"]
    hidden_act = config_dict.get("hidden_act", "silu")
    attention_dropout = config_dict.get("attention_dropout", 0.0)
    rms_norm_eps = config_dict.get("rms_norm_eps", 1e-6)
    training = config_dict.get("training", True)
    
    # 1. Input LayerNorm
    residual = hidden_states
    hidden_states = rms_norm_functional(
        hidden_states, 
        weights_dict["input_layernorm.weight"], 
        rms_norm_eps
    )
    
    # 2. Self Attention
    attn_output = attention_functional(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_embeddings=position_embeddings,
        q_weight=weights_dict["self_attn.q_proj.weight"],
        k_weight=weights_dict["self_attn.k_proj.weight"],
        v_weight=weights_dict["self_attn.v_proj.weight"],
        o_weight=weights_dict["self_attn.o_proj.weight"],
        q_norm_weight=weights_dict["self_attn.q_norm.weight"],
        k_norm_weight=weights_dict["self_attn.k_norm.weight"],
        q_bias=weights_dict.get("self_attn.q_proj.bias"),
        k_bias=weights_dict.get("self_attn.k_proj.bias"),
        v_bias=weights_dict.get("self_attn.v_proj.bias"),
        o_bias=weights_dict.get("self_attn.o_proj.bias"),
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        attention_dropout=attention_dropout,
        rms_norm_eps=rms_norm_eps,
        training=training,
    )
    
    # 3. Residual connection
    hidden_states = residual + attn_output
    
    # 4. Post-attention LayerNorm
    residual = hidden_states
    hidden_states = rms_norm_functional(
        hidden_states,
        weights_dict["post_attention_layernorm.weight"],
        rms_norm_eps
    )
    
    # 5. MLP
    mlp_output = mlp_functional(
        x=hidden_states,
        gate_weight=weights_dict["mlp.gate_proj.weight"],
        up_weight=weights_dict["mlp.up_proj.weight"],
        down_weight=weights_dict["mlp.down_proj.weight"],
        gate_bias=weights_dict.get("mlp.gate_proj.bias"),
        up_bias=weights_dict.get("mlp.up_proj.bias"),
        down_bias=weights_dict.get("mlp.down_proj.bias"),
        activation=hidden_act,
    )
    
    # 6. Final residual connection
    hidden_states = residual + mlp_output
    
    return hidden_states


class RowColumnAttention(nn.Module):
    """Two-stage Row–Column attention for lexicon compression (functional version).

    This version does not hold any parameters internally. All weights are passed
    as dictionaries during forward pass, making it a purely functional module.
    """

    def __init__(self, config: LexiconCompressorConfig) -> None:
        super().__init__()
        self.config = config
        qcfg = config.qwen_config

        self.channels: int = qcfg.hidden_size  # C
        self.num_heads: int = qcfg.num_attention_heads
        self.head_dim: int = getattr(qcfg, "head_dim", self.channels // self.num_heads)

        # Create RoPE for position embeddings
        self.rope = Qwen3RotaryEmbedding(config=qcfg)

        # Store config for functional calls
        self.functional_config = {
            "num_attention_heads": qcfg.num_attention_heads,
            "num_key_value_heads": qcfg.num_key_value_heads,
            "head_dim": self.head_dim,
            "hidden_act": qcfg.hidden_act,
            "attention_dropout": qcfg.attention_dropout,
            "rms_norm_eps": qcfg.rms_norm_eps,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _neg_inf(dtype: torch.dtype) -> float:
        # large negative additive mask value
        return torch.finfo(dtype).min

    def _row_attention_mask(
        self,
        T: int,
        L: int,
        dict_pad_mask_flat: torch.Tensor,  # (B*R_sel, L) bool
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        BxR = dict_pad_mask_flat.size(0)
        total = T + L
        mask = torch.zeros(BxR, 1, total, total, dtype=dtype, device=device)
        neg_inf = self._neg_inf(dtype)

        # Keys that are padded (in the dict segment) are blocked for all queries
        # Dict segment occupies columns [T : T+L)
        pad_cols = dict_pad_mask_flat  # (BxR, L)
        if pad_cols.any():
            mask[:, :, :, T: T + L] = mask[:, :, :, T: T + L].masked_fill(
                pad_cols.unsqueeze(1).unsqueeze(1), neg_inf
            )

        # Causal attention inside the dict block (queries in dict segment cannot see future dict positions)
        if L > 0:
            # Upper-triangular (strictly future) → -inf
            tril = torch.tril(torch.ones(L, L, dtype=torch.bool, device=device))
            causal_block = torch.where(tril, torch.tensor(0.0, dtype=dtype, device=device), torch.tensor(neg_inf, dtype=dtype, device=device))
            mask[:, :, T: T + L, T: T + L] = causal_block

        return mask

    def _col_attention_mask(
        self,
        row_pad_mask_flat: torch.Tensor,  # (B*L_max, R_sel) bool
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        BxL, R_sel = row_pad_mask_flat.shape
        mask = torch.zeros(BxL, 1, R_sel, R_sel, dtype=dtype, device=device)
        neg_inf = self._neg_inf(dtype)
        # Block padded rows as keys for all queries
        if row_pad_mask_flat.any():
            mask[:, :, :, :] = mask[:, :, :, :].masked_fill(
                row_pad_mask_flat.unsqueeze(1).unsqueeze(1), neg_inf
            )
        return mask

    def forward(
        self,
        learned: torch.Tensor,           # (B, R_sel, T, C)
        dict_emb: torch.Tensor,          # (B, R_sel, L_max, C)
        row_weights_dict: Dict[str, torch.Tensor],  # Weights for row-wise attention
        col_weights_dict: Dict[str, torch.Tensor],  # Weights for column-wise attention
        *,
        dict_pad_mask: Optional[torch.Tensor] = None,  # (B, R_sel, L_max) bool
        row_pad_mask: Optional[torch.Tensor] = None,   # (B, R_sel) bool
        **_: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, R_sel, T, C = learned.shape
        L_max = dict_emb.size(2)
        device, dtype = learned.device, learned.dtype

        if dict_pad_mask is None:
            dict_pad_mask = torch.zeros(B, R_sel, L_max, dtype=torch.bool, device=device)
        if row_pad_mask is None:
            row_pad_mask = torch.zeros(B, R_sel, dtype=torch.bool, device=device)

        # Broadcast row mask over dict length (padded rows → all dict positions invalid)
        dict_pad_mask = dict_pad_mask | row_pad_mask.unsqueeze(-1)  # (B, R_sel, L_max)

        # Update config for current training state
        self.functional_config["training"] = self.training

        # ================================================================
        # 1) Row-wise attention
        # ================================================================

        # (B*R_sel, T, C) and (B*R_sel, L_max, C)
        learned_row = learned.contiguous().view(B * R_sel, T, C)
        dict_row = dict_emb.contiguous().view(B * R_sel, L_max, C)
        dict_pad_row = dict_pad_mask.contiguous().view(B * R_sel, L_max)

        # Concatenate [learned; dict] along sequence
        concat_row = torch.cat([learned_row, dict_row], dim=1)  # (B*R_sel, T+L_max, C)
        total_len = T + L_max

        # Position ids for RoPE
        pos_ids_row = torch.arange(total_len, device=device, dtype=torch.long).unsqueeze(0).expand(B * R_sel, -1)
        # Compute position embeddings
        cos_row, sin_row = self.rope(concat_row, pos_ids_row)
        position_embeddings_row = (cos_row, sin_row)
        
        attn_mask_row = self._row_attention_mask(T, L_max, dict_pad_row, dtype, device)

        # Apply functional decoder layer
        hs_row = qwen3_decoder_layer_functional(
            hidden_states=concat_row,
            attention_mask=attn_mask_row,
            position_embeddings=position_embeddings_row,
            weights_dict=row_weights_dict,
            config_dict=self.functional_config,
        )
        
        # Split back
        learned_upd = hs_row[:, :T, :]
        dict_upd_row = hs_row[:, T:, :]

        learned_upd = learned_upd.view(B, R_sel, T, C)
        dict_upd_row = dict_upd_row.view(B, R_sel, L_max, C)

        # ================================================================
        # 2) Column-wise (across rows)
        # ================================================================

        # Transpose first, then view → (B*L_max, R_sel, C)
        dict_for_col = dict_upd_row.transpose(1, 2).contiguous().view(B * L_max, R_sel, C)
        row_pad_for_col = row_pad_mask.unsqueeze(1).expand(B, L_max, R_sel).contiguous().view(B * L_max, R_sel)

        pos_ids_col = torch.arange(R_sel, device=device, dtype=torch.long).unsqueeze(0).expand(B * L_max, -1)
        cos_col, sin_col = self.rope(dict_for_col, pos_ids_col)
        position_embeddings_col = (cos_col, sin_col)
        
        attn_mask_col = self._col_attention_mask(row_pad_for_col, dtype, device)

        # Apply functional decoder layer
        hs_col = qwen3_decoder_layer_functional(
            hidden_states=dict_for_col,
            attention_mask=attn_mask_col,
            position_embeddings=position_embeddings_col,
            weights_dict=col_weights_dict,
            config_dict=self.functional_config,
        )
        
        # Back to (B, R_sel, L_max, C)
        dict_upd_col = hs_col.view(B, L_max, R_sel, C).transpose(1, 2).contiguous()

        return learned_upd, dict_upd_col

    def extra_repr(self) -> str:
        return f"channels={self.channels}, num_heads={self.num_heads} (functional)"


# ================================================================
# Test Code
# ================================================================

if __name__ == "__main__":
    import torch
    import torch.nn as nn
    from dataclasses import dataclass
    
    # Mock configuration classes
    @dataclass
    class MockQwenConfig:
        hidden_size: int = 768
        num_attention_heads: int = 12
        num_key_value_heads: int = 12
        intermediate_size: int = 3072
        hidden_act: str = "silu"
        attention_dropout: float = 0.0
        rms_norm_eps: float = 1e-6
        max_position_embeddings: int = 2048
        rope_scaling: dict = None
        attention_bias: bool = False
        # RoPE related parameters
        rope_theta: float = 10000.0
        rope_type: str = "default"
        # Additional parameters that might be needed
        layer_types: list = None
        sliding_window: int = None
        pad_token_id: int = 0
        vocab_size: int = 32000
        num_hidden_layers: int = 12
        
        def __post_init__(self):
            if self.layer_types is None:
                self.layer_types = ["full_attention"] * self.num_hidden_layers
        
    @dataclass
    class MockLexiconCompressorConfig:
        qwen_config: MockQwenConfig
        
    def create_mock_weights(config: MockQwenConfig) -> Dict[str, torch.Tensor]:
        """Create mock weights for testing."""
        hidden_size = config.hidden_size
        num_heads = config.num_attention_heads
        num_kv_heads = config.num_key_value_heads
        intermediate_size = config.intermediate_size
        head_dim = hidden_size // num_heads
        
        weights = {
            # Input LayerNorm
            "input_layernorm.weight": torch.ones(hidden_size),
            
            # Attention projections
            "self_attn.q_proj.weight": torch.randn(num_heads * head_dim, hidden_size) * 0.02,
            "self_attn.k_proj.weight": torch.randn(num_kv_heads * head_dim, hidden_size) * 0.02,
            "self_attn.v_proj.weight": torch.randn(num_kv_heads * head_dim, hidden_size) * 0.02,
            "self_attn.o_proj.weight": torch.randn(hidden_size, num_heads * head_dim) * 0.02,
            
            # Q/K norm weights
            "self_attn.q_norm.weight": torch.ones(head_dim),
            "self_attn.k_norm.weight": torch.ones(head_dim),
            
            # Post-attention LayerNorm
            "post_attention_layernorm.weight": torch.ones(hidden_size),
            
            # MLP weights
            "mlp.gate_proj.weight": torch.randn(intermediate_size, hidden_size) * 0.02,
            "mlp.up_proj.weight": torch.randn(intermediate_size, hidden_size) * 0.02,
            "mlp.down_proj.weight": torch.randn(hidden_size, intermediate_size) * 0.02,
        }
        return weights
    
    def test_functional_decoder_layer():
        """Test the functional decoder layer implementation."""
        print("Testing qwen3_decoder_layer_functional...")
        
        # Setup
        config = MockQwenConfig()
        batch_size, seq_len = 2, 16
        
        # Create inputs
        hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
        attention_mask = torch.zeros(batch_size, 1, seq_len, seq_len)
        
        # Create position embeddings (mock RoPE)
        pos_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
        cos = torch.cos(pos_ids.unsqueeze(-1).float() * 0.01).expand(-1, -1, config.hidden_size // config.num_attention_heads)
        sin = torch.sin(pos_ids.unsqueeze(-1).float() * 0.01).expand(-1, -1, config.hidden_size // config.num_attention_heads)
        position_embeddings = (cos, sin)
        
        # Create mock weights
        weights_dict = create_mock_weights(config)
        
        # Create config dict
        config_dict = {
            "num_attention_heads": config.num_attention_heads,
            "num_key_value_heads": config.num_key_value_heads,
            "head_dim": config.hidden_size // config.num_attention_heads,
            "hidden_act": config.hidden_act,
            "attention_dropout": config.attention_dropout,
            "rms_norm_eps": config.rms_norm_eps,
            "training": False,
        }
        
        # Test forward pass
        output = qwen3_decoder_layer_functional(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            weights_dict=weights_dict,
            config_dict=config_dict,
        )
        
        # Check output shape
        assert output.shape == hidden_states.shape, f"Output shape {output.shape} != input shape {hidden_states.shape}"
        
        # Check that output is different from input (transformation happened)
        assert not torch.allclose(output, hidden_states, atol=1e-5), "Output should be different from input"
        
        print("✓ Functional decoder layer test passed!")
        print(f"  Input shape: {hidden_states.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Output mean: {output.mean().item():.4f}")
        print(f"  Output std: {output.std().item():.4f}")
        
    def test_row_column_attention():
        """Test the RowColumnAttention module."""
        print("\nTesting RowColumnAttention...")
        
        # Setup
        qwen_config = MockQwenConfig()
        config = MockLexiconCompressorConfig(qwen_config=qwen_config)
        
        # Create module
        row_col_attn = RowColumnAttention(config)
        row_col_attn.eval()  # Set to eval mode
        
        # Check that module has no parameters
        param_count = sum(p.numel() for p in row_col_attn.parameters() if p.requires_grad)
        print(f"  Parameter count: {param_count}")
        # Note: RoPE has some buffers, so param_count might not be exactly 0
        
        # Create inputs
        B, R_sel, T, C = 2, 4, 8, qwen_config.hidden_size
        L_max = 6
        
        learned = torch.randn(B, R_sel, T, C)
        dict_emb = torch.randn(B, R_sel, L_max, C)
        
        # Create padding masks
        dict_pad_mask = torch.zeros(B, R_sel, L_max, dtype=torch.bool)
        dict_pad_mask[:, :, -1] = True  # Mask last position
        
        row_pad_mask = torch.zeros(B, R_sel, dtype=torch.bool)
        row_pad_mask[:, -1] = True  # Mask last row
        
        # Create mock weights (same for row and column)
        row_weights = create_mock_weights(qwen_config)
        col_weights = create_mock_weights(qwen_config)
        
        # Test forward pass
        learned_out, dict_out = row_col_attn(
            learned=learned,
            dict_emb=dict_emb,
            row_weights_dict=row_weights,
            col_weights_dict=col_weights,
            dict_pad_mask=dict_pad_mask,
            row_pad_mask=row_pad_mask,
        )
        
        # Check output shapes
        assert learned_out.shape == learned.shape, f"Learned output shape {learned_out.shape} != input shape {learned.shape}"
        assert dict_out.shape == dict_emb.shape, f"Dict output shape {dict_out.shape} != input shape {dict_emb.shape}"
        
        # Check that outputs are different from inputs
        assert not torch.allclose(learned_out, learned, atol=1e-5), "Learned output should be different from input"
        assert not torch.allclose(dict_out, dict_emb, atol=1e-5), "Dict output should be different from input"
        
        print("✓ RowColumnAttention test passed!")
        print(f"  Input shapes: learned {learned.shape}, dict_emb {dict_emb.shape}")
        print(f"  Output shapes: learned {learned_out.shape}, dict {dict_out.shape}")
        print(f"  Learned output mean: {learned_out.mean().item():.4f}")
        print(f"  Dict output mean: {dict_out.mean().item():.4f}")
        
    def test_gradient_flow():
        """Test that gradients flow correctly through the functional implementation."""
        print("\nTesting gradient flow...")
        
        # Setup with requires_grad=True
        qwen_config = MockQwenConfig()
        config = MockLexiconCompressorConfig(qwen_config=qwen_config)
        
        row_col_attn = RowColumnAttention(config)
        row_col_attn.train()  # Set to training mode
        
        # Create inputs with gradients
        B, R_sel, T, C = 1, 2, 4, qwen_config.hidden_size
        L_max = 3
        
        learned = torch.randn(B, R_sel, T, C, requires_grad=True)
        dict_emb = torch.randn(B, R_sel, L_max, C, requires_grad=True)
        
        # Create weights with gradients
        row_weights = create_mock_weights(qwen_config)
        col_weights = create_mock_weights(qwen_config)
        
        for weights in [row_weights, col_weights]:
            for key, weight in weights.items():
                weight.requires_grad_(True)
        
        # Forward pass
        learned_out, dict_out = row_col_attn(
            learned=learned,
            dict_emb=dict_emb,
            row_weights_dict=row_weights,
            col_weights_dict=col_weights,
        )
        
        # Create a simple loss
        loss = learned_out.mean() + dict_out.mean()
        
        # Backward pass
        loss.backward()
        
        # Check that gradients exist
        assert learned.grad is not None, "Input learned should have gradients"
        assert dict_emb.grad is not None, "Input dict_emb should have gradients"
        
        # Check that weight gradients exist
        for weights in [row_weights, col_weights]:
            for key, weight in weights.items():
                assert weight.grad is not None, f"Weight {key} should have gradients"
        
        print("✓ Gradient flow test passed!")
        print(f"  Loss value: {loss.item():.4f}")
        print(f"  Learned grad norm: {learned.grad.norm().item():.4f}")
        print(f"  Dict_emb grad norm: {dict_emb.grad.norm().item():.4f}")
        
    # Run all tests
    print("Running functional RowColumnAttention tests...\n")
    
    try:
        test_functional_decoder_layer()
        test_row_column_attention()
        test_gradient_flow()
        print("\n🎉 All tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()