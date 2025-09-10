# row_column_attention.py
from __future__ import annotations
from typing import List, Optional, Tuple, Dict, Any
import math

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3DecoderLayer,
    Qwen3RotaryEmbedding,
)
from configuration_lexicon_compressor import LexiconCompressorConfig


class RowColumnAttention(nn.Module):
    """
    Hierarchical Row-Column Attention for lexicon compression.
    
    This module performs two-stage attention:
    1. Row-wise attention: Processes each dictionary entry with its learned tokens
    2. Column-wise attention: Cross-attention between learned tokens across entries
    """

    def __init__(self, config: LexiconCompressorConfig):
        """
        Initialize RowColumnAttention.

        Args:
            config: LexiconCompressorConfig containing Qwen3 config and compressor options
        """
        super().__init__()
        self.config = config
        qwen_config = config.qwen_config

        self.hidden_size = qwen_config.hidden_size
        self.num_heads = qwen_config.num_attention_heads
        self.head_dim = getattr(qwen_config, "head_dim", self.hidden_size // self.num_heads)
        
        # Row attention processes each entry + learned tokens independently
        self.row_layer = Qwen3DecoderLayer(config=qwen_config, layer_idx=0)
        # Column attention processes learned tokens across entries
        self.col_layer = Qwen3DecoderLayer(config=qwen_config, layer_idx=1)
        
        self.rope = Qwen3RotaryEmbedding(config=qwen_config)
        
        # Buffer to track weight loading state
        self.register_buffer("_weights_loaded", torch.tensor(False))

    def load_weights_once(self, row_weights: Dict[str, torch.Tensor], col_weights: Dict[str, torch.Tensor]):
        """
        Load decoder-layer weights once for both row and column attention.

        Args:
            row_weights: State dict for row decoder layer
            col_weights: State dict for column decoder layer
        """
        if self._weights_loaded.item():
            return
            
        if row_weights is None or col_weights is None:
            raise ValueError("Both row_weights and col_weights must be provided.")
            
        self.row_layer.load_state_dict(row_weights, strict=True)
        self.col_layer.load_state_dict(col_weights, strict=True)
        self._weights_loaded.fill_(True)

    def _process_single_row(
        self,
        learned_row: torch.Tensor,
        dict_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process a single row (dict entry + learned tokens) with correct RoPE.

        Args:
            learned_row: (T, H) Learned compression tokens for this row
            dict_tokens: (Li, H) Dictionary entry tokens

        Returns:
            Tuple of:
                - updated_learned: (T, H) Updated learned tokens
                - updated_dict: (Li, H) Updated dictionary tokens
        """
        device, dtype = learned_row.device, learned_row.dtype
        num_learned = learned_row.size(0)
        dict_len = dict_tokens.size(0)
        
        prepend_learned = bool(self.config.learned_tokens_prepend)
        
        # Concatenate tokens
        if prepend_learned:
            # [learned_tokens, dict_tokens]
            if dict_len > 0:
                row_tokens = torch.cat([learned_row, dict_tokens], dim=0)  # (T+Li, H)
            else:
                row_tokens = learned_row  # (T, H)
            total_len = num_learned + dict_len
        else:
            # [dict_tokens, learned_tokens]
            if dict_len > 0:
                row_tokens = torch.cat([dict_tokens, learned_row], dim=0)  # (Li+T, H)
            else:
                row_tokens = learned_row  # (T, H)
            total_len = dict_len + num_learned
        
        # Add batch dimension for processing
        row_tokens = row_tokens.unsqueeze(0)  # (1, total_len, H)
        
        # Build proper causal mask - CORRECTED DIMENSIONS
        attention_mask = self._build_row_causal_mask(
            total_length=total_len,
            dict_length=dict_len,
            prepend_learned=prepend_learned,
            dtype=dtype,
            device=device
        ).unsqueeze(0).unsqueeze(0)  # (L,L) -> (1, 1, L, L)
        
        # Build correct position IDs for RoPE
        position_ids = self._build_row_position_ids(
            total_length=total_len,
            dict_length=dict_len,
            prepend_learned=prepend_learned,
            device=device
        ).unsqueeze(0)  # (1, total_len)
        
        # Get position embeddings
        cos, sin = self.rope(row_tokens, position_ids)
        position_embeddings = (cos, sin)
        
        # Apply row attention
        out = self.row_layer(
        hidden_states=row_tokens,
        attention_mask=attention_mask,
        position_ids=position_ids,
        position_embeddings=position_embeddings,
        past_key_values=None,
        use_cache=False,
        )
        # 支持多种返回格式
        row_hidden = out[0] if isinstance(out, (tuple, list)) else getattr(out, 'hidden_states', out)
        row_output = row_hidden.squeeze(0)  # (total_len, H)
        
        if prepend_learned:
            updated_learned = row_output[:num_learned, :]  # (T, H)
            if dict_len > 0:
                updated_dict = row_output[num_learned:num_learned+dict_len, :]  # (Li, H)
            else:
                updated_dict = torch.empty(0, self.hidden_size, device=device, dtype=dtype)  # (0, H)
        else:
            if dict_len > 0:
                updated_dict = row_output[:dict_len, :]  # (Li, H)
            else:
                updated_dict = torch.empty(0, self.hidden_size, device=device, dtype=dtype)  # (0, H)
            updated_learned = row_output[dict_len:dict_len+num_learned, :]  # (T, H)
        
        return updated_learned, updated_dict

    def _build_row_causal_mask(
        self,
        total_length: int,
        dict_length: int,
        prepend_learned: bool,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Build causal attention mask for a single row.

        Args:
            total_length: Total length of concatenated sequence
            dict_length: Length of dictionary tokens
            prepend_learned: Whether learned tokens come first
            dtype: Tensor data type
            device: Tensor device

        Returns:
            attention_mask: (total_length, total_length) Causal attention mask
        """
        mask = torch.zeros(total_length, total_length, device=device, dtype=dtype)
        neg_inf = torch.finfo(dtype).min
        
        if prepend_learned:
            # Learned tokens (first T positions) can attend to themselves and dict tokens
            num_learned = total_length - dict_length
            # Dict tokens can only attend to learned tokens and previous dict tokens (causal)
            if dict_length > 0:
                causal_dict = torch.tril(torch.ones(dict_length, dict_length, device=device, dtype=torch.bool))
                # Dict tokens can attend to learned tokens (first num_learned positions)
                mask[num_learned:, :num_learned] = 0  # Allow attention to learned tokens
                # Dict tokens have causal attention among themselves
                mask[num_learned:, num_learned:] = torch.where(
                    causal_dict, 
                    torch.tensor(0.0, device=device, dtype=dtype),
                    torch.tensor(neg_inf, device=device, dtype=dtype)
                )
        else:
            # Dict tokens first, then learned tokens
            num_learned = total_length - dict_length
            # Dict tokens have causal attention among themselves
            if dict_length > 0:
                causal_dict = torch.tril(torch.ones(dict_length, dict_length, device=device, dtype=torch.bool))
                mask[:dict_length, :dict_length] = torch.where(
                    causal_dict,
                    torch.tensor(0.0, device=device, dtype=dtype),
                    torch.tensor(neg_inf, device=device, dtype=dtype)
                )
            # Learned tokens can attend to everything (no additional masking needed)
            
        return mask

    def _build_row_position_ids(
        self,
        total_length: int,
        dict_length: int,
        prepend_learned: bool,
        device: torch.device,
    ) -> torch.LongTensor:
        """
        Build position IDs for a single row with correct sequential positions.

        Args:
            total_length: Total length of concatenated sequence
            dict_length: Length of dictionary tokens
            prepend_learned: Whether learned tokens come first
            device: Tensor device

        Returns:
            position_ids: (total_length,) Position IDs for RoPE
        """
        position_ids = torch.arange(total_length, device=device, dtype=torch.long)
        return position_ids

    def _process_column_attention(
        self,
        learned_tokens_batch: torch.Tensor,
    ) -> torch.Tensor:
        """
        Process column-wise attention across learned tokens from all rows.

        Args:
            learned_tokens_batch: (B, T, H) learned tokens from row attention

        Returns:
            updated_learned_tokens: (B, T, H) Updated learned tokens after column attention
        """
        batch_size, num_learned, hidden_size = learned_tokens_batch.shape
        device, dtype = learned_tokens_batch.device, learned_tokens_batch.dtype
        
        # Build column attention mask (full attention) - CORRECTED DIMENSIONS
        attention_mask = torch.zeros(
            batch_size, 1, num_learned, num_learned, 
            device=device, dtype=dtype
        )
        
        # Build identity position embeddings for column attention
        position_ids = torch.arange(num_learned, device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
        dummy_input = torch.empty(batch_size, num_learned, hidden_size, device=device, dtype=dtype)
        cos, sin = self.rope(dummy_input, position_ids)
        # Use identity RoPE (cos=1, sin=0)
        head_dim = self.head_dim
        cos_identity = torch.ones(batch_size, num_learned, head_dim, device=device, dtype=dtype)
        sin_identity = torch.zeros(batch_size, num_learned, head_dim, device=device, dtype=dtype)
        position_embeddings = (cos_identity, sin_identity)
        
        # 修复：正确处理Qwen3DecoderLayer输出
        out = self.col_layer(
            hidden_states=learned_tokens_batch,
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            past_key_values=None,
            use_cache=False,
        )
        col_hidden = out[0] if isinstance(out, (tuple, list)) else getattr(out, 'hidden_states', out)
        
        return col_hidden

    def forward(
        self,
        learned_tokens: torch.Tensor,
        dict_tokens_list: List[torch.Tensor],
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass through hierarchical row-column attention.

        Args:
            learned_tokens: (B, T, H) Learnable compression tokens
            dict_tokens_list: List of (Li, H) dictionary entry tokens

        Returns:
            Tuple of:
                - updated_learned_tokens: (B, T, H) Updated learned tokens (after column attention)
                - updated_dict_list: List of (Li, H) Updated dictionary tokens (after row attention)
        """
        batch_size, num_learned, hidden_size = learned_tokens.shape
        device, dtype = learned_tokens.device, learned_tokens.dtype
        
        # Validate input dimensions
        if len(dict_tokens_list) != batch_size:
            raise ValueError(
                f"Batch size mismatch: learned_tokens has {batch_size} entries "
                f"but dict_tokens_list has {len(dict_tokens_list)} entries"
            )
            
        if hidden_size != self.hidden_size:
            raise ValueError(
                f"Hidden size mismatch: expected {self.hidden_size}, got {hidden_size}"
            )

        # Process each row individually with correct RoPE
        updated_learned_list = []
        updated_dict_list = []
        
        for batch_idx in range(batch_size):
            learned_row = learned_tokens[batch_idx]  # (T, H)
            dict_tokens = dict_tokens_list[batch_idx]  # (Li, H)
            
            updated_learned, updated_dict = self._process_single_row(
                learned_row, dict_tokens
            )
            
            updated_learned_list.append(updated_learned)  # (T, H)
            updated_dict_list.append(updated_dict)  # (Li, H)
        
        # Stack learned tokens for column attention
        learned_tokens_batch = torch.stack(updated_learned_list, dim=0)  # (B, T, H)
        
        # Process column attention across learned tokens
        final_learned_tokens = self._process_column_attention(learned_tokens_batch)
        
        return final_learned_tokens, updated_dict_list

    def extra_repr(self) -> str:
        return f"hidden_size={self.hidden_size}, num_heads={self.num_heads}"