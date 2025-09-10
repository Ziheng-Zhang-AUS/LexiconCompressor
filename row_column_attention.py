from __future__ import annotations
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3RotaryEmbedding
from configuration_lexicon_compressor import LexiconCompressorConfig


class RowColumnAttention(nn.Module):
    """Two-stage Row–Column attention for lexicon compression (vectorized).

    Inputs (shapes):
      learned      : (B, R_sel, T, C)
      dict_emb     : (B, R_sel, L_max, C)
      dict_pad_mask: (B, R_sel, L_max)  # True = padding position in dict_emb
      row_pad_mask : (B, R_sel)         # True = padded/invalid row

    Returns:
      learned_out: (B, R_sel, T, C)
      dict_out   : (B, R_sel, L_max, C)

    Implementation notes:
      • Row-wise pass merges (B, R_sel) → (B*R_sel, ·) so each row is processed independently
        by a Qwen3DecoderLayer over the concatenated sequence [learned; dict].
      • Column-wise pass merges (B, L_max) → (B*L_max, ·) to mix information *across rows*
        for the same column index using another Qwen3DecoderLayer.
      • We build additive attention masks (0 for allow, -inf for block). For padded keys we
        add -inf; causal structure inside the dict block is preserved.
    """

    def __init__(self, config: LexiconCompressorConfig) -> None:
        super().__init__()
        self.config = config
        qcfg = config.qwen_config

        self.channels: int = qcfg.hidden_size  # C
        self.num_heads: int = qcfg.num_attention_heads
        self.head_dim: int = getattr(qcfg, "head_dim", self.channels // self.num_heads)

        # Use two decoder layers from Qwen3 as attention processors
        self.row_layer = Qwen3DecoderLayer(config=qcfg, layer_idx=0)
        self.col_layer = Qwen3DecoderLayer(config=qcfg, layer_idx=1)

        self.rope = Qwen3RotaryEmbedding(config=qcfg)

        # One-time weight loading latch
        self.register_buffer("_weights_loaded", torch.tensor(False), persistent=False)

    # Weight loading (row/col layers)
    def load_weights_once(self, row_weights: Dict[str, torch.Tensor], col_weights: Dict[str, torch.Tensor]) -> None:
        if self._weights_loaded.item():
            return
        if row_weights is None or col_weights is None:
            raise ValueError("Both row_weights and col_weights must be provided.")
        self.row_layer.load_state_dict(row_weights, strict=True)
        self.col_layer.load_state_dict(col_weights, strict=True)
        self._weights_loaded.fill_(True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _neg_inf(dtype: torch.dtype) -> float:
        # large negative additive mask value
        return torch.finfo(dtype).min

    # Build (B*R_sel, 1, L_total, L_total) additive mask for row-wise concat([T], [L])
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

        # Learned tokens (first T) can attend everywhere → no extra blocks
        # Dict queries should be allowed to attend to learned tokens (columns [:T]) → already 0
        return mask

    # Build (B*L_max, 1, R_sel, R_sel) additive mask for column-wise mixing across rows
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
        # (Option) keep non-causal/full attention across rows by not adding triangular mask
        return mask

    def forward(
        self,
        learned: torch.Tensor,           # (B, R_sel, T, C)
        dict_emb: torch.Tensor,          # (B, R_sel, L_max, C)
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





        # Row-wise attention

        # (B*R_sel, T, C) and (B*R_sel, L_max, C)
        learned_row = learned.contiguous().view(B * R_sel, T, C)
        dict_row = dict_emb.contiguous().view(B * R_sel, L_max, C)
        dict_pad_row = dict_pad_mask.contiguous().view(B * R_sel, L_max)

        # Concatenate [learned; dict] along sequence
        concat_row = torch.cat([learned_row, dict_row], dim=1)  # (B*R_sel, T+L_max, C)
        total_len = T + L_max

        # Position ids for RoPE
        pos_ids_row = torch.arange(total_len, device=device, dtype=torch.long).unsqueeze(0).expand(B * R_sel, -1)
        # Qwen3DecoderLayer accepts position_embeddings; compute them
        cos_row, sin_row = self.rope(concat_row, pos_ids_row)
        attn_mask_row = self._row_attention_mask(T, L_max, dict_pad_row, dtype, device)  # (B*R_sel, 1, total, total)

        out_row = self.row_layer(
            hidden_states=concat_row,
            attention_mask=attn_mask_row,
            position_ids=pos_ids_row,
            position_embeddings=(cos_row, sin_row),
            past_key_values=None,
            use_cache=False,
        )
        hs_row = out_row[0] if isinstance(out_row, (tuple, list)) else getattr(out_row, "hidden_states", out_row)
        # Split back
        learned_upd = hs_row[:, :T, :]
        dict_upd_row = hs_row[:, T:, :]

        learned_upd = learned_upd.view(B, R_sel, T, C)
        dict_upd_row = dict_upd_row.view(B, R_sel, L_max, C)




        # 2) Column-wise (across rows)

        # Transpose first, then view → (B*L_max, R_sel, C)
        dict_for_col = dict_upd_row.transpose(1, 2).contiguous().view(B * L_max, R_sel, C)
        row_pad_for_col = row_pad_mask.unsqueeze(1).expand(B, L_max, R_sel).contiguous().view(B * L_max, R_sel)

        pos_ids_col = torch.arange(R_sel, device=device, dtype=torch.long).unsqueeze(0).expand(B * L_max, -1)
        cos_col, sin_col = self.rope(dict_for_col, pos_ids_col)
        attn_mask_col = self._col_attention_mask(row_pad_for_col, dtype, device)  # (B*L_max, 1, R_sel, R_sel)

        out_col = self.col_layer(
            hidden_states=dict_for_col,
            attention_mask=attn_mask_col,
            position_ids=pos_ids_col,
            position_embeddings=(cos_col, sin_col),
            past_key_values=None,
            use_cache=False,
        )
        hs_col = out_col[0] if isinstance(out_col, (tuple, list)) else getattr(out_col, "hidden_states", out_col)
        # Back to (B, R_sel, L_max, C)
        dict_upd_col = hs_col.view(B, L_max, R_sel, C).transpose(1, 2).contiguous()

        return learned_upd, dict_upd_col

    def extra_repr(self) -> str:
        return f"channels={self.channels}, num_heads={self.num_heads}"