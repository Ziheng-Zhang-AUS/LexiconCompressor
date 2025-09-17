from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import Qwen3Config, Qwen3ForCausalLM
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast

from row_column_attention import RowColumnAttention
from configuration_lexicon_compressor import LexiconCompressorConfig


# ---------------------a--------------------------------------------------------
# Outputs
# -----------------------------------------------------------------------------
@dataclass
class LexiconCompressorModelOutput(CausalLMOutputWithPast):
    """Output of LexiconCompressorModel.

    Args:
        loss: Language modeling loss.
        logits: Prediction scores.
        past_key_values: KV cache from the base model.
        hidden_states: Hidden states from the base model.
        attentions: Attention weights from the base model.
        compressed_tokens: List of per-sample compressed token tensors (optional, debug only).
    """

    compressed_tokens: Optional[List[torch.FloatTensor]] = None


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
class LexiconCompressorModel(nn.Module):
    """Vectorized dictionary-to-prefix wrapper over Qwen3.

    Pipeline:RowColumnAttention
      1) pad the full dictionary to a tensor (R, L_max),
      2) embed once as (R, L_max, C) with a pad mask (R, L_max),
      3) gather per-batch rows via advanced indexing (no Python loops),
      4) run Row/Column Attention on batched tensors, and
      5) flatten per-row learned tokens to a prefix, then prepend to Qwen3.

    Notation (shapes):
      B = batch size
      R = total dictionary rows
      R_sel = max selected rows in this batch (after padding)
      L_max = max token length per row (dictionary)
      T = number of learned tokens per row
      C = channels (a.k.a. hidden size)
    """

    def __init__(
        self,
        qwen_model: Qwen3ForCausalLM,
        full_dict: List[List[int]],
        dict_encoder_num_compress_tokens: int,
        dict_encoder_learned_tokens_prepend: bool = True,
        compressor_config: Optional[LexiconCompressorConfig] = None,
    ) -> None:
        super().__init__()
        if not isinstance(qwen_model, Qwen3ForCausalLM):
            raise ValueError("qwen_model must be Qwen3ForCausalLM")

        # Base model refs
        self.qwen = qwen_model
        self.qwen_config: Qwen3Config = qwen_model.config
        self.channels: int = self.qwen_config.hidden_size  # C
        self.num_heads: int = self.qwen_config.num_attention_heads
        self.head_dim: int = getattr(self.qwen_config, "head_dim", self.channels // self.num_heads)

        # Dictionary
        self.full_dict_list = full_dict
        self.num_rows = len(full_dict)
        lens = torch.tensor([len(r) for r in full_dict], dtype=torch.long) if self.num_rows > 0 else torch.zeros(0, dtype=torch.long)
        self.row_max_len = int(lens.max().item()) if self.num_rows > 0 else 1

        # Build (R, L_max) filled with -1, then scatter valid tokens by a boolean mask
        idx_padded = torch.full((self.num_rows, self.row_max_len), -1, dtype=torch.long)  # (R, L_max)
        col = torch.arange(self.row_max_len).expand(self.num_rows, -1)              # (R, L_max)
        valid_mask = col < lens.unsqueeze(1)                                        # (R, L_max) bool
        flat_vals = torch.tensor([x for row in full_dict for x in row], dtype=torch.long)
        idx_padded[valid_mask] = flat_vals                                          # row-major fill
        pad_mask = idx_padded.eq(-1)                                                    # (R, L_max) bool
        self.register_buffer("full_dict_index", idx_padded, persistent=False)
        self.register_buffer("full_dict_pad_mask", pad_mask, persistent=False)

        # Learned tokens per row: (R, T, C)
        self.num_layers = len(self.qwen.model.layers) # equals to the number of qwen decoder layers
        self.num_compress_tokens = dict_encoder_num_compress_tokens  # T
        self.learned_tokens_prepend = dict_encoder_learned_tokens_prepend
        init = torch.randn(self.num_rows, self.num_compress_tokens, self.channels)
        self.learned_tokens_global = nn.Parameter(init)

        # RCA stack (vectorized implementation expected inside RowColumnAttention)
        self.config = compressor_config or LexiconCompressorConfig(
            qwen_config=self.qwen_config,
            num_layers=self.num_layers,
            num_compress_tokens=self.num_compress_tokens,
            learned_tokens_prepend=self.learned_tokens_prepend,
        )
        self.dict_encoder = nn.ModuleList([RowColumnAttention(self.config) for _ in range(self.num_layers)])

        # One-time weight loading flag for RCA
        self._rca_weights_loaded_once: bool = False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _dtype(self) -> torch.dtype:
        try:
            return next(self.qwen.parameters()).dtype
        except StopIteration:
            return torch.float32

    def _device(self) -> torch.device:
        for p in self.parameters():
            return p.device
        for b in self.buffers():
            return b.device
        return torch.device("cpu")

    def _embed_full_dict(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Embed the entire padded dictionary once.

        Returns:
            emb_full: (R, L_max, C)
            pad_mask: (R, L_max) bool
        """
        # Replace -1 with 0 for safe indexing, then mask out padded positions.
        safe_idx = self.full_dict_index.clamp(min=0).to(device=self._device())  # (R, L_max)
        emb_full = self.qwen.model.embed_tokens(safe_idx)  # (R, L_max, C)
        pad_mask = self.full_dict_pad_mask.to(device=self._device())
        emb_full = emb_full.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        return emb_full, pad_mask

    def _build_qwen_inputs_embeds(
        self,
        qwen_input_ids: Optional[torch.LongTensor],
        qwen_inputs_embeds: Optional[torch.FloatTensor],
    ) -> torch.FloatTensor:
        """Convert Qwen inputs to embeddings: (B, S, C)."""
        if qwen_input_ids is None and qwen_inputs_embeds is None:
            raise ValueError("Either qwen_input_ids or qwen_inputs_embeds must be provided")
        if qwen_inputs_embeds is not None:
            return qwen_inputs_embeds.to(device=self._device(), dtype=self._dtype())
        return self.qwen.model.embed_tokens(qwen_input_ids)

    def _estimate_past_length(self, past_key_values: Optional[Cache]) -> int:
        """Robustly estimate KV-cache sequence length."""
        if past_key_values is None:
            return 0
        if hasattr(past_key_values, "get_seq_length"):
            return past_key_values.get_seq_length()
        try:
            return past_key_values[0][0].shape[-2]
        except Exception:
            return 0

    # ------------------------------------------------------------------
    # Vectorized dictionary selection for a batch
    # ------------------------------------------------------------------
    def _gather_batch_rows(
        self,
        idx_padded: torch.LongTensor,  # (B, R_sel) with -1 padding
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Vectorized gather with no Python loops.

        Args:
            idx_padded: (B, R_sel) row indices, padded with -1.

        Returns:
            learned:       (B, R_sel, T, C)
            dict_emb:      (B, R_sel, L_max, C)
            dict_pad_mask: (B, R_sel, L_max) bool
            row_pad_mask:  (B, R_sel) bool  # True for padded rows
        """
        if not torch.is_tensor(idx_padded):
            raise TypeError("idx_padded must be a padded LongTensor of shape (B, R_sel) with -1 for padding.")
        if idx_padded.dim() != 2:
            raise ValueError(f"idx_padded must be 2D (B, R_sel), got shape {tuple(idx_padded.shape)}")

        idx_padded = idx_padded.to(self._get_device())
        row_pad_mask = idx_padded.eq(-1)                  # (B, R_sel)
        idx_safe = idx_padded.clamp(min=0)                # (B, R_sel)

        # Full dict embeddings & pad mask (R, L_max, C), (R, L_max)
        dict_full, pad_full = self._embed_full_dict()

        # Advanced indexing
        dict_emb = dict_full[idx_safe]                    # (B, R_sel, L_max, C)
        dict_pad_mask = pad_full[idx_safe]                # (B, R_sel, L_max)

        # Learned tokens per row
        learned = self.learned_tokens_global[idx_safe]    # (B, R_sel, T, C)

        # Zero out invalid rows completely
        dict_emb = dict_emb.masked_fill(row_pad_mask.unsqueeze(-1).unsqueeze(-1), 0.0)
        dict_pad_mask = dict_pad_mask | row_pad_mask.unsqueeze(-1)
        learned = learned.masked_fill(row_pad_mask.unsqueeze(-1).unsqueeze(-1), 0.0)

        return learned, dict_emb, dict_pad_mask, row_pad_mask


    # ------------------------------------------------------------------
    # Generation integration (first step builds prefix; later steps pass through)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def prepare_inputs_for_generation(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        row_indices_per_sample: Optional[List[List[int]]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        # Later steps: just pass through
        if past_key_values is not None:
            return {
                "input_ids": input_ids,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "row_indices_per_sample": row_indices_per_sample,
            }

        # First step: build prefix then concatenate
        embeds = self._build_qwen_inputs_embeds(input_ids, inputs_embeds)  # (B, S, C)
        B, S, C = embeds.shape

        if not row_indices_per_sample:
            return {"inputs_embeds": embeds, "attention_mask": attention_mask}

        learned, dict_emb, dict_pad_mask, row_pad_mask = self._gather_batch_rows(row_indices_per_sample)
        # RCA stack (vectorized inside)
        for layer in self.dict_encoder:
            out = layer(learned, dict_emb, dict_pad_mask=dict_pad_mask, row_pad_mask=row_pad_mask)
            if isinstance(out, tuple):
                learned, dict_emb = out  # (B, R_sel, T, C), (B, R_sel, L_max, C)
            else:
                learned = out

        Bv, R_sel, T, Cv = learned.shape
        assert Bv == B and Cv == C
        prefix = learned.reshape(B, R_sel * T, C)  # (B, R_sel*T, C)
        valid_rows = (~row_pad_mask).to(learned.dtype)  # (B, R_sel)
        prefix_mask = valid_rows.unsqueeze(-1).expand(B, R_sel, T).reshape(B, R_sel * T)  # (B, R_sel*T)

        base_mask = attention_mask if attention_mask is not None else torch.ones((B, S), device=self._device(), dtype=torch.long)
        pref_mask = prefix_mask.to(dtype=base_mask.dtype)
        final_mask = torch.cat([pref_mask, base_mask], dim=1)  # (B, R_sel*T+S)
        final_embeds = torch.cat([prefix, embeds], dim=1)  # (B, R_sel*T+S, C)

        return {"inputs_embeds": final_embeds, "attention_mask": final_mask}

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        row_indices_per_sample: Optional[List[List[int]]] = None,
        attention_weights: Optional[List[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]]] = None,
        qwen_input_ids: Optional[torch.LongTensor] = None,
        qwen_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        return_dict: Optional[bool] = None,
        **kwargs: Any,
    ) -> Union[Tuple, LexiconCompressorModelOutput]:
        # One-time RCA weight loading
        if not self._rca_weights_loaded_once:
            if attention_weights is None:
                raise ValueError("RowColumnAttention weights must be provided at least once.")
            if len(attention_weights) != self.num_layers:
                raise ValueError(f"Expected {self.num_layers} weight pairs, got {len(attention_weights)}")
            for i, (row_w, col_w) in enumerate(attention_weights):
                self.dict_encoder[i].load_weights_once(row_w, col_w)
            self._rca_weights_loaded_once = True

        return_dict = True if return_dict is None else return_dict
        qwen_embeds = self._build_qwen_inputs_embeds(qwen_input_ids, qwen_inputs_embeds)  # (B, S, C), 'S' is used to describe the seq_len of qwen input, different from L in compressor part
        B, S, C = qwen_embeds.shape

        # Training disables cache
        if self.training and labels is not None:
            use_cache = False

        if past_key_values is None: #first step of decoding
            # First step: build prefix
            if row_indices_per_sample is not None: # select relevant rows for each sample
                if len(row_indices_per_sample) != B:
                    raise ValueError("row_indices_per_sample length must equal batch size")

                learned, dict_emb, dict_pad_mask, row_pad_mask = self._gather_batch_rows(row_indices_per_sample)

                for layer in self.dict_encoder:
                    out = layer(learned, dict_emb, dict_pad_mask=dict_pad_mask, row_pad_mask=row_pad_mask)
                    if isinstance(out, tuple):
                        learned, dict_emb = out
                    else:
                        learned = out

                B2, R_sel, T, C2 = learned.shape
                assert B2 == B and C2 == C
                prefix = learned.reshape(B, R_sel * T, C)  # (B, R_sel*T, C)
                valid_rows = (~row_pad_mask).to(learned.dtype)
                prefix_mask = valid_rows.unsqueeze(-1).expand(B, R_sel, T).reshape(B, R_sel * T)  # (B, R_sel*T)

                base_mask = kwargs.pop("attention_mask", None)
                if base_mask is None:
                    base_mask = torch.ones((B, S), dtype=torch.long, device=self._device())
                final_attention_mask = torch.cat([prefix_mask.to(base_mask.dtype), base_mask], dim=1)  # (B, K+S)
                final_inputs_embeds = torch.cat([prefix, qwen_embeds], dim=1)  # (B, K+S, C)

                if labels is not None:
                    left_pad = torch.full((B, prefix.size(1)), -100, dtype=labels.dtype, device=labels.device)
                    final_labels = torch.cat([left_pad, labels], dim=1)
                else:
                    final_labels = None
                compressed_tokens_list = None

            else:   #prefix from all rows  !!!CUDA OOM
                dict_full, pad_full = self._embed_full_dict()  # (R, L_max, C), (R, L_max)
                learned = self.learned_tokens_global  # (R, T, C)
                dict_emb = dict_full.unsqueeze(0).expand(1, -1, -1, -1)  # (1, R, L_max, C)
                learned = learned.unsqueeze(0)  # (1, R, T, C)
                dummy_row_mask = torch.zeros((1, self.num_rows), dtype=torch.bool, device=self._device())
                dummy_dict_mask = pad_full.unsqueeze(0).expand(1, -1, -1)

                for layer in self.dict_encoder:
                    out = layer(learned, dict_emb, dict_pad_mask=dummy_dict_mask, row_pad_mask=dummy_row_mask)
                    if isinstance(out, tuple):
                        learned, dict_emb = out
                    else:
                        learned = out

                prefix = learned.reshape(1, self.num_rows * self.num_compress_tokens, C)
                pref_mask = torch.ones((1, prefix.size(1)), dtype=torch.long, device=self._device())
                base_mask = kwargs.pop("attention_mask", None)
                if base_mask is None:
                    base_mask = torch.ones((B, S), dtype=torch.long, device=self._device())
                final_inputs_embeds = torch.cat([prefix.expand(B, -1, -1), qwen_embeds], dim=1)
                final_attention_mask = torch.cat([pref_mask.expand(B, -1), base_mask], dim=1)
                final_labels = torch.cat([torch.full((B, prefix.size(1)), -100, dtype=labels.dtype, device=labels.device), labels], dim=1) if labels is not None else None
                compressed_tokens_list = None

            # Length guard
            max_pos = getattr(self.qwen_config, "max_position_embeddings", None)
            if max_pos is not None and final_inputs_embeds.size(1) > max_pos:
                raise ValueError(
                    f"Sequence length {final_inputs_embeds.size(1)} exceeds max_position_embeddings {max_pos}."
                )

        else:    # Later decoding steps
            final_inputs_embeds = qwen_embeds
            attn_step = kwargs.pop("attention_mask", None)
            if attn_step is None:
                attn_step = torch.ones((B, S), dtype=torch.long, device=self._device())
            past_len = self._estimate_past_length(past_key_values)
            if past_len > 0:
                ones = torch.ones((B, past_len), dtype=attn_step.dtype, device=attn_step.device)
                final_attention_mask = torch.cat([ones, attn_step], dim=1)
            else:
                final_attention_mask = attn_step
            final_labels = labels
            compressed_tokens_list = None

        # Base model call
        qwen_kwargs: Dict[str, Any] = {
            "input_ids": None,
            "inputs_embeds": final_inputs_embeds,
            "attention_mask": final_attention_mask,
            "labels": final_labels,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "cache_position": cache_position,
            "logits_to_keep": logits_to_keep,
        }
        for key in ["output_attentions", "output_hidden_states", "return_dict", "position_ids"]: # legal optional args
            if key in kwargs:
                qwen_kwargs[key] = kwargs[key]

        out = self.qwen(**qwen_kwargs)

        if not return_dict:
            items: List[torch.Tensor] = [
                out.loss, out.logits, out.past_key_values, out.hidden_states, out.attentions
            ]
            if compressed_tokens_list is not None:
                items.append(compressed_tokens_list)  # type: ignore[arg-type]
            return tuple(x for x in items if x is not None)

        return LexiconCompressorModelOutput(
            loss=out.loss,
            logits=out.logits,
            past_key_values=out.past_key_values,
            hidden_states=out.hidden_states,
            attentions=out.attentions,
            compressed_tokens=compressed_tokens_list,
        )

    # ------------------------------------------------------------------
    def extra_repr(self) -> str:
        return (
            f"num_rows={self.num_rows}, row_max_len={self.row_max_len}, "
            f"num_layers={self.num_layers}, num_compress_tokens={self.num_compress_tokens}, "
            f"channels={self.channels}"
        )
