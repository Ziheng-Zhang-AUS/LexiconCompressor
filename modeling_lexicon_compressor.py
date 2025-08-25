# lexicon_compressor_model.py
from __future__ import annotations
from typing import List, Optional, Tuple, Dict, Any, Union
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from transformers.cache_utils import Cache
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM, Qwen3Config

from row_column_attention import RowColumnAttention
from configuration_lexicon_compressor import LexiconCompressorConfig


@dataclass
class RCAOutputs:
    """
    Output convention for RowColumnAttention.
    
    Attributes:
        learned_tokens: (B, T, H) compressed tokens
        dict_tokens: (B, Lmax, H) padded dictionary tokens
    """
    learned_tokens: torch.Tensor
    dict_tokens: torch.Tensor


class LexiconCompressorModel(nn.Module):
    """
    A wrapper that compresses each dictionary entry into a fixed number of learned tokens,
    prepends them to Qwen3 input embeddings, and runs through Qwen3ForCausalLM.
    
    Workflow:
        1. Convert dictionary rows (List[List[int]]) into embeddings (B, Lmax, H).
        2. Gather corresponding learned tokens (B, T, H).
        3. Run multiple RowColumnAttention layers.
        4. Keep only compressed tokens (B, T, H), flatten to (B*T, H).
        5. Replicate prefix across N prompts, concatenate with Qwen embeddings (N, BT+S, H).
        6. Run Qwen3ForCausalLM forward.
    """

    def __init__(
        self,
        qwen_model: Qwen3ForCausalLM,
        full_dict: List[List[int]],
        dict_encoder_num_layers: int,
        dict_encoder_num_compress_tokens: int,
        dict_encoder_learned_tokens_prepend: bool = True,
        compressor_config: Optional[LexiconCompressorConfig] = None,
    ):
        """
        Initialize LexiconCompressorModel.
        
        Args:
            qwen_model: Preloaded Qwen3ForCausalLM
            full_dict: Entire dictionary (tokenized), List[List[int]]
            dict_encoder_num_layers: Number of RowColumnAttention layers
            dict_encoder_num_compress_tokens: Number of compress tokens per row
            dict_encoder_learned_tokens_prepend: Whether learned tokens are placed before dict tokens
            compressor_config: Optional custom LexiconCompressorConfig
        """
        super().__init__()
        assert isinstance(qwen_model, Qwen3ForCausalLM), "qwen_model must be Qwen3ForCausalLM"

        self.qwen = qwen_model
        self.qwen_config: Qwen3Config = qwen_model.config
        self.embed_tokens: nn.Embedding = qwen_model.model.embed_tokens
        self.hidden_size: int = self.qwen_config.hidden_size

        self.full_dict = full_dict
        self.num_rows = len(full_dict)
        self.num_layers = dict_encoder_num_layers
        self.num_compress_tokens = dict_encoder_num_compress_tokens
        self.learned_tokens_prepend = dict_encoder_learned_tokens_prepend

        self.config = compressor_config or LexiconCompressorConfig(
            qwen_config=self.qwen_config,
            num_layers=self.num_layers,
            num_compress_tokens=self.num_compress_tokens,
            learned_tokens_prepend=self.learned_tokens_prepend,
        )

        # Learned tokens for each dictionary row
        learned = torch.randn(self.num_rows, self.num_compress_tokens, self.hidden_size) * 0.02
        self.learned_tokens_global = nn.Parameter(learned)

        # Encoder stack
        self.dict_encoder = nn.ModuleList([
            RowColumnAttention(self.config) for _ in range(self.num_layers)
        ])

        self._rca_weights_loaded_once: bool = False

    def _device(self) -> torch.device:
        return self.embed_tokens.weight.device

    def _dtype(self) -> torch.dtype:
        return self.embed_tokens.weight.dtype

    @staticmethod
    def _validate_xor(a: Any, b: Any, name_a: str, name_b: str):
        """
        Ensure exactly one of the two arguments is provided.
        """
        if (a is None) == (b is None):
            raise ValueError(f"Exactly one of `{name_a}` or `{name_b}` must be provided.")

    def _gather_learned_tokens(
        self,
        row_indices: Optional[List[int]],
        batch_rows: int,
    ) -> torch.Tensor:
        """
        Gather learned tokens for the current batch rows.

        Args:
            row_indices: Indices of rows to select from the global table
            batch_rows: Number of rows in the batch

        Returns:
            Tensor (B, T, H)
        """
        if row_indices is not None:
            idx = torch.tensor(row_indices, dtype=torch.long, device=self._device())
        else:
            if batch_rows > self.num_rows:
                raise ValueError("Batch rows exceed global dictionary size without row_indices.")
            idx = torch.arange(batch_rows, device=self._device())

        return self.learned_tokens_global.index_select(dim=0, index=idx)

    def _embed_rows(self, token_ids_list: List[List[int]]) -> Tuple[torch.Tensor, torch.LongTensor]:
        """
        Embed dictionary rows.

        Args:
            token_ids_list: List of tokenized dictionary rows

        Returns:
            dict_tokens: (B, Lmax, H)
            dict_lens: (B,)
        """
        B = len(token_ids_list)
        if B == 0:
            raise ValueError("token_ids_list is empty.")

        device = self._device()
        dtype = self._dtype()

        row_tensors = []
        lengths = []

        for ids in token_ids_list:
            t = torch.tensor(ids, dtype=torch.long, device=device)   # (Li,)
            e = self.embed_tokens(t)                                # (Li, H)
            row_tensors.append(e)
            lengths.append(e.shape[0])

        lengths_t = torch.tensor(lengths, dtype=torch.long, device=device)
        padded = pad_sequence(row_tensors, batch_first=True, padding_value=0.0)
        padded = padded.to(dtype=dtype)

        return padded, lengths_t

    def _build_qwen_inputs_embeds(
        self,
        qwen_input_ids: Optional[torch.LongTensor],
        qwen_inputs_embeds: Optional[torch.FloatTensor],
    ) -> torch.FloatTensor:
        """
        Convert Qwen inputs into embeddings.

        Args:
            qwen_input_ids: Input ids for Qwen
            qwen_inputs_embeds: Precomputed embeddings

        Returns:
            Embeddings (N, S, H)
        """
        self._validate_xor(qwen_input_ids, qwen_inputs_embeds, "qwen_input_ids", "qwen_inputs_embeds")

        if qwen_inputs_embeds is not None:
            return qwen_inputs_embeds.to(device=self._device(), dtype=self._dtype())

        assert qwen_input_ids is not None
        return self.embed_tokens(qwen_input_ids.to(self._device()))

    def _maybe_left_pad_labels(self, labels: Optional[torch.LongTensor], prefix_len: int) -> Optional[torch.LongTensor]:
        """
        Left-pad labels with -100 for prefix tokens.

        Args:
            labels: (N, S)
            prefix_len: Number of prefix tokens

        Returns:
            Padded labels (N, prefix_len+S)
        """
        if labels is None:
            return None
        if prefix_len == 0:
            return labels
        N, S = labels.shape
        pad = labels.new_full((N, prefix_len), fill_value=-100)
        return torch.cat([pad, labels], dim=1)

    def load_attention_weights_once(self, layer_weights: List[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]]):
        """
        Load weights for each Row/Col Attention layer once.
        
        Args:
            layer_weights: List of tuples (row_weights, col_weights)
        """
        if self._rca_weights_loaded_once:
            return
        if len(layer_weights) != self.num_layers:
            raise ValueError(f"Expected {self.num_layers} weight pairs, got {len(layer_weights)}")
        for i, (row_w, col_w) in enumerate(layer_weights):
            self.dict_encoder[i].load_weights_once(row_w, col_w)
        self._rca_weights_loaded_once = True

    def forward(
        self,
        token_ids_list: List[List[int]],
        row_indices: Optional[List[int]] = None,
        attention_weights: Optional[List[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]]] = None,
        rca_kwargs_per_layer: Optional[List[Dict[str, Any]]] = None,
        qwen_input_ids: Optional[torch.LongTensor] = None,
        qwen_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Any,
    ):
        """
        Forward pass for training and inference.

        Args:
            token_ids_list: List of dictionary rows (List[List[int]])
            row_indices: Indices of rows in the full_dict
            attention_weights: Optional Row/Col weights
            rca_kwargs_per_layer: Extra kwargs for each RCA layer
            qwen_input_ids: Qwen input ids
            qwen_inputs_embeds: Qwen embeddings
            labels: Labels for LM loss
            past_key_values, use_cache, cache_position, logits_to_keep, kwargs: Passed to Qwen

        Returns:
            transformers.CausalLMOutputWithPast
        """
        if attention_weights is not None and not self._rca_weights_loaded_once:
            self.load_attention_weights_once(attention_weights)

        dict_tokens, dict_lens = self._embed_rows(token_ids_list)
        B, Lmax, H = dict_tokens.shape
        assert H == self.hidden_size

        learned_tokens = self._gather_learned_tokens(row_indices=row_indices, batch_rows=B)

        if rca_kwargs_per_layer is not None and len(rca_kwargs_per_layer) != self.num_layers:
            raise ValueError("rca_kwargs_per_layer length mismatch.")

        for i, layer in enumerate(self.dict_encoder):
            layer_kwargs = (rca_kwargs_per_layer[i] if rca_kwargs_per_layer is not None else {})
            out: RCAOutputs | Tuple[torch.Tensor, torch.Tensor]
            out = layer(
                learned_tokens=learned_tokens,
                dict_tokens=dict_tokens,
                dict_lens=dict_lens,
                **layer_kwargs,
            )
            if isinstance(out, tuple):
                learned_tokens, dict_tokens = out
            else:
                learned_tokens, dict_tokens = out.learned_tokens, out.dict_tokens

        compressed = learned_tokens.reshape(B * self.num_compress_tokens, H)

        qwen_embeds = self._build_qwen_inputs_embeds(qwen_input_ids, qwen_inputs_embeds)
        N, S, H2 = qwen_embeds.shape
        assert H2 == H

        prefix = compressed.unsqueeze(0).expand(N, -1, -1).contiguous()

        total_len = prefix.size(1) + S
        max_pos = getattr(self.qwen_config, "max_position_embeddings", None)
        if max_pos is not None and total_len > max_pos:
            raise ValueError(f"Total length {total_len} exceeds max_position_embeddings {max_pos}")

        final_inputs_embeds = torch.cat([prefix, qwen_embeds], dim=1)

        attention_mask = None
        final_labels = self._maybe_left_pad_labels(labels, prefix_len=prefix.size(1))

        outputs = self.qwen(
            input_ids=None,
            inputs_embeds=final_inputs_embeds,
            attention_mask=attention_mask,
            labels=final_labels,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )

        return outputs

    # @torch.no_grad()
    # def compress_only(
    #     self,
    #     token_ids_list: List[List[int]],
    #     row_indices: Optional[List[int]] = None,
    #     rca_kwargs_per_layer: Optional[List[Dict[str, Any]]] = None,
    # ) -> torch.Tensor:
    #     """
    #     Only compute compressed prefix for dictionary rows.

    #     Args:
    #         token_ids_list: List of dictionary rows
    #         row_indices: Indices in full_dict
    #         rca_kwargs_per_layer: Extra kwargs per RCA layer

    #     Returns:
    #         Compressed tokens (B*T, H)
    #     """
    #     dict_tokens, dict_lens = self._embed_rows(token_ids_list)
    #     B, _, H = dict_tokens.shape
    #     learned_tokens = self._gather_learned_tokens(row_indices=row_indices, batch_rows=B)

    #     for i, layer in enumerate(self.dict_encoder):
    #         layer_kwargs = (rca_kwargs_per_layer[i] if rca_kwargs_per_layer is not None else {})
    #         lt, dt = layer(
    #             learned_tokens=learned_tokens,
    #             dict_tokens=dict_tokens,
    #             dict_lens=dict_lens,
    #             **layer_kwargs,
    #         )
    #         learned_tokens, dict_tokens = lt, dt

    #     return learned_tokens.reshape(B * self.num_compress_tokens, H)