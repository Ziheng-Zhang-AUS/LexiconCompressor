from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import Qwen3Config, Qwen3ForCausalLM, PreTrainedModel
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast

from row_column_attention import RowColumnAttention
from configuration_lexicon_compressor import LexiconCompressorConfig


# -----------------------------------------------------------------------------
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
class LexiconCompressorModel(PreTrainedModel):
    """Vectorized dictionary-to-prefix wrapper over Qwen3 with functional RowColumnAttention.

    Pipeline:
      1) pad the full dictionary to a tensor (R, L_max),
      2) embed once as (R, L_max, C) with a pad mask (R, L_max),
      3) gather per-batch rows via advanced indexing (no Python loops),
      4) run functional Row/Column Attention on batched tensors with external weights, and
      5) flatten per-row learned tokens to a prefix, then prepend to Qwen3.

    Key improvement: RowColumnAttention modules are now purely functional and don't hold
    any parameters. All weights are passed as dictionaries during forward pass.
    """

    def __init__(
        self,
        qwen_model: Qwen3ForCausalLM,
        full_dict: List[List[int]],
        dict_encoder_num_compress_tokens: int,
        dict_encoder_learned_tokens_prepend: bool = True,
        compressor_config: Optional[LexiconCompressorConfig] = None,
    ) -> None:
        
        # ===== 第一步：纯数据计算，为创建 config 做准备 =====
        if not isinstance(qwen_model, Qwen3ForCausalLM):
            model_class_name = qwen_model.__class__.__name__
            if "Qwen" not in model_class_name:
                raise ValueError(f"qwen_model must be a Qwen model, got {model_class_name}")
            print(f"Warning: Using {model_class_name} instead of Qwen3ForCausalLM for testing")

        # 提取创建 config 所需的信息
        qwen_config_temp = qwen_model.config
        num_layers_temp = len(qwen_model.model.layers)
        channels_temp = qwen_config_temp.hidden_size
        num_heads_temp = qwen_config_temp.num_attention_heads
        head_dim_temp = getattr(qwen_config_temp, "head_dim", channels_temp // num_heads_temp)

        # 处理词典数据
        num_rows_temp = len(full_dict)
        lens_temp = torch.tensor([len(r) for r in full_dict], dtype=torch.long) if num_rows_temp > 0 else torch.zeros(0, dtype=torch.long)
        row_max_len_temp = int(lens_temp.max().item()) if num_rows_temp > 0 else 1

        # 创建 config
        self.config = compressor_config or LexiconCompressorConfig(
            qwen_config=qwen_config_temp,
            num_layers=num_layers_temp,
            num_compress_tokens=dict_encoder_num_compress_tokens,
            learned_tokens_prepend=dict_encoder_learned_tokens_prepend,
        )

        # ===== 第二步：立即调用父类初始化 =====
        super().__init__(self.config)

        # ===== 第三步：现在可以安全地赋值所有模块、参数和缓冲区 =====
        # Base model refs
        self.qwen = qwen_model
        self.qwen.tie_weights()
        print("model.qwen has been tie weights during init.")
        self.qwen_config: Qwen3Config = self.config.qwen_config # <-- 注意：这里从 self.config 读取
        self.channels: int = channels_temp  # C
        self.num_heads: int = num_heads_temp
        self.head_dim: int = head_dim_temp

        # Dictionary
        self.full_dict_list = full_dict
        self.num_rows = num_rows_temp
        self.row_max_len = row_max_len_temp

        # Build (R, L_max) filled with -1, then scatter valid tokens by a boolean mask
        idx_padded = torch.full((self.num_rows, self.row_max_len), -1, dtype=torch.long)  # (R, L_max)
        col = torch.arange(self.row_max_len).expand(self.num_rows, -1)              # (R, L_max)
        valid_mask = col < lens_temp.unsqueeze(1)                                        # (R, L_max) bool
        flat_vals = torch.tensor([x for row in full_dict for x in row], dtype=torch.long)
        idx_padded[valid_mask] = flat_vals                                          # row-major fill
        pad_mask = idx_padded.eq(-1)                                                    # (R, L_max) bool
        self.register_buffer("full_dict_index", idx_padded, persistent=False)
        self.register_buffer("full_dict_pad_mask", pad_mask, persistent=False)

        # Learned tokens per row: (R, T, C)
        self.num_layers = num_layers_temp  # equals to the number of qwen decoder layers
        self.num_compress_tokens = dict_encoder_num_compress_tokens  # T
        self.learned_tokens_prepend = dict_encoder_learned_tokens_prepend
        init = torch.randn(self.num_rows, self.num_compress_tokens, self.channels)
        self.learned_tokens_global = nn.Parameter(init)

        # Functional RCA stack - no parameters, purely functional modules
        self.dict_encoder = nn.ModuleList([
            RowColumnAttention(self.config) for _ in range(self.num_layers)
        ])

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

    def extract_qwen_layer_weights(self, layer_idx: int) -> Dict[str, torch.Tensor]:
        """Extract weights from a specific Qwen3DecoderLayer for functional use."""
        if layer_idx >= len(self.qwen.model.layers):
            raise ValueError(f"Layer index {layer_idx} out of range (max: {len(self.qwen.model.layers)-1})")
        
        layer = self.qwen.model.layers[layer_idx]
        weights = {}
        # qwen_core = getattr(self.qwen, "module", self.qwen)  
        # layer = qwen_core.model.layers[layer_idx]
        # print(sum(p.numel() for p in layer.parameters()))

        # 直接拿参数
        for name, param in layer.named_parameters():
            # param 是 nn.Parameter，需要 .data 或 .detach() 转成 Tensor
            weights[name] = param.detach().clone()
            # print(f"{name}: {tuple(param.shape)}")
        
        return weights


    def get_all_layer_weights(self) -> List[Dict[str, torch.Tensor]]:
        """Extract weights from all Qwen3 layers for functional RCA use.
        
        Returns:
            List of weight dictionaries, one per layer
        """
        return [self.extract_qwen_layer_weights(i) for i in range(self.num_layers)]

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

        idx_padded = idx_padded.to(self._device())
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
    # Functional RCA processing
    # ------------------------------------------------------------------
    def _apply_functional_rca_stack(
        self,
        learned: torch.Tensor,  # (B, R_sel, T, C)
        dict_emb: torch.Tensor,  # (B, R_sel, L_max, C)
        dict_pad_mask: torch.Tensor,  # (B, R_sel, L_max)
        row_pad_mask: torch.Tensor,  # (B, R_sel)
        layer_weights: List[Dict[str, torch.Tensor]],  # Weights for each layer
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply functional RCA stack with external weights.
        
        Args:
            learned: Learned tokens (B, R_sel, T, C)
            dict_emb: Dictionary embeddings (B, R_sel, L_max, C)
            dict_pad_mask: Dictionary padding mask (B, R_sel, L_max)
            row_pad_mask: Row padding mask (B, R_sel)
            layer_weights: List of weight dictionaries for each RCA layer
            
        Returns:
            Updated (learned, dict_emb) tensors
        """
        if len(layer_weights) != len(self.dict_encoder):
            raise ValueError(f"Expected {len(self.dict_encoder)} weight dicts, got {len(layer_weights)}")

        base_dtype = self._dtype()
        learned = learned.to(base_dtype)
        dict_emb = dict_emb.to(base_dtype)
        
        for i, (rca_layer, weights) in enumerate(zip(self.dict_encoder, layer_weights)):
            # For functional RCA, we need to provide both row and column weights
            # In this case, we use the same layer weights for both passes
            # You might want to use different layer weights or create separate weight sets
            # print(f"RCA layer{i} got weights, type of {type(weights)}")
            out = rca_layer(
                learned=learned,
                dict_emb=dict_emb,
                row_weights_dict=weights,
                col_weights_dict=weights,  # Using same weights for both passes
                dict_pad_mask=dict_pad_mask,
                row_pad_mask=row_pad_mask,
            )
            
            if isinstance(out, tuple):
                learned, dict_emb = out
            else:
                learned = out
                
        return learned, dict_emb

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
        layer_weights: Optional[List[Dict[str, torch.Tensor]]] = None,
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
                "layer_weights": layer_weights,
            }

        # First step: build prefix then concatenate
        embeds = self._build_qwen_inputs_embeds(input_ids, inputs_embeds)  # (B, S, C)
        B, S, C = embeds.shape

        if row_indices_per_sample is None:
            return {"inputs_embeds": embeds, "attention_mask": attention_mask}

        # Get layer weights if not provided
        if layer_weights is None:
            layer_weights = self.get_all_layer_weights()

        learned, dict_emb, dict_pad_mask, row_pad_mask = self._gather_batch_rows(row_indices_per_sample)
        
        # Apply functional RCA stack
        learned, dict_emb = self._apply_functional_rca_stack(
            learned, dict_emb, dict_pad_mask, row_pad_mask, layer_weights
        )

        Bv, R_sel, T, Cv = learned.shape
        assert Bv == B and Cv == C
        prefix = learned.reshape(B, R_sel * T, C)  # (B, R_sel*T, C)
        valid_rows = (~row_pad_mask).to(learned.dtype)  # (B, R_sel)
        prefix_mask = valid_rows.unsqueeze(-1).expand(B, R_sel, T).reshape(B, R_sel * T)  # (B, R_sel*T)

        base_mask = attention_mask if attention_mask is not None else torch.ones((B, S), device=self._device(), dtype=torch.long)
        pref_mask = prefix_mask.to(dtype=base_mask.dtype)
        final_mask = torch.cat([pref_mask, base_mask], dim=1)  # (B, R_sel*T+S)
        final_embeds = torch.cat([prefix, embeds], dim=1)  # (B, R_sel*T+S, C)

        return {
            "inputs_embeds": final_embeds, 
            "attention_mask": final_mask,
            "layer_weights": layer_weights,
        }

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        row_indices_per_sample: Optional[List[List[int]]] = None,
        layer_weights: Optional[List[Dict[str, torch.Tensor]]] = None,
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
        """Forward pass with functional RowColumnAttention.
        
        Args:
            row_indices_per_sample: Row indices for each sample in batch
            layer_weights: List of weight dictionaries for each RCA layer.
                          If None, will extract from Qwen model layers.
            ... (other args same as before)
        """
        return_dict = True if return_dict is None else return_dict
        qwen_embeds = self._build_qwen_inputs_embeds(qwen_input_ids, qwen_inputs_embeds)
        B, S, C = qwen_embeds.shape

        # Training disables cache
        if self.training and labels is not None:
            use_cache = False

        # Get layer weights if not provided
        if layer_weights is None:
            layer_weights = self.get_all_layer_weights()
            # print("LexiconCompressorModel automatically get all qwen layer weights.")
            # print(type(layer_weights))

        if past_key_values is None:  # First step of decoding
            # First step: build prefix
            if row_indices_per_sample is not None:  # select relevant rows for each sample
                if len(row_indices_per_sample) != B:
                    raise ValueError("row_indices_per_sample length must equal batch size")

                learned, dict_emb, dict_pad_mask, row_pad_mask = self._gather_batch_rows(row_indices_per_sample)

                # Apply functional RCA stack
                learned, dict_emb = self._apply_functional_rca_stack(
                    learned, dict_emb, dict_pad_mask, row_pad_mask, layer_weights
                )

                B2, R_sel, T, C2 = learned.shape
                assert B2 == B and C2 == C
                prefix = learned.reshape(B, R_sel * T, C)  # (B, R_sel*T, C)
                valid_rows = (~row_pad_mask).to(learned.dtype)
                prefix_mask = valid_rows.unsqueeze(-1).expand(B, R_sel, T).reshape(B, R_sel * T)

                base_mask = kwargs.pop("attention_mask", None)
                if base_mask is None:
                    base_mask = torch.ones((B, S), dtype=torch.long, device=self._device())
                final_attention_mask = torch.cat([prefix_mask.to(base_mask.dtype), base_mask], dim=1)
                final_inputs_embeds = torch.cat([prefix, qwen_embeds], dim=1)

                if labels is not None:
                    left_pad = torch.full((B, prefix.size(1)), -100, dtype=labels.dtype, device=labels.device)
                    final_labels = torch.cat([left_pad, labels], dim=1)
                else:
                    final_labels = None
                compressed_tokens_list = None

            else:   # prefix from all rows (might cause CUDA OOM)
                dict_full, pad_full = self._embed_full_dict()  # (R, L_max, C), (R, L_max)
                learned = self.learned_tokens_global  # (R, T, C)
                dict_emb = dict_full.unsqueeze(0).expand(1, -1, -1, -1)  # (1, R, L_max, C)
                learned = learned.unsqueeze(0)  # (1, R, T, C)
                dummy_row_mask = torch.zeros((1, self.num_rows), dtype=torch.bool, device=self._device())
                dummy_dict_mask = pad_full.unsqueeze(0).expand(1, -1, -1)

                # Apply functional RCA stack
                learned, dict_emb = self._apply_functional_rca_stack(
                    learned, dict_emb, dummy_dict_mask, dummy_row_mask, layer_weights
                )

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
        for key in ["output_attentions", "output_hidden_states", "return_dict", "position_ids"]:
            if key in kwargs:
                qwen_kwargs[key] = kwargs[key]

        out = self.qwen(**qwen_kwargs)

        if not return_dict:
            items: List[torch.Tensor] = [
                out.loss, out.logits, out.past_key_values, out.hidden_states, out.attentions
            ]
            if compressed_tokens_list is not None:
                items.append(compressed_tokens_list)
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
    # Model Saving (Fix for shared tensors)
    # ------------------------------------------------------------------
    def _save(self, output_dir: str):
        """
        Custom save method to handle shared tensors in the wrapped Qwen model.
        This overrides the default behavior in Hugging Face Trainer that causes the safetensors error.
        
        Uses `safetensors.torch.save_model` which correctly handles weight sharing.
        """
        print("Save model using customized '_save' method")
        from safetensors.torch import save_model as safe_save_model
        import os

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)


        # Save the entire model using safetensors' model-aware function
        # This function intelligently handles shared memory tensors.
        safe_save_model(self, os.path.join(output_dir, "model.safetensors"))

        # Save the model's configuration if it has one (highly recommended for reloading)
        if hasattr(self, 'config') and self.config is not None:
            self.config.save_pretrained(output_dir)
        else:
            # If you don't have a config yet, consider creating one based on LexiconCompressorConfig
            print("Warning: No 'config' attribute found. Consider adding one for easier model reloading.")

        # Optional: Save any additional custom components
        # For example, if your dictionary or tokenizer state needs to be saved, do it here.
        # You might want to save `self.full_dict_list` or metadata about the learned tokens.
        # torch.save(self.full_dict_list, os.path.join(output_dir, "dictionary.pt"))

    def save_pretrained(self, save_directory: str, **kwargs):
        print(f"Received kwargs: {list(kwargs.keys())}") # 打印收到的所有额外参数

        # 如果传入了 state_dict，我们可以选择警告用户
        if 'state_dict' in kwargs:
            print("Warning: Ignoring provided 'state_dict' in favor of custom save logic.")

        # 如果 safe_serialization 是 False，你可以选择回退到 torch.save
        # 但在你的情况下，你的 _save 方法已经强制使用 safetensors，所以可以忽略
        # safe_serialization = kwargs.get('safe_serialization', True)

        # 执行你的自定义保存逻辑
        self._save(save_directory)
        print(f"Model saved to {save_directory}")
    

    def extra_repr(self) -> str:
        return (
            f"num_rows={self.num_rows}, row_max_len={self.row_max_len}, "
            f"num_layers={self.num_layers}, num_compress_tokens={self.num_compress_tokens}, "
            f"channels={self.channels} (functional RCA)"
        )


# ================================================================
# Real Test with Actual Qwen Model and Dictionary
# ================================================================

if __name__ == "__main__":
    import os
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    def load_lexicon_from_csv(tokenizer, csv_path: str = "data/cleaned_lexicon_tiny.csv") -> List[List[int]]:
        """Load lexicon from CSV file and tokenize entries."""
        import pandas as pd
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Lexicon file not found: {csv_path}")
        
        # Load CSV
        print(f"Loading lexicon from {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"CSV columns: {df.columns.tolist()}")
        print(f"CSV shape: {df.shape}")
        
        # Assume the lexicon entries are in a column (adjust column name as needed)
        # Common column names: 'text', 'entry', 'word', 'phrase', etc.
        text_columns = ['text', 'entry', 'word', 'phrase', 'lexicon_entry', 'content']
        text_column = None
        
        for col in text_columns:
            if col in df.columns:
                text_column = col
                break
        
        if text_column is None:
            # If no standard column found, use the first string column
            for col in df.columns:
                if df[col].dtype == 'object':
                    text_column = col
                    print(f"Using column '{col}' as text column")
                    break
        
        if text_column is None:
            raise ValueError(f"No suitable text column found in CSV. Available columns: {df.columns.tolist()}")
        
        # Extract text entries and remove NaN
        texts = df[text_column].dropna().astype(str).tolist()
        print(f"Loaded {len(texts)} lexicon entries")
        print(f"Sample entries: {texts[:5]}")
        
        # Tokenize each entry
        dictionary = []
        max_entries = min(len(texts), 100)  # Limit for testing
        
        for i, text in enumerate(texts[:max_entries]):
            # Clean and tokenize
            text = text.strip()
            if len(text) > 0:
                tokens = tokenizer.encode(text, add_special_tokens=False)
                if len(tokens) > 0:  # Only add non-empty tokenizations
                    dictionary.append(tokens)
        
        print(f"Successfully tokenized {len(dictionary)} dictionary entries")
        if len(dictionary) > 0:
            avg_len = sum(len(entry) for entry in dictionary) / len(dictionary)
            max_len = max(len(entry) for entry in dictionary)
            print(f"Average tokens per entry: {avg_len:.1f}")
            print(f"Max tokens per entry: {max_len}")
        
        return dictionary
    
    def create_sample_batch() -> Tuple[torch.Tensor, List[List[int]]]:
        """Create a sample batch for testing."""
        # Sample input text
        texts = [
            "Tell me about artificial intelligence",
            "What is machine learning?"
        ]
        
        # Sample row indices (which dictionary entries to use for each sample)
        row_indices_per_sample = [
            [0, 1, 2, 5],    # First sample uses rows 0,1,2,5
            [3, 4, 6, 7, 8]  # Second sample uses rows 3,4,6,7,8
        ]
        
        return texts, row_indices_per_sample
    
    def pad_row_indices(row_indices_per_sample: List[List[int]]) -> torch.Tensor:
        """Pad row indices to same length with -1."""
        if not row_indices_per_sample:
            return torch.empty(0, 0, dtype=torch.long)
            
        max_len = max(len(indices) for indices in row_indices_per_sample)
        B = len(row_indices_per_sample)
        
        padded = torch.full((B, max_len), -1, dtype=torch.long)
        for i, indices in enumerate(row_indices_per_sample):
            if len(indices) > 0:
                padded[i, :len(indices)] = torch.tensor(indices, dtype=torch.long)
                
        return padded
    
    def test_functional_lexicon_compressor():
        """Test the functional LexiconCompressorModel with real Qwen model."""
        print("Testing Functional LexiconCompressorModel with real Qwen...")
        
        # Configuration
        model_name = "Qwen/Qwen3-0.6B"  # Use base model instead of instruct for compatibility
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        try:
            # Load tokenizer and model
            print("Loading tokenizer and model...")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
            # Use generic AutoModelForCausalLM to avoid Qwen3 dependency
            qwen_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                device_map=device if device == "cuda" else None,
                trust_remote_code=True
            )
            
            print(f"Loaded model: {qwen_model.__class__.__name__}")
            
            # Create sample dictionary
            print("Loading dictionary from CSV...")
            dictionary = load_lexicon_from_csv(tokenizer)
            print(f"Dictionary size: {len(dictionary)} entries")
            print(f"Sample entries: {dictionary[:3]}")
            
            # Initialize LexiconCompressor
            print("Initializing LexiconCompressorModel...")
            compressor = LexiconCompressorModel(
                qwen_model=qwen_model,
                full_dict=dictionary,
                dict_encoder_num_compress_tokens=4,  # 4 compressed tokens per row
                dict_encoder_learned_tokens_prepend=True,
            ).to(device)
            
            # Check parameter counts
            total_params = sum(p.numel() for p in compressor.parameters())
            qwen_params = sum(p.numel() for p in compressor.qwen.parameters())
            rca_params = sum(p.numel() for p in compressor.dict_encoder.parameters())
            learned_params = compressor.learned_tokens_global.numel()
            
            print(f"Parameter breakdown:")
            print(f"  Total params: {total_params:,}")
            print(f"  Qwen params: {qwen_params:,}")
            print(f"  RCA params: {rca_params:,}")  # Should be near 0
            print(f"  Learned tokens: {learned_params:,}")
            print(f"  RCA is functional: {rca_params < 1000}")  # Should be True
            
            # Create sample batch
            print("\nPreparing test inputs...")
            texts, row_indices_per_sample = create_sample_batch()
            
            # Tokenize inputs
            inputs = tokenizer(
                texts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=128
            ).to(device)
            
            # Pad row indices
            row_indices_padded = pad_row_indices(row_indices_per_sample).to(device)
            
            print(f"Input shapes:")
            print(f"  Input IDs: {inputs.input_ids.shape}")
            print(f"  Attention mask: {inputs.attention_mask.shape}")
            print(f"  Row indices: {row_indices_padded.shape}")
            
            # Test forward pass
            print("\nRunning forward pass...")
            compressor.eval()
            
            with torch.no_grad():
                # Method 1: Auto-extract weights (recommended)
                outputs = compressor(
                    qwen_input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    row_indices_per_sample=row_indices_padded,
                )
                
                print(f"Output shapes:")
                print(f"  Logits: {outputs.logits.shape}")
                if outputs.hidden_states is not None:
                    print(f"  Hidden states: {outputs.hidden_states.shape}")
                
                # Check that prefix was added
                original_seq_len = inputs.input_ids.shape[1]
                output_seq_len = outputs.logits.shape[1]
                prefix_len = output_seq_len - original_seq_len
                print(f"  Original seq len: {original_seq_len}")
                print(f"  Output seq len: {output_seq_len}")
                print(f"  Prefix length: {prefix_len}")
                
                # Method 2: Manual weight extraction
                print("\nTesting manual weight extraction...")
                layer_weights = compressor.get_all_layer_weights()
                print(f"Extracted weights for {len(layer_weights)} layers")
                
                # Check a sample weight dict
                sample_weights = layer_weights[0]
                print(f"Sample layer weights keys: {list(sample_weights.keys())}")
                
                outputs2 = compressor(
                    qwen_input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    row_indices_per_sample=row_indices_padded,
                    layer_weights=layer_weights,
                )
                
                # Results should be identical
                logits_diff = torch.abs(outputs.logits - outputs2.logits).max().item()
                print(f"Logits difference between methods: {logits_diff}")
                assert logits_diff < 1e-5, "Results should be identical"
                
            print("\n✓ All functional tests passed!")
            
            # Test generation (optional)
            print("\nTesting generation...")
            generation_inputs = compressor.prepare_inputs_for_generation(
                input_ids=inputs.input_ids[:1],  # Just first sample
                attention_mask=inputs.attention_mask[:1],
                row_indices_per_sample=row_indices_padded[:1],
            )
            
            print(f"Generation input shapes:")
            print(f"  Embeds: {generation_inputs['inputs_embeds'].shape}")
            print(f"  Attention mask: {generation_inputs['attention_mask'].shape}")
            
            # Try actual generation (short)
            with torch.no_grad():
                generated = qwen_model.generate(
                    inputs_embeds=generation_inputs['inputs_embeds'],
                    attention_mask=generation_inputs['attention_mask'],
                    max_new_tokens=10,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
                
                # Decode (skip the prefix part)
                prefix_len = generation_inputs['inputs_embeds'].shape[1] - inputs.input_ids.shape[1]
                generated_text = tokenizer.decode(generated[0, prefix_len:], skip_special_tokens=True)
                print(f"Generated text: {generated_text}")
            
            print("\n🎉 All tests completed successfully!")
            
        except Exception as e:
            print(f"\n❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
            
        finally:
            # Cleanup
            if 'qwen_model' in locals():
                del qwen_model
            if 'compressor' in locals():
                del compressor
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Run the test
    print("="*60)
    print("FUNCTIONAL LEXICON COMPRESSOR REAL TEST")
    print("="*60)
    test_functional_lexicon_compressor()