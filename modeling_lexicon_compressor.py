# modeling_lexicon_compressor.py
from __future__ import annotations
from typing import List, Optional, Tuple, Dict, Any, Union
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from transformers.cache_utils import Cache
from transformers import Qwen3ForCausalLM, Qwen3Config, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast

from row_column_attention import RowColumnAttention
from configuration_lexicon_compressor import LexiconCompressorConfig
from tokenization_lexicon_compressor import LexiconCompressorTokenizor


@dataclass
class LexiconCompressorModelOutput(CausalLMOutputWithPast):
    """
    Output class for LexiconCompressorModel.
    
    Args:
        loss: Language modeling loss
        logits: Prediction scores
        past_key_values: Key/value states from the model
        hidden_states: Hidden-states of the model
        attentions: Attentions weights
        compressed_tokens: Compressed dictionary tokens for each sample
    """
    compressed_tokens: Optional[List[torch.FloatTensor]] = None


class LexiconCompressorModel(nn.Module):
    """
    A wrapper that compresses dictionary entries into fixed learned tokens,
    prepends them to Qwen3 input embeddings, and runs through Qwen3ForCausalLM.
    
    Supports per-sample dictionary entries for personalized compression.
    """

    def __init__(
        self,
        qwen_model: Qwen3ForCausalLM,
        full_dict: List[List[int]],
        dict_encoder_num_compress_tokens: int,
        dict_encoder_learned_tokens_prepend: bool = True,
        compressor_config: Optional[LexiconCompressorConfig] = None,
    ):
        """
        Initialize LexiconCompressorModel.
        
        Args:
            qwen_model: Preloaded Qwen3ForCausalLM
            full_dict: Entire dictionary (tokenized), List[List[int]]
            dict_encoder_num_compress_tokens: Number of compress tokens per row
            dict_encoder_learned_tokens_prepend: Whether learned tokens are placed before dict tokens
            compressor_config: Optional custom LexiconCompressorConfig
        """
        super().__init__()
        
        if not isinstance(qwen_model, Qwen3ForCausalLM):
            raise ValueError("qwen_model must be Qwen3ForCausalLM")

        self.qwen = qwen_model 
        self.qwen_config: Qwen3Config = qwen_model.config
        # self.embed_tokens: nn.Embedding = qwen_model.model.embed_tokens
        self.hidden_size: int = self.qwen_config.hidden_size
        self.num_heads: int = self.qwen_config.num_attention_heads
        self.head_dim: int = getattr(self.qwen_config, "head_dim", self.hidden_size // self.num_heads)

        self.full_dict = full_dict
        self.num_rows = len(full_dict)
        self.num_layers = len(self.qwen.model.layers) # equals to num of decoder layers in qwen3
        self.num_compress_tokens = dict_encoder_num_compress_tokens
        self.learned_tokens_prepend = dict_encoder_learned_tokens_prepend #bool

        self.config = compressor_config or LexiconCompressorConfig(
            qwen_config=self.qwen_config,
            num_layers=self.num_layers,
            num_compress_tokens=self.num_compress_tokens,
            learned_tokens_prepend=self.learned_tokens_prepend,
        )

        # Learned tokens for each dictionary row
        learned = torch.randn(self.num_rows, self.num_compress_tokens, self.hidden_size)
        self.learned_tokens_global = nn.Parameter(learned) # make it learnable

        # Encoder stack
        self.dict_encoder = nn.ModuleList([
            RowColumnAttention(self.config) for _ in range(self.num_layers)
        ])

        self._rca_weights_loaded_once: bool = False

    def _get_device(self) -> torch.device:
        """Get device of model parameters."""
        for p in self.parameters():
            return p.device
        for b in self.buffers():
            return b.device
        return torch.device("cpu")

    def _get_dtype(self) -> torch.dtype:
        """Get dtype of model parameters."""
        try:
            return next(self.qwen.parameters()).dtype
        except StopIteration:
            return torch.float32

    def _embed_dictionary_rows(self, token_ids_list: List[List[int]]) -> List[torch.Tensor]:
        """
        Embed dictionary rows into token embeddings.
        
        Args:
            token_ids_list: List of token ID lists for each dictionary entry
            
        Returns:
            List of embedded tensors, each (Li, H)
        """
        if not token_ids_list:
            raise ValueError("token_ids_list is empty.")
            
        device = self._get_device()
        embedded_rows = []
        
        for ids in token_ids_list:
            if len(ids) == 0:
                # Handle empty entries
                embedded_rows.append(torch.empty(0, self.hidden_size, device=device, dtype=self._get_dtype()))
            else:
                token_ids = torch.tensor(ids, dtype=torch.long, device=device)
                embeddings = self.qwen.model.embed_tokens(token_ids)  # (Li, H)
                embedded_rows.append(embeddings)
                
        return embedded_rows

    def _build_prefix_attention_mask(
        self, 
        qwen_mask: torch.LongTensor, 
        prefix_len: int
    ) -> torch.LongTensor:
        """
        Build attention mask including prefix tokens.
        
        Args:
            qwen_mask: (N, S) Qwen attention mask, 1=valid, 0=pad
            prefix_len: Number of prefix tokens
            
        Returns:
            Extended attention mask (N, prefix_len+S)
        """
        if prefix_len == 0:
            return qwen_mask
            
        N, S = qwen_mask.shape
        device, dtype = qwen_mask.device, qwen_mask.dtype
        prefix_mask = torch.ones((N, prefix_len), dtype=dtype, device=device)
        return torch.cat([prefix_mask, qwen_mask], dim=1)

    def _extend_mask_for_past(
        self, 
        attn_mask_step: torch.LongTensor, 
        past_len: int
    ) -> torch.LongTensor:
        """
        Extend attention mask for past tokens in generation.
        
        Args:
            attn_mask_step: (N, S_cur) Current step attention mask
            past_len: Length of past tokens
            
        Returns:
            Extended attention mask (N, past_len + S_cur)
        """
        if past_len == 0:
            return attn_mask_step
            
        N, S = attn_mask_step.shape
        device, dtype = attn_mask_step.device, attn_mask_step.dtype
        ones = torch.ones((N, past_len), dtype=dtype, device=device)
        return torch.cat([ones, attn_mask_step], dim=1)

    def _gather_learned_tokens(
        self,
        row_indices: List[int],
    ) -> torch.Tensor:
        """
        Gather learned tokens for specified rows.
        
        Args:
            row_indices: Indices of rows to select
            
        Returns:
            Learned tokens tensor (B, T, H)
        """
        if len(row_indices) == 0:
            raise ValueError("row_indices cannot be empty")
            
        device = self._get_device()
        idx = torch.tensor(row_indices, dtype=torch.long, device=device)
        return self.learned_tokens_global.index_select(dim=0, index=idx)

    def _build_qwen_inputs_embeds(
        self,
        qwen_input_ids: Optional[torch.LongTensor],
        qwen_inputs_embeds: Optional[torch.FloatTensor],
    ) -> torch.FloatTensor:
        """
        Convert Qwen inputs to embeddings.
        
        Args:
            qwen_input_ids: Input IDs for Qwen
            qwen_inputs_embeds: Precomputed embeddings
            
        Returns:
            Embeddings tensor (N, S, H)
        """
        if qwen_input_ids is None and qwen_inputs_embeds is None:
            raise ValueError("Either qwen_input_ids or qwen_inputs_embeds must be provided")

        if qwen_inputs_embeds is not None:
            return qwen_inputs_embeds.to(device=self._get_device(), dtype=self._get_dtype())

        assert qwen_input_ids is not None
        return self.qwen.model.embed_tokens(qwen_input_ids)

    def _maybe_left_pad_labels(
        self, 
        labels: Optional[torch.LongTensor], 
        prefix_len: int
    ) -> Optional[torch.LongTensor]:
        """
        Left-pad labels with -100 for prefix tokens.
        
        Args:
            labels: (N, S) Labels tensor
            prefix_len: Number of prefix tokens to pad
            
        Returns:
            Padded labels (N, prefix_len+S) or None
        """
        if labels is None:
            return None
        if prefix_len == 0:
            return labels
            
        N, S = labels.shape
        device = labels.device
        pad = torch.full((N, prefix_len), fill_value=-100, device=device, dtype=labels.dtype)
        return torch.cat([pad, labels], dim=1)

    def load_attention_weights_once(
        self, 
        layer_weights: List[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]]
    ):
        """
        Load weights for RowColumnAttention layers once.
        
        Args:
            layer_weights: List of (row_weights, col_weights) tuples
        """
        if self._rca_weights_loaded_once:
            return
            
        if len(layer_weights) != self.num_layers:
            raise ValueError(
                f"Expected {self.num_layers} weight pairs, got {len(layer_weights)}"
            )
            
        for i, (row_weights, col_weights) in enumerate(layer_weights):
            self.dict_encoder[i].load_weights_once(row_weights, col_weights)
            
        self._rca_weights_loaded_once = True

    def _process_sample_dictionary_compression(
        self,
        row_indices: List[int],
    ) -> torch.Tensor:
        """
        Process dictionary entries for a single sample through RowColumnAttention.
        
        Args:
            row_indices: Row indices for this sample's dictionary entries
            
        Returns:
            Compressed tokens (num_selected_rows, T, H)
        """
        if len(row_indices) == 0:
            raise ValueError("row_indices cannot be empty for sample")
            
        # Get dictionary entries for this sample
        token_ids_list = [self.full_dict[i] for i in row_indices]
        
        # Embed dictionary entries
        dict_tokens_list = self._embed_dictionary_rows(token_ids_list)
        
        # Get corresponding learned tokens
        learned_tokens = self._gather_learned_tokens(row_indices)
        
        # Apply RowColumnAttention layers
        for i in range(self.num_layers):
            learned_tokens, dict_tokens_list = self.dict_encoder[i](learned_tokens, dict_tokens_list)
            
        return learned_tokens

    def _create_sample_prefix(
        self,
        compressed_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Create prefix from compressed tokens for a single sample.
        
        Args:
            compressed_tokens: (B_rows, T, H) Compressed tokens for sample's entries
            
        Returns:
            Flattened prefix (K, H) where K = B_rows * T
        """
        # Flatten all compressed tokens for this sample
        num_total_tokens = compressed_tokens.size(0) * compressed_tokens.size(1)
        flattened = compressed_tokens.reshape(num_total_tokens, self.hidden_size)  # (K, H)
        return flattened

    def _estimate_past_length(self, past_key_values: Optional[Cache]) -> int:
        """
        Estimate past sequence length from cache.
        
        Args:
            past_key_values: Cache object or None
            
        Returns:
            Estimated past length
        """
        if past_key_values is None:
            return 0
            
        # Use HF's standard method if available
        if hasattr(past_key_values, "get_seq_length"):
            return past_key_values.get_seq_length()
            
        # Fallback to direct access (less reliable)
        try:
            return past_key_values[0][0].shape[-2]
        except Exception:
            return 0

    @torch.no_grad()
    def prepare_inputs_for_generation(
        self,
        input_ids: Optional[torch.LongTensor] = None,          # ✅ 标准键名
        inputs_embeds: Optional[torch.FloatTensor] = None,     # ✅ 标准键名
        attention_mask: Optional[torch.LongTensor] = None,     # ✅ 标准键名
        past_key_values: Optional[Cache] = None,               # ✅ 标准键名
        row_indices_per_sample: Optional[List[List[int]]] = None,
        **kwargs,
    ) -> Dict[str, torch.Any]:
        """
        供 HF generate() 调用的输入构造。第一步把压缩前缀拼进 inputs_embeds；
        之后（有 past_key_values）返回本步需要的最小字段。
        """
        # 后续步：有 cache 就直接返回本步（不重复加前缀）
        if past_key_values is not None:
            return {
                "input_ids": input_ids,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "row_indices_per_sample": row_indices_per_sample,
            }

        # 第一步：把压缩前缀拼接进输入
        embeds = self._build_qwen_inputs_embeds(input_ids, inputs_embeds)  # (N, S, H)

        # 没有词典索引就原样返回
        if not row_indices_per_sample:
            return {
                "inputs_embeds": embeds,
                "attention_mask": attention_mask,
                "row_indices_per_sample": row_indices_per_sample,
            }

        # 每个样本生成前缀
        sample_prefixes = []
        for rows in row_indices_per_sample:
            compressed = self._process_sample_dictionary_compression(rows)  # (B_rows, T, H)
            prefix = self._create_sample_prefix(compressed)                 # (K, H)
            sample_prefixes.append(prefix)

        max_pref = max(p.size(0) for p in sample_prefixes)
        batch_inputs, batch_masks = [], []

        for i, prefix in enumerate(sample_prefixes):
            cur = prefix.size(0)
            if cur < max_pref:
                pad = torch.zeros((max_pref - cur, self.hidden_size),
                                  dtype=prefix.dtype, device=prefix.device)
                prefix = torch.cat([prefix, pad], dim=0)  # (max_pref, H)

            # 拼前缀 + 原始嵌入
            sample_embeds = torch.cat([prefix.unsqueeze(0), embeds[i:i+1]], dim=1)  # (1, max_pref+S, H)
            batch_inputs.append(sample_embeds)

            # 拼 attention_mask
            if attention_mask is None:
                base_mask = torch.ones((1, embeds.size(1)), device=embeds.device, dtype=torch.long)
            else:
                base_mask = attention_mask[i:i+1]
            pref_mask = torch.ones((1, max_pref), device=base_mask.device, dtype=base_mask.dtype)
            batch_masks.append(torch.cat([pref_mask, base_mask], dim=1))

        final_inputs_embeds = torch.cat(batch_inputs, dim=0)
        final_attention_mask = torch.cat(batch_masks, dim=0)

        return {
            "inputs_embeds": final_inputs_embeds,
            "attention_mask": final_attention_mask,
            "row_indices_per_sample": row_indices_per_sample,
        }


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
        """
        Forward pass through LexiconCompressorModel.
        
        Args:
            row_indices_per_sample: Per-sample dictionary row indices, List[List[int]] of length N
            attention_weights: Pre-trained weights for RowColumnAttention layers
            qwen_input_ids: Input IDs for Qwen model
            qwen_inputs_embeds: Precomputed embeddings for Qwen model
            labels: Labels for computing loss
            past_key_values: Key/value states for generation
            use_cache: Whether to use key/value cache
            cache_position: Position of cache
            logits_to_keep: Number of logits to keep
            return_dict: Whether to return dict or tuple
            
        Returns:
            Model output with logits, loss, and compressed tokens
        """
        # Load attention weights if provided
        if not self._rca_weights_loaded_once:
            if attention_weights is not None:
                self.load_attention_weights_once(attention_weights)
            else:
                raise ValueError("Lexicon encoder decoder layers have not been loaded weights, and the attention weights are None. You should pass attention weights at least once.")

        return_dict = return_dict if return_dict is not None else True
        
        # Prepare Qwen inputs
        qwen_inputs_embeds = self._build_qwen_inputs_embeds(qwen_input_ids, qwen_inputs_embeds)
        batch_size_qwen = qwen_inputs_embeds.size(0)
        
        # Disable cache during training
        if self.training and labels is not None:
            use_cache = False

        # Handle cache vs non-cache paths
        if past_key_values is None:
            # First step: process dictionary compression for each sample
            if row_indices_per_sample is not None:
                # 处理DataParallel情况下的长度不匹配
                if len(row_indices_per_sample) != batch_size_qwen:
                    raise ValueError("row_indices_per_sample should equal to batch_size_qwen")
                actual_batch_size = len(row_indices_per_sample)
                
                # Process each sample with its own dictionary entries
                sample_prefixes = []
                compressed_tokens_list = []
                
                for sample_idx in range(actual_batch_size):
                    row_indices = row_indices_per_sample[sample_idx]
                    # Process dictionary compression for this sample
                    compressed_tokens = self._process_sample_dictionary_compression(row_indices)
                    compressed_tokens_list.append(compressed_tokens)
                    
                    # Create prefix for this sample
                    sample_prefix = self._create_sample_prefix(compressed_tokens)
                    sample_prefixes.append(sample_prefix)
                
                # Get base attention mask
                base_attention_mask = kwargs.pop("attention_mask", None)
                if base_attention_mask is None:
                    base_attention_mask = torch.ones(
                        (actual_batch_size, qwen_inputs_embeds.size(1)),
                        dtype=torch.long, device=qwen_inputs_embeds.device
                    )
                
                # 处理不同长度的前缀
                if sample_prefixes:
                    # 计算所有前缀的最大长度
                    max_prefix_len = max(prefix.size(0) for prefix in sample_prefixes)
                    qwen_seq_len = qwen_inputs_embeds.size(1)
                    
                    # 创建填充后的输入和mask
                    batched_inputs = []
                    batched_masks = []
                    batched_labels = []
                    
                    for sample_idx in range(actual_batch_size):
                        # 获取当前样本的前缀和Qwen输入
                        sample_prefix = sample_prefixes[sample_idx]
                        sample_qwen_input = qwen_inputs_embeds[sample_idx:sample_idx+1]  # (1, S, H)
                        
                        # 计算需要填充的长度
                        current_prefix_len = sample_prefix.size(0)
                        padding_len = max_prefix_len - current_prefix_len
                        
                        # 填充前缀到最大长度
                        if padding_len > 0:
                            padding = torch.zeros(
                                (padding_len, sample_prefix.size(1)), 
                                device=sample_prefix.device, 
                                dtype=sample_prefix.dtype
                            )
                            padded_prefix = torch.cat([sample_prefix, padding], dim=0)  # (max_prefix_len, H)
                        else:
                            padded_prefix = sample_prefix  # (max_prefix_len, H)
                        
                        # 拼接前缀和Qwen输入
                        sample_input = torch.cat([
                            padded_prefix.unsqueeze(0),  # (1, max_prefix_len, H)
                            sample_qwen_input  # (1, S, H)
                        ], dim=1)  # (1, max_prefix_len+S, H)
                        
                        batched_inputs.append(sample_input)
                        
                        # 构建对应的attention mask
                        sample_base_mask = base_attention_mask[sample_idx:sample_idx+1]  # (1, S)
                        prefix_mask = torch.ones(
                            (1, max_prefix_len), 
                            dtype=sample_base_mask.dtype, 
                            device=sample_base_mask.device
                        )
                        sample_attention_mask = torch.cat([prefix_mask, sample_base_mask], dim=1)  # (1, max_prefix_len+S)
                        batched_masks.append(sample_attention_mask)
                        
                        # 处理labels
                        if labels is not None and sample_idx < labels.size(0):
                            sample_label = labels[sample_idx:sample_idx+1]  # (1, S)
                            prefix_label_padding = torch.full(
                                (1, max_prefix_len), 
                                fill_value=-100, 
                                dtype=sample_label.dtype, 
                                device=sample_label.device
                            )
                            sample_padded_label = torch.cat([prefix_label_padding, sample_label], dim=1)  # (1, max_prefix_len+S)
                            batched_labels.append(sample_padded_label)
                    
                    # 拼接所有样本
                    final_inputs_embeds = torch.cat(batched_inputs, dim=0)  # (N, max_prefix_len+S, H)
                    attention_mask = torch.cat(batched_masks, dim=0)  # (N, max_prefix_len+S)
                    
                    if batched_labels:
                        final_labels = torch.cat(batched_labels, dim=0)  # (N, max_prefix_len+S)
                    else:
                        final_labels = None
                else:
                    # Fallback
                    final_inputs_embeds = qwen_inputs_embeds[:actual_batch_size]
                    attention_mask = base_attention_mask[:actual_batch_size] if base_attention_mask.size(0) >= actual_batch_size else base_attention_mask
                    final_labels = labels[:actual_batch_size] if labels is not None and labels.size(0) >= actual_batch_size else labels
                    compressed_tokens_list = None
                    
            else:
                # Fallback: use all dictionary entries
                compressed_tokens = self._process_sample_dictionary_compression(
                    list(range(self.num_rows))
                )
                compressed_tokens_list = [compressed_tokens]
                
                # Create single prefix for all samples
                global_prefix = self._create_sample_prefix(compressed_tokens)
                prefix_len = global_prefix.size(0)
                
                # Use actual batch size
                actual_batch_size = batch_size_qwen
                prefix = global_prefix.unsqueeze(0).expand(actual_batch_size, -1, -1).contiguous()
                final_inputs_embeds = torch.cat([prefix, qwen_inputs_embeds[:actual_batch_size]], dim=1)
                
                # Build attention mask
                qwen_mask = kwargs.pop("attention_mask", None)
                if qwen_mask is None:
                    qwen_mask = torch.ones(
                        (actual_batch_size, qwen_inputs_embeds.size(1)),
                        dtype=torch.long, device=final_inputs_embeds.device
                    )
                attention_mask = self._build_prefix_attention_mask(qwen_mask, prefix_len=prefix_len)
                
                # Handle labels
                final_labels = self._maybe_left_pad_labels(
                    labels[:actual_batch_size] if labels is not None and labels.size(0) >= actual_batch_size else labels, 
                    prefix_len=prefix_len
                )
            
            # Check sequence length limits
            max_pos = getattr(self.qwen_config, "max_position_embeddings", None)
            if max_pos is not None:
                actual_max_len = final_inputs_embeds.size(1)
                if actual_max_len > max_pos:
                    raise ValueError(
                        f"Max sequence length {actual_max_len} exceeds "
                        f"max_position_embeddings {max_pos}. "
                        f"Consider reducing number of dictionary entries per sample."
                    )
                
        else:
            # Subsequent steps: no prefix needed, just process current step
            final_inputs_embeds = qwen_inputs_embeds

            attn_step = kwargs.pop("attention_mask", None)
            if attn_step is None:
                attn_step = torch.ones(
                    (qwen_inputs_embeds.size(0), qwen_inputs_embeds.size(1)),
                    dtype=torch.long, device=final_inputs_embeds.device
                )

            # Estimate past length using robust method
            past_len = self._estimate_past_length(past_key_values)
            attention_mask = self._extend_mask_for_past(attn_step, past_len)
            final_labels = labels
            compressed_tokens_list = None

        # Forward through Qwen model - 修复：只传递Qwen3认识的参数
        qwen_forward_kwargs = {
            "input_ids": None,
            "inputs_embeds": final_inputs_embeds,
            "attention_mask": attention_mask,
            "labels": final_labels,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "cache_position": cache_position,
            "logits_to_keep": logits_to_keep,
        }
        
        # 添加Qwen3可能需要的其他参数
        for key, value in kwargs.items():
            if key in ["output_attentions", "output_hidden_states", "return_dict", "position_ids"]:
                qwen_forward_kwargs[key] = value
        
        qwen_outputs = self.qwen(**qwen_forward_kwargs)
        
        if not return_dict:
            # Return tuple with compressed tokens appended
            output_values = [
                qwen_outputs.loss, 
                qwen_outputs.logits, 
                qwen_outputs.past_key_values, 
                qwen_outputs.hidden_states,
                qwen_outputs.attentions
            ]
            if compressed_tokens_list is not None:
                output_values.append(compressed_tokens_list)
                
            return tuple(v for v in output_values if v is not None)
        
        return LexiconCompressorModelOutput(
            loss=qwen_outputs.loss,
            logits=qwen_outputs.logits,
            past_key_values=qwen_outputs.past_key_values,
            hidden_states=qwen_outputs.hidden_states,
            attentions=qwen_outputs.attentions,
            compressed_tokens=compressed_tokens_list,
        )

    def extra_repr(self) -> str:
        return (
            f"num_rows={self.num_rows}, "
            f"num_layers={self.num_layers}, "
            f"num_compress_tokens={self.num_compress_tokens}, "
            f"hidden_size={self.hidden_size}"
        )