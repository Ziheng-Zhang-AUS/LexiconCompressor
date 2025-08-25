# row_column_attention.py
from __future__ import annotations
from typing import List, Optional, Tuple, Dict, Any

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
    One row-column attention block.

    Row pass: self-attention over per-row sequences after concatenation and right padding.
    Column pass: self-attention over learned (compressed) tokens with identity RoPE.

    Inputs:
        learned_tokens: (B, T, H)
        dict_tokens_list: List[(Li, H)] of length B

    Outputs:
        updated_learned: (B, T, H)
        updated_dict_list: List[(Li, H)] of length B
    """

    def __init__(self, config: LexiconCompressorConfig):
        """
        Initialize RowColumnAttention.

        Args:
            config: LexiconCompressorConfig carrying the underlying Qwen3Config and compressor options
        """
        super().__init__()
        self.config = config
        qcfg = config.qwen_config

        self.hidden_size = qcfg.hidden_size
        self.row_layer = Qwen3DecoderLayer(config=qcfg, layer_idx=0)
        self.col_layer = Qwen3DecoderLayer(config=qcfg, layer_idx=1)
        self.rope = Qwen3RotaryEmbedding(config=qcfg)

        self._weights_loaded = False

    def load_weights_once(self, row_weights: Dict[str, torch.Tensor], col_weights: Dict[str, torch.Tensor]):
        """
        Load decoder-layer weights once.

        Args:
            row_weights: state_dict for row decoder layer
            col_weights: state_dict for column decoder layer
        """
        if self._weights_loaded:
            return
        if row_weights is None or col_weights is None:
            raise ValueError("Both row_weights and col_weights must be provided.")
        self.row_layer.load_state_dict(row_weights, strict=True)
        self.col_layer.load_state_dict(col_weights, strict=True)
        self._weights_loaded = True

    @staticmethod
    def _concat_then_right_pad(
        learned: torch.Tensor,
        dict_list: List[torch.Tensor],
        prepend: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Concatenate per row then right-pad to batch.

        Args:
            learned: (B, T, H)
            dict_list: list of per-row tensors, each (Li, H)
            prepend: if True, learned precedes dict; else dict precedes learned

        Returns:
            padded: (B, Lmax, H)
            total_lens: (B,)
            learned_beg: (B,)
            dict_beg: (B,)
        """
        B, T, H = learned.shape
        concat_rows, total_lens, learned_beg, dict_beg = [], [], [], []
        for i in range(B):
            Li = dict_list[i].size(0)
            if prepend:
                row = torch.cat([learned[i], dict_list[i]], dim=0)  # (T+Li, H)
                learned_beg.append(0)
                dict_beg.append(T)
            else:
                row = torch.cat([dict_list[i], learned[i]], dim=0)  # (Li+T, H)
                learned_beg.append(Li)
                dict_beg.append(0)
            concat_rows.append(row)
            total_lens.append(row.size(0))
        padded = pad_sequence(concat_rows, batch_first=True, padding_value=0.0)  # (B, Lmax, H)
        total_lens = torch.tensor(total_lens, device=learned.device, dtype=torch.long)  # (B,)
        learned_beg = torch.tensor(learned_beg, device=learned.device, dtype=torch.long)  # (B,)
        dict_beg = torch.tensor(dict_beg, device=learned.device, dtype=torch.long)  # (B,)
        return padded, total_lens, learned_beg, dict_beg

    def _build_row_mask_and_rope(
        self,
        total_lens: torch.Tensor,
        Lmax: int,
        prepend: bool,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Build additive attention mask and RoPE for row pass.

        Args:
            total_lens: (B,)
            Lmax: int, max padded length
            prepend: whether learned tokens are placed before dict tokens
            dtype: torch dtype
            device: torch device

        Returns:
            mask: (B, 1, Lmax, Lmax)
            rope: tuple(cos, sin) where each is (B, Lmax, head_dim)
        """
        B = total_lens.size(0)
        k_range = torch.arange(Lmax, device=device).unsqueeze(0).expand(B, -1)  # (B, Lmax)
        key_valid_1d = (k_range < total_lens.unsqueeze(1))  # (B, Lmax)

        mask = torch.zeros(B, 1, Lmax, Lmax, device=device, dtype=dtype)  # (B, 1, Lmax, Lmax)
        neg_inf = torch.finfo(dtype).min
        key_valid_4d = key_valid_1d.view(B, 1, 1, Lmax).expand(B, 1, Lmax, Lmax)  # (B,1,Lmax,Lmax)
        mask = torch.where(key_valid_4d, mask, torch.tensor(neg_inf, device=device, dtype=dtype))
        if not prepend:
            causal = torch.tril(torch.ones(Lmax, Lmax, device=device, dtype=torch.bool)).view(1, 1, Lmax, Lmax)
            mask = torch.where(causal, mask, torch.tensor(neg_inf, device=device, dtype=dtype))

        pos_ids = torch.zeros(B, Lmax, device=device, dtype=torch.long)  # (B, Lmax)
        for b in range(B):
            Lb = total_lens[b].item()
            if Lb > 0:
                pos_ids[b, :Lb] = torch.arange(Lb, device=device, dtype=torch.long)

        dummy = torch.empty(B, Lmax, self.hidden_size, device=device, dtype=dtype)  # (B, Lmax, H)
        cos, sin = self.rope(dummy, pos_ids)  # (B, Lmax, head_dim)
        return mask, (cos, sin)

    @staticmethod
    def _slice_back_to_list(
        row_out: torch.Tensor,
        dict_lens: torch.Tensor,
        dict_beg: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        Slice updated dict segments back into a list.

        Args:
            row_out: (B, Lmax, H)
            dict_lens: (B,)
            dict_beg: (B,)

        Returns:
            list of tensors, each (Li, H)
        """
        B, Lmax, H = row_out.shape
        out_list: List[torch.Tensor] = []
        for b in range(B):
            beg = dict_beg[b].item()
            Li = dict_lens[b].item()
            out_list.append(row_out[b, beg:beg + Li, :])
        return out_list

    def _identity_rope(
        self,
        B: int,
        T: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build identity RoPE for column pass.

        Args:
            B: batch size
            T: number of learned tokens
            dtype: torch dtype
            device: torch device

        Returns:
            tuple(cos, sin) where cos=1 and sin=0, each (B, T, head_dim)
        """
        dummy = torch.empty(B, T, self.hidden_size, device=device, dtype=dtype)  # (B, T, H)
        cos_ref, _ = self.rope(dummy, torch.arange(T, device=device).unsqueeze(0).expand(B, -1))  # (B,T,head_dim)
        head_dim = cos_ref.size(-1)
        cos = torch.ones(B, T, head_dim, device=device, dtype=dtype)  # (B, T, head_dim)
        sin = torch.zeros(B, T, head_dim, device=device, dtype=dtype)  # (B, T, head_dim)
        return cos, sin

    def forward(
        self,
        learned_tokens: torch.Tensor,
        dict_tokens_list: List[torch.Tensor],
        **_: Any,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass.

        Args:
            learned_tokens: (B, T, H)
            dict_tokens_list: list of per-row tensors, each (Li, H)

        Returns:
            updated_learned: (B, T, H)
            updated_dict_list: List[(Li, H)]
        """
        if len(dict_tokens_list) != learned_tokens.size(0):
            raise ValueError("Batch size mismatch between learned_tokens and dict_tokens_list")

        B, T, H = learned_tokens.shape
        device, dtype = learned_tokens.device, learned_tokens.dtype
        dict_lens = torch.tensor([t.size(0) for t in dict_tokens_list], device=device, dtype=torch.long)  # (B,)

        prepend = bool(self.config.learned_tokens_prepend)
        row_padded, total_lens, learned_beg, dict_beg = self._concat_then_right_pad(
            learned=learned_tokens, dict_list=dict_tokens_list, prepend=prepend
        )  # (B, Lmax, H), (B,), (B,), (B,)
        Lmax = row_padded.size(1)

        row_mask, row_rope = self._build_row_mask_and_rope(
            total_lens=total_lens, Lmax=Lmax, prepend=prepend, dtype=dtype, device=device
        )

        row_out = self.row_layer(
            hidden_states=row_padded,            # (B, Lmax, H)
            attention_mask=row_mask,             # (B, 1, Lmax, Lmax)
            position_ids=None,
            position_embeddings=row_rope,        # (cos, sin)
            past_key_values=None,
            use_cache=False,
        )
        # row_out = _last_hidden(row_out)          # (B, Lmax, H)

        idx = (learned_beg.unsqueeze(1) + torch.arange(T, device=device).unsqueeze(0)).unsqueeze(-1).expand(B, T, H)
        updated_learned = torch.gather(row_out, dim=1, index=idx)  # (B, T, H)
        updated_dict_list = self._slice_back_to_list(row_out, dict_lens=dict_lens, dict_beg=dict_beg)

        col_mask = torch.zeros(B, 1, T, T, device=device, dtype=dtype)  # (B, 1, T, T)
        col_rope = self._identity_rope(B=B, T=T, dtype=dtype, device=device)

        col_out = self.col_layer(
            hidden_states=updated_learned,       # (B, T, H)
            attention_mask=col_mask,             # (B, 1, T, T)
            position_ids=None,
            position_embeddings=col_rope,        # (cos=1, sin=0)
            past_key_values=None,
            use_cache=False,
        )
        # updated_learned = _last_hidden(col_out)  # (B, T, H)
        updated_learned = col_out

        return updated_learned, updated_dict_list
    
    
    
    
# test_row_column_attention.py
"""
Test script for RowColumnAttention module using real Qwen3-0.6B configuration.
"""

import torch
import torch.nn as nn
from transformers import Qwen3Config
from configuration_lexicon_compressor import LexiconCompressorConfig
from row_column_attention import RowColumnAttention

def create_valid_qwen_config():
    """
    Create a valid Qwen3 configuration that works with RowColumnAttention.
    """
    # Load base config
    config = Qwen3Config.from_pretrained("Qwen/Qwen3-0.6B")
    
    # Set attention implementation
    config._attn_implementation = "eager"
    
    # Ensure all required parameters are set properly
    if not hasattr(config, 'sliding_window') or config.sliding_window is None:
        config.sliding_window = 4096
    
    return config

def test_row_column_attention():
    """
    Test basic functionality of RowColumnAttention with real Qwen3-0.6B config.
    """
    
    # 1. Load real Qwen3-0.6B configuration from HuggingFace
    print("Loading real Qwen3-0.6B configuration...")
    qwen_config = create_valid_qwen_config()
    
    print(f"Loaded Qwen3 config: hidden_size={qwen_config.hidden_size}, num_heads={qwen_config.num_attention_heads}")
    print(f"Attention implementation: {qwen_config._attn_implementation}")
    
    # 2. Create compressor configuration
    compressor_config = LexiconCompressorConfig(
        qwen_config=qwen_config,
        num_layers=2,
        num_compress_tokens=4,
        learned_tokens_prepend=True,  # Test prepend mode
    )
    
    # 3. Create model
    rca = RowColumnAttention(compressor_config)
    
    # 4. Prepare test data
    B = 3  # batch size
    T = 4  # compress tokens per row
    H = qwen_config.hidden_size  # hidden size from real config
    
    # learned tokens: (B, T, H)
    learned_tokens = torch.randn(B, T, H)
    
    # dict tokens: List of (Li, H)
    dict_tokens_list = [
        torch.randn(5, H),   # Row 1, length 5
        torch.randn(3, H),   # Row 2, length 3
        torch.randn(7, H),   # Row 3, length 7
    ]
    
    print("=== Test Data ===")
    print(f"learned_tokens shape: {learned_tokens.shape}")
    print(f"dict_tokens_list lengths: {[t.shape[0] for t in dict_tokens_list]}")
    
    # 5. Forward pass
    print("\n=== Forward Pass Test ===")
    try:
        updated_learned, updated_dict_list = rca(
            learned_tokens=learned_tokens,
            dict_tokens_list=dict_tokens_list
        )
        
        print(f"updated_learned shape: {updated_learned.shape}")
        print(f"updated_dict_list lengths: {[t.shape[0] for t in updated_dict_list]}")
        
        # 6. Verify output shapes
        assert updated_learned.shape == (B, T, H), f"Expected {(B, T, H)}, got {updated_learned.shape}"
        assert len(updated_dict_list) == B, f"Expected {B} dict tensors, got {len(updated_dict_list)}"
        
        for i, (orig, updated) in enumerate(zip(dict_tokens_list, updated_dict_list)):
            assert updated.shape == orig.shape, f"Dict tensor {i}: expected {orig.shape}, got {updated.shape}"
        
        print("✅ Shape validation passed!")
        
        # 7. Verify numerical changes (should not be identical)
        assert not torch.allclose(updated_learned, learned_tokens), "Learned tokens should be updated"
        print("✅ Numerical update validation passed!")
        
    except Exception as e:
        print(f"❌ Forward pass failed with error: {e}")
        raise
    
    # 8. Test append mode
    print("\n=== Test Append Mode ===")
    compressor_config_append = LexiconCompressorConfig(
        qwen_config=qwen_config,
        num_layers=2,
        num_compress_tokens=4,
        learned_tokens_prepend=False,  # append mode
    )
    
    rca_append = RowColumnAttention(compressor_config_append)
    
    try:
        updated_learned_append, updated_dict_list_append = rca_append(
            learned_tokens=learned_tokens,
            dict_tokens_list=dict_tokens_list
        )
        
        assert updated_learned_append.shape == (B, T, H)
        assert len(updated_dict_list_append) == B
        print("✅ Append mode test passed!")
        
    except Exception as e:
        print(f"❌ Append mode test failed with error: {e}")
        raise
    
    # 9. Test gradient flow
    print("\n=== Gradient Flow Test ===")
    learned_tokens.requires_grad_(True)
    
    try:
        updated_learned, _ = rca(
            learned_tokens=learned_tokens,
            dict_tokens_list=dict_tokens_list
        )
        
        loss = updated_learned.sum()
        loss.backward()
        
        assert learned_tokens.grad is not None, "Gradient should flow back to learned_tokens"
        print("✅ Gradient flow test passed!")
        
    except Exception as e:
        print(f"❌ Gradient flow test failed with error: {e}")
        raise
    
    print("\n🎉 All basic tests passed!")

def test_edge_cases():
    """
    Test edge cases for RowColumnAttention.
    """
    print("\n=== Edge Cases Test ===")
    
    # Load real configuration
    qwen_config = create_valid_qwen_config()
    H = qwen_config.hidden_size
    
    compressor_config = LexiconCompressorConfig(
        qwen_config=qwen_config,
        num_layers=1,
        num_compress_tokens=2,
        learned_tokens_prepend=True,
    )
    
    rca = RowColumnAttention(compressor_config)
    
    # Test single sample
    B, T = 1, 2
    learned_tokens = torch.randn(B, T, H)
    dict_tokens_list = [torch.randn(10, H)]  # Single long sequence
    
    try:
        updated_learned, updated_dict_list = rca(
            learned_tokens=learned_tokens,
            dict_tokens_list=dict_tokens_list
        )
        
        assert updated_learned.shape == (1, 2, H)
        assert len(updated_dict_list) == 1
        assert updated_dict_list[0].shape == (10, H)
        print("✅ Single sample test passed!")
        
    except Exception as e:
        print(f"❌ Single sample test failed with error: {e}")
        raise
    
    # Test short sequences
    dict_tokens_list_short = [
        torch.randn(1, H),   # Length 1
        torch.randn(2, H),   # Length 2
    ]
    
    try:
        updated_learned, updated_dict_list = rca(
            learned_tokens=torch.randn(2, T, H),
            dict_tokens_list=dict_tokens_list_short
        )
        
        assert len(updated_dict_list) == 2
        assert updated_dict_list[0].shape == (1, H)
        assert updated_dict_list[1].shape == (2, H)
        print("✅ Short sequences test passed!")
        
    except Exception as e:
        print(f"❌ Short sequences test failed with error: {e}")
        raise

def test_consistent_dimensions():
    """
    Test with consistent dimensions using the real Qwen3 config without modifications.
    """
    print("\n=== Consistent Dimensions Test ===")
    
    # Use real Qwen3-0.6B config without modifications
    qwen_config = create_valid_qwen_config()
    H = qwen_config.hidden_size
    
    compressor_config = LexiconCompressorConfig(
        qwen_config=qwen_config,
        num_layers=1,
        num_compress_tokens=3,
        learned_tokens_prepend=True,
    )
    
    rca = RowColumnAttention(compressor_config)
    
    B, T = 2, 3
    learned_tokens = torch.randn(B, T, H)
    dict_tokens_list = [
        torch.randn(4, H),
        torch.randn(6, H),
    ]
    
    try:
        updated_learned, updated_dict_list = rca(
            learned_tokens=learned_tokens,
            dict_tokens_list=dict_tokens_list
        )
        
        assert updated_learned.shape == (B, T, H)
        assert len(updated_dict_list) == B
        assert updated_dict_list[0].shape == (4, H)
        assert updated_dict_list[1].shape == (6, H)
        print("✅ Consistent dimensions test passed!")
        
    except Exception as e:
        print(f"❌ Consistent dimensions test failed with error: {e}")
        raise

if __name__ == "__main__":
    print("🚀 Starting RowColumnAttention tests with real Qwen3-0.6B configuration...")
    try:
        test_row_column_attention()
        test_edge_cases()
        test_consistent_dimensions()
        print("\n🎊 All tests completed successfully!")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()