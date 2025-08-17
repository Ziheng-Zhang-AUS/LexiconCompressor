# lexicon_compressor_model.py
import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Dict, Any
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config
from row_column_attention_stack import RowColumnAttentionStack

class LexiconCompressorModel(nn.Module):
    def __init__(self, config: Qwen3Config, num_attention_layers: int = 2):
        """
        Lexicon compressor model.
        
        Args:
            config: Qwen3 configuration
            num_attention_layers: Number of row-column attention layers
        """
        super().__init__()
        self.config = config
        self.num_attention_layers = num_attention_layers
        
        # Row-column attention stack
        self.attention_stack = RowColumnAttentionStack(config, num_attention_layers)
        
        # Embedding layer placeholder (weights loaded via load_embeddings_weights)
        self.embeddings = None
        self._embeddings_loaded = False
        
    def load_embeddings_weights(self, embedding_weights: Dict[str, torch.Tensor]):
        """
        Load embedding layer weights.
        
        Args:
            embedding_weights: Dictionary containing embedding weights
        """
        # Create embedding layer and load weights
        vocab_size = embedding_weights['weight'].shape[0]
        hidden_size = embedding_weights['weight'].shape[1]
        
        self.embeddings = nn.Embedding(vocab_size, hidden_size)
        self.embeddings.load_state_dict(embedding_weights)
        self.embeddings.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self._embeddings_loaded = True
    
    def load_attention_weights(self, attention_weights_list: List[Tuple[Dict, Dict]]):
        """
        Load weights for all attention layers.
        
        Args:
            attention_weights_list: List of attention weights, each element is (row_weights, col_weights) tuple
        """
        self.attention_stack.load_weights_once(attention_weights_list)
        # Move attention stack to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attention_stack.to(device)
    
    def tokenize_and_embed(self, token_ids_list: List[List[int]]) -> List[torch.Tensor]:
        """
        Convert token IDs to embedding vectors.
        
        Args:
            token_ids_list: List of token IDs for each row
            
        Returns:
            List of embedding vectors for each row
        """
        if not self._embeddings_loaded:
            raise ValueError("Embeddings weights must be loaded first")
        
        device = self.embeddings.weight.device
        embedded_rows = []
        for token_ids in token_ids_list:
            # Convert to tensor
            token_tensor = torch.tensor(token_ids, dtype=torch.long, device=device)
            # Get embeddings
            embedded = self.embeddings(token_tensor)  # shape: [seq_len, hidden_size]
            embedded_rows.append(embedded)
        
        return embedded_rows
    
    def forward(
        self,
        token_ids_list: List[List[int]],  # Input token IDs list
        attention_weights: Optional[List[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]]] = None,
        embeddings_weights: Optional[Dict[str, torch.Tensor]] = None,
        row_attention_masks: Optional[List[List[torch.Tensor]]] = None,
        column_attention_masks: Optional[List[torch.Tensor]] = None,
        row_position_embeddings: Optional[List[List[Tuple[torch.Tensor, torch.Tensor]]]] = None,
        column_position_embeddings: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ):
        """
        Forward pass - lexicon compression.
        
        Args:
            token_ids_list: List of token IDs for each row
            attention_weights: Attention layer weights list
            embeddings_weights: Embedding layer weights
            Other attention-related parameters...
        """
        
        # Load weights if provided
        if embeddings_weights is not None and not self._embeddings_loaded:
            self.load_embeddings_weights(embeddings_weights)
        
        if attention_weights is not None:
            self.load_attention_weights(attention_weights)
        
        # 1. Tokenize and embed
        embedded_rows = self.tokenize_and_embed(token_ids_list)
        
        # 2. Process through row-column attention stack
        compressed_rows = self.attention_stack(
            embedded_rows=embedded_rows,
            row_attention_masks=row_attention_masks,
            column_attention_masks=column_attention_masks,
            row_position_embeddings=row_position_embeddings,
            column_position_embeddings=column_position_embeddings
        )
        
        return compressed_rows


def main():
    import torch
    from transformers import AutoTokenizer, Qwen3ForCausalLM
    from tokenization_lexicon import LexiconTokenizer
    import os

    # Disable tokenizer parallelism warning
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    MODEL_NAME = "Qwen/Qwen3-0.6B"
    CSV_PATH   = "cleaned_lexicon_tiny.csv"
    COLUMNS    = ["lexical_unit", "pos", "gloss", "variant"]
    NUM_LAYERS = 6
    SHOW_N     = 100

    torch.set_grad_enabled(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1) Tokenizer + [COMP]
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tok.add_tokens(["[COMP]"], special_tokens=False)
    comp_id = tok.convert_tokens_to_ids("[COMP]")

    # 2) CSV → token ids (prepend [COMP] to each row)
    lt = LexiconTokenizer(
        csv_file_path=CSV_PATH,
        tokenizer=tok,
        column_names=COLUMNS,
        compress_token_id=comp_id,
        delimiter=",",
        add_special_tokens=False
    )
    token_ids_list = lt.process_lexicon()[:SHOW_N]
    assert token_ids_list, "No entries parsed from CSV."

    # 3) Qwen weights (embedding / layers / RoPE)
    qwen   = Qwen3ForCausalLM.from_pretrained(MODEL_NAME).to(device)
    cfg    = qwen.config
    rotary = qwen.model.rotary_emb
    H      = cfg.hidden_size

    lcm = LexiconCompressorModel(config=cfg, num_attention_layers=NUM_LAYERS).to(device)
    emb_weights = qwen.get_input_embeddings().state_dict()
    attn_weights = []
    for i in range(NUM_LAYERS):
        row_sd = qwen.model.layers[2*i].state_dict()
        col_sd = qwen.model.layers[2*i+1].state_dict()
        attn_weights.append((row_sd, col_sd))

    # --------- Helpers: Build RoPE / mask and forward once ----------
    def full_vis_mask(L: int):
        return torch.zeros(1, 1, L, L, dtype=torch.float32, device=device)

    def row_rope(L: int):
        dummy = torch.zeros(1, L, H, device=device)
        pos   = torch.arange(L, device=device).unsqueeze(0)
        return rotary(dummy, pos)  # Ordered RoPE

    def col_identity_rope(N: int):
        dummy = torch.zeros(1, N, H, device=device)
        pos0  = torch.zeros(1, N, dtype=torch.long, device=device)
        return rotary(dummy, pos0)  # Identity (unordered)

    def build_inputs(tids):
        # Rows: each layer needs a copy (simple replication)
        row_pos = []; row_msk = []
        for _ in range(NUM_LAYERS):
            row_pos.append([row_rope(len(x)) for x in tids])
            row_msk.append([full_vis_mask(len(x)) for x in tids])
        # Columns: one per layer
        N = len(tids)
        col_pos = [col_identity_rope(N) for _ in range(NUM_LAYERS)]
        col_msk = [full_vis_mask(N)     for _ in range(NUM_LAYERS)]
        return row_pos, row_msk, col_pos, col_msk

    def fwd_once(tids):
        row_pos, row_msk, col_pos, col_msk = build_inputs(tids)
        out_rows = lcm(
            token_ids_list=tids,
            attention_weights=attn_weights,
            embeddings_weights=emb_weights,
            row_attention_masks=row_msk,
            column_attention_masks=col_msk,
            row_position_embeddings=row_pos,
            column_position_embeddings=col_pos
        )
        # Extract container vectors ([COMP] tokens) from each row
        heads = torch.stack([r[0].detach().cpu() for r in out_rows], dim=0)
        return out_rows, heads
    # ------------------------------------------------------------

    print("Forward once (baseline)...")
    out_rows, heads = fwd_once(token_ids_list)
    for i, (ids, row) in enumerate(zip(token_ids_list, out_rows)):
        print(f"Row {i}: ids_len={len(ids)} -> out_shape={tuple(row.shape)}; head_norm={row[0].norm().item():.4f}")

    # ============== Test 1: Column invariance (column should be insensitive to row order) ==============
    import random
    perm = list(range(len(token_ids_list)))
    random.shuffle(perm)
    perm_tids = [token_ids_list[i] for i in perm]
    _, heads_perm = fwd_once(perm_tids)

    assert torch.allclose(heads_perm, heads[perm], atol=1e-5, rtol=1e-5), \
        "Column invariance FAILED: shuffling rows changed container vectors beyond permutation."
    print("Test#1 Column invariance: OK ✅")

    # ============ Test 2: Row order sensitivity (swapping tokens within row should change output) ============
    # Select first row, try swapping two non-[COMP] positions (ensure length≥3)
    swapped_tids = [x[:] for x in token_ids_list]
    if len(swapped_tids[0]) >= 3:
        i0, j0 = 1, 2  # Swap 1st/2nd real tokens (0 is [COMP])
        swapped_tids[0][i0], swapped_tids[0][j0] = swapped_tids[0][j0], swapped_tids[0][i0]
        _, heads_swapped = fwd_once(swapped_tids)
        assert not torch.allclose(heads_swapped[0], heads[0], atol=1e-6), \
            "Row order FAILED: swapping tokens did not change row container."
        print("Test#2 Row order sensitivity: OK ✅")
    else:
        print("Test#2 Row order sensitivity: SKIP (row too short)")

    # ====== Test 3: Non-causal (right-side information should affect left-side container [COMP]) ======
    # In first row, replace a right-side token with another id
    noncausal_tids = [x[:] for x in token_ids_list]
    if len(noncausal_tids[0]) >= 4:
        # Swap more right-side positions, check if [COMP] is affected
        i1, j1 = 2, 3
        noncausal_tids[0][i1], noncausal_tids[0][j1] = noncausal_tids[0][j1], noncausal_tids[0][i1]
        _, heads_noncausal = fwd_once(noncausal_tids)
        assert not torch.allclose(heads_noncausal[0], heads[0], atol=1e-6), \
            "Non-causal FAILED: changing right-side tokens did not affect [COMP]."
        print("Test#3 Non-causal (full visibility): OK ✅")
    else:
        print("Test#3 Non-causal (full visibility): SKIP (row too short)")

    print("All tests passed 🎉")


if __name__ == "__main__":
    main()