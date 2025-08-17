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
        
        # 2. Process through row-column attention stack.
        compressed_rows = self.attention_stack(
            embedded_rows=embedded_rows,
            row_attention_masks=row_attention_masks,
            column_attention_masks=column_attention_masks,
            row_position_embeddings=row_position_embeddings,
            column_position_embeddings=column_position_embeddings
        )
        
        return compressed_rows


def main():
    """
    A rigorous, reproducible test harness for LexiconCompressorModel.
    Focus: self-consistency, column permutation-equivariance, row order sensitivity, non-causality.
    """
    import os, random, numpy as np, time
    import torch
    from transformers import AutoTokenizer, Qwen3ForCausalLM
    from tokenization_lexicon import LexiconTokenizer

    # ---------------- Config ----------------
    MODEL_NAME   = "Qwen/Qwen3-0.6B"
    CSV_PATH     = "cleaned_lexicon_tiny.csv"
    COLUMNS      = ["lexical_unit", "pos", "gloss", "variant"]
    NUM_LAYERS   = 6           
    SHOW_N       = 100         
    ATOL         = 1e-5        
    RTOL         = 1e-5
    SEED         = 0
    PRINT_DIFF   = True        
    STRESS_ROUNDS= 8           

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # --------------- Determinism ---------------
    def fix_seeds(seed=SEED):
        random.seed(seed); np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    fix_seeds(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Env] device={device}, torch={torch.__version__}")

    # --------------- Helper: masks & RoPE ---------------
    def full_vis_mask(L: int, device=device):
        return torch.ones(1, 1, L, L, dtype=torch.float32, device=device)

    def build_row_rope(rotary, L: int, H: int, device=device):
        # 行 RoPE：顺序编码
        dummy = torch.ones(1, L, H, device=device)
        pos   = torch.arange(L, device=device).unsqueeze(0)
        return rotary(dummy, pos)

    def build_col_identity_rope(rotary, N: int, H: int, device=device):
        # 列 RoPE：恒等（同相位），保证置换等变
        dummy = torch.ones(1, N, H, device=device)
        pos0  = torch.ones(1, N, dtype=torch.long, device=device)
        return rotary(dummy, pos0)

    # --------------- Load tokenizer & lexicon ---------------
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tok.add_tokens(["[COMP]"], special_tokens=False)
    comp_id = tok.convert_tokens_to_ids("[COMP]")

    lt = LexiconTokenizer(
        csv_file_path=CSV_PATH,
        tokenizer=tok,
        column_names=COLUMNS,
        compress_token_id=comp_id,
        delimiter=",",
        add_special_tokens=False
    )
    token_ids_list = lt.process_lexicon()
    assert token_ids_list, "No entries parsed from CSV."
    token_ids_list = token_ids_list[:SHOW_N]

    # --------------- Load Qwen weights for init ---------------
    qwen   = Qwen3ForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True).to(device)
    cfg    = qwen.config
    rotary = qwen.model.rotary_emb
    H      = cfg.hidden_size

    # --------------- Build LCM and load weights ---------------
    lcm = LexiconCompressorModel(config=cfg, num_attention_layers=NUM_LAYERS).to(device)
    emb_weights = qwen.get_input_embeddings().state_dict()

    attn_weights = []
    for i in range(NUM_LAYERS):
        # 约定：偶数层 = row 分支，奇数层 = column 分支（按你实现调整）
        row_sd = qwen.model.layers[2*i].state_dict()
        col_sd = qwen.model.layers[2*i + 1].state_dict()
        attn_weights.append((row_sd, col_sd))

    # 确保评估模式（禁用 Dropout/DropPath）
    lcm.eval()
    # 若你有自定义 Dropout，可强制置 0
    for m in lcm.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0

    # 一次性加载权重
    lcm.load_embeddings_weights(emb_weights)
    lcm.load_attention_weights(attn_weights)

    # --------------- Build inputs for one forward ---------------
    def build_inputs(tids_list):
        # row：每一层都要一份（可复用同构造）
        row_pos_layers, row_msk_layers = [], []
        for _ in range(NUM_LAYERS):
            row_pos = [build_row_rope(rotary, len(x), H) for x in tids_list]
            row_msk = [full_vis_mask(len(x)) for x in tids_list]
            row_pos_layers.append(row_pos)
            row_msk_layers.append(row_msk)
        # column：每层一份
        N = len(tids_list)
        col_pos_layers = [build_col_identity_rope(rotary, N, H) for _ in range(NUM_LAYERS)]
        col_msk_layers = [full_vis_mask(N) for _ in range(NUM_LAYERS)]
        return row_pos_layers, row_msk_layers, col_pos_layers, col_msk_layers

    # --------------- Forward wrapper ---------------
    @torch.no_grad()
    def forward_heads(tids_list):
        row_pos, row_msk, col_pos, col_msk = build_inputs(tids_list)
        out_rows = lcm(
            token_ids_list=tids_list,
            attention_weights=None,            # 已在 lcm 内部装载
            embeddings_weights=None,
            row_attention_masks=row_msk,
            column_attention_masks=col_msk,
            row_position_embeddings=row_pos,
            column_position_embeddings=col_pos
        )
        # 取每行第 0 个 token（[COMP]）的向量作为“容器向量”
        heads = torch.stack([r[0].detach().cpu() for r in out_rows], dim=0)
        return heads

    # --------------- Test 0: Self-consistency ---------------
    print("\n[Test 0] Self-consistency...")
    fix_seeds(SEED)  # 保证一致
    hA1 = forward_heads(token_ids_list)
    fix_seeds(SEED)
    hA2 = forward_heads(token_ids_list)

    self_ok = torch.allclose(hA1, hA2, atol=ATOL, rtol=RTOL)
    max_self_diff = (hA1 - hA2).abs().max().item()
    print(f" -> allclose={self_ok}, max_abs_diff={max_self_diff:.3e}")
    assert self_ok, f"Self-consistency FAILED (max diff {max_self_diff:.3e})"

    # --------------- Test 1: Column permutation-equivariance ---------------
    print("\n[Test 1] Column permutation-equivariance...")
    fix_seeds(SEED)
    base_heads = forward_heads(token_ids_list)

    perm = list(range(len(token_ids_list)))
    random.shuffle(perm)
    perm_tids = [token_ids_list[i] for i in perm]

    fix_seeds(SEED)   # 重要：相同随机态（尽管 eval+no-dropout 应当已无随机性）
    perm_heads = forward_heads(perm_tids)

    # 应满足 perm_heads ≈ base_heads[perm]
    ref = base_heads[perm]
    eq_ok = torch.allclose(perm_heads, ref, atol=ATOL, rtol=RTOL)
    max_perm_diff = (perm_heads - ref).abs().max().item()
    print(f" -> allclose={eq_ok}, max_abs_diff={max_perm_diff:.3e}")
    if not eq_ok and PRINT_DIFF:
        print("   (DIAG) perm example idx 0 diff norm:",
              torch.norm(perm_heads[0] - ref[0]).item())
    assert eq_ok, f"Column invariance FAILED (max diff {max_perm_diff:.3e}). " \
                  f"检查列注意力是否非因果、是否无绝对列 position_ids、掩码是否全可见。"

    # --------------- Test 2: Row order sensitivity ---------------
    print("\n[Test 2] Row order sensitivity (intra-row swap should change head)...")
    swapped = [x[:] for x in token_ids_list]
    # 选第一行，交换第 1/2 个真实 token（0 是 [COMP]）
    if len(swapped[0]) >= 3:
        i0, j0 = 1, 2
        swapped[0][i0], swapped[0][j0] = swapped[0][j0], swapped[0][i0]
        fix_seeds(SEED)
        h_swap = forward_heads(swapped)
        change_norm = torch.norm(h_swap[0] - base_heads[0]).item()
        changed = change_norm > 1e-6
        print(f" -> changed={changed}, head_delta_norm={change_norm:.3e}")
        assert changed, "Row order sensitivity FAILED: swapping tokens did not change the container."
    else:
        print(" -> SKIP (first row too short)")

    # --------------- Test 3: Non-causality ---------------
    print("\n[Test 3] Non-causality (right-side change should affect [COMP])...")
    noncausal = [x[:] for x in token_ids_list]
    if len(noncausal[0]) >= 4:
        i1, j1 = 2, 3
        noncausal[0][i1], noncausal[0][j1] = noncausal[0][j1], noncausal[0][i1]
        fix_seeds(SEED)
        h_nc = forward_heads(noncausal)
        delta = torch.norm(h_nc[0] - base_heads[0]).item()
        affected = delta > 1e-6
        print(f" -> affected={affected}, head_delta_norm={delta:.3e}")
        assert affected, "Non-causality FAILED: changing right-side tokens did not affect [COMP]."
    else:
        print(" -> SKIP (first row too short)")

    # --------------- (Optional) Stress test ---------------
    if STRESS_ROUNDS > 0:
        print(f"\n[Stress] {STRESS_ROUNDS} random permutations ...")
        pass_cnt, max_seen = 0, 0.0
        for r in range(STRESS_ROUNDS):
            perm = list(range(len(token_ids_list)))
            random.shuffle(perm)
            perm_tids = [token_ids_list[i] for i in perm]
            fix_seeds(SEED)
            ph = forward_heads(perm_tids)
            ref = base_heads[perm]
            ok = torch.allclose(ph, ref, atol=ATOL, rtol=RTOL)
            diff = (ph - ref).abs().max().item()
            max_seen = max(max_seen, diff)
            pass_cnt += int(ok)
        print(f" -> pass {pass_cnt}/{STRESS_ROUNDS}, max_abs_diff_seen={max_seen:.3e}")
        assert pass_cnt == STRESS_ROUNDS, \
            f"Permutation equivariance flaky: only {pass_cnt}/{STRESS_ROUNDS} passes (max diff {max_seen:.3e})."

    print("\nAll tests passed 🎉")



if __name__ == "__main__":
    main()