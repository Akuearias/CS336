from transformer_block import run_transformer_block
from RMSNorm import run_rmsnorm
import torch

def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    attn_pdrop: float,
    residual_pdrop: float,
    weights: dict[str, torch.FloatTensor],
    in_indices: torch.LongTensor,
) -> torch.FloatTensor:
    B, L = in_indices.shape
    assert L <= context_length
    assert weights["tok_emb"].shape[0] == vocab_size, "Mismatch in vocab size"

    tok_emb = weights["tok_emb"]
    pos_emb = weights["pos_emb"]
    h = tok_emb[in_indices] + pos_emb[:L]

    for i in range(num_layers):
        h = run_transformer_block(
            d_model, num_heads, d_ff, attn_pdrop, residual_pdrop, weights[f"blocks.{i}"], h
        )

    h = run_rmsnorm(d_model, 1e-5, weights["final_norm"], h)
    return h @ weights["head"].T  # (B, L, vocab_size)