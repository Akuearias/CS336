from attention import run_multihead_self_attention
from positionwise_feedforward import run_positionwise_feedforward
from RMSNorm import run_rmsnorm
import torch

def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    attn_pdrop: float,
    residual_pdrop: float,
    weights: dict[str, torch.FloatTensor],
    in_features: torch.FloatTensor,
) -> torch.FloatTensor:
    norm1_out = run_rmsnorm(d_model, 1e-5, weights["rms1"], in_features)
    attn_out = run_multihead_self_attention(
        d_model, num_heads, attn_pdrop, weights["attn"], norm1_out
    )
    attn_out = torch.dropout(attn_out, residual_pdrop, True)
    x = in_features + attn_out

    norm2_out = run_rmsnorm(d_model, 1e-5, weights["rms2"], x)
    ffn_out = run_positionwise_feedforward(d_model, d_ff, weights["ffn"], norm2_out)
    ffn_out = torch.dropout(ffn_out, residual_pdrop, True)

    return x + ffn_out