import torch
import math
import torch.nn.functional as F
from typing import Optional


def run_scaled_dot_product_attention(
    K: torch.FloatTensor,
    Q: torch.FloatTensor,
    V: torch.FloatTensor,
    mask: Optional[torch.BoolTensor] = None,
    pdrop: Optional[float] = None,
) -> torch.FloatTensor:
    """
    Q, K, V: (B, H, L, D)
    mask: (B, 1, L, L), optional
    pdrop: dropout rate
    """
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    attn_weights = F.softmax(scores, dim=-1)

    if pdrop is not None and pdrop > 0:
        attn_weights = F.dropout(attn_weights, p=pdrop, training=True)

    output = torch.matmul(attn_weights, V)
    return output


from typing import Optional
import attention
import torch


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    attn_pdrop: float,
    weights: dict[str, torch.FloatTensor],
    in_features: torch.FloatTensor,
    mask: Optional[torch.BoolTensor] = None
) -> torch.FloatTensor:
    """
    Args:
        in_features: (B, L, d_model)
    Returns:
        output: (B, L, d_model)
    """
    B, L, _ = in_features.shape
    H = num_heads
    d_head = d_model // H

    qkv = in_features @ weights["qkv"].T  # (B, L, 3 * d_model)
    q, k, v = torch.chunk(qkv, 3, dim=-1)  # Each: (B, L, d_model)

    # Reshape to (B, H, L, d_head)
    q = q.view(B, L, H, d_head).transpose(1, 2)
    k = k.view(B, L, H, d_head).transpose(1, 2)
    v = v.view(B, L, H, d_head).transpose(1, 2)

    # Scaled dot-product attention
    attn_out = scaled_dot_product_attention.run_scaled_dot_product_attention(k, q, v, mask, attn_pdrop)  # (B, H, L, d_head)

    # Concatenate heads
    attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, d_model)
    out = attn_out @ weights["out"].T

    return out