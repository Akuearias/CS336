import math
import torch


def run_gelu(x: torch.FloatTensor) -> torch.FloatTensor:
    return 0.5 * x * (1.0 + torch.tanh(
        math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))
    ))

def run_positionwise_feedforward(
    d_model: int,
    d_ff: int,
    weights: dict[str, torch.FloatTensor],
    in_features: torch.FloatTensor,
) -> torch.FloatTensor:
    """
    Args:
        in_features: (B, L, d_model)
    Returns:
        output: (B, L, d_model)
    """
    # Optional: validate weight shapes
    assert weights["fc1"].shape == (d_ff, d_model), "fc1 weight shape mismatch"
    assert weights["fc2"].shape == (d_model, d_ff), "fc2 weight shape mismatch"

    hidden = in_features @ weights["fc1"].T  # (B, L, d_ff)
    hidden = run_gelu(hidden)
    output = hidden @ weights["fc2"].T  # (B, L, d_model)
    return output
