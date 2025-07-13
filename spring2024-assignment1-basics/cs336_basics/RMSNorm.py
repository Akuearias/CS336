import torch, math


def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: dict[str, torch.FloatTensor],
    in_features: torch.FloatTensor,
) -> torch.FloatTensor:
    scale = weights["scale"]  # (d_model,)
    norm = in_features.norm(dim=-1, keepdim=True) / math.sqrt(d_model)
    return (in_features / (norm + eps)) * scale