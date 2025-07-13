import os
import torch
import numpy as np
import torch.nn.functional as F
from torch.optim import AdamW
from typing import Iterable, Type, BinaryIO, IO
import numpy.typing as npt
import math

def run_get_batch(dataset: npt.NDArray, batch_size: int, context_length: int, device: str):
    ix = np.random.randint(0, dataset.shape[0] - context_length, size=batch_size)
    x = torch.tensor([dataset[i:i+context_length] for i in ix], dtype=torch.long, device=device)
    y = torch.tensor([dataset[i+1:i+context_length+1] for i in ix], dtype=torch.long, device=device)
    return x, y

def run_softmax(in_features: torch.FloatTensor, dim: int) -> torch.FloatTensor:
    return F.softmax(in_features, dim=dim)

def run_cross_entropy(inputs: torch.FloatTensor, targets: torch.LongTensor):
    return F.cross_entropy(inputs.view(-1, inputs.size(-1)), targets.view(-1))

def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float):
    torch.nn.utils.clip_grad_norm_(parameters, max_l2_norm)

def get_adamw_cls() -> Type[torch.optim.Optimizer]:
    return AdamW

def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters
    progress = (it - warmup_iters) / max(1, cosine_cycle_iters - warmup_iters)
    cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
    return min_learning_rate + (max_learning_rate - min_learning_rate) * cosine_decay

def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }, out)

def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
):
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["iteration"]