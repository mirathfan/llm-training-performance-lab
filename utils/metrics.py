import math
from contextlib import nullcontext
from pathlib import Path
from typing import Dict

import torch


def autocast_context(device: torch.device | str, enabled: bool):
    device = torch.device(device)
    if enabled and device.type == "cuda":
        return torch.cuda.amp.autocast(enabled=True)
    return nullcontext()


@torch.no_grad()
def estimate_loss(
    model: torch.nn.Module,
    dataset,
    eval_iters: int,
    batch_size: int,
    block_size: int,
    device: torch.device | str,
    amp_enabled: bool = False,
) -> Dict[str, float]:
    model_was_training = model.training
    model.eval()
    losses = {}

    for split in ["train", "val"]:
        split_losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = dataset.get_batch(split, batch_size, block_size)
            with autocast_context(device, amp_enabled):
                _, loss = model(xb, yb)
            split_losses[k] = loss.item()
        losses[split] = float(split_losses.mean().item())

    if model_was_training:
        model.train()
    return losses


def perplexity_from_loss(loss: float) -> float:
    try:
        return float(math.exp(loss))
    except OverflowError:
        return float("inf")


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    params = model.parameters()
    if trainable_only:
        params = (p for p in params if p.requires_grad)
    return sum(p.numel() for p in params)


def get_model_size_mb(checkpoint_path: str | Path) -> float | None:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        return None
    return checkpoint_path.stat().st_size / (1024**2)
