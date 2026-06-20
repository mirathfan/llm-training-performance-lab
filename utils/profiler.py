import time
from contextlib import contextmanager
from typing import Iterator

import torch


def synchronize_device(device: torch.device | str) -> None:
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def reset_peak_memory(device: torch.device | str) -> None:
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


@contextmanager
def step_timer(device: torch.device | str) -> Iterator[list[float]]:
    timings: list[float] = []
    synchronize_device(device)
    start = time.perf_counter()
    try:
        yield timings
    finally:
        synchronize_device(device)
        timings.append(time.perf_counter() - start)
