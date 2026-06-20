import platform
from typing import Dict

import torch


def get_device(requested: str = "auto") -> torch.device:
    if requested is None or requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def get_device_name(device: torch.device | str | None = None) -> str:
    device = get_device("auto") if device is None else torch.device(device)
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    cpu_name = platform.processor() or platform.machine() or "CPU"
    return f"CPU ({cpu_name})"


def get_gpu_memory_stats(device: torch.device | str | None = None) -> Dict[str, float]:
    device = get_device("auto") if device is None else torch.device(device)
    if device.type != "cuda":
        return {
            "gpu_memory_allocated_mb": 0.0,
            "gpu_memory_reserved_mb": 0.0,
            "max_gpu_memory_allocated_mb": 0.0,
            "max_gpu_memory_reserved_mb": 0.0,
        }

    return {
        "gpu_memory_allocated_mb": torch.cuda.memory_allocated(device) / (1024**2),
        "gpu_memory_reserved_mb": torch.cuda.memory_reserved(device) / (1024**2),
        "max_gpu_memory_allocated_mb": torch.cuda.max_memory_allocated(device) / (1024**2),
        "max_gpu_memory_reserved_mb": torch.cuda.max_memory_reserved(device) / (1024**2),
    }
