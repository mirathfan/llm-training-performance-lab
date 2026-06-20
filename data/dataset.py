import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch


class CharDataset:
    """Memory-mapped character-level dataset with random batch sampling."""

    def __init__(self, data_dir: str = "data/processed", device: torch.device | str = "cpu") -> None:
        self.data_dir = Path(data_dir)
        self.device = torch.device(device)
        meta_path = self.data_dir / "meta.json"

        if not meta_path.exists():
            raise FileNotFoundError(
                f"Processed dataset metadata not found at {meta_path}. "
                "Run: python data/prepare_tinyshakespeare.py"
            )

        self.meta: Dict = json.loads(meta_path.read_text(encoding="utf-8"))
        dtype = np.dtype(self.meta["dtype"])
        self.train_data = np.memmap(self.data_dir / "train.bin", dtype=dtype, mode="r")
        self.val_data = np.memmap(self.data_dir / "val.bin", dtype=dtype, mode="r")

    @property
    def vocab_size(self) -> int:
        return int(self.meta["vocab_size"])

    def get_batch(self, split: str, batch_size: int, block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        data = self.train_data if split == "train" else self.val_data
        if len(data) <= block_size + 1:
            raise ValueError(f"{split} split is too small for block_size={block_size}")

        ix = torch.randint(len(data) - block_size - 1, (batch_size,))
        x = torch.stack(
            [torch.from_numpy(np.array(data[int(i) : int(i) + block_size], dtype=np.int64)) for i in ix]
        )
        y = torch.stack(
            [
                torch.from_numpy(np.array(data[int(i) + 1 : int(i) + block_size + 1], dtype=np.int64))
                for i in ix
            ]
        )

        if self.device.type == "cuda":
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
        else:
            x = x.to(self.device)
            y = y.to(self.device)
        return x, y
