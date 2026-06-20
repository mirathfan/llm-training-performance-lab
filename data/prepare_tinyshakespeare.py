import argparse
import json
import urllib.request
from pathlib import Path

import numpy as np


DEFAULT_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def download_if_needed(url: str, raw_path: Path) -> None:
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    if raw_path.exists():
        print(f"Using existing raw dataset: {raw_path}")
        return

    print(f"Downloading Tiny Shakespeare from {url}")
    urllib.request.urlretrieve(url, raw_path)
    print(f"Saved raw dataset to {raw_path}")


def build_dataset(raw_path: Path, processed_dir: Path, val_fraction: float) -> None:
    text = raw_path.read_text(encoding="utf-8")
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}

    dtype = np.uint16 if len(chars) <= np.iinfo(np.uint16).max else np.int32
    encoded = np.array([stoi[ch] for ch in text], dtype=dtype)

    split_idx = int((1.0 - val_fraction) * len(encoded))
    train_data = encoded[:split_idx]
    val_data = encoded[split_idx:]

    processed_dir.mkdir(parents=True, exist_ok=True)
    train_data.tofile(processed_dir / "train.bin")
    val_data.tofile(processed_dir / "val.bin")

    meta = {
        "dataset": "tinyshakespeare",
        "vocab_size": len(chars),
        "chars": chars,
        "stoi": stoi,
        "itos": {str(k): v for k, v in itos.items()},
        "dtype": np.dtype(dtype).name,
        "total_tokens": int(len(encoded)),
        "train_tokens": int(len(train_data)),
        "val_tokens": int(len(val_data)),
        "val_fraction": val_fraction,
        "raw_file": str(raw_path),
    }
    (processed_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Vocabulary size: {len(chars)}")
    print(f"Train tokens: {len(train_data):,}")
    print(f"Validation tokens: {len(val_data):,}")
    print(f"Processed files saved under {processed_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and encode Tiny Shakespeare.")
    parser.add_argument("--url", default=DEFAULT_URL, help="Tiny Shakespeare source URL.")
    parser.add_argument("--raw-dir", default="data/raw", help="Directory for raw text data.")
    parser.add_argument("--processed-dir", default="data/processed", help="Output directory for encoded data.")
    parser.add_argument("--val-fraction", type=float, default=0.1, help="Fraction of tokens used for validation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.val_fraction < 1.0:
        raise ValueError("--val-fraction must be between 0 and 1")

    raw_path = Path(args.raw_dir) / "tinyshakespeare.txt"
    download_if_needed(args.url, raw_path)
    build_dataset(raw_path, Path(args.processed_dir), args.val_fraction)


if __name__ == "__main__":
    main()
