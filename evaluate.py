import argparse
from pathlib import Path

import torch

from data.dataset import CharDataset
from train import build_model
from utils.device import get_device, get_device_name
from utils.logging_utils import create_run_dir, save_json
from utils.metrics import count_parameters, estimate_loss, get_model_size_mb, perplexity_from_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved GPT checkpoint on validation data.")
    parser.add_argument("--checkpoint", required=True, help="Path to a checkpoint, e.g. checkpoints/best_gpt_tiny.pt")
    parser.add_argument("--device", help="Override device: auto, cpu, cuda, cuda:0, etc.")
    parser.add_argument("--eval-iters", type=int, help="Override number of validation batches.")
    parser.add_argument("--batch-size", type=int, help="Override evaluation batch size.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]

    if args.device is not None:
        config["device"] = args.device
    if args.eval_iters is not None:
        config["eval_iters"] = args.eval_iters
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size

    device = get_device(config.get("device", "auto"))
    dataset = CharDataset(config["data_dir"], device=device)
    model = build_model(config, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    amp_enabled = bool(config.get("amp_enabled_resolved", config.get("amp", False))) and device.type == "cuda"
    losses = estimate_loss(
        model,
        dataset,
        int(config["eval_iters"]),
        int(config["batch_size"]),
        int(config["block_size"]),
        device,
        amp_enabled,
    )

    run_name = config.get("run_name", checkpoint_path.stem.replace("best_", "").replace("last_", ""))
    run_dir = create_run_dir(config.get("results_dir", "results"), run_name)
    metrics = {
        "checkpoint": str(checkpoint_path),
        "run_name": run_name,
        "device": str(device),
        "device_name": get_device_name(device),
        "val_loss": losses["val"],
        "val_perplexity": perplexity_from_loss(losses["val"]),
        "train_loss_estimate": losses["train"],
        "parameter_count": count_parameters(model),
        "checkpoint_size_mb": get_model_size_mb(checkpoint_path),
    }
    output_path = run_dir / "eval_metrics.json"
    save_json(metrics, output_path)

    print(f"Validation loss: {metrics['val_loss']:.4f}")
    print(f"Validation perplexity: {metrics['val_perplexity']:.2f}")
    print(f"Saved evaluation metrics to {output_path}")


if __name__ == "__main__":
    main()
