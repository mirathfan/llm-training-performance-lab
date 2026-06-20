import argparse
import copy
import math
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import yaml

from data.dataset import CharDataset
from models.gpt import GPT, GPTConfig
from utils.device import get_device, get_device_name, get_gpu_memory_stats
from utils.logging_utils import append_json_log, create_run_dir, save_config_copy, save_json
from utils.metrics import count_parameters, estimate_loss, perplexity_from_loss
from utils.profiler import reset_peak_memory, step_timer
from utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a small GPT-style Transformer.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    parser.add_argument("--amp", action="store_true", help="Enable CUDA AMP mixed precision.")
    parser.add_argument("--compile", action="store_true", dest="torch_compile", help="Enable torch.compile.")
    parser.add_argument(
        "--activation-checkpointing",
        action="store_true",
        help="Enable activation checkpointing in Transformer blocks.",
    )
    parser.add_argument("--max-iters", type=int, help="Override max training iterations.")
    parser.add_argument("--batch-size", type=int, help="Override batch size.")
    parser.add_argument("--gradient-accumulation-steps", type=int, help="Override gradient accumulation steps.")
    parser.add_argument("--learning-rate", type=float, help="Override learning rate.")
    parser.add_argument("--device", help="Override device: auto, cpu, cuda, cuda:0, etc.")
    parser.add_argument("--run-name", help="Override run name for logs and checkpoints.")
    return parser.parse_args()


def load_config(config_path: str | Path) -> Dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Config at {config_path} must contain a YAML mapping")
    config["config_path"] = str(config_path)
    return config


def load_vocab_size(data_dir: str) -> int | None:
    meta_path = Path(data_dir) / "meta.json"
    if not meta_path.exists():
        return None
    import json

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return int(meta["vocab_size"])


def resolve_config(config: Dict[str, Any], args: argparse.Namespace | None = None) -> Dict[str, Any]:
    resolved = copy.deepcopy(config)
    resolved.setdefault("data_dir", "data/processed")
    resolved.setdefault("results_dir", "results")
    resolved.setdefault("checkpoint_dir", "checkpoints")
    resolved.setdefault("run_name", resolved.get("model_name", "gpt_run"))
    resolved.setdefault("warmup_iters", max(1, int(resolved.get("max_iters", 1000)) // 20))
    resolved.setdefault("min_lr", float(resolved.get("learning_rate", 3e-4)) * 0.1)
    resolved.setdefault("log_interval", 10)
    resolved.setdefault("gradient_accumulation_steps", 1)
    resolved.setdefault("amp", False)
    resolved.setdefault("torch_compile", False)
    resolved.setdefault("activation_checkpointing", False)

    if args is not None:
        if getattr(args, "amp", False):
            resolved["amp"] = True
        if getattr(args, "torch_compile", False):
            resolved["torch_compile"] = True
        if getattr(args, "activation_checkpointing", False):
            resolved["activation_checkpointing"] = True
        if getattr(args, "max_iters", None) is not None:
            resolved["max_iters"] = args.max_iters
        if getattr(args, "batch_size", None) is not None:
            resolved["batch_size"] = args.batch_size
        if getattr(args, "gradient_accumulation_steps", None) is not None:
            resolved["gradient_accumulation_steps"] = args.gradient_accumulation_steps
        if getattr(args, "learning_rate", None) is not None:
            resolved["learning_rate"] = args.learning_rate
        if getattr(args, "device", None) is not None:
            resolved["device"] = args.device
        if getattr(args, "run_name", None) is not None:
            resolved["run_name"] = args.run_name

    if resolved.get("vocab_size") == "auto":
        vocab_size = load_vocab_size(resolved["data_dir"])
        if vocab_size is None:
            raise FileNotFoundError(
                f"vocab_size is set to auto, but {resolved['data_dir']}/meta.json was not found. "
                "Run: python data/prepare_tinyshakespeare.py"
            )
        resolved["vocab_size"] = vocab_size

    resolved["vocab_size"] = int(resolved["vocab_size"])
    resolved["block_size"] = int(resolved["block_size"])
    resolved["n_layer"] = int(resolved["n_layer"])
    resolved["n_head"] = int(resolved["n_head"])
    resolved["n_embd"] = int(resolved["n_embd"])
    resolved["batch_size"] = int(resolved["batch_size"])
    resolved["gradient_accumulation_steps"] = int(resolved["gradient_accumulation_steps"])
    if resolved["gradient_accumulation_steps"] < 1:
        raise ValueError("gradient_accumulation_steps must be >= 1")
    resolved["max_iters"] = int(resolved["max_iters"])
    resolved["eval_interval"] = int(resolved["eval_interval"])
    resolved["eval_iters"] = int(resolved["eval_iters"])
    resolved["seed"] = int(resolved["seed"])
    return resolved


def build_model(config: Dict[str, Any], device: torch.device) -> GPT:
    model_config = GPTConfig(
        vocab_size=config["vocab_size"],
        block_size=config["block_size"],
        n_layer=config["n_layer"],
        n_head=config["n_head"],
        n_embd=config["n_embd"],
        dropout=float(config["dropout"]),
        activation_checkpointing=bool(config.get("activation_checkpointing", False)),
    )
    return GPT(model_config).to(device)


def build_optimizer(model: torch.nn.Module, config: Dict[str, Any]) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        betas=(float(config["beta1"]), float(config["beta2"])),
        weight_decay=float(config["weight_decay"]),
    )


def get_lr(iter_num: int, config: Dict[str, Any]) -> float:
    learning_rate = float(config["learning_rate"])
    min_lr = float(config.get("min_lr", learning_rate * 0.1))
    warmup_iters = int(config.get("warmup_iters", 0))
    max_iters = int(config["max_iters"])

    if warmup_iters > 0 and iter_num < warmup_iters:
        return learning_rate * float(iter_num + 1) / float(warmup_iters)
    if iter_num >= max_iters:
        return min_lr

    decay_iters = max(1, max_iters - warmup_iters)
    decay_ratio = min(1.0, (iter_num - warmup_iters) / decay_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


def autocast_context(device: torch.device, enabled: bool):
    from contextlib import nullcontext

    if enabled and device.type == "cuda":
        return torch.cuda.amp.autocast(enabled=True)
    return nullcontext()


def make_grad_scaler(enabled: bool):
    return torch.cuda.amp.GradScaler(enabled=enabled)


def maybe_compile_model(model: GPT, config: Dict[str, Any]) -> Tuple[torch.nn.Module, bool]:
    if not config.get("torch_compile", False):
        return model, False
    if not hasattr(torch, "compile"):
        print("torch.compile requested, but this PyTorch build does not provide torch.compile. Continuing eager.")
        return model, False
    try:
        return torch.compile(model), True
    except Exception as exc:
        print(f"torch.compile requested but failed to initialize: {exc}. Continuing eager.")
        return model, False


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: Dict[str, Any],
    iter_num: int,
    best_val_loss: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "iter_num": iter_num,
            "best_val_loss": best_val_loss,
        },
        path,
    )


def train(config: Dict[str, Any]) -> None:
    set_seed(config["seed"])
    device = get_device(config.get("device", "auto"))
    amp_enabled = bool(config.get("amp", False)) and device.type == "cuda"
    if config.get("amp", False) and not amp_enabled:
        print("AMP requested, but CUDA is unavailable. Training will use FP32.")

    dataset = CharDataset(config["data_dir"], device=device)
    raw_model = build_model(config, device)
    optimizer = build_optimizer(raw_model, config)
    model, compile_enabled = maybe_compile_model(raw_model, config)
    scaler = make_grad_scaler(amp_enabled)

    run_name = config["run_name"]
    run_dir = create_run_dir(config["results_dir"], run_name)
    log_path = run_dir / "train_log.json"
    save_json([], log_path)

    config["device_resolved"] = str(device)
    config["device_name"] = get_device_name(device)
    config["amp_enabled_resolved"] = amp_enabled
    config["torch_compile_enabled_resolved"] = compile_enabled
    config["parameter_count"] = count_parameters(raw_model)
    save_config_copy(config, run_dir)

    checkpoint_dir = Path(config["checkpoint_dir"])
    best_path = checkpoint_dir / f"best_{run_name}.pt"
    last_path = checkpoint_dir / f"last_{run_name}.pt"
    best_val_loss = float("inf")

    print(f"Run: {run_name}")
    print(f"Device: {config['device_name']}")
    print(f"Parameters: {config['parameter_count']:,}")

    for iter_num in range(config["max_iters"]):
        lr = get_lr(iter_num, config)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        reset_peak_memory(device)
        grad_accum_steps = int(config.get("gradient_accumulation_steps", 1))
        train_loss_value = 0.0

        with step_timer(device) as timings:
            optimizer.zero_grad(set_to_none=True)
            for _ in range(grad_accum_steps):
                xb, yb = dataset.get_batch("train", config["batch_size"], config["block_size"])
                with autocast_context(device, amp_enabled):
                    _, loss = model(xb, yb)
                    loss_for_backward = loss / grad_accum_steps
                train_loss_value += float(loss.item())
                scaler.scale(loss_for_backward).backward()
            if float(config["grad_clip"]) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(raw_model.parameters(), float(config["grad_clip"]))
            scaler.step(optimizer)
            scaler.update()

        train_loss_value /= grad_accum_steps
        step_time_s = timings[0]
        tokens_per_step = config["batch_size"] * config["block_size"] * grad_accum_steps
        tokens_per_sec = tokens_per_step / max(step_time_s, 1e-12)
        memory_stats = get_gpu_memory_stats(device)

        should_eval = (iter_num + 1) % config["eval_interval"] == 0 or iter_num == 0 or (
            iter_num + 1 == config["max_iters"]
        )
        if should_eval:
            losses = estimate_loss(
                model,
                dataset,
                config["eval_iters"],
                config["batch_size"],
                config["block_size"],
                device,
                amp_enabled,
            )
            val_ppl = perplexity_from_loss(losses["val"])
            record = {
                "iteration": iter_num + 1,
                "train_loss": train_loss_value,
                "train_loss_estimate": losses["train"],
                "val_loss": losses["val"],
                "val_perplexity": val_ppl,
                "learning_rate": lr,
                "step_time_ms": step_time_s * 1000.0,
                "tokens_per_sec": tokens_per_sec,
                "gradient_accumulation_steps": grad_accum_steps,
                **memory_stats,
            }
            append_json_log(record, log_path)

            print(
                f"iter {iter_num + 1:5d} | train {train_loss_value:.4f} | "
                f"val {losses['val']:.4f} | ppl {val_ppl:.2f} | lr {lr:.2e} | "
                f"{step_time_s * 1000.0:.1f} ms | {tokens_per_sec:.0f} tok/s | "
                f"max mem {memory_stats['max_gpu_memory_allocated_mb']:.1f} MB"
            )

            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                save_checkpoint(best_path, raw_model, optimizer, config, iter_num + 1, best_val_loss)
            save_checkpoint(last_path, raw_model, optimizer, config, iter_num + 1, best_val_loss)

        elif (iter_num + 1) % int(config.get("log_interval", 10)) == 0:
            print(
                f"iter {iter_num + 1:5d} | train {train_loss_value:.4f} | lr {lr:.2e} | "
                f"{step_time_s * 1000.0:.1f} ms | {tokens_per_sec:.0f} tok/s"
            )

    print(f"Saved best checkpoint to {best_path}")
    print(f"Saved last checkpoint to {last_path}")
    print(f"Saved training log to {log_path}")


def main() -> None:
    args = parse_args()
    config = resolve_config(load_config(args.config), args)
    train(config)


if __name__ == "__main__":
    main()
