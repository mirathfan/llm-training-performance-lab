import argparse
import csv
import itertools
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch

from data.dataset import CharDataset
from train import (
    autocast_context,
    build_model,
    build_optimizer,
    configure_tf32,
    get_lr,
    load_config,
    make_grad_scaler,
    maybe_compile_model,
    precision_is_available,
    resolve_config,
)
from utils.device import get_device, get_device_name, get_gpu_memory_stats
from utils.logging_utils import save_json
from utils.metrics import count_parameters
from utils.profiler import reset_peak_memory, step_timer
from utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark short GPT training runs for throughput and memory.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    parser.add_argument("--modes", nargs="+", default=["fp32", "amp"], help="Modes such as fp32 amp compile checkpoint.")
    parser.add_argument("--batch-sizes", nargs="+", type=int, help="Batch sizes to benchmark.")
    parser.add_argument("--seq-lens", nargs="+", type=int, help="Sequence lengths to benchmark.")
    parser.add_argument("--warmup-steps", type=int, default=5, help="Warmup steps before timing.")
    parser.add_argument("--benchmark-steps", "--steps", type=int, default=20, help="Timed benchmark steps.")
    parser.add_argument("--attention-backend", choices=["manual", "sdpa"], help="Attention backend to use.")
    parser.add_argument("--tf32", action="store_true", help="Enable TF32 matmul/convolution on CUDA.")
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], help="Training precision/autocast mode.")
    parser.add_argument("--fused-adamw", action="store_true", help="Use fused AdamW when supported.")
    parser.add_argument("--device", help="Override device: auto, cpu, cuda, cuda:0, etc.")
    return parser.parse_args()


def mode_flags(mode: str) -> Dict[str, bool]:
    normalized = mode.lower().replace("-", "_")
    return {
        "amp": "amp" in normalized,
        "precision": "fp16" if "amp" in normalized or "fp16" in normalized else "fp32",
        "torch_compile": "compile" in normalized,
        "activation_checkpointing": "checkpoint" in normalized,
        "attention_backend": "sdpa" if "sdpa" in normalized else None,
    }


def benchmark_one(
    base_config: Dict[str, Any],
    dataset: CharDataset,
    mode: str,
    batch_size: int,
    seq_len: int,
    warmup_steps: int,
    benchmark_steps: int,
    device: torch.device,
) -> Dict[str, Any]:
    config = dict(base_config)
    flags = mode_flags(mode)
    for key, value in flags.items():
        if value is not None:
            config[key] = value
    config["batch_size"] = batch_size
    config["block_size"] = seq_len
    config["max_iters"] = warmup_steps + benchmark_steps
    grad_accum_steps = int(config.get("gradient_accumulation_steps", 1))

    set_seed(int(config["seed"]))
    raw_model = build_model(config, device)
    optimizer = build_optimizer(raw_model, config)
    model, compile_enabled = maybe_compile_model(raw_model, config)
    precision = config.get("precision", "fp32")
    scaler = make_grad_scaler(precision, device)

    timings: List[float] = []
    reset_peak_memory(device)

    for step in range(warmup_steps + benchmark_steps):
        if step == warmup_steps:
            reset_peak_memory(device)

        lr = get_lr(step, config)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        with step_timer(device) as elapsed:
            optimizer.zero_grad(set_to_none=True)
            for _ in range(grad_accum_steps):
                xb, yb = dataset.get_batch("train", batch_size, seq_len)
                with autocast_context(device, precision):
                    _, loss = model(xb, yb)
                    loss_for_backward = loss / grad_accum_steps
                scaler.scale(loss_for_backward).backward()
            if float(config["grad_clip"]) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(raw_model.parameters(), float(config["grad_clip"]))
            scaler.step(optimizer)
            scaler.update()

        if step >= warmup_steps:
            timings.append(elapsed[0])

    if not timings:
        raise ValueError("benchmark_steps must be greater than zero")

    avg_step_time_s = statistics.mean(timings)
    memory_stats = get_gpu_memory_stats(device)
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "status": "ok",
        "mode": mode,
        "precision": precision,
        "amp_enabled": precision in {"fp16", "bf16"} and device.type == "cuda",
        "torch_compile_requested": flags["torch_compile"],
        "torch_compile_enabled": compile_enabled,
        "activation_checkpointing": flags["activation_checkpointing"],
        "attention_backend": config.get("attention_backend", "manual"),
        "tf32_enabled": bool(config.get("tf32", False)),
        "fused_adamw_requested": bool(config.get("fused_adamw", False)),
        "fused_adamw_enabled": bool(config.get("fused_adamw_enabled_resolved", False)),
        "fused_adamw_error": config.get("fused_adamw_error", ""),
        "batch_size": batch_size,
        "seq_len": seq_len,
        "avg_step_time_ms": avg_step_time_s * 1000.0,
        "tokens_per_sec": (batch_size * seq_len * grad_accum_steps) / avg_step_time_s,
        "samples_per_sec": (batch_size * grad_accum_steps) / avg_step_time_s,
        "max_gpu_memory_allocated_mb": memory_stats["max_gpu_memory_allocated_mb"],
        "max_gpu_memory_reserved_mb": memory_stats["max_gpu_memory_reserved_mb"],
        "parameter_count": count_parameters(raw_model),
        "device": str(device),
        "device_name": get_device_name(device),
        "warmup_steps": warmup_steps,
        "benchmark_steps": benchmark_steps,
        "gradient_accumulation_steps": grad_accum_steps,
        "model_config": {
            "vocab_size": config["vocab_size"],
            "block_size": seq_len,
            "n_layer": config["n_layer"],
            "n_head": config["n_head"],
            "n_embd": config["n_embd"],
            "dropout": config["dropout"],
        },
        "note": "" if (mode.lower() != "amp" or device.type == "cuda") else "AMP requested but disabled because CUDA is unavailable.",
    }


def write_csv(rows: Iterable[Dict[str, Any]], path: Path) -> None:
    rows = list(rows)
    if not rows:
        return

    flat_rows = []
    for row in rows:
        flat = dict(row)
        flat["model_config"] = str(flat.get("model_config", {}))
        flat_rows.append(flat)

    fieldnames = sorted({key for row in flat_rows for key in row.keys()})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat_rows)


def make_plots(rows: List[Dict[str, Any]], results_dir: Path) -> None:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    if not ok_rows:
        return

    import matplotlib.pyplot as plt

    labels = [f"{row['mode']}\nbs={row['batch_size']}, seq={row['seq_len']}" for row in ok_rows]

    def bar_plot(metric: str, ylabel: str, filename: str) -> None:
        width = max(8, min(20, len(ok_rows) * 1.2))
        plt.figure(figsize=(width, 5))
        plt.bar(labels, [row[metric] for row in ok_rows], color="#2563eb")
        plt.ylabel(ylabel)
        plt.xticks(rotation=35, ha="right")
        plt.tight_layout()
        plt.savefig(results_dir / filename, dpi=160)
        plt.close()

    bar_plot("tokens_per_sec", "Tokens/sec", "tokens_per_sec.png")
    bar_plot("avg_step_time_ms", "Step time (ms)", "step_time_ms.png")
    bar_plot("max_gpu_memory_allocated_mb", "Max allocated GPU memory (MB)", "memory_usage.png")


def main() -> None:
    args = parse_args()
    base_config = resolve_config(load_config(args.config), args)
    if args.batch_sizes is None:
        args.batch_sizes = [int(base_config["batch_size"])]
    if args.seq_lens is None:
        args.seq_lens = [int(base_config["block_size"])]

    device = get_device(base_config.get("device", "auto"))
    configure_tf32(bool(base_config.get("tf32", False)), device)
    dataset = CharDataset(base_config["data_dir"], device=device)
    results_dir = Path(base_config.get("results_dir", "results"))
    results_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for mode, batch_size, seq_len in itertools.product(args.modes, args.batch_sizes, args.seq_lens):
        print(f"Benchmarking mode={mode}, batch_size={batch_size}, seq_len={seq_len} on {get_device_name(device)}")
        try:
            flags = mode_flags(mode)
            precision = flags.get("precision") or base_config.get("precision", "fp32")
            available, error = precision_is_available(precision, device)
            if not available:
                raise RuntimeError(error)
            row = benchmark_one(
                base_config,
                dataset,
                mode,
                batch_size,
                seq_len,
                args.warmup_steps,
                args.benchmark_steps,
                device,
            )
        except RuntimeError as exc:
            row = {
                "status": "failed",
                "mode": mode,
                "batch_size": batch_size,
                "seq_len": seq_len,
                "device": str(device),
                "device_name": get_device_name(device),
                "error": str(exc),
            }
            if device.type == "cuda":
                torch.cuda.empty_cache()
        rows.append(row)

    save_json(rows, results_dir / "benchmark_results.json")
    write_csv(rows, results_dir / "benchmark_results.csv")
    make_plots(rows, results_dir)
    print(f"Saved benchmark results under {results_dir}")


if __name__ == "__main__":
    main()
