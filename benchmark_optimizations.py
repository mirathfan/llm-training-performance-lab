import argparse
import csv
import statistics
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
    parser = argparse.ArgumentParser(description="Benchmark cumulative GPT training optimizations.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    parser.add_argument("--steps", type=int, default=20, help="Timed benchmark steps per repeat.")
    parser.add_argument("--warmup-steps", type=int, default=5, help="Warmup steps before timing.")
    parser.add_argument("--repeats", type=int, default=3, help="Repeats per optimization row.")
    parser.add_argument("--device", help="Override device: auto, cpu, cuda, cuda:0, etc.")
    return parser.parse_args()


def optimization_plan(device: torch.device) -> List[Dict[str, Any]]:
    bf16_supported = device.type == "cuda" and torch.cuda.is_bf16_supported()
    return [
        {
            "name": "baseline_fp32_manual",
            "precision": "fp32",
            "attention_backend": "manual",
            "tf32": False,
            "fused_adamw": False,
            "torch_compile": False,
        },
        {
            "name": "tf32_manual",
            "precision": "fp32",
            "attention_backend": "manual",
            "tf32": True,
            "fused_adamw": False,
            "torch_compile": False,
        },
        {
            "name": "tf32_sdpa",
            "precision": "fp32",
            "attention_backend": "sdpa",
            "tf32": True,
            "fused_adamw": False,
            "torch_compile": False,
        },
        {
            "name": "tf32_sdpa_fp16",
            "precision": "fp16",
            "attention_backend": "sdpa",
            "tf32": True,
            "fused_adamw": False,
            "torch_compile": False,
        },
        {
            "name": "tf32_sdpa_bf16",
            "precision": "bf16",
            "attention_backend": "sdpa",
            "tf32": True,
            "fused_adamw": False,
            "torch_compile": False,
            "skip": not bf16_supported,
            "skip_reason": "BF16 is not supported by this CUDA device." if not bf16_supported else "",
        },
        {
            "name": "tf32_sdpa_fp16_fused_adamw",
            "precision": "fp16",
            "attention_backend": "sdpa",
            "tf32": True,
            "fused_adamw": True,
            "torch_compile": False,
        },
        {
            "name": "tf32_sdpa_fp16_fused_adamw_compile",
            "precision": "fp16",
            "attention_backend": "sdpa",
            "tf32": True,
            "fused_adamw": True,
            "torch_compile": True,
        },
    ]


def check_attention_backend(config: Dict[str, Any], device: torch.device, backend: str) -> Tuple[bool, str]:
    smoke_config = dict(config)
    smoke_config["attention_backend"] = backend
    smoke_config["block_size"] = min(int(config["block_size"]), 16)
    smoke_config["dropout"] = 0.0
    try:
        model = build_model(smoke_config, device)
        model.train()
        x = torch.randint(0, smoke_config["vocab_size"], (2, smoke_config["block_size"]), device=device)
        y = torch.randint(0, smoke_config["vocab_size"], (2, smoke_config["block_size"]), device=device)
        _, loss = model(x, y)
        loss.backward()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        del model, x, y, loss
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return True, ""
    except Exception as exc:
        return False, str(exc)


def check_fused_adamw_support(device: torch.device) -> Tuple[bool, str]:
    if device.type != "cuda":
        return False, "Fused AdamW requires CUDA."
    try:
        param = torch.nn.Parameter(torch.zeros(1, device=device))
        optimizer = torch.optim.AdamW([param], lr=1e-3, fused=True)
        loss = param.sum()
        loss.backward()
        optimizer.step()
        del param, optimizer, loss
        torch.cuda.empty_cache()
        return True, ""
    except Exception as exc:
        return False, str(exc)


def run_training_repeat(
    base_config: Dict[str, Any],
    dataset: CharDataset,
    opt_config: Dict[str, Any],
    repeat: int,
    warmup_steps: int,
    benchmark_steps: int,
    device: torch.device,
    fused_supported: bool,
    fused_error: str,
) -> Dict[str, Any]:
    config = dict(base_config)
    config.update(
        {
            "precision": opt_config["precision"],
            "attention_backend": opt_config["attention_backend"],
            "tf32": opt_config["tf32"],
            "fused_adamw": opt_config["fused_adamw"],
            "torch_compile": opt_config["torch_compile"],
            "max_iters": warmup_steps + benchmark_steps,
        }
    )

    common = {
        "name": opt_config["name"],
        "repeat": repeat,
        "precision": config["precision"],
        "attention_backend": config["attention_backend"],
        "tf32_enabled": bool(config["tf32"]),
        "fused_adamw_requested": bool(config["fused_adamw"]),
        "compile_requested": bool(config["torch_compile"]),
        "batch_size": int(config["batch_size"]),
        "seq_len": int(config["block_size"]),
        "warmup_steps": warmup_steps,
        "benchmark_steps": benchmark_steps,
        "device": str(device),
        "device_name": get_device_name(device),
    }

    if opt_config.get("skip", False):
        return {**common, "status": "skipped", "error": opt_config.get("skip_reason", "Skipped.")}

    precision_available, precision_error = precision_is_available(config["precision"], device)
    if not precision_available:
        return {**common, "status": "skipped", "error": precision_error}

    if config["fused_adamw"] and not fused_supported:
        return {**common, "status": "skipped", "error": fused_error}

    try:
        set_seed(int(config["seed"]) + repeat)
        configure_tf32(bool(config["tf32"]), device)
        raw_model = build_model(config, device)
        optimizer = build_optimizer(raw_model, config)
        model, compile_enabled = maybe_compile_model(raw_model, config)
        scaler = make_grad_scaler(config["precision"], device)
        grad_accum_steps = int(config.get("gradient_accumulation_steps", 1))

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
                    xb, yb = dataset.get_batch("train", int(config["batch_size"]), int(config["block_size"]))
                    with autocast_context(device, config["precision"]):
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

        avg_step_time_s = statistics.mean(timings)
        memory_stats = get_gpu_memory_stats(device)
        tokens_per_step = int(config["batch_size"]) * int(config["block_size"]) * grad_accum_steps
        row = {
            **common,
            "status": "ok",
            "error": "",
            "compile_enabled": compile_enabled,
            "fused_adamw_enabled": bool(config.get("fused_adamw_enabled_resolved", False)),
            "fused_adamw_error": config.get("fused_adamw_error", ""),
            "avg_step_time_ms": avg_step_time_s * 1000.0,
            "tokens_per_sec": tokens_per_step / avg_step_time_s,
            "samples_per_sec": (int(config["batch_size"]) * grad_accum_steps) / avg_step_time_s,
            "max_gpu_memory_allocated_mb": memory_stats["max_gpu_memory_allocated_mb"],
            "max_gpu_memory_reserved_mb": memory_stats["max_gpu_memory_reserved_mb"],
            "parameter_count": count_parameters(raw_model),
        }
        del model, raw_model, optimizer, scaler
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return row
    except Exception as exc:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return {**common, "status": "failed", "error": str(exc)}


def aggregate_rows(plan: List[Dict[str, Any]], raw_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    summaries: List[Dict[str, Any]] = []
    baseline_step_ms = None
    previous_success_step_ms = None

    for opt_config in plan:
        rows = [row for row in raw_rows if row["name"] == opt_config["name"]]
        ok_rows = [row for row in rows if row["status"] == "ok"]
        failed_rows = [row for row in rows if row["status"] == "failed"]
        skipped_rows = [row for row in rows if row["status"] == "skipped"]
        first_row = rows[0]
        if ok_rows and len(ok_rows) == len(rows):
            status = "ok"
            error = ""
        elif failed_rows:
            status = "failed"
            error = failed_rows[0].get("error", "")
        elif skipped_rows:
            status = "skipped"
            error = skipped_rows[0].get("error", "")
        else:
            status = first_row["status"]
            error = first_row.get("error", "")
        summary = {
            "name": opt_config["name"],
            "status": status,
            "error": error,
            "precision": first_row["precision"],
            "attention_backend": first_row["attention_backend"],
            "tf32_enabled": first_row["tf32_enabled"],
            "fused_adamw_requested": first_row["fused_adamw_requested"],
            "compile_requested": first_row["compile_requested"],
            "batch_size": first_row["batch_size"],
            "seq_len": first_row["seq_len"],
            "device_name": first_row["device_name"],
            "repeats": len(rows),
            "successful_repeats": len(ok_rows),
        }

        if ok_rows and len(ok_rows) == len(rows):
            step_times = [row["avg_step_time_ms"] for row in ok_rows]
            tokens = [row["tokens_per_sec"] for row in ok_rows]
            samples = [row["samples_per_sec"] for row in ok_rows]
            allocated = [row["max_gpu_memory_allocated_mb"] for row in ok_rows]
            reserved = [row["max_gpu_memory_reserved_mb"] for row in ok_rows]
            summary.update(
                {
                    "avg_step_time_ms_mean": statistics.mean(step_times),
                    "avg_step_time_ms_std": statistics.stdev(step_times) if len(step_times) > 1 else 0.0,
                    "tokens_per_sec_mean": statistics.mean(tokens),
                    "tokens_per_sec_std": statistics.stdev(tokens) if len(tokens) > 1 else 0.0,
                    "samples_per_sec_mean": statistics.mean(samples),
                    "samples_per_sec_std": statistics.stdev(samples) if len(samples) > 1 else 0.0,
                    "max_gpu_memory_allocated_mb_mean": statistics.mean(allocated),
                    "max_gpu_memory_allocated_mb_std": statistics.stdev(allocated) if len(allocated) > 1 else 0.0,
                    "max_gpu_memory_reserved_mb_mean": statistics.mean(reserved),
                    "max_gpu_memory_reserved_mb_std": statistics.stdev(reserved) if len(reserved) > 1 else 0.0,
                    "parameter_count": ok_rows[0]["parameter_count"],
                    "compile_enabled": ok_rows[0]["compile_enabled"],
                    "fused_adamw_enabled": ok_rows[0]["fused_adamw_enabled"],
                }
            )
            if opt_config["name"] == "baseline_fp32_manual":
                baseline_step_ms = summary["avg_step_time_ms_mean"]
                previous_success_step_ms = summary["avg_step_time_ms_mean"]

            if baseline_step_ms is not None:
                speedup = baseline_step_ms / summary["avg_step_time_ms_mean"]
                summary["speedup_vs_baseline"] = speedup
                summary["percent_change_vs_baseline"] = (speedup - 1.0) * 100.0
            else:
                summary["speedup_vs_baseline"] = None
                summary["percent_change_vs_baseline"] = None

            if previous_success_step_ms is not None:
                previous_speedup = previous_success_step_ms / summary["avg_step_time_ms_mean"]
                summary["percent_change_vs_previous_success"] = (previous_speedup - 1.0) * 100.0
            else:
                summary["percent_change_vs_previous_success"] = None
            previous_success_step_ms = summary["avg_step_time_ms_mean"]
        else:
            summary.update(
                {
                    "avg_step_time_ms_mean": None,
                    "avg_step_time_ms_std": None,
                    "tokens_per_sec_mean": None,
                    "tokens_per_sec_std": None,
                    "samples_per_sec_mean": None,
                    "samples_per_sec_std": None,
                    "max_gpu_memory_allocated_mb_mean": None,
                    "max_gpu_memory_allocated_mb_std": None,
                    "max_gpu_memory_reserved_mb_mean": None,
                    "max_gpu_memory_reserved_mb_std": None,
                    "parameter_count": None,
                    "compile_enabled": False,
                    "fused_adamw_enabled": False,
                    "speedup_vs_baseline": None,
                    "percent_change_vs_baseline": None,
                    "percent_change_vs_previous_success": None,
                }
            )
        summaries.append(summary)

    return summaries


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_plots(summary_rows: List[Dict[str, Any]], results_dir: Path, suffix: str = "") -> None:
    ok_rows = [row for row in summary_rows if row["status"] == "ok"]
    if not ok_rows:
        return

    import matplotlib.pyplot as plt

    labels = [row["name"] for row in ok_rows]
    speedups = [row["speedup_vs_baseline"] for row in ok_rows]
    memory = [row["max_gpu_memory_allocated_mb_mean"] for row in ok_rows]
    width = max(9, min(22, len(ok_rows) * 1.8))

    plt.figure(figsize=(width, 5))
    plt.bar(labels, speedups, color="#2563eb")
    plt.ylabel("Cumulative speedup vs baseline")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(results_dir / f"cumulative_speedup{suffix}.png", dpi=160)
    plt.close()

    plt.figure(figsize=(width, 5))
    plt.bar(labels, memory, color="#16a34a")
    plt.ylabel("Max allocated GPU memory (MB)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(results_dir / f"optimization_memory{suffix}.png", dpi=160)
    plt.close()


def save_result_set(raw_rows: List[Dict[str, Any]], summary_rows: List[Dict[str, Any]], results_dir: Path, name: str) -> None:
    suffix = f"_{name}" if name else ""
    raw_path = results_dir / f"optimization_benchmark{suffix}_raw.json"
    summary_path = results_dir / f"optimization_benchmark{suffix}_summary.json"
    csv_path = results_dir / f"optimization_benchmark{suffix}_summary.csv"

    save_json(raw_rows, raw_path)
    save_json(summary_rows, summary_path)
    write_csv(summary_rows, csv_path)
    make_plots(summary_rows, results_dir, suffix)

    print(f"Saved raw results to {raw_path}")
    print(f"Saved summary results to {summary_path}")
    print(f"Saved summary CSV to {csv_path}")


def main() -> None:
    args = parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1")
    if args.steps < 1:
        raise ValueError("--steps must be >= 1")

    config = resolve_config(load_config(args.config), args)
    device = get_device(config.get("device", "auto"))
    dataset = CharDataset(config["data_dir"], device=device)
    results_dir = Path(config.get("results_dir", "results"))
    results_dir.mkdir(parents=True, exist_ok=True)

    manual_ok, manual_error = check_attention_backend(config, device, "manual")
    sdpa_ok, sdpa_error = check_attention_backend(config, device, "sdpa")
    print(f"Manual attention smoke: {'ok' if manual_ok else 'failed'} {manual_error}")
    print(f"SDPA attention smoke: {'ok' if sdpa_ok else 'failed'} {sdpa_error}")

    fused_supported, fused_error = check_fused_adamw_support(device)
    print(f"Fused AdamW support: {'yes' if fused_supported else 'no'} {fused_error}")

    plan = optimization_plan(device)
    raw_rows: List[Dict[str, Any]] = []
    for opt_config in plan:
        print(f"Benchmarking {opt_config['name']}")
        for repeat in range(args.repeats):
            row = run_training_repeat(
                config,
                dataset,
                opt_config,
                repeat,
                args.warmup_steps,
                args.steps,
                device,
                fused_supported,
                fused_error,
            )
            raw_rows.append(row)
            status = row["status"]
            if status == "ok":
                print(
                    f"  repeat {repeat + 1}/{args.repeats}: "
                    f"{row['tokens_per_sec']:.0f} tok/s, {row['avg_step_time_ms']:.2f} ms"
                )
            else:
                print(f"  repeat {repeat + 1}/{args.repeats}: {status} - {row.get('error', '')}")
                break

    summary_rows = aggregate_rows(plan, raw_rows)

    run_name = str(config.get("run_name") or config.get("model_name") or "run")
    save_result_set(raw_rows, summary_rows, results_dir, run_name)

    # Also keep the original filenames as "latest run" outputs for compatibility.
    save_json(raw_rows, results_dir / "optimization_benchmark_raw.json")
    save_json(summary_rows, results_dir / "optimization_benchmark_summary.json")
    write_csv(summary_rows, results_dir / "optimization_benchmark_summary.csv")
    make_plots(summary_rows, results_dir)


if __name__ == "__main__":
    main()
