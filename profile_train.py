import argparse
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, schedule, tensorboard_trace_handler

from data.dataset import CharDataset
from train import (
    autocast_context,
    build_model,
    build_optimizer,
    configure_tf32,
    load_config,
    make_grad_scaler,
    maybe_compile_model,
    precision_is_available,
    resolve_config,
)
from utils.device import get_device, get_device_name
from utils.logging_utils import create_run_dir
from utils.profiler import synchronize_device
from utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile a short GPT training run with PyTorch Profiler.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    parser.add_argument("--amp", action="store_true", help="Enable CUDA AMP mixed precision.")
    parser.add_argument("--compile", action="store_true", dest="torch_compile", help="Enable torch.compile.")
    parser.add_argument("--attention-backend", choices=["manual", "sdpa"], help="Attention backend to use.")
    parser.add_argument("--tf32", action="store_true", help="Enable TF32 matmul/convolution on CUDA.")
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], help="Training precision/autocast mode.")
    parser.add_argument("--fused-adamw", action="store_true", help="Use fused AdamW when supported.")
    parser.add_argument(
        "--activation-checkpointing",
        action="store_true",
        help="Enable activation checkpointing in Transformer blocks.",
    )
    parser.add_argument("--steps", type=int, default=6, help="Number of active profiler steps.")
    parser.add_argument("--warmup-steps", type=int, default=2, help="Profiler warmup steps.")
    parser.add_argument("--wait-steps", type=int, default=1, help="Profiler wait steps.")
    parser.add_argument("--device", help="Override device: auto, cpu, cuda, cuda:0, etc.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = resolve_config(load_config(args.config), args)
    set_seed(config["seed"])

    device = get_device(config.get("device", "auto"))
    configure_tf32(bool(config.get("tf32", False)), device)
    precision_requested = config.get("precision", "fp32")
    precision_available, precision_error = precision_is_available(precision_requested, device)
    precision = precision_requested if precision_available else "fp32"
    if not precision_available:
        print(f"{precision_requested} requested but unavailable: {precision_error} Profiling will use FP32.")

    dataset = CharDataset(config["data_dir"], device=device)
    raw_model = build_model(config, device)
    optimizer = build_optimizer(raw_model, config)
    model, _ = maybe_compile_model(raw_model, config)
    scaler = make_grad_scaler(precision, device)

    run_name = config["run_name"]
    run_dir = create_run_dir(config["results_dir"], run_name)
    trace_dir = Path(config["results_dir"]) / "profiler" / run_name
    trace_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "profiler_summary.txt"

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    total_steps = args.wait_steps + args.warmup_steps + args.steps
    grad_accum_steps = int(config.get("gradient_accumulation_steps", 1))
    print(f"Profiling {total_steps} steps on {get_device_name(device)}")

    with profile(
        activities=activities,
        schedule=schedule(wait=args.wait_steps, warmup=args.warmup_steps, active=args.steps, repeat=1),
        on_trace_ready=tensorboard_trace_handler(str(trace_dir)),
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        for _ in range(total_steps):
            optimizer.zero_grad(set_to_none=True)
            for _ in range(grad_accum_steps):
                xb, yb = dataset.get_batch("train", config["batch_size"], config["block_size"])
                with autocast_context(device, precision):
                    _, loss = model(xb, yb)
                    loss_for_backward = loss / grad_accum_steps
                scaler.scale(loss_for_backward).backward()
            if float(config["grad_clip"]) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(raw_model.parameters(), float(config["grad_clip"]))
            scaler.step(optimizer)
            scaler.update()
            synchronize_device(device)
            prof.step()

    sort_by = "cuda_time_total" if device.type == "cuda" else "cpu_time_total"
    table = prof.key_averages().table(sort_by=sort_by, row_limit=30)
    summary_path.write_text(table, encoding="utf-8")

    print(table)
    print(f"Saved profiler traces to {trace_dir}")
    print(f"Saved profiler summary to {summary_path}")
    print("Open traces with:")
    print("tensorboard --logdir results/profiler")


if __name__ == "__main__":
    main()
