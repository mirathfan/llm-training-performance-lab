# LLM Training Performance Lab

## Overview

LLM Training Performance Lab is a reproducible PyTorch benchmark project for GPT-style Transformer training performance. It focuses on small decoder-only language models that can be trained, benchmarked, and profiled on local NVIDIA GPUs while still running basic commands on CPU-only machines.

The project is intentionally performance-oriented: it tracks throughput, step latency, validation loss, perplexity, GPU memory, and profiler traces across training configurations.

## Why This Project Matters

High-performance AI training work is about more than model quality. Engineers need to understand GPU utilization, memory pressure, mixed precision, profiler output, and the tradeoffs between speed and memory. This repository gives a compact, resume-ready lab for practicing those skills without requiring a large cluster or proprietary dataset.

## Architecture Summary

The model is a small decoder-only GPT-style Transformer implemented in PyTorch. It includes token embeddings, learned positional embeddings, stacked pre-LayerNorm Transformer blocks, multi-head causal self-attention, an MLP feed-forward block, residual connections, dropout, and a language modeling head. Model size is configurable through YAML files.

## Dataset

The default dataset is Tiny Shakespeare for character-level language modeling. The preparation script downloads the raw text, builds a character vocabulary, encodes text into token IDs, creates train/validation splits, and saves metadata under `data/processed/`.

## Benchmark Methodology

Benchmarks measure training performance rather than final model quality. The benchmark runner can compare:

- FP32 baseline training
- AMP mixed precision on CUDA
- `torch.compile` when requested and available
- Activation checkpointing when requested
- Batch size scaling
- Sequence length scaling

Each run records average step time, tokens/sec, samples/sec, GPU memory, parameter count, model configuration, device name, warmup steps, and benchmark steps. If CUDA is unavailable, benchmarks run on CPU and record that device clearly.

## How To Run

```bash
pip install -r requirements.txt
python data/prepare_tinyshakespeare.py
python train.py --config configs/gpt_tiny.yaml
python train.py --config configs/gpt_tiny.yaml --amp
python benchmark_train.py --config configs/gpt_tiny.yaml
python profile_train.py --config configs/gpt_tiny.yaml --amp
python evaluate.py --checkpoint checkpoints/best_gpt_tiny.pt
```

Useful variants:

```bash
python train.py --config configs/gpt_tiny.yaml --compile
python train.py --config configs/gpt_tiny.yaml --activation-checkpointing
python train.py --config configs/gpt_tiny.yaml --max-iters 1000
python train.py --config configs/gpt_tiny.yaml --gradient-accumulation-steps 2
python benchmark_train.py --config configs/gpt_tiny.yaml --modes fp32 amp compile
python benchmark_train.py --config configs/gpt_tiny.yaml --batch-sizes 8 16 32 64
python benchmark_train.py --config configs/gpt_tiny.yaml --seq-lens 64 128 256
```

## Results

Smoke training was run for 20 iterations with `configs/gpt_tiny.yaml` and AMP on CUDA. These are smoke-test metrics, not final model-quality claims.

| run | config | validation loss | validation perplexity | checkpoint |
| --- | --- | ---: | ---: | --- |
| gpt_tiny | configs/gpt_tiny.yaml | 3.8066 | 45.00 | checkpoints/last_gpt_tiny.pt |

## Key Findings

- CUDA was successfully enabled on an NVIDIA GeForce RTX 3060 Laptop GPU.
- FP32 at batch 32 / seq 128 reached 187,882.71 tokens/sec with 21.80 ms/step and 249.40 MB max GPU memory.
- AMP at batch 32 / seq 128 reached 179,190.97 tokens/sec with 22.86 ms/step and 181.41 MB max GPU memory.
- AMP reduced memory usage in this setting, but did not outperform FP32 at the small benchmark size.
- Best batch-scaling result was AMP batch 64 / seq 128 at 375,771.54 tokens/sec.
- Best sequence-scaling result was AMP batch 32 / seq 256 at 354,869.84 tokens/sec.
- BF16 + fused AdamW was tested after the initial optimization benchmark to complete the `gpt_small` optimization matrix, producing the new best `gpt_small` result at 135,606.99 +/- 1,563.28 tokens/sec and 2.67x speedup over baseline.
- `torch.compile` failed on this Windows setup because Triton was missing, so compiled results are recorded as unavailable instead of fabricated.

## Benchmark Results

All benchmark runs used `gpt_tiny`, 5 warmup steps, 20 benchmark steps, and gradient accumulation of 1 unless noted otherwise.

### Mode Comparison

| mode | batch size | seq len | tokens/sec | step time ms | max memory MB | device |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| fp32 | 32 | 128 | 187882.71 | 21.80 | 249.40 | NVIDIA GeForce RTX 3060 Laptop GPU |
| amp | 32 | 128 | 179190.97 | 22.86 | 181.41 | NVIDIA GeForce RTX 3060 Laptop GPU |
| compile | 32 | 128 | failed | failed | failed | NVIDIA GeForce RTX 3060 Laptop GPU |

`torch.compile` failed because a working Triton installation was not available.

### Batch Size Scaling

| mode | batch size | seq len | tokens/sec | step time ms | max memory MB | device |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| fp32 | 8 | 128 | 53693.27 | 19.07 | 82.92 | NVIDIA GeForce RTX 3060 Laptop GPU |
| fp32 | 16 | 128 | 92146.24 | 22.23 | 137.87 | NVIDIA GeForce RTX 3060 Laptop GPU |
| fp32 | 32 | 128 | 229507.65 | 17.85 | 249.40 | NVIDIA GeForce RTX 3060 Laptop GPU |
| fp32 | 64 | 128 | 320613.11 | 25.55 | 470.21 | NVIDIA GeForce RTX 3060 Laptop GPU |
| amp | 8 | 128 | 49799.57 | 20.56 | 65.67 | NVIDIA GeForce RTX 3060 Laptop GPU |
| amp | 16 | 128 | 91734.49 | 22.33 | 103.00 | NVIDIA GeForce RTX 3060 Laptop GPU |
| amp | 32 | 128 | 184304.26 | 22.22 | 176.91 | NVIDIA GeForce RTX 3060 Laptop GPU |
| amp | 64 | 128 | 375771.54 | 21.80 | 317.09 | NVIDIA GeForce RTX 3060 Laptop GPU |

### Sequence Length Scaling

| mode | batch size | seq len | tokens/sec | step time ms | max memory MB | device |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| fp32 | 32 | 64 | 99027.14 | 20.68 | 120.42 | NVIDIA GeForce RTX 3060 Laptop GPU |
| fp32 | 32 | 128 | 212341.58 | 19.29 | 251.27 | NVIDIA GeForce RTX 3060 Laptop GPU |
| fp32 | 32 | 256 | 249888.58 | 32.78 | 616.84 | NVIDIA GeForce RTX 3060 Laptop GPU |
| amp | 32 | 64 | 94506.27 | 21.67 | 90.54 | NVIDIA GeForce RTX 3060 Laptop GPU |
| amp | 32 | 128 | 176682.26 | 23.18 | 178.41 | NVIDIA GeForce RTX 3060 Laptop GPU |
| amp | 32 | 256 | 354869.84 | 23.08 | 447.75 | NVIDIA GeForce RTX 3060 Laptop GPU |

## Performance Analysis

AMP reduced max GPU memory in the batch 32 / sequence length 128 comparison because mixed precision stores and computes many tensors in lower precision. For this small GPT benchmark, AMP did not improve throughput over FP32 at the same shape. Small models can be dominated by framework overhead, kernel launch overhead, memory movement, and limited tensor core saturation, so lower precision does not automatically translate into higher tokens/sec.

Larger batch sizes improved tokens/sec in the batch-scaling sweep because they increased the amount of work available per optimizer step. That can improve GPU utilization by giving the device more parallel work, although memory usage also rises as activations, gradients, and optimizer-related tensors scale with batch size.

Sequence length changes both compute and memory behavior. Longer sequences increase the number of tokens processed per step, but causal self-attention also grows with sequence length, so GPU memory and step latency can rise quickly. In this run, AMP at batch 32 / sequence length 256 produced the best sequence-scaling throughput while still using less memory than FP32 at the same shape.

`torch.compile` was requested, but this Windows environment did not have a working Triton installation. Compiled results are therefore treated as unavailable for this run. A future WSL2 or Linux benchmark may be a better environment for evaluating `torch.compile` speedups honestly.

## Optimization Roadmap

The `benchmark_optimizations.py` harness stacks optimizations cumulatively so each row can be compared against the same baseline. The full runs below used 10 warmup steps, 50 measured steps, and 3 repeats per stage.

SDPA uses PyTorch's `scaled_dot_product_attention` dispatch instead of the manual attention implementation. Depending on the GPU, tensor shapes, PyTorch build, and sequence length, SDPA may use optimized kernels that reduce memory use and latency, especially as sequence length grows.

BF16 is tested only when `torch.cuda.is_bf16_supported()` returns true. BF16 performance and memory behavior should be reported empirically from generated benchmark files, not assumed.

### gpt_tiny Optimization Results

These `gpt_tiny` optimization results were generated before the BF16 + fused AdamW stage was added.

| optimization stage | status | precision | attention | tokens/sec mean +/- std | step ms mean +/- std | max memory MB mean +/- std | speedup |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| baseline_fp32_manual | ok | fp32 | manual | 231516.45 +/- 5646.74 | 17.70 +/- 0.43 | 249.27 +/- 0.00 | 1.00x |
| tf32_manual | ok | fp32 | manual | 242115.49 +/- 18786.29 | 16.99 +/- 1.38 | 249.27 +/- 0.00 | 1.04x |
| tf32_sdpa | ok | fp32 | sdpa | 299150.60 +/- 12641.55 | 13.71 +/- 0.57 | 177.46 +/- 0.00 | 1.29x |
| tf32_sdpa_fp16 | ok | fp16 | sdpa | 236668.60 +/- 20173.88 | 17.40 +/- 1.56 | 121.72 +/- 0.00 | 1.02x |
| tf32_sdpa_bf16 | ok | bf16 | sdpa | 261788.10 +/- 2830.20 | 15.65 +/- 0.17 | 121.72 +/- 0.00 | 1.13x |
| tf32_sdpa_fp16_fused_adamw | ok | fp16 | sdpa | 256684.82 +/- 10029.47 | 15.97 +/- 0.63 | 121.75 +/- 0.00 | 1.11x |
| tf32_sdpa_fp16_fused_adamw_compile | failed: Triton missing | fp16 | sdpa | unavailable | unavailable | unavailable | unavailable |

### gpt_small Optimization Results

| optimization stage | status | precision | attention | tokens/sec mean +/- std | step ms mean +/- std | max memory MB mean +/- std | speedup |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| baseline_fp32_manual | ok | fp32 | manual | 50769.52 +/- 615.14 | 80.69 +/- 0.98 | 1120.49 +/- 1.63 | 1.00x |
| tf32_manual | ok | fp32 | manual | 67324.47 +/- 298.66 | 60.84 +/- 0.27 | 1120.07 +/- 0.25 | 1.33x |
| tf32_sdpa | ok | fp32 | sdpa | 78726.20 +/- 266.05 | 52.03 +/- 0.18 | 795.95 +/- 2.81 | 1.55x |
| tf32_sdpa_fp16 | ok | fp16 | sdpa | 126008.84 +/- 444.65 | 32.51 +/- 0.11 | 553.17 +/- 1.92 | 2.48x |
| tf32_sdpa_bf16 | ok | bf16 | sdpa | 125467.02 +/- 2849.93 | 32.66 +/- 0.74 | 551.66 +/- 2.96 | 2.47x |
| tf32_sdpa_fp16_fused_adamw | ok | fp16 | sdpa | 133410.04 +/- 2083.13 | 30.71 +/- 0.48 | 551.62 +/- 0.36 | 2.63x |
| tf32_sdpa_bf16_fused_adamw | ok | bf16 | sdpa | 135606.99 +/- 1563.28 | 30.21 +/- 0.35 | 551.83 +/- 1.79 | 2.67x |
| tf32_sdpa_bf16_fused_adamw_compile | failed: Triton missing | bf16 | sdpa | unavailable | unavailable | unavailable | unavailable |

## Optimization Findings

- Fastest `gpt_tiny` stage: `tf32_sdpa` at 299150.60 +/- 12641.55 tokens/sec, a 1.29x speedup over baseline.
- Fastest `gpt_small` stage: `tf32_sdpa_bf16_fused_adamw` at 135606.99 +/- 1563.28 tokens/sec, a 2.67x speedup over baseline.
- Lowest `gpt_tiny` memory: `tf32_sdpa_bf16` at 121.72 +/- 0.00 MB, a 127.55 MB reduction from the 249.27 MB baseline.
- Lowest `gpt_small` memory: `tf32_sdpa_fp16_fused_adamw` at 551.62 +/- 0.36 MB, a 568.87 MB reduction from the 1120.49 MB baseline.
- SDPA improved both throughput and memory versus the corresponding TF32 manual-attention stage in both configs.
- FP16 and BF16 reduced memory in both configs. For `gpt_tiny`, they did not beat the TF32+SDPA throughput result. For the refreshed `gpt_small` run, FP16 and BF16 both improved throughput over TF32+SDPA, with FP16 slightly ahead of plain BF16.
- Fused AdamW improved throughput versus the matching non-fused precision stages in the refreshed `gpt_small` run. BF16 + fused AdamW beat both plain BF16 and FP16 + fused AdamW in this run.
- `torch.compile` remained unavailable on this Windows setup because Triton was missing, so compiled results are recorded as failed rather than treated as a speedup.

## Optional WSL2 torch.compile validation

`torch.compile` failed on Windows because Triton was unavailable. WSL2/Linux is the intended environment for validating `torch.compile` on this project. Add results to this README only after the WSL2 benchmark actually runs successfully.

```bash
cd ~/llm-training-performance-lab
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA')"
python benchmark_optimizations.py --config configs/gpt_small.yaml --steps 50 --warmup-steps 10 --repeats 3
```

## Profiling

Profiler traces are written to `results/profiler/<run_name>/`. Open them with:

```bash
tensorboard --logdir results/profiler
```

The profiler summary table is also saved to `results/<run_name>/profiler_summary.txt` for quick operator-level inspection.

## Hardware Disclosure

| component | value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 3060 Laptop GPU, 6144 MiB |
| CPU | AMD Ryzen 7 5800HS with Radeon Graphics, 8 cores / 16 logical processors |
| RAM | 16,542,683,136 bytes |
| CUDA availability | true |
| CUDA version reported by PyTorch | 12.8 |
| PyTorch version | 2.11.0+cu128 |
| Driver version | 610.47 |
| OS | Microsoft Windows 11 Home Single Language, version 10.0.26200, build 26200 |

## NVIDIA Role Alignment

This project maps directly to performance-focused AI engineering work:

- Profiling AI training workloads with PyTorch Profiler
- Measuring and improving training throughput
- Using PyTorch training loops, optimizers, AMP, and `torch.compile`
- Understanding CUDA-aware execution and CPU fallback behavior
- Analyzing GPU memory allocation and reservation
- Producing MLPerf-inspired benchmark artifacts with reproducible configs, logs, plots, and hardware disclosure

## Resume Version

### NVIDIA / ML Systems Version

- Built a PyTorch LLM training performance benchmark for GPT-style Transformer workloads on an NVIDIA RTX 3060 Laptop GPU, measuring tokens/sec, step latency, GPU memory, validation loss, and perplexity.
- Profiled and benchmarked FP32, TF32, SDPA, FP16, BF16, and fused AdamW training configurations, reaching a measured 2.67x `gpt_small` speedup with BF16 + fused AdamW while documenting memory and throughput tradeoffs.
- Created MLPerf-inspired benchmark reports with hardware disclosure, JSON/CSV outputs, plots, profiler traces, and reproducible configs for GPU training analysis.

### General AI Engineer Version

- Built a reproducible GPT-style Transformer training lab in PyTorch with configurable model size, sequence length, batch size, AMP, checkpointing, profiling, and evaluation.
- Developed end-to-end training, evaluation, and benchmarking scripts with Tiny Shakespeare data preparation, validation perplexity tracking, GPU memory reporting, and TensorBoard profiler traces.
- Analyzed training performance tradeoffs across precision modes, batch sizes, and sequence lengths to connect LLM model design with practical deployment constraints.

## Limitations and Next Steps

- Current benchmark uses a small character-level GPT model, not a production LLM.
- Single-GPU benchmark only; no distributed training yet.
- `torch.compile` unavailable due to Triton setup on Windows.
- Future improvements: WSL2/Linux benchmarking, distributed data parallel, gradient accumulation experiments, activation checkpointing comparison, custom CUDA kernel, Nsight Systems profiling.
