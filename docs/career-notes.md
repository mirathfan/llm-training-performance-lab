# Career notes

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
- Profiled and benchmarked FP32, TF32, SDPA, FP16, BF16, fused AdamW, and `torch.compile` training configurations, reaching a measured 3.02x `gpt_small` speedup under WSL2 with BF16 + fused AdamW + `torch.compile` while documenting memory and throughput tradeoffs.
- Created MLPerf-inspired benchmark reports with hardware disclosure, JSON/CSV outputs, plots, profiler traces, and reproducible configs for GPU training analysis.

### General AI Engineer Version

- Built a reproducible GPT-style Transformer training lab in PyTorch with configurable model size, sequence length, batch size, AMP, checkpointing, profiling, and evaluation.
- Developed end-to-end training, evaluation, and benchmarking scripts with Tiny Shakespeare data preparation, validation perplexity tracking, GPU memory reporting, and TensorBoard profiler traces.
- Analyzed training performance tradeoffs across precision modes, batch sizes, and sequence lengths to connect LLM model design with practical deployment constraints.
