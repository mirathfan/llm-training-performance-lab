# Interview notes

## 60-second explanation

I built a small GPT-style PyTorch training performance lab to measure how implementation choices affect throughput, step latency and GPU memory. It uses character-level Tiny Shakespeare data and MLPerf-inspired measurement practices: warmup, synchronized timing, repeated runs and explicit hardware disclosure. It is not an MLPerf-compliant submission or production-scale training system.

The strongest saved result is a WSL2 `gpt_small` experiment on an RTX 3060 Laptop GPU. At batch 16 and sequence length 256, TF32, SDPA, BF16, fused AdamW and `torch.compile` reduced mean step latency from 80.32 to 26.56 milliseconds, a 3.02× speedup. Throughput reached 154,197.54 ± 587.16 tokens per second, and peak allocated memory fell 52.41%. Those values come from three successful repeats per stage, not a new training run or a dashboard demo.

I then added a registry that preserves each run, an offline report command, CPU correctness tests and a read-only dashboard. Historical provenance gaps stay unknown, failed stages remain visible, and incompatible runs are kept separate. The main engineering lesson is that performance claims need both measurements and the context that makes comparisons valid.

## Three-minute technical walkthrough

**0:00–0:35 — Model and workload.** The model is a small decoder-only Transformer: token and learned position embeddings, pre-LayerNorm blocks, causal multi-head attention, an MLP and a tied language-model head. Manual attention and PyTorch SDPA implement the same causal operation. CPU tests check shapes, causality and forward/gradient parity with dropout disabled. The historical small run records 10,770,816 parameters, batch 16 and sequence 256. The current YAML describes six layers, six heads and embedding width 384; I avoid claiming that today's YAML reconstructs the unsaved historical resolved configuration.

**0:35–1:15 — Measurement.** The baseline is FP32 manual attention, TF32 disabled, ordinary AdamW and compilation disabled. Each repeat initializes a fresh model and optimizer with a fixed seed. The harness performs 10 warmup steps and 50 measured optimizer steps, with CUDA synchronization around a wall-clock timer. Sampling and transfer are included. Peak memory counters reset after warmup. Three successful repeats provide arithmetic means and sample standard deviations. Compilation happens during warmup, so the reported result excludes first-compilation cost. The 3.02362× speedup is mean baseline latency divided by mean optimized latency; it is not computed by dividing the separately averaged throughputs.

**1:15–1:55 — Optimization and evidence.** The stages introduce TF32, SDPA, FP16/BF16 autocast, fused AdamW and finally compilation. The FP16 and BF16 branches are alternatives, not a single monotonic precision ladder. The compiled BF16 stack is fastest in this saved WSL2 run: 154,197.54 ± 587.16 tokens/s and 26.56 ± 0.10 ms/step. Allocated peak falls from 1,119.76 ± 1.23 to 532.91 ± 1.23 MiB. This is a measured 586.85 MiB reduction. Improved attention dispatch, tensor-core math and less launch overhead are plausible explanations; these measurements do not isolate their kernel-level contributions. Native Windows tiny compilation failed because a working Triton installation was unavailable. WSL2 raw rows explicitly record successful compilation. The development-header remediation is documented in the earlier README, while the separate original missing-header failure artifact is absent.

**1:55–2:35 — Reproducibility.** New benchmark runs receive unique timestamp/UUID directories containing the existing JSON/CSV/plot formats and a manifest. The manifest records resolved config and hash, dataset hashes, UTC time, Git revision/dirty state, software/hardware and stage settings/outcomes. Historical import uses content fingerprints to deduplicate aliases without changing source files. It does not invent timestamps, Git state or configs. Offline reports exclude failed/invalid stages from rankings and use a compatible baseline per workload. Windows and WSL2 are separate runs, and missing provenance prevents defensible cross-run comparisons.

**2:35–3:00 — Dashboard and limitations.** FastAPI serves a read-only API and a React/TypeScript/Vite interface with run selection, filters, sortable measurements, uncertainty charts, exact failures and registered artifact downloads. Path containment and SHA256 checks restrict downloads to saved results. It never launches training. CPU tests and the frontend build have local and GitHub verification records in `docs/verification.md`; I check that record before claiming hosted CI passed. Remaining limits include modest repeat counts, a single laptop GPU, no convergence or distributed-training study, and unknown historical power/thermal state.

## Five likely questions

1. **What exactly does 3.02× compare?** The FP32/manual/no-TF32/no-compile baseline and the TF32 + SDPA + BF16 + fused AdamW + compile stage within the same saved WSL2 `gpt_small` run. Both use batch 16, sequence 256, 10 warmup steps, 50 measured steps and three successful repeats. The ratio is 80.31826 / 26.56359 = 3.02362. It is not a Windows-versus-WSL2 ratio.

2. **How do you know it is not one lucky measurement?** I recomputed every saved stage mean and sample SD from its raw repeats. The compiled stage's throughput SD is 587.16 tokens/s across three repeats. Three repeats are still a modest sample, and I do not call SD a confidence interval or claim universal significance.

3. **Why does lower precision not always win?** Observed: Windows `gpt_tiny` was fastest with TF32 + SDPA, while lower precision reduced memory. WSL2 `gpt_small` benefited from mixed precision and compilation. Hypothesis: model size changes the balance between arithmetic throughput, memory traffic and framework/kernel-launch overhead. These runs do not prove a single cause or generalize to larger models.

4. **How did you validate `torch.compile`?** The final WSL2 raw result set has three successful compiled-stage repeats with `compile_enabled=true` and `fused_adamw_enabled=true`. The native Windows tiny result records its Triton failure and no compiled metrics. Earlier notes describe supplying Python development headers in WSL2. First-compilation time is excluded by warmup, and this task did not rerun compilation benchmarks.

5. **What would you improve next?** Capture power/temperature and driver details, increase repeat counts and measurement duration, separately measure compilation cost, and profile kernel behavior before attributing causes. Then test larger models and distributed workloads. The registry, dataset hashes, immutable run directories and CPU correctness tests establish the foundation, while the dashboard keeps evidence and failure states inspectable.

## Two evidence-supported resume bullets

- Built a PyTorch GPT-style training performance lab on an RTX 3060 Laptop GPU; verified a 3.02× steady-state WSL2 speedup with TF32, SDPA, BF16, fused AdamW and `torch.compile`, reaching 154,197.54 ± 587.16 tokens/s across three repeats and reducing peak allocated GPU memory by 52.41%.
- Developed an experiment registry with configuration/dataset hashes and Git/environment provenance, deterministic offline reports, CPU correctness tests and a read-only React/FastAPI dashboard with compatibility checks, failure inspection, uncertainty charts and integrity-checked artifact downloads.

Numerical evidence: [performance case study](performance_case_study.md), [saved raw repeats](../results/optimization_benchmark_gpt_small_raw.json), [saved summary](../results/optimization_benchmark_gpt_small_summary.json). Scope and executed checks: [verification record](verification.md).
