# Performance case study: small GPT training on a laptop GPU

This is a small character-level GPT-style workload with MLPerf-inspired benchmarking. It is not an MLPerf submission, a production-scale training result, or evidence of a general speedup across model sizes and hardware. No benchmark was rerun for this case study.

## Evidence and workload

The numerical source is the saved `results/optimization_benchmark_gpt_small_raw.json` and its summary JSON/CSV. All eight summary means and sample standard deviations for throughput, latency, allocated memory and reserved memory were checked against the three successful raw repeats per stage. The native Windows evidence is `results/optimization_benchmark_gpt_tiny_raw.json` and its summary. Content fingerprints and documentary environment disclosures are in `docs/historical_evidence.json`.

The WSL2 workload records batch 16, sequence length 256 (4,096 tokens per optimizer step), and 10,770,816 parameters. The current `configs/gpt_small.yaml` specifies six layers, six heads, embedding width 384, dropout 0.1 and gradient accumulation 1. Those dimensions match the intended workload, but the legacy optimization runner did not save a resolved config or dataset hash; the current YAML is context, not proof of the exact historical configuration. The native Windows tiny workload records batch 32, sequence length 128 and 818,048 parameters.

Environment disclosure in the pre-existing README identifies an NVIDIA GeForce RTX 3060 Laptop GPU. The WSL2 run used Ubuntu, Python 3.12.3, PyTorch 2.12.1+cu130, and CUDA 13.0 as reported by PyTorch. Native Windows disclosure lists Windows 11, PyTorch 2.11.0+cu128, CUDA 12.8, an AMD Ryzen 7 5800HS and a 6,144 MiB GPU. These are documentary disclosures, not fields captured in the raw rows. Historical timestamps, Git revision/dirty state, WSL2 OS build and exact driver configuration remain unknown.

## Methodology

The harness uses random character batches from the saved Tiny Shakespeare split, loss/backward, gradient clipping and an AdamW optimizer update. Each repeat builds a fresh model and optimizer, resets the seed, warms up for 10 steps and measures 50 steps. CUDA synchronization brackets the wall-clock step timer. Data sampling and transfer are inside that timer. Peak memory counters reset after warmup. Compilation occurs during warmup, so the result describes steady-state throughput and excludes first-compilation cost.

The tables give the arithmetic mean ± sample standard deviation across repeat-level measurements, not confidence intervals. Throughput is computed for each repeat from its mean step time, then averaged; therefore mean throughput need not equal tokens per step divided by mean latency. Speedup is mean baseline step latency / mean stage latency. Memory reduction is baseline allocated peak mean minus stage allocated peak mean. Legacy fields use an `mb` suffix but the implementation divides bytes by 1024²: the correct unit here is MiB.

## Observed WSL2 gpt_small results

| Stage | Precision / attention | Tokens/s mean ± SD | ms/step mean ± SD | Peak allocated MiB mean ± SD | Speedup |
| --- | --- | ---: | ---: | ---: | ---: |
| baseline_fp32_manual | fp32 / manual | 50,997.73 ± 216.45 | 80.32 ± 0.34 | 1,119.76 ± 1.23 | 1.00× |
| tf32_manual | fp32 / manual | 63,681.44 ± 3,003.08 | 64.41 ± 2.96 | 1,119.18 ± 0.00 | 1.25× |
| tf32_sdpa | fp32 / sdpa | 74,763.86 ± 644.83 | 54.79 ± 0.47 | 795.30 ± 2.50 | 1.47× |
| tf32_sdpa_fp16 | fp16 / sdpa | 120,042.78 ± 3,727.50 | 34.14 ± 1.05 | 549.92 ± 1.15 | 2.35× |
| tf32_sdpa_bf16 | bf16 / sdpa | 125,110.92 ± 5,085.65 | 32.77 ± 1.32 | 552.67 ± 1.25 | 2.45× |
| tf32_sdpa_fp16_fused_adamw | fp16 / sdpa | 123,912.96 ± 2,919.22 | 33.07 ± 0.77 | 552.79 ± 0.00 | 2.43× |
| tf32_sdpa_bf16_fused_adamw | bf16 / sdpa | 130,571.59 ± 5,554.74 | 31.41 ± 1.33 | 552.79 ± 0.00 | 2.56× |
| tf32_sdpa_bf16_fused_adamw_compile | bf16 / sdpa | 154,197.54 ± 587.16 | 26.56 ± 0.10 | 532.91 ± 1.23 | 3.02× |

All stages completed three repeats. The compiled BF16 + fused AdamW stage was both fastest and lowest in allocated peak memory: 3.02362× baseline and 586.85 MiB (52.41%) lower allocated peak. Reserved peak memory was 1160.00 ± 8.00 MiB at baseline versus 546.00 ± 6.93 MiB compiled.

TF32 kept FP32 storage and manual attention while enabling TF32 math. SDPA changed attention dispatch; FP16 and BF16 added autocast. The fused stages requested fused AdamW, and their successful raw rows confirm the optimizer was enabled. The final stage requested compilation and its three successful raw rows record `compile_enabled=true`. The FP16 and BF16 branches are alternative precision choices; the listed stages do not form one strictly monotonic precision ladder.

## Native Windows evidence and compile failure

| Stage | Status | Tokens/s mean ± SD | Peak allocated MiB mean ± SD |
| --- | --- | ---: | ---: |
| baseline_fp32_manual | ok (3 repeats) | 231,516.45 ± 5,646.74 | 249.27 ± 0.00 |
| tf32_manual | ok (3 repeats) | 242,115.49 ± 18,786.29 | 249.27 ± 0.00 |
| tf32_sdpa | ok (3 repeats) | 299,150.60 ± 12,641.55 | 177.46 ± 0.00 |
| tf32_sdpa_fp16 | ok (3 repeats) | 236,668.60 ± 20,173.88 | 121.72 ± 0.00 |
| tf32_sdpa_bf16 | ok (3 repeats) | 261,788.10 ± 2,830.20 | 121.72 ± 0.00 |
| tf32_sdpa_fp16_fused_adamw | ok (3 repeats) | 256,684.82 ± 10,029.47 | 121.75 ± 0.00 |
| tf32_sdpa_fp16_fused_adamw_compile | failed (0/1 successful) | unavailable | unavailable |

The saved Windows compile error begins: `Cannot find a working triton installation.` It contains no valid compiled timing or memory result and is excluded from rankings. The tiny run was fastest with TF32 + SDPA; lower precision reduced allocated memory but did not beat that stage. This is a measured example of precision benefits depending on workload size.

The pre-existing README documents an initial WSL2 compiler failure, `fatal error: Python.h: No such file or directory`, followed by development-header setup and a successful compile smoke test. That remediation history is documentary evidence; a separate initial failure artifact is not present. The final saved WSL2 raw measurements directly verify successful compilation.

The README also mentions an earlier native Windows `gpt_small` BF16 + fused AdamW run. A separate raw result set for that historical run is no longer present at the existing artifact paths, so this case study does not reuse its numerical claim. Windows tiny and WSL2 small are different workloads and software environments; no cross-environment performance ratio is calculated.

## Interpretation and remaining limitations

Observed: SDPA, mixed precision and the compiled BF16 stack improved throughput and memory for this WSL2 small-model run. Compilation improved over the noncompiled BF16 fused stage from 130,571.59 ± 5,554.74 to 154,197.54 ± 587.16 tokens/s and reduced allocated peak from 552.79 to 532.91 MiB.

Explanations/hypotheses: optimized attention dispatch may reduce attention memory traffic; tensor-core math may improve arithmetic throughput; fusion and compilation may reduce launch overhead. No kernel-level profiling of these stages establishes how much each mechanism contributed. The benchmark does not isolate every optimization causally, and stage order, laptop thermal/power state, allocator behavior and software changes can affect results.

Three repeats and 50 steps are modest samples. There is no power or temperature telemetry, multi-GPU scaling, end-to-end convergence comparison, compilation amortization analysis, or production-model validation. Tiny Shakespeare is a character-level workload. Data/version and Git provenance were not captured historically. The new registry captures resolved configuration, dataset hashes, software, hardware, UTC time, Git state and individual stage outcomes for future runs; it cannot reconstruct missing past metadata.
