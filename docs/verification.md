# Local verification record

Verified on October 9, 2026 (America/Chicago) in the existing Windows repository. No GPU benchmarks, long training, or training-data downloads were performed. The original checkout had no uncommitted changes at task start.

| Check | Result |
| --- | --- |
| CPU suite, `python -m pytest -q` | 14 passed; synthetic CPU inputs and temporary fixtures only |
| Python compilation, `python -m compileall .` | Exit 0 |
| `train.py`, `benchmark_train.py`, `benchmark_optimizations.py`, `profile_train.py`, `evaluate.py`, `experiment.py` help | All exit 0 |
| Locked pnpm installation | Passed with esbuild explicitly allowed in root workspace configuration |
| TypeScript and Vite production build | Passed; app and chart chunks are below the default 500 kB warning threshold |
| Historical import | Five distinct registered runs; subsequent imports add zero |
| Report regeneration | All report files and README generated block are byte-identical on regeneration |
| WSL2 small and Windows tiny optimization summaries | Every saved throughput, samples/s, latency, allocated/reserved memory mean and sample SD verified against raw successful repeats |
| Original artifacts | All 41 original result files preserved byte-for-byte using SHA256 checks |
| README preservation | Git diff shows only the appended marked generated results block |
| Diff whitespace | `git diff --check` passed |

The CPU suite covers model shapes and loss, both causal attention implementations, manual/SDPA forward and gradient parity with dropout disabled, benchmark aggregation and baseline calculations, partial/total failures, incompatible workloads, invalid metric relationships, legacy duplicate and summary-only imports, missing provenance, deterministic reports, unique future benchmark paths using a stubbed measurement function, read-only API routes, safe artifact paths, changed/missing downloads, and cross-run compatibility. The runner orchestration test does not perform training.

The local Python environment was Python 3.10.0 with PyTorch 2.11.0+cu128. Tests ran entirely on CPU. FastAPI 0.143.0 / Starlette 1.7.0 emit one TestClient deprecation warning for httpx; the tests passed. GitHub Actions workflows were added but have not been executed on GitHub in this task.

## Browser checks

The actual built dashboard was served on <http://127.0.0.1:8000> and tested through the Codex in-app browser at a 1440 × 1000 desktop viewport and a 390 × 844 mobile viewport. The viewport override was reset afterward.

- The first loaded run showed the real WSL2 `gpt_small` dataset, eight stages, and the verified 3.02× result.
- All four charts rendered; desktop inspection found 21 nonzero-error-bar stage groups for the three metrics with measured dispersion. Throughput, latency, allocated memory and speedup axes/labels have explicit units.
- Precision filtering reduced the WSL2 run to three BF16 stages. Sorting tokens/s ascending changed the first visible stage as expected.
- Model, environment and status filters produced the expected no-match screen. Reset restored stages.
- Windows tiny compile filtering showed one failed stage, zero charts, zero successful repeats and the exact saved Triton error; resetting filters restored its six successful stages and kept the failed stage visible.
- Attempting Windows/WSL2 comparison produced a compatibility rejection explaining that the environment differed; no combined ranking was shown.
- An isolated server with conspicuously labelled `TEST FIXTURE` records verified compatible comparison: four table rows from two runs, per-row run labels and four charts. A separate empty fixture server verified “No registered experiments.” Those fixtures used temporary directories, were never put under measured `results/`, and both fixture servers were stopped after testing.
- JSON, CSV, PNG and Markdown download controls saved all four artifact types through the browser. Backend tests also verify content-disposition, path restrictions, 404 for missing files and 409 for changed files.
- Mobile controls stacked, tables scrolled within their container and all four chart widths fitted the screen (313 px). Page width remained within the viewport. Mobile precision filtering worked.
- No browser console warnings or errors were reported during the inspected real and fixture sessions.

The screenshot is saved locally at `.verification/dashboard-desktop.jpg`; generated verification evidence and frontend output are ignored. The actual dashboard remains running on port 8000 for review.

## Files changed

- `benchmark_train.py`, `benchmark_optimizations.py`: unique experiment directories and provenance/finalization hooks; focused argument and aggregation handling.
- `experiments/registry.py`, `experiments/reports.py`, `experiments/__init__.py`, `experiment.py`: registry, normalization, validation, compatibility, import and offline report CLI.
- `dashboard/server.py`, `dashboard/__init__.py`: read-only FastAPI API, integrity-checked registered downloads and static frontend hosting.
- `dashboard/frontend/package.json`, `tsconfig.json`, `vite.config.ts`, `index.html`, `src/main.tsx`, `src/style.css`: React/TypeScript/Vite interface, charts, icons, filters, comparison, sortable table, failure inspection and responsive styling.
- `pnpm-workspace.yaml`, `pnpm-lock.yaml`: locked frontend dependencies and explicit esbuild installation permission.
- `tests/`: CPU correctness and artifact/API tests; isolated browser test fixtures. `pytest.ini` selects the test directory.
- `.github/workflows/cpu.yml`: CPU correctness/CLI and frontend build jobs.
- `docs/experiments.md`, `docs/historical_evidence.json`, `docs/performance_case_study.md`, this file: usage, explicit documentary disclosures, verified measurements and validation evidence.
- `README.md`: generated block only. `requirements.txt`, `.gitignore`: backend/test dependencies and ignored generated files.
- Local generated files: five manifests and five reports under `results/runs/`; original measured artifacts were not changed.

## Limits

Historical timestamps, Git state, resolved configs and dataset hashes are unknown. Current historical runs therefore cannot be compared across runs with complete provenance. The prior Windows `gpt_small` artifact and the original WSL2 missing-header failure artifact are absent; the case study identifies their README-only documentation. Speedup uncertainty is not propagated; ratio charts explicitly state that it is unavailable. Real future GPU execution was not exercised in this task, in accordance with the request. GitHub-hosted CI still needs its first run. The local server serves a read-only review dashboard and is not a public deployment.
