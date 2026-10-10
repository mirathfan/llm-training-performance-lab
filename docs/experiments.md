# Reproducible experiments and dashboard

Run all commands from the repository root. Python 3.10+ is required (CI uses 3.11). The dashboard uses FastAPI and a React/TypeScript/Vite frontend with Recharts and Lucide icons. Node 24 and pnpm 11.25.0 were used for the verified build; the root `pnpm-lock.yaml` pins the resolved frontend dependencies.

```powershell
python -m pip install -r requirements.txt
pnpm install --frozen-lockfile
pnpm --dir dashboard/frontend build
python experiment.py import --evidence docs/historical_evidence.json
python experiment.py report --readme README.md
python -m uvicorn dashboard.server:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000>. The first screen selects the saved WSL2 `gpt_small` result when present, otherwise the first registered run. Nothing in the dashboard starts training, downloads data, edits artifacts, or runs a benchmark. The API has GET routes only. Run selector and filters determine displayed stages; failed/skipped stages stay visible until filtered out. Select a stage to inspect its settings, repeat counts, and exact saved errors. Sort by clicking a table heading. Charts show successful, validated measurements with ± one sample standard deviation where repeat dispersion is available. Speedup is a latency ratio; its uncertainty is unavailable and is labelled accordingly. Tables provide the numeric alternative to charts for accessibility.

The comparison selector checks the saved workload, full software environment, dataset content hashes, model/optimizer configuration, and measurement protocol. Incompatible or incomplete provenance produces an explanation and keeps runs separate. The current legacy records lack enough metadata for a defensible comparison across runs; Windows and WSL2 never share a baseline. Individual sweep baselines must match batch and sequence sizes. Precision and implementation optimizations may differ within a valid comparison.

## Registry and artifact locations

`results/runs/<run-id>/manifest.json` is the registry. It contains schema version, unique ID, UTC timestamp, resolved configuration and SHA256, Git revision/dirty state, model dimensions, batch/sequence sizes, protocol, software/hardware disclosure, dataset file hashes, stage settings, repeat success counts, exact errors, throughput/latency/allocated and reserved memory, validation findings, and a file allowlist with SHA256 hashes. `null` means unknown. Memory fields with a legacy `_mb` suffix are **MiB** (bytes divided by 1024²), not decimal MB.

Future invocations of `benchmark_train.py` and `benchmark_optimizations.py` reserve a new timestamp/UUID directory. They save their original JSON/CSV/plot formats inside it, then finalize its manifest. The scripts no longer overwrite the top-level historical artifacts. Original top-level file paths remain readable, and the optimization runner keeps both model-specific and generic filenames inside each new directory. There is no mutable top-level "latest" alias. An interrupted run may retain a manifest with no stages; inspect the benchmark process/log before treating it as complete.

The registry itself uses only the Python standard library. Torch is imported only when a future benchmark captures its environment. CPU tests and offline artifact operations never need CUDA.

```powershell
python experiment.py list
python experiment.py import
python experiment.py import --evidence docs/historical_evidence.json
python experiment.py report
python experiment.py report --readme README.md
python experiment.py --results-dir path/to/other/results import
python experiment.py --results-dir path/to/other/results report
```

Historical import keeps original files in place. It fingerprints parsed JSON content to prevent duplicate alias imports and associates matching saved CSV, summary JSON, and PNG files. Raw repeat sets take precedence; a summary-only optimization result can also be imported if its raw counterpart is missing. It does not infer a timestamp from file modification times, a Git revision from the current checkout, or a historical resolved configuration from today's YAML. The optional evidence file explicitly binds documented model/environment labels to content fingerprints. Its disclosures cite the pre-existing README and leave unrecorded details unknown. Import can enrich an already registered record using this evidence without creating another run.

Reports are generated at `results/runs/<run-id>/report.md`, include settings, protocol, hardware/software, failures, fastest/lowest-memory successful stages per workload, baseline speedup/memory reduction, and validation issues. Single-observation standard deviations are unknown. Partially failed stages preserve successful measurements and error details but are excluded from rankings. Offline reports use a deterministic sort order and contain no generation-time timestamp. The README updater changes only `<!-- BEGIN GENERATED EXPERIMENT RESULTS -->` through its corresponding END marker, appending that marked block if absent. All existing prose, historical tables, and limitations remain intact.

Downloads resolve paths inside the results root, use registered file indices rather than user-supplied filesystem paths, restrict extensions to JSON/CSV/PNG/Markdown, and verify content hashes. Missing files produce 404; changed files produce 409 and need re-import; unsafe paths produce 403. WSL2 artifact contents use repository-relative paths and work from the existing Windows checkout.

For another root, set `LAB_RESULTS_DIR` before starting the server:

```powershell
$env:LAB_RESULTS_DIR = 'C:\path\to\saved-results'
python -m uvicorn dashboard.server:app --host 127.0.0.1 --port 8001
```

Under WSL2, use `export LAB_RESULTS_DIR=/mnt/c/path/to/saved-results`. For frontend development, run the backend on port 8000, then `pnpm --dir dashboard/frontend dev`. Vite proxies `/api` to that backend. To serve a production build, build **before** starting Uvicorn. Changing the listening port is supported; the built frontend uses same-origin API URLs.

## Verification

```powershell
python -m pytest -q
python -m compileall .
python train.py --help
python benchmark_train.py --help
python benchmark_optimizations.py --help
python profile_train.py --help
python evaluate.py --help
python experiment.py --help
pnpm --dir dashboard/frontend build
```

Tests use small synthetic CPU inputs and temporary artifact fixtures, clearly separate from measured `results/`. GitHub Actions runs CPU tests, compile checks, CLI help, and a locked frontend build. It installs the CPU Torch wheel and never downloads a dataset or runs a GPU benchmark. The frontend dependency tree and build output are ignored. Original benchmark JSON/CSV/PNG files retain the repository's existing ignore policy; transfer them separately when moving the project to another machine, then import them.

See [performance_case_study.md](performance_case_study.md) for the verified historical measurements and their limits. Local verification details are recorded in [verification.md](verification.md).
