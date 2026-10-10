import re
from pathlib import Path
from .registry import artifact, comparisons, manifests, validate, write_json

START = "<!-- BEGIN GENERATED EXPERIMENT RESULTS -->"
END = "<!-- END GENERATED EXPERIMENT RESULTS -->"


def cell(value):
    return str(value).replace("|", "\\|").replace("\r", "").replace("\n", "<br>")


def number(value):
    return "unknown" if value is None else f"{value:,.2f}"


def measurement(s, key):
    m = s["metrics"][key]
    return f"{number(m['mean'])} ± {number(m['std'])}"


def render(run):
    stages = comparisons(run["stages"])
    errors = validate(stages)
    lines = [f"# Experiment {run['run_id']}", "", f"Model: {run.get('model') or 'unknown'}; environment: {run['environment'].get('label') or 'unknown'}; origin: {run['origin']}.",
             f"UTC timestamp: {run.get('timestamp_utc') or 'unknown'}. Config hash: {run.get('config_hash') or 'unknown'}; Git: {run.get('git_revision') or 'unknown'}; dirty: {run.get('git_dirty') if run.get('git_dirty') is not None else 'unknown'}.", "",
             "Memory fields named MB in legacy JSON are MiB (bytes / 1024²). Standard deviations describe successful repeats; a single observation has unknown repeat dispersion. Speedup uses mean baseline latency / mean stage latency, not a ratio of mean throughput.", "", "## Hardware and software", ""]
    lines += [f"- {key}: {value if value is not None else 'unknown'}" for key, value in sorted(run["environment"].items())]
    lines += ["", str(run.get("provenance_notes") or "Provenance captured at benchmark start."), "", "## Stages", "",
              "| Stage | Status | Successful / attempted | Batch / seq | Warmup / measured | Settings | Tokens/s mean ± std | ms/step mean ± std | Allocated MiB mean ± std | Reserved MiB mean ± std | Speedup | Memory reduction MiB (%) |",
              "| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for s in stages:
        settings = ", ".join(f"{k}={v if v is not None else 'unknown'}" for k, v in s["settings"].items())
        values = [s["name"], s["status"], f"{s['successful_repeats']} / {s['repeats']}", f"{s['batch_size']} / {s['sequence_length']}",
                  f"{s['warmup_steps']} / {s['measurement_steps']}", settings,
                  *[measurement(s, key) for key in ("tokens_per_sec", "avg_step_time_ms", "max_gpu_memory_allocated_mb", "max_gpu_memory_reserved_mb")],
                  number(s["speedup_vs_baseline"]), f"{number(s['memory_reduction_mib'])} ({number(s['memory_reduction_percent'])}%)"]
        lines.append("| " + " | ".join(cell(v) for v in values) + " |")
    ok = [s for s in stages if s["status"] == "ok" and not validate([s])]
    # Sweeps are ranked separately for each workload; no batch/sequence conflation.
    from .registry import workload
    for group in sorted({workload(s) for s in ok}, key=str):
        rows = [s for s in ok if workload(s) == group]
        fastest = max(rows, key=lambda s: s["metrics"]["tokens_per_sec"]["mean"])
        lowest = min(rows, key=lambda s: s["metrics"]["max_gpu_memory_allocated_mb"]["mean"])
        lines += ["", f"Batch {group[0]}, seq {group[1]}: fastest successful stage **{fastest['name']}** ({measurement(fastest, 'tokens_per_sec')} tokens/s, {number(fastest['speedup_vs_baseline'])}×). Lowest allocated memory: **{lowest['name']}** ({measurement(lowest, 'max_gpu_memory_allocated_mb')} MiB)."]
    lines += ["", "## Failures and validation", ""]
    for s in stages:
        for error in s["errors"]:
            lines += [f"### {s['name']} ({s['status']})", "", "```text", error.replace("```", "'''"), "```", ""]
    lines += [f"- {e}" for e in errors] or ["No metric validation errors."]
    lines += ["", "Historical unknown fields remain unknown. Rankings are within each saved run and workload. Windows and WSL2 are separate runs; missing provenance prevents cross-run compatibility claims.", ""]
    return "\n".join(lines)


def generate(root, readme=None):
    root = Path(root).resolve()
    runs = manifests(root)
    block = [START, "", "Saved experiments (no benchmarks run during report generation).", "",
             "| Run | Model | Environment | Stages | Validation errors |", "| --- | --- | --- | ---: | ---: |"]
    for run in runs:
        report = root / "runs" / run["run_id"] / "report.md"
        report.write_text(render(run), encoding="utf-8")
        run["artifacts"] = [a for a in run["artifacts"] if a["path"] != report.relative_to(root).as_posix()] + [artifact(root, report)]
        run["artifacts"].sort(key=lambda a: a["path"])
        write_json(report.parent / "manifest.json", run)
        block.append(f"| {run['run_id']} | {cell(run.get('model') or 'unknown')} | {cell(run['environment'].get('label') or 'unknown')} | {len(run['stages'])} | {len(validate(run['stages']))} |")
    block += ["", "Each row below ranks a single workload within one saved run. Failed and invalid stages are excluded; Windows and WSL2 remain separate.", "",
              "| Run | Batch / seq | Fastest stage | Tokens/s mean ± SD | Speedup × | Lowest-memory stage | Allocated MiB mean ± SD |",
              "| --- | --- | --- | ---: | ---: | --- | ---: |"]
    from .registry import workload
    for run in runs:
        ok = [s for s in comparisons(run["stages"]) if s["status"] == "ok" and not validate([s])]
        for group in sorted({workload(s) for s in ok}, key=str):
            rows = [s for s in ok if workload(s) == group]
            fastest = max(rows, key=lambda s: s["metrics"]["tokens_per_sec"]["mean"])
            lowest = min(rows, key=lambda s: s["metrics"]["max_gpu_memory_allocated_mb"]["mean"])
            values = [f"{run.get('model') or 'unknown'} / {run['environment'].get('label') or 'unknown'} / {run['run_id']}",
                      f"{group[0]} / {group[1]}", fastest["name"], measurement(fastest, "tokens_per_sec"), number(fastest["speedup_vs_baseline"]),
                      lowest["name"], measurement(lowest, "max_gpu_memory_allocated_mb")]
            block.append("| " + " | ".join(cell(v) for v in values) + " |")
    block += ["", "Run reports: `results/runs/<run-id>/report.md`. Unknown historical provenance is disclosed in each report. Cross-environment ratios are not reported.", "", "[Registry, reports, dashboard startup, and CPU verification](docs/experiments.md) · [Verified performance case study](docs/performance_case_study.md)", "", END]
    if readme:
        path = Path(readme)
        text = path.read_text(encoding="utf-8")
        if text.count(START) != text.count(END) or text.count(START) > 1:
            raise ValueError("README generated markers are malformed")
        generated = "\n".join(block)
        if START in text:
            text = re.sub(re.escape(START) + r".*?" + re.escape(END), lambda _: generated, text, flags=re.S)
        else:
            text = text.rstrip() + "\n\n" + generated + "\n"
        path.write_text(text, encoding="utf-8")
    return len(runs)
