"""One manifest per run; legacy files stay in place and are never rewritten."""
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

METRICS = {
    "tokens_per_sec": "tokens/s", "samples_per_sec": "samples/s",
    "avg_step_time_ms": "ms/step", "max_gpu_memory_allocated_mb": "MiB",
    "max_gpu_memory_reserved_mb": "MiB",
}
SETTINGS = ("precision", "attention_backend", "tf32_enabled", "fused_adamw_requested",
            "fused_adamw_enabled", "compile_requested", "compile_enabled", "activation_checkpointing")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def git_value(*args):
    try:
        return subprocess.check_output(["git", *args], stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def begin_run(root, config, args, device, kind):
    """Capture provenance before measurements and reserve a unique directory."""
    import torch
    root = Path(root).resolve()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:12]
    directory = root / "runs" / run_id
    directory.mkdir(parents=True, exist_ok=False)
    status = git_value("status", "--porcelain")
    cuda = device.type == "cuda"
    release = platform.release()
    environment = "WSL2" if "wsl2" in release.lower() else ("WSL (version unknown)" if "microsoft" in release.lower() else platform.system())
    manifest = {
        "schema_version": 1, "run_id": run_id, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "kind": kind, "origin": "measured", "config": config, "config_hash": digest(config),
        "git_revision": git_value("rev-parse", "HEAD"), "git_dirty": None if status is None else bool(status),
        "model": config.get("model_name"),
        "model_dimensions": {k: config.get(k) for k in ("vocab_size", "n_layer", "n_head", "n_embd", "block_size", "dropout")},
        "batch_size": config.get("batch_size"), "sequence_length": config.get("block_size"),
        "measurement_steps": getattr(args, "steps", getattr(args, "benchmark_steps", None)),
        "warmup_steps": args.warmup_steps, "repeats": getattr(args, "repeats", 1),
        "optimization_settings": {k: config.get(k) for k in ("precision", "attention_backend", "tf32", "fused_adamw", "torch_compile", "activation_checkpointing")},
        "environment": {"label": environment, "os": platform.platform(), "python": sys.version,
                        "pytorch": torch.__version__, "cuda": torch.version.cuda, "cpu": platform.processor() or platform.machine() or None,
                        "device": str(device), "gpu": torch.cuda.get_device_name(device) if cuda else None,
                        "gpu_total_memory_bytes": torch.cuda.get_device_properties(device).total_memory if cuda else None},
        "stages": [], "artifacts": [], "validation_errors": [],
    }
    manifest["environment"]["logical_cpu_count"] = os.cpu_count()
    manifest["dataset_hashes"] = {
        name: hashlib.sha256((Path(config.get("data_dir", "data/processed")) / name).read_bytes()).hexdigest()
        if (Path(config.get("data_dir", "data/processed")) / name).is_file() else None
        for name in ("meta.json", "train.bin", "val.bin")
    }
    write_json(directory / "manifest.json", manifest)
    return directory


def normalize(rows):
    """Normalize either individual measurements or the historical summary format."""
    groups = {}
    for row in rows:
        key = (row.get("name", row.get("mode", "unknown")), row.get("batch_size"), row.get("seq_len"))
        groups.setdefault(key, []).append(row)
    stages = []
    for (name, batch, seq), repeats in groups.items():
        first = repeats[0]
        is_summary = "successful_repeats" in first
        ok = [r for r in repeats if r.get("status") == "ok"]
        status = "ok" if len(ok) == len(repeats) else ("failed" if any(r.get("status") == "failed" for r in repeats) else "skipped")
        if any(r.get("status") not in {"ok", "failed", "skipped"} for r in repeats):
            status = "invalid"
        stage = {"name": name, "status": status, "batch_size": batch, "sequence_length": seq,
                 "repeats": first.get("repeats") if is_summary else len(repeats),
                 "successful_repeats": first.get("successful_repeats") if is_summary else len(ok),
                 "errors": list(dict.fromkeys(r["error"] for r in repeats if r.get("error"))),
                 "settings": {k: first.get(k) for k in SETTINGS},
                 "parameter_count": first.get("parameter_count"), "model_dimensions": first.get("model_config"),
                 "warmup_steps": first.get("warmup_steps"), "measurement_steps": first.get("benchmark_steps"),
                 "gradient_accumulation_steps": first.get("gradient_accumulation_steps"),
                 "device_name": first.get("device_name"), "metrics": {}}
        stage["settings"]["compile_requested"] = first.get("compile_requested", first.get("torch_compile_requested"))
        stage["settings"]["compile_enabled"] = first.get("compile_enabled", first.get("torch_compile_enabled"))
        for metric, unit in METRICS.items():
            values = [r[metric] for r in ok if r.get(metric) is not None]
            mean = first.get(metric + "_mean") if is_summary else (statistics.mean(values) if values else None)
            std = first.get(metric + "_std") if is_summary else (statistics.stdev(values) if len(values) > 1 else None)
            if is_summary and (first.get("successful_repeats") or 0) < 2:
                std = None
            stage["metrics"][metric] = {"mean": mean, "std": std, "unit": unit}
        stages.append(stage)
    return stages


def workload(stage):
    return tuple(stage.get(k) for k in ("batch_size", "sequence_length", "parameter_count", "gradient_accumulation_steps", "measurement_steps", "warmup_steps")) + (digest(stage.get("model_dimensions")),)


def validate(stages):
    errors = []
    for s in stages:
        name = s["name"]
        if s["status"] not in {"ok", "failed", "skipped"}:
            errors.append(f"{name}: invalid status")
        for field in ("batch_size", "sequence_length", "repeats", "successful_repeats"):
            if s.get(field) is None:
                errors.append(f"{name}: missing {field}")
            elif not isinstance(s[field], int) or s[field] < (0 if field == "successful_repeats" else 1):
                errors.append(f"{name}: invalid {field}")
        if s.get("successful_repeats", 0) > (s.get("repeats") or 0):
            errors.append(f"{name}: successful repeats exceed attempted repeats")
        for key, metric in s["metrics"].items():
            for part in ("mean", "std"):
                value = metric[part]
                if value is not None and (not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0):
                    errors.append(f"{name}: invalid {key} {part}")
            if s["status"] == "ok" and metric["mean"] is None:
                errors.append(f"{name}: missing {key}")
        t = s["metrics"]["tokens_per_sec"]["mean"]
        samples = s["metrics"]["samples_per_sec"]["mean"]
        ms = s["metrics"]["avg_step_time_ms"]["mean"]
        if s["status"] == "ok":
            if not s.get("successful_repeats") or s.get("successful_repeats") != s.get("repeats"):
                errors.append(f"{name}: ok status requires all repeats to succeed")
            if not t or not ms:
                errors.append(f"{name}: throughput and latency must be positive")
            if t and samples and s["sequence_length"] and not math.isclose(t, samples * s["sequence_length"], rel_tol=0.001):
                errors.append(f"{name}: tokens/s inconsistent with samples/s × sequence length")
            accum = s.get("gradient_accumulation_steps")
            if t and ms and accum and s["batch_size"] and s["sequence_length"]:
                # mean(1 / latency) differs from 1 / mean(latency); allow measured dispersion.
                std = s["metrics"]["avg_step_time_ms"]["std"] or 0
                tolerance = max(0.001, 5 * (std / ms) ** 2)
                expected = s["batch_size"] * s["sequence_length"] * accum * 1000 / ms
                if not math.isclose(t, expected, rel_tol=tolerance):
                    errors.append(f"{name}: throughput inconsistent with step workload/latency")
            allocated = s["metrics"]["max_gpu_memory_allocated_mb"]["mean"]
            reserved = s["metrics"]["max_gpu_memory_reserved_mb"]["mean"]
            if allocated is not None and reserved is not None and allocated > reserved:
                errors.append(f"{name}: allocated memory exceeds reserved memory")
        if s["status"] != "ok" and not s["errors"]:
            errors.append(f"{name}: unavailable stage has no error explanation")
    return errors


def comparisons(stages):
    """Use a successful FP32 baseline with the same workload inside this run."""
    for s in stages:
        s["speedup_vs_baseline"] = None
        s["memory_reduction_mib"] = None
        s["memory_reduction_percent"] = None
        if s["status"] != "ok" or validate([s]):
            continue
        base = next((b for b in stages if b["name"] in {"baseline_fp32_manual", "fp32"} and b["status"] == "ok" and workload(b) == workload(s) and not validate([b])), None)
        if base:
            s["speedup_vs_baseline"] = base["metrics"]["avg_step_time_ms"]["mean"] / s["metrics"]["avg_step_time_ms"]["mean"]
            mem = base["metrics"]["max_gpu_memory_allocated_mb"]["mean"]
            s["memory_reduction_mib"] = mem - s["metrics"]["max_gpu_memory_allocated_mb"]["mean"]
            s["memory_reduction_percent"] = s["memory_reduction_mib"] / mem * 100 if mem else None
    return stages


def artifact(root, path):
    path = Path(path).resolve()
    return {"path": path.relative_to(Path(root).resolve()).as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def finish_run(root, directory, rows):
    manifest = read_json(directory / "manifest.json")
    manifest["stages"] = normalize(rows)
    manifest["validation_errors"] = validate(manifest["stages"])
    comparisons(manifest["stages"])
    manifest["artifacts"] = [artifact(root, p) for p in sorted(directory.iterdir()) if p.suffix in {".json", ".csv", ".png"} and p.name != "manifest.json"]
    write_json(directory / "manifest.json", manifest)


def manifests(root):
    return [read_json(p) for p in sorted((Path(root) / "runs").glob("*/manifest.json"))]


def import_legacy(root, evidence_path=None):
    root = Path(root).resolve()
    existing = {m.get("source_fingerprint"): m for m in manifests(root) if m.get("source_fingerprint")}
    evidence = read_json(evidence_path) if evidence_path else {}
    added = []
    # Raw repeat sets are authoritative; summaries and alias files are associated downloads.
    candidates = sorted(root.glob("optimization_benchmark*_raw.json")) + sorted(root.glob("benchmark*.json"))
    candidates += [p for p in sorted(root.glob("optimization_benchmark*_summary.json")) if not p.with_name(p.name.replace("_summary", "_raw")).exists()]
    for path in candidates:
        rows = read_json(path)
        if not isinstance(rows, list) or not rows:
            continue
        fingerprint = digest(rows)
        if fingerprint in existing:
            m = existing[fingerprint]
            directory = root / "runs" / m["run_id"]
        else:
            run_id = "legacy-" + fingerprint[:16]
            directory = root / "runs" / run_id
            directory.mkdir(parents=True, exist_ok=True)
            stages = normalize(rows)
            disclosure = evidence.get(fingerprint, {})
            m = {"schema_version": 1, "run_id": run_id, "origin": "historical", "kind": "optimizations" if "optimization" in path.name else "sweep",
                 "timestamp_utc": None, "config": None, "config_hash": None, "git_revision": None, "git_dirty": None,
                 "model": disclosure.get("model"), "model_dimensions": rows[0].get("model_config"),
                 "batch_size": None, "sequence_length": None, "measurement_steps": None, "warmup_steps": None, "repeats": None,
                 "optimization_settings": None, "environment": {"label": None, "os": None, "python": None, "pytorch": None, "cuda": None,
                 "cpu": None, "gpu": rows[0].get("device_name"), "gpu_total_memory_bytes": None, **disclosure.get("environment", {})},
                 "provenance_notes": disclosure.get("notes", "Historical metadata not recorded; null means unknown. Stage fields come from source rows."),
                 "source_fingerprint": fingerprint, "stages": comparisons(stages), "validation_errors": validate(stages), "artifacts": []}
            existing[fingerprint] = m
            added.append(run_id)
        disclosure = evidence.get(fingerprint)
        if disclosure:
            m["model"] = disclosure.get("model", m["model"])
            m["environment"].update(disclosure.get("environment", {}))
            m["provenance_notes"] = disclosure.get("notes", m["provenance_notes"])
        for target, source in (("batch_size", "batch_size"), ("sequence_length", "sequence_length"), ("measurement_steps", "measurement_steps"), ("warmup_steps", "warmup_steps"), ("repeats", "repeats")):
            values = {s.get(source) for s in m["stages"]}
            m[target] = next(iter(values)) if len(values) == 1 else None
        stem = path.stem.removesuffix("_raw").removesuffix("_summary")
        paths = [path]
        if "optimization" in stem:
            suffix = stem.removeprefix("optimization_benchmark")
            paths += [root / (stem + "_summary" + ext) for ext in (".json", ".csv")]
            paths += [root / (prefix + suffix + ".png") for prefix in ("cumulative_speedup", "optimization_memory")]
        else:
            paths += [path.with_suffix(".csv")]
            suffix = stem.removeprefix("benchmark").replace("_results", "")
            paths += [root / (prefix + suffix + ".png") for prefix in ("tokens_per_sec", "step_time_ms", "memory_usage")]
        for p in paths:
            if p.is_file():
                a = artifact(root, p)
                if a not in m["artifacts"]:
                    m["artifacts"].append(a)
        m["artifacts"].sort(key=lambda a: a["path"])
        write_json(directory / "manifest.json", m)
    return added


def compatible(a, b):
    """Cross-run comparison requires complete, identical workload + environment metadata."""
    cpu = a["environment"].get("device") == b["environment"].get("device") == "cpu"
    required_env = ("label", "os", "python", "pytorch", "cpu", "logical_cpu_count") if cpu else ("label", "os", "python", "pytorch", "cuda", "gpu")
    for key in required_env:
        if a["environment"].get(key) is None or a["environment"].get(key) != b["environment"].get(key):
            return False, f"Environment field {key} differs or is unknown"
    if not a.get("dataset_hashes") or any(v is None for v in a["dataset_hashes"].values()) or a.get("dataset_hashes") != b.get("dataset_hashes"):
        return False, "Dataset content hashes differ or are unknown"
    required_config = ("vocab_size", "n_layer", "n_head", "n_embd", "dropout", "seed", "data_dir", "gradient_accumulation_steps", "learning_rate", "weight_decay", "beta1", "beta2", "grad_clip")
    for key in required_config:
        if not a.get("config") or not b.get("config") or a["config"].get(key) is None or a["config"].get(key) != b["config"].get(key):
            return False, f"Workload/config field {key} differs or is unknown"
    wa, wb = {workload(s) for s in a["stages"]}, {workload(s) for s in b["stages"]}
    if wa != wb or any(any(x is None for x in w[:-1]) for w in wa):
        return False, "Workload or measurement protocol differs or is unknown"
    return True, "Matching workload, measurement protocol, and environment"
