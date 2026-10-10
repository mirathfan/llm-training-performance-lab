import copy
import json
from pathlib import Path
import pytest
from experiments.registry import (artifact, begin_run, comparisons, compatible, digest, finish_run,
                                  import_legacy, manifests, normalize, validate, write_json)
from experiments.reports import END, START, generate, render
from benchmark_optimizations import aggregate_rows


def test_aggregation_baseline_and_failures(measured_rows):
    stages = comparisons(normalize(measured_rows))
    baseline, optimized, failed = stages
    assert baseline["metrics"]["avg_step_time_ms"]["mean"] == 11
    assert optimized["metrics"]["avg_step_time_ms"]["std"] == pytest.approx(2**-.5)
    assert optimized["successful_repeats"] == 2
    assert optimized["speedup_vs_baseline"] == 2
    assert optimized["memory_reduction_mib"] == 10
    assert failed["successful_repeats"] == 0 and failed["speedup_vs_baseline"] is None
    assert not validate(stages)
    report = render(dict(run_id="synthetic", model="fixture", environment={}, origin="fixture", stages=stages))
    assert "fastest successful stage **optimized**" in report
    assert "Lowest allocated memory: **optimized**" in report
    assert "Synthetic failure: Triton missing" in report


def test_original_benchmark_aggregation(measured_rows):
    for r in measured_rows:
        r.update(tf32_enabled=False, fused_adamw_requested=False, compile_requested=False,
                 compile_enabled=False, fused_adamw_enabled=False)
    rows = aggregate_rows([{"name": n} for n in ("baseline_fp32_manual", "optimized", "compile", "no_measurements")], measured_rows)
    assert len(rows) == 3
    assert rows[1]["speedup_vs_baseline"] == 2
    assert rows[2]["tokens_per_sec_mean"] is None


def test_partial_failure_excluded_and_baseline_shape(measured_rows):
    measured_rows.append(dict(measured_rows[2], status="failed", error="repeat failed"))
    s = comparisons(normalize(measured_rows))
    assert s[1]["status"] == "failed" and s[1]["successful_repeats"] == 2
    assert s[1]["speedup_vs_baseline"] is None
    changed = normalize(measured_rows[:4])
    changed[1]["sequence_length"] = 8
    assert comparisons(changed)[1]["speedup_vs_baseline"] is None
    measured_rows[0]["status"] = "failed"
    measured_rows[0]["error"] = "baseline failed"
    assert comparisons(normalize(measured_rows))[1]["speedup_vs_baseline"] is None


def test_invalid_metrics_and_required_fields(measured_rows):
    s = normalize(measured_rows)
    s[0]["metrics"]["tokens_per_sec"]["mean"] = 3
    s[0]["metrics"]["max_gpu_memory_allocated_mb"]["mean"] = 1000
    s[0]["repeats"] = 1
    errors = validate(s)
    assert any("inconsistent" in e for e in errors)
    assert any("exceeds reserved" in e for e in errors)
    assert any("exceed attempted" in e for e in errors)
    assert comparisons(s)[0]["speedup_vs_baseline"] is None
    s[0]["batch_size"] = None
    assert any("missing batch_size" in e for e in validate(s))


def test_legacy_import_unknowns_aliases_and_deterministic_report(tmp_path, measured_rows):
    original = json.dumps(measured_rows)
    for name in ("optimization_benchmark_fixture_raw.json", "optimization_benchmark_raw.json"):
        (tmp_path/name).write_text(original)
    assert len(import_legacy(tmp_path)) == 1
    assert not import_legacy(tmp_path)
    run = manifests(tmp_path)[0]
    assert run["timestamp_utc"] is None and run["config_hash"] is None
    assert run["environment"]["python"] is None and run["model"] is None
    assert len(run["artifacts"]) == 2
    readme = tmp_path/"README.md"
    readme.write_text(f"manual introduction\n{START}\nold\n{END}\nmanual limitations\n")
    generate(tmp_path, readme)
    first = readme.read_bytes()
    report = (tmp_path/"runs"/run["run_id"]/"report.md").read_bytes()
    generate(tmp_path, readme)
    assert first == readme.read_bytes()
    assert report == (tmp_path/"runs"/run["run_id"]/"report.md").read_bytes()
    assert readme.read_text().startswith("manual introduction\n") and readme.read_text().endswith("manual limitations\n")
    assert (tmp_path/"optimization_benchmark_raw.json").read_text() == original
    readme.write_text(START)
    with pytest.raises(ValueError, match="markers"):
        generate(tmp_path, readme)


def test_unique_future_runs_and_metadata(tmp_path, measured_rows):
    from argparse import Namespace
    import torch
    config = dict(model_name="fixture",batch_size=2,block_size=4,n_layer=1,n_head=1,n_embd=8,vocab_size=19,dropout=0)
    args = Namespace(steps=5,warmup_steps=2,repeats=2)
    one = begin_run(tmp_path, config, args, torch.device("cpu"), "synthetic test")
    two = begin_run(tmp_path, config, args, torch.device("cpu"), "synthetic test")
    assert one != two
    write_json(one/"raw.json", measured_rows)
    finish_run(tmp_path, one, measured_rows)
    run = next(r for r in manifests(tmp_path) if r["run_id"] == one.name)
    assert run["config_hash"] == digest(config) and run["measurement_steps"] == 5
    assert run["timestamp_utc"].endswith("+00:00") and run["environment"]["python"]
    assert run["artifacts"][0] == artifact(tmp_path,one/"raw.json")


def test_cross_run_compatibility(measured_rows):
    config = dict(vocab_size=19,n_layer=1,n_head=1,n_embd=8,dropout=0,seed=1,data_dir="fixture",gradient_accumulation_steps=1,
                  learning_rate=.001,weight_decay=.1,beta1=.9,beta2=.95,grad_clip=1)
    run = dict(config=config,dataset_hashes={"train.bin":"synthetic content hash"},environment=dict(label="Windows",os="fixture",python="3",pytorch="2",cuda="12",gpu="fixture"), stages=normalize(measured_rows))
    assert compatible(run,copy.deepcopy(run))[0]
    other = copy.deepcopy(run); other["environment"]["label"] = "WSL2"
    assert not compatible(run,other)[0]
    other = copy.deepcopy(run); other["config"] = None
    assert not compatible(run,other)[0]


def test_summary_only_import(tmp_path, measured_rows):
    from benchmark_optimizations import aggregate_rows
    for r in measured_rows:
        r.update(tf32_enabled=False,fused_adamw_requested=False,compile_requested=False,compile_enabled=False,fused_adamw_enabled=False)
    summary = aggregate_rows([{"name":n} for n in ("baseline_fp32_manual","optimized","compile")],measured_rows)
    write_json(tmp_path/"optimization_benchmark_fixture_summary.json",summary)
    assert len(import_legacy(tmp_path))==1
    assert not import_legacy(tmp_path)
    run=manifests(tmp_path)[0]
    assert run["stages"][1]["metrics"]["tokens_per_sec"]["mean"]==summary[1]["tokens_per_sec_mean"]
    assert run["stages"][2]["status"]=="failed"
