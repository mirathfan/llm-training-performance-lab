"""Exercise CLI orchestration with synthetic measurements; no training occurs."""
from argparse import Namespace
from experiments.registry import manifests
import benchmark_train


def test_benchmark_cli_saves_unique_runs(tmp_path, monkeypatch, measured_rows):
    args = Namespace(config="unused.yaml", benchmark_steps=5, warmup_steps=2, batch_sizes=[2],seq_lens=[4],modes=["fp32"])
    config = dict(results_dir=str(tmp_path), model_name="fixture", batch_size=2, block_size=4,device="cpu",data_dir=str(tmp_path/"unused"))
    monkeypatch.setattr(benchmark_train,"parse_args",lambda:args)
    monkeypatch.setattr(benchmark_train,"load_config",lambda _:config)
    monkeypatch.setattr(benchmark_train,"resolve_config",lambda config,args:config)
    monkeypatch.setattr(benchmark_train,"CharDataset",lambda *a,**kw:None)
    monkeypatch.setattr(benchmark_train,"benchmark_one",lambda *a,**kw:dict(measured_rows[0], mode="fp32"))
    monkeypatch.setattr(benchmark_train,"make_plots",lambda *a:None)
    original=tmp_path/"benchmark_results.json"
    original.write_text("historical placeholder preserved")
    benchmark_train.main(); benchmark_train.main()
    assert len(manifests(tmp_path)) == 2
    assert len(list((tmp_path/"runs").glob("*/benchmark_results.json"))) == 2
    assert original.read_text() == "historical placeholder preserved"
    assert all(r["stages"][0]["successful_repeats"]==1 for r in manifests(tmp_path))
