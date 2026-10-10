"""Isolated synthetic browser fixtures. Never imports or writes measured results.

Run: python -m uvicorn tests.browser_fixture_server:app --port 8002
Every visible record is labelled TEST FIXTURE. For the empty-state server use
tests.browser_fixture_server:empty_app on another port.
"""
import tempfile
from pathlib import Path
from dashboard.server import create_app
from experiments.registry import comparisons, normalize, write_json

root=Path(tempfile.mkdtemp(prefix="llm-lab-browser-test-fixtures-"))
common=dict(batch_size=2,seq_len=4,precision="fp32",attention_backend="manual",parameter_count=100,
            gradient_accumulation_steps=1,warmup_steps=2,benchmark_steps=5,status="ok",samples_per_sec=200,tokens_per_sec=800,
            avg_step_time_ms=10,max_gpu_memory_allocated_mb=20,max_gpu_memory_reserved_mb=22)
config=dict(vocab_size=19,n_layer=1,n_head=1,n_embd=8,dropout=0,seed=1,data_dir="TEST FIXTURE",gradient_accumulation_steps=1,
            learning_rate=.001,weight_decay=.1,beta1=.9,beta2=.95,grad_clip=1)
for run_id in ("test-fixture-a","test-fixture-b"):
    write_json(root/"runs"/run_id/"manifest.json",dict(run_id=run_id,origin="TEST FIXTURE — synthetic, not measured",model="TEST FIXTURE",timestamp_utc=None,
        config=config,config_hash=None,git_revision=None,git_dirty=None,artifacts=[],validation_errors=[],dataset_hashes={"train.bin":"synthetic fixture hash"},
        environment=dict(label="TEST FIXTURE",os="synthetic",python="synthetic",pytorch="synthetic",cuda="synthetic",gpu="synthetic"),
        stages=comparisons(normalize([dict(common,name="baseline_fp32_manual"),dict(common,name="optimized",avg_step_time_ms=5,tokens_per_sec=1600,samples_per_sec=400,max_gpu_memory_allocated_mb=10)]))))
app=create_app(root)
empty_app=create_app(root/"empty")
