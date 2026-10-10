import pytest


@pytest.fixture
def measured_rows():
    """Synthetic fixture; never saved into the project's measured results directory."""
    common = dict(batch_size=2, seq_len=4, warmup_steps=2, benchmark_steps=5,
                  gradient_accumulation_steps=1, parameter_count=100,
                  device_name="Synthetic CPU fixture", precision="fp32", attention_backend="manual")
    def row(name, ms, allocated, repeat=0):
        return dict(common, name=name, status="ok", repeat=repeat, avg_step_time_ms=ms,
                    tokens_per_sec=8000/ms, samples_per_sec=2000/ms,
                    max_gpu_memory_allocated_mb=allocated, max_gpu_memory_reserved_mb=allocated+2)
    return [row("baseline_fp32_manual",10,20),row("baseline_fp32_manual",12,22,1),
            row("optimized",5,10),row("optimized",6,12,1),
            dict(common,name="compile",status="failed",error="Synthetic failure: Triton missing")]
