import json
from pathlib import Path
from typing import Any, Dict


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "item"):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def create_run_dir(results_dir: str, run_name: str) -> Path:
    run_dir = Path(results_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=_json_default), encoding="utf-8")


def append_json_log(record: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    if path.exists():
        records = json.loads(path.read_text(encoding="utf-8"))
    else:
        records = []
    records.append(record)
    save_json(records, path)


def save_config_copy(config: Dict[str, Any], run_dir: str | Path) -> None:
    save_json(config, Path(run_dir) / "config_resolved.json")
