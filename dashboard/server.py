"""Serve registered experiment results and the built frontend; no benchmark routes."""
import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from experiments.registry import compatible, manifests

REPO = Path(__file__).resolve().parents[1]


def create_app(results_dir=None, frontend_dir=None):
    root = Path(results_dir or os.environ.get("LAB_RESULTS_DIR", REPO / "results")).resolve()
    dist = Path(frontend_dir or REPO / "dashboard" / "frontend" / "dist")
    app = FastAPI(title="LLM Training Performance Lab", docs_url="/api/docs")

    def find(run_id):
        run = next((r for r in manifests(root) if r["run_id"] == run_id), None)
        if run is None:
            raise HTTPException(404, "Registered run not found")
        return run

    @app.get("/api/runs")
    def runs():
        return manifests(root)

    @app.get("/api/compare")
    def compare(left: str, right: str):
        a, b = find(left), find(right)
        match, reason = compatible(a, b)
        return {"compatible": match, "reason": reason, "runs": [a, b] if match else []}

    @app.get("/api/runs/{run_id}/files/{file_index}")
    def download(run_id: str, file_index: int):
        run = find(run_id)
        if file_index < 0 or file_index >= len(run["artifacts"]):
            raise HTTPException(404, "Registered file not found")
        entry = run["artifacts"][file_index]
        relative = Path(entry["path"])
        path = (root / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(root) or path.suffix.lower() not in {".json", ".csv", ".png", ".md"}:
            raise HTTPException(403, "Unsafe artifact path")
        if not path.is_file():
            raise HTTPException(404, "Saved artifact is missing")
        import hashlib
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise HTTPException(409, "Saved artifact changed since registration; re-import before downloading")
        return FileResponse(path, filename=path.name)

    if dist.is_dir():
        app.mount("/", StaticFiles(directory=dist, html=True), name="dashboard")
    return app


app = create_app()
