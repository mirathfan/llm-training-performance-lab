from fastapi.testclient import TestClient
from dashboard.server import create_app
from experiments.registry import import_legacy, manifests, read_json, write_json


def test_read_only_downloads_and_empty_state(tmp_path, measured_rows):
    app = create_app(tmp_path, tmp_path/"no_frontend")
    client = TestClient(app)
    assert client.get("/api/runs").json() == []
    write_json(tmp_path/"benchmark_results.json", measured_rows)
    import_legacy(tmp_path)
    run = manifests(tmp_path)[0]
    url = f"/api/runs/{run['run_id']}/files/0"
    response = client.get(url)
    assert response.status_code == 200 and "attachment" in response.headers["content-disposition"]
    assert client.post("/api/runs").status_code == 405
    assert client.get(url[:-1]+"-1").status_code == 404
    assert client.get("/api/runs/unknown/files/0").status_code == 404
    assert client.get(f"/api/compare?left={run['run_id']}&right={run['run_id']}").json()["compatible"] is False
    (tmp_path/"benchmark_results.json").write_text("[]")
    assert client.get(url).status_code == 409
    (tmp_path/"benchmark_results.json").unlink()
    assert client.get(url).status_code == 404


def test_registered_path_traversal_blocked(tmp_path, measured_rows):
    write_json(tmp_path/"benchmark_results.json", measured_rows)
    import_legacy(tmp_path)
    manifest = next((tmp_path/"runs").glob("*/manifest.json"))
    run = read_json(manifest)
    run["artifacts"][0]["path"] = "../../README.md"
    write_json(manifest,run)
    client = TestClient(create_app(tmp_path,tmp_path/"no_frontend"))
    assert client.get(f"/api/runs/{run['run_id']}/files/0").status_code == 403
