import json
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
FIXTURES = Path(__file__).resolve().parent / "fixtures"


def run_script(script_name: str, *args: str) -> subprocess.CompletedProcess[str]:
    script = SCRIPTS / script_name
    return subprocess.run(
        [sys.executable, str(script), *args],
        check=True,
        text=True,
        capture_output=True,
    )


def materialize_fixture(name: str, tmp_path: Path) -> Path:
    src = FIXTURES / name
    dst = tmp_path / name
    shutil.copytree(src, dst)
    return dst


def test_training_workspace_fixture_supports_discovery_closure_and_task_smoke(tmp_path: Path):
    workspace = materialize_fixture("training_workspace", tmp_path)
    target_json = tmp_path / "target.json"
    closure_json = tmp_path / "closure.json"
    task_smoke_json = tmp_path / "task-smoke.json"
    checks_json = tmp_path / "checks.json"

    run_script(
        "discover_execution_target.py",
        "--working-dir",
        str(workspace),
        "--target",
        "training",
        "--dataset-path",
        "dataset",
        "--model-path",
        "model",
        "--task-smoke-cmd",
        "python train.py --smoke-test --config train.yaml --dataset dataset --model-path model",
        "--output-json",
        str(target_json),
    )
    run_script(
        "build_dependency_closure.py",
        "--target-json",
        str(target_json),
        "--output-json",
        str(closure_json),
    )
    run_script(
        "run_task_smoke.py",
        "--target-json",
        str(target_json),
        "--closure-json",
        str(closure_json),
        "--output-json",
        str(task_smoke_json),
    )
    run_script(
        "collect_readiness_checks.py",
        "--target-json",
        str(target_json),
        "--closure-json",
        str(closure_json),
        "--task-smoke-json",
        str(task_smoke_json),
        "--output-json",
        str(checks_json),
    )

    target = json.loads(target_json.read_text(encoding="utf-8"))
    closure = json.loads(closure_json.read_text(encoding="utf-8"))
    task_smoke = json.loads(task_smoke_json.read_text(encoding="utf-8"))
    checks = {item["id"]: item for item in json.loads(checks_json.read_text(encoding="utf-8"))}

    assert target["entry_script"] == "train.py"
    assert target["config_path"] == "train.yaml"
    assert target["task_smoke_cmd"].startswith("python train.py")
    assert closure["layers"]["workspace_assets"]["dataset_path"]["exists"] is True
    assert closure["layers"]["workspace_assets"]["model_path"]["exists"] is True
    task_smoke_by_id = {item["id"]: item for item in task_smoke}
    assert task_smoke_by_id["task-smoke-script-parse"]["status"] == "ok"
    assert task_smoke_by_id["task-smoke-executed"]["status"] == "ok"
    assert task_smoke_by_id["task-smoke-executed"]["stdout_head"] == "training smoke ok"
    assert checks["workspace-dataset_path"]["status"] == "ok"
    assert checks["task-smoke-executed"]["status"] == "ok"


def test_inference_workspace_fixture_exposes_model_markers_and_task_smoke(tmp_path: Path):
    workspace = materialize_fixture("inference_workspace", tmp_path)
    target_json = tmp_path / "target.json"
    closure_json = tmp_path / "closure.json"
    task_smoke_json = tmp_path / "task-smoke.json"
    checks_json = tmp_path / "checks.json"

    run_script(
        "discover_execution_target.py",
        "--working-dir",
        str(workspace),
        "--target",
        "inference",
        "--model-path",
        "model",
        "--task-smoke-cmd",
        "python infer.py --smoke-test --model-path model",
        "--output-json",
        str(target_json),
    )
    run_script(
        "build_dependency_closure.py",
        "--target-json",
        str(target_json),
        "--output-json",
        str(closure_json),
    )
    run_script(
        "run_task_smoke.py",
        "--target-json",
        str(target_json),
        "--closure-json",
        str(closure_json),
        "--output-json",
        str(task_smoke_json),
    )
    run_script(
        "collect_readiness_checks.py",
        "--target-json",
        str(target_json),
        "--closure-json",
        str(closure_json),
        "--task-smoke-json",
        str(task_smoke_json),
        "--output-json",
        str(checks_json),
    )

    target = json.loads(target_json.read_text(encoding="utf-8"))
    closure = json.loads(closure_json.read_text(encoding="utf-8"))
    task_smoke = {item["id"]: item for item in json.loads(task_smoke_json.read_text(encoding="utf-8"))}
    checks = {item["id"]: item for item in json.loads(checks_json.read_text(encoding="utf-8"))}

    assert target["entry_script"] == "infer.py"
    assert "model/config.json" in target["model_markers"]
    assert "model/tokenizer_config.json" in target["model_markers"]
    assert closure["layers"]["workspace_assets"]["model_path"]["exists"] is True
    assert task_smoke["task-smoke-script-parse"]["status"] == "ok"
    assert task_smoke["task-smoke-executed"]["status"] == "ok"
    assert task_smoke["task-smoke-executed"]["stdout_head"] == "inference smoke ok"
    assert checks["workspace-model_path"]["status"] == "ok"
    assert checks["task-smoke-executed"]["status"] == "ok"
