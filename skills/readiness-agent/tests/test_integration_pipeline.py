import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
READINESS_VERDICT_REF = Path("meta/readiness-verdict.json")


def run_script(script_name: str, *args: str) -> subprocess.CompletedProcess[str]:
    script = SCRIPTS / script_name
    return subprocess.run(
        [sys.executable, str(script), *args],
        check=True,
        text=True,
        capture_output=True,
    )


def load_report_pair(report_json: Path) -> tuple[dict, dict]:
    envelope = json.loads(report_json.read_text(encoding="utf-8"))
    verdict_json = report_json.parent / READINESS_VERDICT_REF
    verdict = json.loads(verdict_json.read_text(encoding="utf-8"))
    return envelope, verdict


def test_end_to_end_pipeline_reaches_warn_for_ambiguous_or_incomplete_workspace(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "train.py").write_text(
        "import mindspore as ms\noptimizer = object()\n",
        encoding="utf-8",
    )
    (workspace / "config.yaml").write_text("epochs: 1\n", encoding="utf-8")
    (workspace / "model").mkdir()

    target_json = tmp_path / "target.json"
    closure_json = tmp_path / "closure.json"
    task_smoke_json = tmp_path / "task-smoke.json"
    checks_json = tmp_path / "checks.json"
    normalized_json = tmp_path / "normalized.json"
    plan_json = tmp_path / "plan.json"
    execution_json = tmp_path / "execution.json"
    report_json = tmp_path / "report.json"
    report_md = tmp_path / "report.md"

    run_script(
        "discover_execution_target.py",
        "--working-dir",
        str(workspace),
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
    run_script(
        "normalize_blockers.py",
        "--input-json",
        str(checks_json),
        "--output-json",
        str(normalized_json),
    )
    run_script(
        "plan_env_fix.py",
        "--blockers-json",
        str(normalized_json),
        "--closure-json",
        str(closure_json),
        "--output-json",
        str(plan_json),
    )
    run_script(
        "execute_env_fix.py",
        "--plan-json",
        str(plan_json),
        "--output-json",
        str(execution_json),
    )
    run_script(
        "build_readiness_report.py",
        "--target-json",
        str(target_json),
        "--normalized-json",
        str(normalized_json),
        "--checks-json",
        str(checks_json),
        "--closure-json",
        str(closure_json),
        "--fix-applied-json",
        str(execution_json),
        "--output-json",
        str(report_json),
        "--output-md",
        str(report_md),
    )

    report, verdict = load_report_pair(report_json)
    plan = json.loads(plan_json.read_text(encoding="utf-8"))
    execution = json.loads(execution_json.read_text(encoding="utf-8"))

    assert report["status"] == "partial"
    assert verdict["status"] in {"WARN", "BLOCKED"}
    assert verdict["target"] == "training"
    assert isinstance(verdict["blockers"], list)
    assert isinstance(verdict["warnings"], list)
    assert verdict["dependency_closure"]["target_type"] == "training"
    assert verdict["fix_applied"]["execute"] is False
    assert isinstance(plan["actions"], list)
    assert execution["execute"] is False
    assert "next_action" in verdict


def test_end_to_end_pipeline_reaches_ready_with_strong_evidence_and_no_blockers(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "infer.py").write_text(
        "import torch\nimport torch_npu\nimport transformers\nprint('smoke ok')\n",
        encoding="utf-8",
    )
    (workspace / "model").mkdir()

    target_json = tmp_path / "target.json"
    closure_json = tmp_path / "closure.json"
    task_smoke_json = tmp_path / "task-smoke.json"
    normalized_json = tmp_path / "normalized.json"
    checks_json = tmp_path / "checks.json"
    report_json = tmp_path / "report.json"
    report_md = tmp_path / "report.md"

    run_script(
        "discover_execution_target.py",
        "--working-dir",
        str(workspace),
        "--target",
        "inference",
        "--model-path",
        "model",
        "--task-smoke-cmd",
        "python -c \"print('smoke ok')\"",
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
    checks_json.write_text(
        json.dumps(
            [
                {
                    "id": "target-stability",
                    "status": "ok",
                    "summary": "target resolved",
                },
                {
                    "id": "framework-importability",
                    "status": "ok",
                    "summary": "framework importable",
                },
                {
                    "id": "framework-smoke-prerequisite",
                    "status": "ok",
                    "summary": "framework smoke ok",
                },
                *json.loads(task_smoke_json.read_text(encoding="utf-8")),
            ]
        ),
        encoding="utf-8",
    )
    normalized_json.write_text(
        json.dumps(
            {
                "blockers": [],
                "warnings": [],
                "blockers_detailed": [],
                "warnings_detailed": [],
            }
        ),
        encoding="utf-8",
    )

    run_script(
        "build_readiness_report.py",
        "--target-json",
        str(target_json),
        "--normalized-json",
        str(normalized_json),
        "--checks-json",
        str(checks_json),
        "--closure-json",
        str(closure_json),
        "--output-json",
        str(report_json),
        "--output-md",
        str(report_md),
    )

    report, verdict = load_report_pair(report_json)
    assert report["status"] == "success"
    assert verdict["status"] == "READY"
    assert verdict["can_run"] is True
    assert verdict["target"] == "inference"
    assert verdict["evidence_level"] == "task_smoke"
    assert verdict["dependency_closure"]["target_type"] == "inference"
