import json
from pathlib import Path

from .helpers import check_by_id, run_pipeline, stdout_payload


def test_pipeline_writes_full_bundle_and_surfaces_cann_paths(tmp_path: Path, fake_selected_python: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "train.py").write_text("import torch\nimport torch_npu\nfrom transformers import Trainer\n", encoding="utf-8")
    (workspace / "train.yaml").write_text("model_name_or_path: model\ntrain_file: dataset/sample.txt\n", encoding="utf-8")
    (workspace / "model").mkdir()
    (workspace / "dataset").mkdir()
    (workspace / "dataset" / "sample.txt").write_text("hello\n", encoding="utf-8")
    cann_root = tmp_path / "cann"
    cann_root.mkdir()
    (cann_root / "version.cfg").write_text("version=8.5.0\n", encoding="utf-8")
    output_dir = tmp_path / "out"

    completed = run_pipeline(
        "--working-dir",
        str(workspace),
        "--output-dir",
        str(output_dir),
        "--target",
        "training",
        "--framework-hint",
        "pta",
        "--launcher-hint",
        "torchrun",
        "--selected-python",
        str(fake_selected_python),
        "--entry-script",
        "train.py",
        "--config-path",
        "train.yaml",
        "--model-path",
        "model",
        "--dataset-path",
        "dataset",
        "--cann-path",
        str(cann_root),
        cwd=workspace,
    )

    verdict = json.loads((output_dir / "meta" / "readiness-verdict.json").read_text(encoding="utf-8"))
    summary = stdout_payload(completed)
    report = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
    latest_root = workspace / "readiness-output" / "latest" / "new-readiness-agent"
    latest_lock = json.loads((latest_root / "workspace-readiness.lock.json").read_text(encoding="utf-8"))
    confirmation = json.loads((latest_root / "confirmation-latest.json").read_text(encoding="utf-8"))

    assert verdict["status"] == "READY"
    assert verdict["can_run"] is True
    assert summary["cann_path"] == str(cann_root)
    assert summary["ascend_env_script_path"] is None
    assert summary["artifact_refs"] == {
        "verdict": "meta/readiness-verdict.json",
        "lock": "artifacts/workspace-readiness.lock.json",
        "confirmation": "artifacts/confirmation-step.json",
        "report": "report.json",
        "markdown": "report.md",
        "env": "meta/env.json",
        "inputs": "meta/inputs.json",
        "run_log": "logs/run.log",
    }
    assert report["artifacts"] == [
        "report.md",
        "meta/env.json",
        "meta/inputs.json",
        "meta/readiness-verdict.json",
        "artifacts/workspace-readiness.lock.json",
        "artifacts/confirmation-step.json",
    ]
    assert report["logs"] == ["logs/run.log"]
    assert (output_dir / "report.md").exists()
    assert (output_dir / "meta" / "env.json").exists()
    assert (output_dir / "meta" / "inputs.json").exists()
    assert (output_dir / "logs" / "run.log").exists()
    assert verdict["cann_path"] == str(cann_root)
    assert verdict["ascend_env_script_path"] is None
    assert "torchrun" in str(verdict["launcher"]["command_template"])
    assert latest_lock["cann_path"] == str(cann_root)
    assert latest_lock["ascend_env_script_path"] is None
    assert latest_lock["launcher"] == "torchrun"
    assert latest_lock["selected_python"] == str(fake_selected_python)
    assert confirmation["current_confirmation"] is None
    assert verdict["latest_cache_ref"] == {
        "root": "readiness-output/latest/new-readiness-agent",
        "lock": "readiness-output/latest/new-readiness-agent/workspace-readiness.lock.json",
        "confirmation": "readiness-output/latest/new-readiness-agent/confirmation-latest.json",
        "run_ref": "readiness-output/latest/new-readiness-agent/run-ref.json",
    }
    compatibility_check = check_by_id(verdict, "framework-compatibility")
    cann_check = check_by_id(verdict, "cann-version")
    ascend_runtime_check = check_by_id(verdict, "ascend-runtime")
    assert compatibility_check["status"] == "ok"
    assert "match a local compatibility row" in compatibility_check["summary"]
    assert compatibility_check["details"]["installed_versions"]["torch"] == "2.9.0"
    assert str(cann_root) in cann_check["summary"]
    assert str(cann_root) in ascend_runtime_check["summary"]
    report_markdown = (output_dir / "report.md").read_text(encoding="utf-8")
    assert f"- cann_path: `{cann_root}`" in report_markdown


def test_repeated_run_refreshes_latest_run_ref(tmp_path: Path, fake_selected_python: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "infer.py").write_text("import torch\nimport torch_npu\nprint('infer')\n", encoding="utf-8")
    (workspace / "model").mkdir()
    cann_root = tmp_path / "cann"
    cann_root.mkdir()
    (cann_root / "version.cfg").write_text("version=8.5.0\n", encoding="utf-8")

    out1 = tmp_path / "out1"
    run_pipeline(
        "--working-dir",
        str(workspace),
        "--output-dir",
        str(out1),
        "--target",
        "inference",
        "--framework-hint",
        "pta",
        "--launcher-hint",
        "python",
        "--selected-python",
        str(fake_selected_python),
        "--entry-script",
        "infer.py",
        "--model-path",
        "model",
        "--cann-path",
        str(cann_root),
        "--launch-command",
        "python infer.py",
        cwd=workspace,
    )

    latest_root = workspace / "readiness-output" / "latest" / "new-readiness-agent"
    first_run_ref = json.loads((latest_root / "run-ref.json").read_text(encoding="utf-8"))

    out2 = tmp_path / "out2"
    run_pipeline(
        "--working-dir",
        str(workspace),
        "--output-dir",
        str(out2),
        "--target",
        "inference",
        "--framework-hint",
        "pta",
        "--launcher-hint",
        "python",
        "--selected-python",
        str(fake_selected_python),
        "--entry-script",
        "infer.py",
        "--model-path",
        "model",
        "--cann-path",
        str(cann_root),
        "--launch-command",
        "python infer.py",
        cwd=workspace,
    )

    second_run_ref = json.loads((latest_root / "run-ref.json").read_text(encoding="utf-8"))
    assert first_run_ref["output_dir"] != second_run_ref["output_dir"]
