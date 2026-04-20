import json
from pathlib import Path

from .helpers import current_field, current_options, run_pipeline, stdout_payload


def test_pipeline_requires_confirmation_before_final_verdict(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "infer.py").write_text("print('infer')\n", encoding="utf-8")
    (workspace / "model").mkdir()
    output_dir = tmp_path / "out"

    completed = run_pipeline(
        "--working-dir",
        str(workspace),
        "--output-dir",
        str(output_dir),
        "--target",
        "inference",
        "--entry-script",
        "infer.py",
        "--model-path",
        "model",
        "--launch-command",
        "python infer.py",
        cwd=workspace,
    )

    verdict = json.loads((output_dir / "meta" / "readiness-verdict.json").read_text(encoding="utf-8"))
    confirmation_step = json.loads((output_dir / "artifacts" / "confirmation-step.json").read_text(encoding="utf-8"))
    summary = stdout_payload(completed)
    assert verdict["status"] == "NEEDS_CONFIRMATION"
    assert verdict["phase"] == "awaiting_confirmation"
    assert verdict["confirmation_required"] is True
    assert verdict["can_run"] is False
    assert verdict["current_confirmation"]["field"] == "launcher"
    assert summary["confirmation_required"] is True
    assert current_field(summary) == "launcher"
    assert summary["artifact_refs"] == {
        "verdict": "meta/readiness-verdict.json",
        "lock": "artifacts/workspace-readiness.lock.json",
        "confirmation": "artifacts/confirmation-step.json",
    }
    assert confirmation_step["portable_question"]["response_binding"]["field"] == "launcher"
    assert len(confirmation_step["portable_question"]["options"]) <= 4
    assert "__manual__" not in {str(item.get("value")) for item in confirmation_step["portable_question"]["options"]}
    assert (output_dir / "meta" / "readiness-verdict.json").exists()
    assert (output_dir / "artifacts" / "workspace-readiness.lock.json").exists()
    assert (output_dir / "artifacts" / "confirmation-step.json").exists()
    assert not (output_dir / "report.json").exists()
    assert not (output_dir / "report.md").exists()
    assert not (output_dir / "meta" / "env.json").exists()
    assert not (output_dir / "meta" / "inputs.json").exists()
    assert not (output_dir / "logs" / "run.log").exists()
    assert (workspace / "readiness-output" / "latest" / "new-readiness-agent" / "workspace-readiness.lock.json").exists()


def test_pipeline_offers_catalog_options_when_workspace_has_no_runtime_evidence(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    output_dir = tmp_path / "out"

    summary = stdout_payload(
        run_pipeline(
            "--working-dir",
            str(workspace),
            "--output-dir",
            str(output_dir),
            cwd=workspace,
        )
    )

    assert summary["status"] == "NEEDS_CONFIRMATION"
    assert current_field(summary) == "target"
    assert "training" in current_options(summary)
    assert "inference" in current_options(summary)
    assert "__unknown__" in current_options(summary)
    assert summary["current_confirmation"]["options"][-1]["label"] == "Skip for now"
    portable_question = summary["current_confirmation"]["portable_question"]
    portable_values = [str(option.get("value")) for option in portable_question["options"]]
    portable_unknown = next(
        option for option in portable_question["options"] if str(option.get("value")) == "__unknown__"
    )
    assert portable_question["header"] == "Target"
    assert 2 <= len(portable_values) <= 4
    assert "__manual__" not in portable_values
    assert "__unknown__" in portable_values
    assert portable_unknown["label"] == "Skip for now"


def test_pipeline_advances_one_confirmation_step_at_a_time(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    output_dir_1 = tmp_path / "out1"
    output_dir_2 = tmp_path / "out2"
    output_dir_3 = tmp_path / "out3"

    first = stdout_payload(run_pipeline("--working-dir", str(workspace), "--output-dir", str(output_dir_1), cwd=workspace))
    second = stdout_payload(
        run_pipeline(
            "--working-dir",
            str(workspace),
            "--output-dir",
            str(output_dir_2),
            "--confirm",
            "target=training",
            cwd=workspace,
        )
    )
    third = stdout_payload(
        run_pipeline(
            "--working-dir",
            str(workspace),
            "--output-dir",
            str(output_dir_3),
            "--confirm",
            "launcher=python",
            cwd=workspace,
        )
    )

    assert current_field(first) == "target"
    assert current_field(second) == "launcher"
    assert current_field(third) == "framework"


def test_pipeline_reuses_one_attempt_directory_across_default_confirmation_steps(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    first = stdout_payload(run_pipeline("--working-dir", str(workspace), cwd=workspace))
    second = stdout_payload(
        run_pipeline(
            "--working-dir",
            str(workspace),
            "--confirm",
            "target=training",
            cwd=workspace,
        )
    )

    attempts_root = workspace / "readiness-output" / "attempts"
    attempt_dirs = [path for path in attempts_root.iterdir() if path.is_dir()]

    assert current_field(first) == "target"
    assert current_field(second) == "launcher"
    assert len(attempt_dirs) == 1
    first_output_dir = Path(first["output_dir"])
    second_output_dir = Path(second["output_dir"])
    assert first_output_dir.parent == second_output_dir.parent
    assert first_output_dir.name == "current"
    assert second_output_dir.name == "current"


def test_pipeline_detects_llamafactory_launcher_from_explicit_command(tmp_path: Path, fake_selected_python: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "train.py").write_text("import torch\nimport torch_npu\nimport transformers\n", encoding="utf-8")
    (workspace / "llama_sft.yaml").write_text("stage: sft\nmodel_name_or_path: model\ntrain_file: dataset/sample.txt\n", encoding="utf-8")
    (workspace / "model").mkdir()
    (workspace / "dataset").mkdir()
    (workspace / "dataset" / "sample.txt").write_text("hello\n", encoding="utf-8")
    output_dir = tmp_path / "out"

    run_pipeline(
        "--working-dir",
        str(workspace),
        "--output-dir",
        str(output_dir),
        "--selected-python",
        str(fake_selected_python),
        "--entry-script",
        "train.py",
        "--config-path",
        "llama_sft.yaml",
        "--model-path",
        "model",
        "--dataset-path",
        "dataset",
        "--launch-command",
        "uv run llamafactory-cli train --config llama_sft.yaml",
        cwd=workspace,
    )

    verdict = json.loads((output_dir / "meta" / "readiness-verdict.json").read_text(encoding="utf-8"))
    assert verdict["status"] == "NEEDS_CONFIRMATION"
    assert verdict["launcher"]["value"] == "llamafactory-cli"
    assert verdict["evidence_summary"]["uses_llamafactory"] is True
