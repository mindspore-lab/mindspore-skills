import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def run_pipeline(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPTS / "run_new_readiness_pipeline.py"), *args],
        cwd=str(cwd),
        check=True,
        text=True,
        capture_output=True,
    )


def stdout_payload(completed: subprocess.CompletedProcess[str]) -> dict:
    return json.loads(completed.stdout)


def current_field(summary: dict) -> Optional[str]:
    current_confirmation = summary.get("current_confirmation")
    if not isinstance(current_confirmation, dict):
        return None
    return current_confirmation.get("field")


def current_options(summary: dict) -> list[str]:
    current_confirmation = summary.get("current_confirmation")
    if not isinstance(current_confirmation, dict):
        return []
    return [str(option.get("value")) for option in current_confirmation.get("options", [])]


def check_by_id(verdict: dict, check_id: str) -> dict:
    for item in verdict.get("checks", []):
        if item.get("id") == check_id:
            return item
    raise AssertionError(f"missing check: {check_id}")


def make_fake_selected_python_with_import_error(tmp_path: Path, failing_package: str, error_message: str) -> Path:
    real_python = json.dumps(sys.executable)
    script = tmp_path / "fake-import-error-python.py"
    script.write_text(
        f"""#!/usr/bin/env python3
import json
import subprocess
import sys

REAL_PYTHON = {real_python}
FAILING_PACKAGE = {json.dumps(failing_package)}
ERROR_MESSAGE = {json.dumps(error_message)}
VERSION_OVERRIDES = {{
    "torch": "2.9.0",
    "torch_npu": "2.9.0",
    "mindspore": "2.6.0",
}}

if len(sys.argv) >= 3 and sys.argv[1] == "-c":
    code = sys.argv[2]
    if "platform.python_version" in code and "version_info" in code:
        print(json.dumps({{"version_info": [3, 10, 20], "version": "3.10.20"}}))
        raise SystemExit(0)
    if "importlib.util" in code and len(sys.argv) >= 5:
        mode = sys.argv[3]
        payload = json.loads(sys.argv[4])
        if mode == "import":
            packages = payload.get("packages", [])
            result = {{"imports": {{}}, "errors": {{}}}}
            for name in packages:
                if name == FAILING_PACKAGE:
                    result["imports"][name] = False
                    result["errors"][name] = ERROR_MESSAGE
                else:
                    result["imports"][name] = True
            print(json.dumps(result))
            raise SystemExit(0)
        if mode == "package_versions":
            packages = payload.get("packages", [])
            print(json.dumps({{"versions": {{name: VERSION_OVERRIDES.get(name, "1.0.0") for name in packages}}, "errors": {{}}}}))
            raise SystemExit(0)
    completed = subprocess.run([REAL_PYTHON, *sys.argv[1:]])
    raise SystemExit(completed.returncode)

completed = subprocess.run([REAL_PYTHON, *sys.argv[1:]])
raise SystemExit(completed.returncode)
""",
        encoding="utf-8",
    )
    if os.name == "nt":
        launcher = tmp_path / "fake-import-error-python.cmd"
        launcher.write_text(f'@echo off\r\n"{sys.executable}" "%~dp0fake-import-error-python.py" %*\r\n', encoding="utf-8")
        return launcher
    script.chmod(script.stat().st_mode | 0o111)
    return script


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
    assert (output_dir / "meta" / "readiness-verdict.json").exists()
    assert (output_dir / "artifacts" / "workspace-readiness.lock.json").exists()
    assert (output_dir / "artifacts" / "confirmation-step.json").exists()
    assert not (output_dir / "report.json").exists()
    assert not (output_dir / "report.md").exists()
    assert not (output_dir / "meta" / "env.json").exists()
    assert not (output_dir / "meta" / "inputs.json").exists()
    assert not (output_dir / "logs" / "run.log").exists()
    assert (workspace / "runs" / "latest" / "new-readiness-agent" / "workspace-readiness.lock.json").exists()


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
        "--launch-command",
        "torchrun train.py --config train.yaml",
        cwd=workspace,
    )

    verdict = json.loads((output_dir / "meta" / "readiness-verdict.json").read_text(encoding="utf-8"))
    summary = stdout_payload(completed)
    report = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
    latest_root = workspace / "runs" / "latest" / "new-readiness-agent"
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
    assert latest_lock["cann_path"] == str(cann_root)
    assert latest_lock["ascend_env_script_path"] is None
    assert latest_lock["launcher"] == "torchrun"
    assert latest_lock["selected_python"] == str(fake_selected_python)
    assert confirmation["current_confirmation"] is None
    assert verdict["latest_cache_ref"] == {
        "root": "runs/latest/new-readiness-agent",
        "lock": "runs/latest/new-readiness-agent/workspace-readiness.lock.json",
        "confirmation": "runs/latest/new-readiness-agent/confirmation-latest.json",
        "run_ref": "runs/latest/new-readiness-agent/run-ref.json",
    }
    compatibility_check = check_by_id(verdict, "framework-compatibility")
    assert compatibility_check["status"] == "ok"
    assert "match a local compatibility row" in compatibility_check["summary"]
    assert compatibility_check["details"]["installed_versions"]["torch"] == "2.9.0"
    report_markdown = (output_dir / "report.md").read_text(encoding="utf-8")
    assert f"- cann_path: `{cann_root}`" in report_markdown


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


def test_pipeline_blocks_when_framework_import_fails_at_runtime(tmp_path: Path):
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
    selected_python = make_fake_selected_python_with_import_error(tmp_path, "torch_npu", "ImportError: libhccl.so: cannot open shared object file")
    output_dir = tmp_path / "out"

    run_pipeline(
        "--working-dir",
        str(workspace),
        "--output-dir",
        str(output_dir),
        "--target",
        "training",
        "--framework-hint",
        "pta",
        "--launcher-hint",
        "python",
        "--selected-python",
        str(selected_python),
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
        "--launch-command",
        "python train.py --config train.yaml",
        cwd=workspace,
    )

    verdict = json.loads((output_dir / "meta" / "readiness-verdict.json").read_text(encoding="utf-8"))
    framework_importability = check_by_id(verdict, "framework-importability")
    runtime_dependencies = check_by_id(verdict, "runtime-dependencies")
    runtime_smoke = check_by_id(verdict, "runtime-smoke")

    assert verdict["status"] == "BLOCKED"
    assert verdict["can_run"] is False
    assert framework_importability["status"] == "block"
    assert "libhccl.so" in framework_importability["summary"]
    assert framework_importability["import_errors"]["torch_npu"] == "ImportError: libhccl.so: cannot open shared object file"
    assert runtime_dependencies["status"] == "block"
    assert runtime_smoke["status"] == "block"


def test_pipeline_advances_one_confirmation_step_at_a_time(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    output_dir_1 = tmp_path / "out1"
    output_dir_2 = tmp_path / "out2"
    output_dir_3 = tmp_path / "out3"

    first = stdout_payload(
        run_pipeline(
            "--working-dir",
            str(workspace),
            "--output-dir",
            str(output_dir_1),
            cwd=workspace,
        )
    )
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

    latest_root = workspace / "runs" / "latest" / "new-readiness-agent"
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


def test_pipeline_surfaces_hf_asset_options_for_script_managed_dataset(tmp_path: Path, fake_selected_python: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "train_qwen3.py").write_text(
        "\n".join(
            [
                "from datasets import load_dataset",
                "from transformers import AutoModelForCausalLM, TrainingArguments",
                'TrainingArguments(output_dir="qwen3-finetuned")',
                'dataset = load_dataset("karthiksagarn/astro_horoscope", split="train")',
                'model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (workspace / "huggingface-cache" / "datasets" / "karthiksagarn___astro_horoscope").mkdir(parents=True)

    summary = stdout_payload(
        run_pipeline(
            "--working-dir",
            str(workspace),
            "--output-dir",
            str(tmp_path / "out"),
            "--target",
            "training",
            "--framework-hint",
            "pta",
            "--launcher-hint",
            "python",
            "--selected-python",
            str(fake_selected_python),
            "--entry-script",
            "train_qwen3.py",
            "--confirm",
            "config_asset=inline_config",
            "--confirm",
            "model_asset=hf_hub:Qwen/Qwen3-0.6B",
            cwd=workspace,
        )
    )

    assert current_field(summary) == "dataset_asset"
    current_confirmation = summary["current_confirmation"]
    options = current_confirmation["options"]
    source_types = {str(option.get("source_type")) for option in options if option.get("source_type")}
    assert "hf_cache" in source_types
    assert "hf_hub" in source_types
    assert "script_managed_remote" in source_types
    assert any((option.get("locator") or {}).get("repo_id") == "karthiksagarn/astro_horoscope" for option in options if isinstance(option.get("locator"), dict))


def test_pipeline_treats_hf_cache_dataset_as_satisfied_in_final_verdict(tmp_path: Path, fake_selected_python: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "train_qwen3.py").write_text(
        "\n".join(
            [
                "from datasets import load_dataset",
                "from transformers import AutoModelForCausalLM, TrainingArguments",
                'TrainingArguments(output_dir="qwen3-finetuned")',
                'dataset = load_dataset("karthiksagarn/astro_horoscope", split="train")',
                'model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (workspace / "huggingface-cache" / "datasets" / "karthiksagarn___astro_horoscope").mkdir(parents=True)
    (workspace / "huggingface-cache" / "hub" / "models--Qwen--Qwen3-0.6B").mkdir(parents=True)
    cann_root = tmp_path / "cann"
    cann_root.mkdir()
    (cann_root / "version.cfg").write_text("version=8.5.0\n", encoding="utf-8")
    output_dir = tmp_path / "out"

    run_pipeline(
        "--working-dir",
        str(workspace),
        "--output-dir",
        str(output_dir),
        "--target",
        "training",
        "--framework-hint",
        "pta",
        "--launcher-hint",
        "python",
        "--selected-python",
        str(fake_selected_python),
        "--entry-script",
        "train_qwen3.py",
        "--model-hub-id",
        "Qwen/Qwen3-0.6B",
        "--dataset-hub-id",
        "karthiksagarn/astro_horoscope",
        "--cann-path",
        str(cann_root),
        "--launch-command",
        "python train_qwen3.py",
        "--confirm",
        "config_asset=inline_config",
        cwd=workspace,
    )

    verdict = json.loads((output_dir / "meta" / "readiness-verdict.json").read_text(encoding="utf-8"))
    dataset_check = check_by_id(verdict, "workspace-dataset-asset")
    model_check = check_by_id(verdict, "workspace-model-asset")

    assert verdict["status"] in {"WARN", "READY"}
    assert dataset_check["status"] == "ok"
    assert model_check["status"] == "ok"
    assert not any("dataset asset is required but unresolved" in item for item in verdict["missing_items"])


def test_pipeline_treats_inline_config_as_a_valid_asset(tmp_path: Path, fake_selected_python: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "train.py").write_text(
        "\n".join(
            [
                "from transformers import TrainingArguments",
                'TrainingArguments(output_dir="out")',
                "print('train')",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (workspace / "model").mkdir()
    (workspace / "dataset").mkdir()
    (workspace / "dataset" / "sample.txt").write_text("hello\n", encoding="utf-8")
    cann_root = tmp_path / "cann"
    cann_root.mkdir()
    (cann_root / "version.cfg").write_text("version=8.5.0\n", encoding="utf-8")
    output_dir = tmp_path / "out"

    run_pipeline(
        "--working-dir",
        str(workspace),
        "--output-dir",
        str(output_dir),
        "--target",
        "training",
        "--framework-hint",
        "pta",
        "--launcher-hint",
        "python",
        "--selected-python",
        str(fake_selected_python),
        "--entry-script",
        "train.py",
        "--model-path",
        "model",
        "--dataset-path",
        "dataset",
        "--cann-path",
        str(cann_root),
        "--launch-command",
        "python train.py",
        "--confirm",
        "config_asset=inline_config",
        cwd=workspace,
    )

    verdict = json.loads((output_dir / "meta" / "readiness-verdict.json").read_text(encoding="utf-8"))
    config_check = check_by_id(verdict, "workspace-config-asset")
    assert config_check["status"] == "ok"
