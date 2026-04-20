import json
import os
from pathlib import Path

import pytest

from runtime_env import detect_ascend_runtime

from .helpers import (
    check_by_id,
    make_fake_selected_python_requiring_runtime_env,
    make_fake_selected_python_with_import_error,
    make_fake_selected_python_with_torch_autoload_conflict,
    run_pipeline,
)


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


def test_pipeline_sources_cann_script_on_top_of_selected_runtime_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    if os.name == "nt":
        pytest.skip("set_env.sh sourcing regression is Linux-specific")

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "train.py").write_text("import torch\nimport torch_npu\n", encoding="utf-8")
    (workspace / "train.yaml").write_text("model_name_or_path: model\ntrain_file: dataset/sample.txt\n", encoding="utf-8")
    (workspace / "model").mkdir()
    (workspace / "dataset").mkdir()
    (workspace / "dataset" / "sample.txt").write_text("hello\n", encoding="utf-8")

    fake_env_root = tmp_path / "selected-env"
    (fake_env_root / "bin").mkdir(parents=True)
    selected_python = make_fake_selected_python_requiring_runtime_env(tmp_path, "FAKE_ASCEND_READY", "1")

    cann_root = tmp_path / "cann"
    toolkit_root = cann_root / "ascend-toolkit"
    toolkit_root.mkdir(parents=True)
    (toolkit_root / "set_env.sh").write_text(
        "\n".join(
            [
                "export FAKE_ASCEND_READY=1",
                "export ASCEND_HOME_PATH=/fake/ascend",
                "export ASCEND_OPP_PATH=/fake/ascend/opp",
                "export LD_LIBRARY_PATH=/fake/ascend/lib:${LD_LIBRARY_PATH}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (cann_root / "version.cfg").write_text("version=8.5.0\n", encoding="utf-8")

    monkeypatch.setenv("ASCEND_HOME_PATH", "/stale/ascend")
    monkeypatch.setenv("ASCEND_OPP_PATH", "/stale/ascend/opp")

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
        "--selected-env-root",
        str(fake_env_root),
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
    framework_importability = check_by_id(verdict, "framework-importability")
    compatibility_check = check_by_id(verdict, "framework-compatibility")

    assert framework_importability["status"] == "ok"
    assert compatibility_check["status"] == "ok"
    assert verdict["evidence_summary"]["package_versions"]["torch"] == "2.8.0"
    assert verdict["evidence_summary"]["package_versions"]["torch_npu"] == "2.8.0.post2"


def test_pipeline_probes_pta_packages_in_isolated_processes(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "train.py").write_text("import torch\nimport torch_npu\n", encoding="utf-8")
    (workspace / "train.yaml").write_text("model_name_or_path: model\ntrain_file: dataset/sample.txt\n", encoding="utf-8")
    (workspace / "model").mkdir()
    (workspace / "dataset").mkdir()
    (workspace / "dataset" / "sample.txt").write_text("hello\n", encoding="utf-8")
    cann_root = tmp_path / "cann"
    cann_root.mkdir()
    (cann_root / "version.cfg").write_text("version=8.5.0\n", encoding="utf-8")
    selected_python = make_fake_selected_python_with_torch_autoload_conflict(tmp_path)
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
        cwd=workspace,
    )

    verdict = json.loads((output_dir / "meta" / "readiness-verdict.json").read_text(encoding="utf-8"))
    framework_importability = check_by_id(verdict, "framework-importability")
    compatibility_check = check_by_id(verdict, "framework-compatibility")

    assert verdict["status"] == "READY"
    assert framework_importability["status"] == "ok"
    assert compatibility_check["status"] == "ok"


def test_detect_ascend_runtime_accepts_explicit_set_env_script_path(tmp_path: Path):
    script_path = tmp_path / "cann" / "ascend-toolkit" / "set_env.sh"
    script_path.parent.mkdir(parents=True)
    script_path.write_text("export ASCEND_HOME_PATH=/fake/ascend\n", encoding="utf-8")

    system_layer = detect_ascend_runtime({"cann_path": str(script_path)})

    assert system_layer["ascend_env_script_present"] is True
    assert system_layer["ascend_env_script_path"] == str(script_path)
    assert system_layer["ascend_env_selection_source"] == "explicit_cann_path"
