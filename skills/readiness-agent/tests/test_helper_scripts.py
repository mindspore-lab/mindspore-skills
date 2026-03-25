import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
READINESS_VERDICT_REF = Path("meta/readiness-verdict.json")


def run_script(script_name: str, *args: str, env: Optional[dict] = None) -> subprocess.CompletedProcess[str]:
    script = SCRIPTS / script_name
    return subprocess.run(
        [sys.executable, str(script), *args],
        check=True,
        text=True,
        capture_output=True,
        env=env,
    )


def load_report_pair(report_json: Path) -> tuple[dict, dict]:
    envelope = json.loads(report_json.read_text(encoding="utf-8"))
    verdict_json = report_json.parent / READINESS_VERDICT_REF
    verdict = json.loads(verdict_json.read_text(encoding="utf-8"))
    return envelope, verdict


def test_discover_execution_target_finds_training_script(tmp_path: Path):
    (tmp_path / "train.py").write_text(
        "import mindspore as ms\nfrom dataset import build_dataset\noptimizer = object()\n",
        encoding="utf-8",
    )
    (tmp_path / "train.yaml").write_text("epochs: 1\n", encoding="utf-8")
    output = tmp_path / "target.json"

    run_script(
        "discover_execution_target.py",
        "--working-dir",
        str(tmp_path),
        "--output-json",
        str(output),
    )
    target = json.loads(output.read_text(encoding="utf-8"))
    assert target["target_type"] == "training"
    assert target["entry_script"] == "train.py"
    assert target["framework_path"] == "mindspore"
    assert target["confidence"] in {"medium", "high"}


def test_resolve_selected_python_prefers_explicit_python(tmp_path: Path):
    output = tmp_path / "selected-python.json"

    run_script(
        "resolve_selected_python.py",
        "--working-dir",
        str(tmp_path),
        "--selected-python",
        sys.executable,
        "--output-json",
        str(output),
    )
    selected = json.loads(output.read_text(encoding="utf-8"))
    assert selected["selection_status"] == "selected"
    assert selected["selection_source"] == "explicit_python"
    assert selected["selected_python"] == sys.executable
    assert selected["helper_python_compatible"] is True


def test_discover_execution_target_preserves_task_smoke_cmd(tmp_path: Path):
    (tmp_path / "infer.py").write_text(
        "import torch\nimport torch_npu\n",
        encoding="utf-8",
    )
    output = tmp_path / "target.json"

    run_script(
        "discover_execution_target.py",
        "--working-dir",
        str(tmp_path),
        "--target",
        "inference",
        "--task-smoke-cmd",
        "python infer.py --smoke-test",
        "--output-json",
        str(output),
    )
    target = json.loads(output.read_text(encoding="utf-8"))
    assert target["task_smoke_cmd"] == "python infer.py --smoke-test"


def test_discover_execution_target_records_selected_python(tmp_path: Path):
    (tmp_path / "infer.py").write_text(
        "import torch\nimport torch_npu\n",
        encoding="utf-8",
    )
    output = tmp_path / "target.json"

    run_script(
        "discover_execution_target.py",
        "--working-dir",
        str(tmp_path),
        "--selected-python",
        sys.executable,
        "--output-json",
        str(output),
    )
    target = json.loads(output.read_text(encoding="utf-8"))
    assert target["selected_python"] == sys.executable
    assert target["selected_python_status"] == "selected"


def test_discover_execution_target_infers_model_path_from_markers(tmp_path: Path):
    (tmp_path / "infer.py").write_text(
        "import torch\nimport torch_npu\nmodel_path = './models/Qwen3.5-0.8B'\n",
        encoding="utf-8",
    )
    model_dir = tmp_path / "models" / "Qwen3.5-0.8B"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    (model_dir / "model.safetensors").write_text("", encoding="utf-8")
    output = tmp_path / "target.json"

    run_script(
        "discover_execution_target.py",
        "--working-dir",
        str(tmp_path),
        "--output-json",
        str(output),
    )
    target = json.loads(output.read_text(encoding="utf-8"))
    assert target["model_path"] == "models/Qwen3.5-0.8B"
    assert "model path inferred from workspace model markers" in target["evidence"]


def test_normalize_blockers_maps_categories(tmp_path: Path):
    checks_path = tmp_path / "checks.json"
    checks_path.write_text(
        json.dumps(
            [
                {
                    "id": "uv-missing",
                    "status": "block",
                    "summary": "uv is missing",
                    "category_hint": "env",
                    "revalidation_scope": ["tool-resolution"],
                },
                {
                    "id": "target-ambiguous",
                    "status": "warn",
                    "summary": "multiple target candidates remain",
                    "evidence": ["train.py", "infer.py"],
                },
            ]
        ),
        encoding="utf-8",
    )
    output = tmp_path / "normalized.json"

    run_script(
        "normalize_blockers.py",
        "--input-json",
        str(checks_path),
        "--output-json",
        str(output),
    )
    normalized = json.loads(output.read_text(encoding="utf-8"))
    assert normalized["blockers"] == ["uv is missing"]
    assert normalized["warnings"] == ["multiple target candidates remain"]
    assert normalized["blockers_detailed"][0]["category"] == "env_remediable"
    assert normalized["blockers_detailed"][0]["remediable"] is True


def test_build_dependency_closure_tracks_required_assets(tmp_path: Path):
    (tmp_path / "infer.py").write_text(
        "import torch\nimport torch_npu\nimport transformers\n",
        encoding="utf-8",
    )
    (tmp_path / "model").mkdir()
    target_path = tmp_path / "target.json"
    closure_path = tmp_path / "closure.json"
    target_path.write_text(
        json.dumps(
            {
                "working_dir": str(tmp_path),
                "target_type": "inference",
                "entry_script": "infer.py",
                "framework_path": "pta",
                "selected_python": sys.executable,
                "model_path": "model",
                "config_path": None,
                "dataset_path": None,
                "checkpoint_path": None,
                "launch_cmd": "python infer.py",
            }
        ),
        encoding="utf-8",
    )

    run_script(
        "build_dependency_closure.py",
        "--target-json",
        str(target_path),
        "--output-json",
        str(closure_path),
    )
    closure = json.loads(closure_path.read_text(encoding="utf-8"))
    assert closure["target_type"] == "inference"
    assert closure["layers"]["framework"]["framework_path"] == "pta"
    assert closure["layers"]["python_environment"]["probe_source"] == "explicit_python"
    assert closure["layers"]["python_environment"]["probe_python_path"] == sys.executable
    assert "import_probes" in closure["layers"]["framework"]
    assert "torch" in closure["layers"]["runtime_dependencies"]["required_imports"]
    assert "import_probes" in closure["layers"]["runtime_dependencies"]
    assert closure["layers"]["workspace_assets"]["entry_script"]["exists"] is True
    assert closure["layers"]["workspace_assets"]["model_path"]["exists"] is True
    assert closure["complete_for_static_validation"] is True


def test_build_dependency_closure_prefers_selected_env_probe_python(tmp_path: Path):
    (tmp_path / "infer.py").write_text(
        "import torch\nimport torch_npu\nimport transformers\n",
        encoding="utf-8",
    )
    (tmp_path / "model").mkdir()
    fake_python = tmp_path / ".venv" / "bin" / "python"
    fake_python.parent.mkdir(parents=True)
    fake_python.write_text(
        """#!/usr/bin/env python3
import json
import sys

mode = sys.argv[3]
payload = json.loads(sys.argv[4])

if mode == "import":
    available = {"torch", "torch_npu"}
    requested = payload.get("packages", [])
    print(json.dumps({name: name in available for name in requested}))
elif mode == "framework_smoke":
    print(json.dumps({"success": True, "details": ["pta smoke ok"], "error": None}))
else:
    print(json.dumps({"error": "unexpected mode"}))
""",
        encoding="utf-8",
    )
    fake_python.chmod(fake_python.stat().st_mode | 0o111)

    target_path = tmp_path / "target.json"
    closure_path = tmp_path / "closure.json"
    target_path.write_text(
        json.dumps(
            {
                "working_dir": str(tmp_path),
                "target_type": "inference",
                "entry_script": "infer.py",
                "framework_path": "pta",
                "model_path": "model",
                "launch_cmd": "python infer.py",
            }
        ),
        encoding="utf-8",
    )

    run_script(
        "build_dependency_closure.py",
        "--target-json",
        str(target_path),
        "--output-json",
        str(closure_path),
    )
    closure = json.loads(closure_path.read_text(encoding="utf-8"))
    python_env = closure["layers"]["python_environment"]
    framework = closure["layers"]["framework"]
    runtime = closure["layers"]["runtime_dependencies"]

    assert python_env["selected_env_root"] == str(tmp_path / ".venv")
    assert python_env["probe_source"] == "workspace_env"
    assert python_env["probe_python_path"] == str(fake_python)
    assert framework["import_probes"] == {"torch": True, "torch_npu": True}
    assert framework["smoke_prerequisite"]["status"] == "passed"
    assert "pta smoke ok" in framework["smoke_prerequisite"]["details"]
    assert runtime["import_probes"]["transformers"] is False


def test_build_dependency_closure_sources_detected_ascend_env_for_pta_probe(tmp_path: Path):
    if os.name == "nt" or shutil.which("bash") is None:
        return

    (tmp_path / "infer.py").write_text(
        "import torch\nimport torch_npu\n",
        encoding="utf-8",
    )
    (tmp_path / "model").mkdir()

    fake_python = tmp_path / ".venv" / "bin" / "python"
    fake_python.parent.mkdir(parents=True)
    fake_python.write_text(
        """#!/usr/bin/env python3
import json
import os
import subprocess
import sys

if len(sys.argv) >= 3 and sys.argv[1] == "-c":
    code = sys.argv[2]
    if "platform.python_version" in code and "version_info" in code:
        print(json.dumps({"version_info": [3, 10, 0], "version": "3.10.0"}))
        raise SystemExit(0)
    mode = sys.argv[3]
    payload = json.loads(sys.argv[4])
    runtime_ready = os.environ.get("FAKE_ASCEND_READY") == "1"
    if mode == "import":
        packages = payload.get("packages", [])
        available = {"torch", "torch_npu"} if runtime_ready else {"torch"}
        print(json.dumps({name: name in available for name in packages}))
        raise SystemExit(0)
    if mode == "framework_smoke":
        print(json.dumps({"success": runtime_ready, "details": ["runtime ready"] if runtime_ready else [], "error": None if runtime_ready else "missing runtime"}))
        raise SystemExit(0)

completed = subprocess.run([sys.executable, *sys.argv[1:]])
raise SystemExit(completed.returncode)
""",
        encoding="utf-8",
    )
    fake_python.chmod(fake_python.stat().st_mode | 0o111)

    ascend_home = tmp_path / "ascend"
    ascend_home.mkdir()
    (ascend_home / "set_env.sh").write_text(
        "export FAKE_ASCEND_READY=1\n",
        encoding="utf-8",
    )

    target_path = tmp_path / "target.json"
    closure_path = tmp_path / "closure.json"
    target_path.write_text(
        json.dumps(
            {
                "working_dir": str(tmp_path),
                "target_type": "inference",
                "entry_script": "infer.py",
                "framework_path": "pta",
                "model_path": "model",
                "launch_cmd": "python infer.py",
            }
        ),
        encoding="utf-8",
    )

    env = dict(os.environ)
    env["ASCEND_HOME_PATH"] = str(ascend_home)

    run_script(
        "build_dependency_closure.py",
        "--target-json",
        str(target_path),
        "--output-json",
        str(closure_path),
        env=env,
    )
    closure = json.loads(closure_path.read_text(encoding="utf-8"))
    system = closure["layers"]["system"]
    framework = closure["layers"]["framework"]

    assert system["ascend_env_script_present"] is True
    assert system["ascend_env_script_path"] == str(ascend_home / "set_env.sh")
    assert system["probe_env_source"] == "sourced_script"
    assert framework["import_probes"]["torch_npu"] is True
    assert framework["smoke_prerequisite"]["status"] == "passed"


def test_collect_readiness_checks_flags_missing_uv_and_missing_entry(tmp_path: Path):
    target_path = tmp_path / "target.json"
    closure_path = tmp_path / "closure.json"
    checks_path = tmp_path / "checks.json"

    target_path.write_text(
        json.dumps(
            {
                "target_type": "training",
                "working_dir": str(tmp_path),
            }
        ),
        encoding="utf-8",
    )
    closure_path.write_text(
        json.dumps(
            {
                "layers": {
                    "system": {
                        "requires_ascend": False,
                    },
                    "python_environment": {
                        "tooling": {
                            "uv_available": False,
                            "uv_path": None,
                        }
                    },
                    "framework": {
                        "framework_path": "unknown",
                        "required_packages": [],
                        "import_probes": {},
                    },
                    "runtime_dependencies": {
                        "required_imports": [],
                        "import_probes": {},
                    },
                    "workspace_assets": {
                        "entry_script": {"required": True, "exists": False},
                        "model_path": {"required": True, "exists": True},
                        "dataset_path": {"required": True, "exists": False},
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    run_script(
        "collect_readiness_checks.py",
        "--target-json",
        str(target_path),
        "--closure-json",
        str(closure_path),
        "--output-json",
        str(checks_path),
    )
    checks = json.loads(checks_path.read_text(encoding="utf-8"))
    by_id = {item["id"]: item for item in checks}
    assert by_id["python-uv"]["status"] == "block"
    assert by_id["python-uv"]["category_hint"] == "env"
    assert by_id["workspace-entry_script"]["status"] == "block"
    assert by_id["workspace-entry_script"]["category_hint"] == "workspace"


def test_collect_readiness_checks_flags_missing_framework_packages(tmp_path: Path):
    target_path = tmp_path / "target.json"
    closure_path = tmp_path / "closure.json"
    checks_path = tmp_path / "checks.json"

    target_path.write_text(
        json.dumps(
            {
                "target_type": "inference",
                "working_dir": str(tmp_path),
            }
        ),
        encoding="utf-8",
    )
    closure_path.write_text(
        json.dumps(
            {
                "layers": {
                    "system": {
                        "requires_ascend": False,
                    },
                    "python_environment": {
                        "tooling": {
                            "uv_available": True,
                            "uv_path": "/usr/bin/uv",
                        }
                    },
                    "framework": {
                        "framework_path": "mindspore",
                        "required_packages": ["mindspore"],
                        "import_probes": {
                            "mindspore": False,
                        },
                    },
                    "runtime_dependencies": {
                        "required_imports": ["transformers"],
                        "import_probes": {
                            "transformers": True,
                        },
                    },
                    "workspace_assets": {
                        "entry_script": {"required": True, "exists": True},
                        "model_path": {"required": True, "exists": True},
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    run_script(
        "collect_readiness_checks.py",
        "--target-json",
        str(target_path),
        "--closure-json",
        str(closure_path),
        "--output-json",
        str(checks_path),
    )
    checks = json.loads(checks_path.read_text(encoding="utf-8"))
    by_id = {item["id"]: item for item in checks}
    assert by_id["framework-importability"]["status"] == "block"
    assert by_id["framework-importability"]["category_hint"] == "framework"


def test_plan_env_fix_filters_non_package_framework_evidence(tmp_path: Path):
    blockers_path = tmp_path / "blockers.json"
    closure_path = tmp_path / "closure.json"
    output = tmp_path / "plan.json"

    blockers_path.write_text(
        json.dumps(
            {
                "blockers_detailed": [
                    {
                        "id": "framework-importability",
                        "category": "framework_remediable",
                        "summary": "Required framework packages are unavailable in the selected Python interpreter: torch, torch_npu.",
                        "evidence": [
                            "probe_source=workspace_env",
                            "torch",
                            "torch_npu",
                            "probe_error=probe python path is unavailable",
                        ],
                        "remediable": True,
                        "revalidation_scope": ["framework", "task-smoke"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    closure_path.write_text(
        json.dumps(
            {
                "layers": {
                    "framework": {
                        "framework_path": "pta",
                        "required_packages": ["torch", "torch_npu"],
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    run_script(
        "plan_env_fix.py",
        "--blockers-json",
        str(blockers_path),
        "--closure-json",
        str(closure_path),
        "--output-json",
        str(output),
    )
    plan = json.loads(output.read_text(encoding="utf-8"))
    assert plan["actions"][0]["action_type"] == "repair_pta_framework"
    assert plan["actions"][0]["package_names"] == ["torch", "torch_npu"]
    assert "mindspore" in by_id["framework-importability"]["summary"]


def test_collect_readiness_checks_flags_unusable_selected_env(tmp_path: Path):
    target_path = tmp_path / "target.json"
    closure_path = tmp_path / "closure.json"
    checks_path = tmp_path / "checks.json"

    target_path.write_text(
        json.dumps(
            {
                "target_type": "inference",
                "working_dir": str(tmp_path),
            }
        ),
        encoding="utf-8",
    )
    closure_path.write_text(
        json.dumps(
            {
                "layers": {
                    "system": {
                        "requires_ascend": False,
                    },
                    "python_environment": {
                        "tooling": {
                            "uv_available": True,
                            "uv_path": "/usr/bin/uv",
                        },
                        "selected_env_root": str(tmp_path / ".venv"),
                        "selected_python": None,
                        "selection_status": "missing",
                        "selection_reason": "selected environment root does not contain a Python executable",
                        "probe_source": "explicit_env",
                        "probe_python_path": None,
                    },
                    "framework": {
                        "framework_path": "pta",
                        "required_packages": ["torch", "torch_npu"],
                        "import_probes": {
                            "torch": False,
                            "torch_npu": False,
                        },
                        "probe_source": "explicit_env",
                    },
                    "runtime_dependencies": {
                        "required_imports": ["transformers"],
                        "import_probes": {
                            "transformers": False,
                        },
                        "probe_source": "explicit_env",
                    },
                    "workspace_assets": {
                        "entry_script": {"required": True, "exists": True},
                        "model_path": {"required": True, "exists": True},
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    run_script(
        "collect_readiness_checks.py",
        "--target-json",
        str(target_path),
        "--closure-json",
        str(closure_path),
        "--output-json",
        str(checks_path),
    )
    checks = json.loads(checks_path.read_text(encoding="utf-8"))
    by_id = {item["id"]: item for item in checks}
    assert by_id["python-selected-python"]["status"] == "block"
    assert by_id["python-selected-python"]["category_hint"] == "env"
    assert "selected environment" in by_id["framework-importability"]["summary"].lower()


def test_collect_readiness_checks_flags_framework_smoke_failure(tmp_path: Path):
    target_path = tmp_path / "target.json"
    closure_path = tmp_path / "closure.json"
    checks_path = tmp_path / "checks.json"

    target_path.write_text(
        json.dumps(
            {
                "target_type": "inference",
                "working_dir": str(tmp_path),
            }
        ),
        encoding="utf-8",
    )
    closure_path.write_text(
        json.dumps(
            {
                "layers": {
                    "system": {
                        "requires_ascend": False,
                    },
                    "python_environment": {
                        "tooling": {
                            "uv_available": True,
                            "uv_path": "/usr/bin/uv",
                        },
                        "selected_env_root": str(tmp_path / ".venv"),
                        "selected_python": str(tmp_path / ".venv" / "bin" / "python"),
                        "selection_status": "selected",
                        "selection_reason": "selected python is usable for readiness-agent helpers",
                        "probe_source": "workspace_env",
                        "probe_python_path": str(tmp_path / ".venv" / "bin" / "python"),
                    },
                    "framework": {
                        "framework_path": "pta",
                        "required_packages": ["torch", "torch_npu"],
                        "import_probes": {
                            "torch": True,
                            "torch_npu": True,
                        },
                        "probe_source": "workspace_env",
                        "smoke_prerequisite": {
                            "status": "failed",
                            "details": [],
                            "error": "ImportError: torch_npu bootstrap failed",
                        },
                    },
                    "runtime_dependencies": {
                        "required_imports": ["transformers"],
                        "import_probes": {
                            "transformers": True,
                        },
                        "probe_source": "workspace_env",
                    },
                    "workspace_assets": {
                        "entry_script": {"required": True, "exists": True},
                        "model_path": {"required": True, "exists": True},
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    run_script(
        "collect_readiness_checks.py",
        "--target-json",
        str(target_path),
        "--closure-json",
        str(closure_path),
        "--output-json",
        str(checks_path),
    )
    checks = json.loads(checks_path.read_text(encoding="utf-8"))
    by_id = {item["id"]: item for item in checks}
    assert by_id["framework-smoke-prerequisite"]["status"] == "block"
    assert by_id["framework-smoke-prerequisite"]["category_hint"] == "framework"
    assert "selected environment" in by_id["framework-smoke-prerequisite"]["summary"].lower()


def test_run_task_smoke_executes_explicit_command(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "infer.py").write_text("print('ok')\n", encoding="utf-8")
    target_path = tmp_path / "target.json"
    closure_path = tmp_path / "closure.json"
    smoke_path = tmp_path / "task-smoke.json"

    target_path.write_text(
        json.dumps(
            {
                "working_dir": str(workspace),
                "target_type": "inference",
                "entry_script": "infer.py",
                "task_smoke_cmd": "python infer.py",
            }
        ),
        encoding="utf-8",
    )
    closure_path.write_text(
        json.dumps(
            {
                "layers": {
                    "python_environment": {
                        "probe_python_path": sys.executable,
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    run_script(
        "run_task_smoke.py",
        "--target-json",
        str(target_path),
        "--closure-json",
        str(closure_path),
        "--output-json",
        str(smoke_path),
    )
    checks = json.loads(smoke_path.read_text(encoding="utf-8"))
    by_id = {item["id"]: item for item in checks}
    assert by_id["task-smoke-script-parse"]["status"] == "ok"
    assert by_id["task-smoke-script-parse"]["command_preview"].endswith(" -m py_compile " + str(workspace / "infer.py"))
    assert by_id["task-smoke-script-parse"]["exit_code"] == 0
    assert by_id["task-smoke-script-parse"]["timed_out"] is False
    assert by_id["task-smoke-executed"]["status"] == "ok"
    assert "infer.py" in by_id["task-smoke-executed"]["command_preview"]
    assert by_id["task-smoke-executed"]["exit_code"] == 0
    assert by_id["task-smoke-executed"]["stdout_head"] == "ok"
    assert by_id["task-smoke-executed"]["timed_out"] is False


def test_run_task_smoke_failed_command_has_workspace_taxonomy(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "infer.py").write_text("print('ok')\n", encoding="utf-8")
    target_path = tmp_path / "target.json"
    closure_path = tmp_path / "closure.json"
    smoke_path = tmp_path / "task-smoke.json"
    normalized_path = tmp_path / "normalized.json"

    target_path.write_text(
        json.dumps(
            {
                "working_dir": str(workspace),
                "target_type": "inference",
                "entry_script": "infer.py",
                "task_smoke_cmd": "python -c \"import sys; sys.exit(3)\"",
            }
        ),
        encoding="utf-8",
    )
    closure_path.write_text(
        json.dumps(
            {
                "layers": {
                    "python_environment": {
                        "probe_python_path": sys.executable,
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    run_script(
        "run_task_smoke.py",
        "--target-json",
        str(target_path),
        "--closure-json",
        str(closure_path),
        "--output-json",
        str(smoke_path),
    )
    checks = json.loads(smoke_path.read_text(encoding="utf-8"))
    by_id = {item["id"]: item for item in checks}
    assert by_id["task-smoke-executed"]["status"] == "block"
    assert by_id["task-smoke-executed"]["category_hint"] == "workspace"
    assert by_id["task-smoke-executed"]["remediable"] is False
    assert by_id["task-smoke-executed"]["remediation_owner"] == "workspace"
    assert by_id["task-smoke-executed"]["exit_code"] == 3
    assert by_id["task-smoke-executed"]["timed_out"] is False
    assert "sys.exit(3)" in by_id["task-smoke-executed"]["command_preview"]


def test_run_task_smoke_timeout_records_structured_result(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "infer.py").write_text("print('ok')\n", encoding="utf-8")
    target_path = tmp_path / "target.json"
    closure_path = tmp_path / "closure.json"
    smoke_path = tmp_path / "task-smoke.json"
    normalized_path = tmp_path / "normalized.json"

    target_path.write_text(
        json.dumps(
            {
                "working_dir": str(workspace),
                "target_type": "inference",
                "entry_script": "infer.py",
                "task_smoke_cmd": "python -c \"import time; time.sleep(1.0)\"",
            }
        ),
        encoding="utf-8",
    )
    closure_path.write_text(
        json.dumps(
            {
                "layers": {
                    "python_environment": {
                        "probe_python_path": sys.executable,
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    run_script(
        "run_task_smoke.py",
        "--target-json",
        str(target_path),
        "--closure-json",
        str(closure_path),
        "--output-json",
        str(smoke_path),
        "--timeout-seconds",
        "0",
    )
    checks = json.loads(smoke_path.read_text(encoding="utf-8"))
    by_id = {item["id"]: item for item in checks}
    assert by_id["task-smoke-executed"]["status"] == "block"
    assert by_id["task-smoke-executed"]["timed_out"] is True
    assert by_id["task-smoke-executed"]["command_preview"]

    run_script(
        "normalize_blockers.py",
        "--input-json",
        str(smoke_path),
        "--output-json",
        str(normalized_path),
    )
    normalized = json.loads(normalized_path.read_text(encoding="utf-8"))
    blocker = next(item for item in normalized["blockers_detailed"] if item["id"] == "task-smoke-executed")
    assert blocker["category"] == "workspace_manual"


def test_plan_env_fix_maps_remediable_blockers_to_actions(tmp_path: Path):
    blockers_path = tmp_path / "normalized.json"
    closure_path = tmp_path / "closure.json"
    plan_path = tmp_path / "plan.json"

    blockers_path.write_text(
        json.dumps(
            {
                "blockers_detailed": [
                    {
                        "id": "uv-missing",
                        "category": "env_remediable",
                        "summary": "uv is missing",
                        "remediable": True,
                        "revalidation_scope": ["tool-resolution", "python-environment"],
                    },
                    {
                        "id": "framework-missing",
                        "category": "framework_remediable",
                        "summary": "mindspore package is missing",
                        "remediable": True,
                        "evidence": ["mindspore"],
                        "revalidation_scope": ["framework", "task-smoke"],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    closure_path.write_text(
        json.dumps(
            {
                "layers": {
                    "framework": {
                        "framework_path": "mindspore",
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    run_script(
        "plan_env_fix.py",
        "--blockers-json",
        str(blockers_path),
        "--closure-json",
        str(closure_path),
        "--output-json",
        str(plan_path),
    )
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert len(plan["actions"]) == 2
    assert plan["actions"][0]["action_type"] == "install_uv"
    assert plan["actions"][0]["requires_confirmation"] is True
    assert plan["actions"][1]["action_type"] == "repair_mindspore_framework"
    assert plan["actions"][1]["allowed"] is True
    assert plan["actions"][1]["package_names"] == ["mindspore"]


def test_execute_env_fix_supports_dry_run_and_path_repair(tmp_path: Path):
    plan_path = tmp_path / "plan.json"
    output_path = tmp_path / "result.json"
    profile_path = tmp_path / ".zshrc"
    plan_path.write_text(
        json.dumps(
            {
                "actions": [
                    {
                        "id": "action-1",
                        "action_type": "repair_uv_path",
                        "allowed": True,
                        "requires_confirmation": True,
                        "reason": "uv exists but path is incomplete",
                        "revalidation_scope": ["tool-resolution"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    run_script(
        "execute_env_fix.py",
        "--plan-json",
        str(plan_path),
        "--output-json",
        str(output_path),
    )
    dry_run = json.loads(output_path.read_text(encoding="utf-8"))
    assert dry_run["results"][0]["status"] == "planned"

    run_script(
        "execute_env_fix.py",
        "--plan-json",
        str(plan_path),
        "--output-json",
        str(output_path),
        "--execute",
        "--confirm-path-edit",
        "--path-profile",
        str(profile_path),
    )
    executed = json.loads(output_path.read_text(encoding="utf-8"))
    assert executed["results"][0]["status"] == "executed"
    assert "PATH" in profile_path.read_text(encoding="utf-8")
    assert executed["needs_revalidation"] == ["tool-resolution"]


def test_execute_env_fix_retains_package_level_framework_repair_in_dry_run(tmp_path: Path):
    plan_path = tmp_path / "plan.json"
    output_path = tmp_path / "result.json"
    plan_path.write_text(
        json.dumps(
            {
                "actions": [
                    {
                        "id": "action-1",
                        "action_type": "repair_mindspore_framework",
                        "allowed": True,
                        "requires_confirmation": True,
                        "reason": "MindSpore packages are missing",
                        "revalidation_scope": ["framework", "task-smoke"],
                        "package_names": ["mindspore"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    run_script(
        "execute_env_fix.py",
        "--plan-json",
        str(plan_path),
        "--output-json",
        str(output_path),
    )
    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["results"][0]["status"] == "planned"
    assert "framework package_names" in result["results"][0]["command_preview"]


def test_build_readiness_report_emits_ready_for_strong_evidence(tmp_path: Path):
    target_path = tmp_path / "target.json"
    normalized_path = tmp_path / "normalized.json"
    checks_path = tmp_path / "checks.json"
    closure_path = tmp_path / "closure.json"
    fix_path = tmp_path / "fix.json"
    report_json = tmp_path / "report.json"
    report_md = tmp_path / "report.md"

    target_path.write_text(
        json.dumps(
            {
                "target_type": "training",
                "entry_script": "train.py",
                "framework_path": "mindspore",
            }
        ),
        encoding="utf-8",
    )
    normalized_path.write_text(
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
    checks_path.write_text(
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
                    "summary": "framework packages importable",
                },
                {
                    "id": "framework-smoke-prerequisite",
                    "status": "ok",
                    "summary": "framework smoke prerequisite passed",
                },
            ]
        ),
        encoding="utf-8",
    )
    closure_path.write_text(
        json.dumps(
            {
                "working_dir": str(tmp_path),
                "target_type": "training",
                "layers": {
                    "framework": {
                        "framework_path": "mindspore",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    fix_path.write_text(
        json.dumps(
            {
                "execute": True,
                "results": [
                    {
                        "action_id": "action-1",
                        "status": "executed",
                    }
                ],
                "executed_actions": ["action-1"],
                "failed_actions": [],
                "needs_revalidation": ["framework"],
            }
        ),
        encoding="utf-8",
    )

    run_script(
        "build_readiness_report.py",
        "--target-json",
        str(target_path),
        "--normalized-json",
        str(normalized_path),
        "--checks-json",
        str(checks_path),
        "--closure-json",
        str(closure_path),
        "--fix-applied-json",
        str(fix_path),
        "--output-json",
        str(report_json),
        "--output-md",
        str(report_md),
    )
    envelope, verdict = load_report_pair(report_json)
    assert envelope["schema_version"] == "1.0.0"
    assert envelope["status"] == "success"
    assert "meta/readiness-verdict.json" in envelope["artifacts"]
    assert verdict["status"] == "READY"
    assert verdict["can_run"] is True
    assert verdict["target"] == "training"
    assert verdict["evidence_level"] == "runtime_smoke"
    assert len(verdict["checks"]) == 3
    assert verdict["dependency_closure"]["layers"]["framework"]["framework_path"] == "mindspore"
    assert verdict["fix_applied"]["executed_actions"] == ["action-1"]
    assert verdict["revalidated"] is True
    assert verdict["revalidation_required_scopes"] == ["framework"]
    assert verdict["revalidation_covered_scopes"] == ["framework", "target"]
    assert "ready for training" in verdict["summary"].lower()


def test_build_readiness_report_uses_explicit_evidence_level_override(tmp_path: Path):
    target_path = tmp_path / "target.json"
    normalized_path = tmp_path / "normalized.json"
    checks_path = tmp_path / "checks.json"
    report_json = tmp_path / "report.json"
    report_md = tmp_path / "report.md"

    target_path.write_text(
        json.dumps(
            {
                "target_type": "inference",
                "entry_script": "infer.py",
                "framework_path": "pta",
            }
        ),
        encoding="utf-8",
    )
    normalized_path.write_text(
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
    checks_path.write_text(
        json.dumps(
            [
                {
                    "id": "framework-smoke-prerequisite",
                    "status": "ok",
                    "summary": "framework smoke prerequisite passed",
                }
            ]
        ),
        encoding="utf-8",
    )

    run_script(
        "build_readiness_report.py",
        "--target-json",
        str(target_path),
        "--normalized-json",
        str(normalized_path),
        "--checks-json",
        str(checks_path),
        "--output-json",
        str(report_json),
        "--output-md",
        str(report_md),
        "--evidence-level",
        "task_smoke",
    )
    envelope, verdict = load_report_pair(report_json)
    assert envelope["status"] == "success"
    assert verdict["status"] == "READY"
    assert verdict["evidence_level"] == "task_smoke"


def test_build_readiness_report_downgrades_when_explicit_task_smoke_missing(tmp_path: Path):
    target_path = tmp_path / "target.json"
    normalized_path = tmp_path / "normalized.json"
    checks_path = tmp_path / "checks.json"
    report_json = tmp_path / "report.json"
    report_md = tmp_path / "report.md"

    target_path.write_text(
        json.dumps(
            {
                "target_type": "inference",
                "entry_script": "infer.py",
                "framework_path": "pta",
                "task_smoke_cmd": "python infer.py --smoke-test",
            }
        ),
        encoding="utf-8",
    )
    normalized_path.write_text(
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
    checks_path.write_text(
        json.dumps(
            [
                {
                    "id": "framework-smoke-prerequisite",
                    "status": "ok",
                    "summary": "framework smoke prerequisite passed",
                }
            ]
        ),
        encoding="utf-8",
    )

    run_script(
        "build_readiness_report.py",
        "--target-json",
        str(target_path),
        "--normalized-json",
        str(normalized_path),
        "--checks-json",
        str(checks_path),
        "--output-json",
        str(report_json),
        "--output-md",
        str(report_md),
    )
    envelope, verdict = load_report_pair(report_json)
    assert envelope["status"] == "partial"
    assert verdict["status"] == "WARN"
    assert verdict["can_run"] is False
    assert verdict["task_smoke_state"] == "missing_result"
    assert "task-smoke" in verdict["summary"].lower()
    assert verdict["next_action"] == "Run the explicit task smoke command and rerun readiness."


def test_build_readiness_report_blocks_when_explicit_task_smoke_failed(tmp_path: Path):
    target_path = tmp_path / "target.json"
    normalized_path = tmp_path / "normalized.json"
    checks_path = tmp_path / "checks.json"
    report_json = tmp_path / "report.json"
    report_md = tmp_path / "report.md"

    target_path.write_text(
        json.dumps(
            {
                "target_type": "training",
                "entry_script": "train.py",
                "framework_path": "mindspore",
                "task_smoke_cmd": "python train.py --max_steps 1",
            }
        ),
        encoding="utf-8",
    )
    normalized_path.write_text(
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
    checks_path.write_text(
        json.dumps(
            [
                {
                    "id": "framework-smoke-prerequisite",
                    "status": "ok",
                    "summary": "framework smoke prerequisite passed",
                },
                {
                    "id": "task-smoke-executed",
                    "status": "block",
                    "summary": "explicit task smoke failed",
                },
            ]
        ),
        encoding="utf-8",
    )

    run_script(
        "build_readiness_report.py",
        "--target-json",
        str(target_path),
        "--normalized-json",
        str(normalized_path),
        "--checks-json",
        str(checks_path),
        "--output-json",
        str(report_json),
        "--output-md",
        str(report_md),
    )
    envelope, verdict = load_report_pair(report_json)
    assert envelope["status"] == "partial"
    assert verdict["status"] == "BLOCKED"
    assert verdict["can_run"] is False
    assert verdict["task_smoke_state"] == "failed"
    assert verdict["next_action"] == "Inspect the task smoke failure, fix the target path, and rerun readiness."


def test_build_readiness_report_marks_revalidation_false_when_required_scope_missing(tmp_path: Path):
    target_path = tmp_path / "target.json"
    normalized_path = tmp_path / "normalized.json"
    checks_path = tmp_path / "checks.json"
    fix_path = tmp_path / "fix.json"
    report_json = tmp_path / "report.json"
    report_md = tmp_path / "report.md"

    target_path.write_text(
        json.dumps(
            {
                "target_type": "training",
                "entry_script": "train.py",
                "framework_path": "mindspore",
            }
        ),
        encoding="utf-8",
    )
    normalized_path.write_text(
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
    checks_path.write_text(
        json.dumps(
            [
                {
                    "id": "target-stability",
                    "status": "ok",
                    "summary": "target resolved",
                }
            ]
        ),
        encoding="utf-8",
    )
    fix_path.write_text(
        json.dumps(
            {
                "execute": True,
                "results": [
                    {
                        "action_id": "action-1",
                        "status": "executed",
                    }
                ],
                "executed_actions": ["action-1"],
                "failed_actions": [],
                "needs_revalidation": ["framework", "runtime-dependencies"],
            }
        ),
        encoding="utf-8",
    )

    run_script(
        "build_readiness_report.py",
        "--target-json",
        str(target_path),
        "--normalized-json",
        str(normalized_path),
        "--checks-json",
        str(checks_path),
        "--fix-applied-json",
        str(fix_path),
        "--output-json",
        str(report_json),
        "--output-md",
        str(report_md),
    )
    envelope, verdict = load_report_pair(report_json)
    assert envelope["status"] == "partial"
    assert verdict["revalidated"] is False
    assert verdict["status"] == "WARN"
    assert verdict["can_run"] is False
    assert "revalidation" in verdict["summary"].lower()
    assert verdict["next_action"] == "Rerun the required readiness checks before certification."
    assert verdict["revalidation_required_scopes"] == ["framework", "runtime-dependencies"]
    assert verdict["revalidation_covered_scopes"] == ["target"]


def test_build_readiness_report_warn_with_can_run_true_for_strong_evidence_plus_warnings(tmp_path: Path):
    target_path = tmp_path / "target.json"
    normalized_path = tmp_path / "normalized.json"
    checks_path = tmp_path / "checks.json"
    report_json = tmp_path / "report.json"
    report_md = tmp_path / "report.md"

    target_path.write_text(
        json.dumps(
            {
                "target_type": "inference",
                "entry_script": "infer.py",
                "framework_path": "pta",
            }
        ),
        encoding="utf-8",
    )
    normalized_path.write_text(
        json.dumps(
            {
                "blockers": [],
                "warnings": ["Compatibility evidence is mixed."],
                "blockers_detailed": [],
                "warnings_detailed": [
                    {
                        "id": "compat-warning",
                        "summary": "Compatibility evidence is mixed.",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    checks_path.write_text(
        json.dumps(
            [
                {
                    "id": "framework-smoke-prerequisite",
                    "status": "ok",
                    "summary": "framework smoke prerequisite passed",
                }
            ]
        ),
        encoding="utf-8",
    )

    run_script(
        "build_readiness_report.py",
        "--target-json",
        str(target_path),
        "--normalized-json",
        str(normalized_path),
        "--checks-json",
        str(checks_path),
        "--output-json",
        str(report_json),
        "--output-md",
        str(report_md),
    )
    envelope, verdict = load_report_pair(report_json)
    assert envelope["status"] == "partial"
    assert verdict["status"] == "WARN"
    assert verdict["can_run"] is True
    assert "warnings" in verdict["summary"].lower()
    assert verdict["next_action"] == "Inspect warnings before proceeding with the intended task."


def test_execute_env_fix_prefers_cpu_torch_for_pta_framework(tmp_path: Path):
    plan_path = tmp_path / "plan.json"
    result_path = tmp_path / "result.json"
    log_path = tmp_path / "uv-log.jsonl"
    env_root = tmp_path / ".venv"
    python_path = env_root / "bin" / "python"
    python_path.parent.mkdir(parents=True)
    python_path.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    python_path.chmod(python_path.stat().st_mode | 0o111)

    plan_path.write_text(
        json.dumps(
            {
                "actions": [
                    {
                        "id": "action-1",
                        "action_type": "repair_pta_framework",
                        "allowed": True,
                        "requires_confirmation": True,
                        "reason": "PTA framework path requires repair inside the selected environment.",
                        "revalidation_scope": ["framework"],
                        "package_names": ["torch", "torch_npu"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    uv_dir = tmp_path / "fake-uv"
    uv_dir.mkdir()
    uv_py = uv_dir / "uv"
    uv_py.write_text(
        f"""#!/usr/bin/env python3
import json
import sys
from pathlib import Path

log_path = Path(r"{log_path}")
log_path.parent.mkdir(parents=True, exist_ok=True)
log_path.open("a", encoding="utf-8").write(json.dumps(sys.argv[1:]) + "\\n")
raise SystemExit(0)
""",
        encoding="utf-8",
    )
    uv_py.chmod(uv_py.stat().st_mode | 0o111)
    uv_cmd = uv_dir / "uv.cmd"
    uv_cmd.write_text(f'@echo off\r\n"{sys.executable}" "%~dp0uv" %*\r\n', encoding="utf-8")

    env = dict(os.environ)
    env["PATH"] = str(uv_dir) + os.pathsep + env.get("PATH", "")

    run_script(
        "execute_env_fix.py",
        "--plan-json",
        str(plan_path),
        "--output-json",
        str(result_path),
        "--execute",
        "--working-dir",
        str(tmp_path),
        "--selected-env-root",
        str(env_root),
        "--confirm-framework-repair",
        env=env,
    )

    calls = [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(calls) == 2
    assert "--index-url" in calls[0]
    assert "https://download.pytorch.org/whl/cpu" in calls[0]
    assert "torch" in calls[0]
    assert "torch_npu" not in calls[0]
    assert "torch_npu" in calls[1]
