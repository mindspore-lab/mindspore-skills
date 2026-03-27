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


def test_discover_execution_target_prefers_explicit_framework_hint(tmp_path: Path):
    (tmp_path / "train.py").write_text(
        "import mindspore as ms\n",
        encoding="utf-8",
    )
    output = tmp_path / "target.json"

    run_script(
        "discover_execution_target.py",
        "--working-dir",
        str(tmp_path),
        "--framework-hint",
        "pta",
        "--output-json",
        str(output),
    )
    target = json.loads(output.read_text(encoding="utf-8"))
    assert target["framework_path"] == "pta"
    assert target["framework_hint"] == "pta"
    assert "explicit framework_hint input provided" in target["evidence"]
    assert any(
        "local workspace evidence suggests mindspore, but explicit framework hint requested pta" in item
        for item in target["evidence"]
    )


def test_discover_execution_target_records_explicit_cann_path(tmp_path: Path):
    output = tmp_path / "target.json"
    cann_root = tmp_path / "custom-cann" / "8.5.0"

    run_script(
        "discover_execution_target.py",
        "--working-dir",
        str(tmp_path),
        "--cann-path",
        str(cann_root),
        "--output-json",
        str(output),
    )
    target = json.loads(output.read_text(encoding="utf-8"))
    assert target["cann_path"] == str(cann_root)
    assert "explicit cann_path input provided" in target["evidence"]


def test_discover_execution_target_applies_qwen_huggingface_recipe(tmp_path: Path):
    output = tmp_path / "target.json"

    run_script(
        "discover_execution_target.py",
        "--working-dir",
        str(tmp_path),
        "--target",
        "training",
        "--model-hub-id",
        "qwen3-0.6b",
        "--output-json",
        str(output),
    )
    target = json.loads(output.read_text(encoding="utf-8"))
    assert target["target_type"] == "training"
    assert target["entry_script"] == "workspace-assets/examples/train_qwen3_0_6b.py"
    assert target["model_hub_id"] == "Qwen/Qwen3-0.6B"
    assert target["dataset_hub_id"] == "karthiksagarn/astro_horoscope"
    assert target["model_path"] is None
    assert target["dataset_path"] is None
    assert target["dataset_split"] == "train"
    assert target["reference_transformers_version"] == "4.57.6"
    assert target["example_recipe_id"] == "qwen3-0.6b-hf-training"
    assert any(item.get("package_name") == "transformers==4.57.6" for item in target["expected_runtime_profile"])


def test_discover_execution_target_keeps_existing_training_script_when_recipe_matches(tmp_path: Path):
    (tmp_path / "train.py").write_text("import torch\nimport torch_npu\n", encoding="utf-8")
    output = tmp_path / "target.json"

    run_script(
        "discover_execution_target.py",
        "--working-dir",
        str(tmp_path),
        "--target",
        "training",
        "--model-hub-id",
        "qwen3-0.6b",
        "--output-json",
        str(output),
    )
    target = json.loads(output.read_text(encoding="utf-8"))
    assert target["entry_script"] == "train.py"
    assert target["example_recipe_id"] == "qwen3-0.6b-hf-training"
    assert target["model_path"] is None
    assert target["dataset_path"] is None
    assert target["model_hub_id"] == "Qwen/Qwen3-0.6B"
    assert target["dataset_hub_id"] == "karthiksagarn/astro_horoscope"


def test_discover_execution_target_extracts_remote_huggingface_assets_from_script(tmp_path: Path):
    (tmp_path / "train.py").write_text(
        (
            'from datasets import load_dataset\n'
            'from transformers import AutoTokenizer\n'
            'model_name = "Qwen/Qwen3-0.6B"\n'
            'dataset_name = "karthiksagarn/astro_horoscope"\n'
            'tokenizer = AutoTokenizer.from_pretrained(model_name)\n'
            'dataset = load_dataset(dataset_name, split="train")\n'
        ),
        encoding="utf-8",
    )
    output = tmp_path / "target.json"

    run_script(
        "discover_execution_target.py",
        "--working-dir",
        str(tmp_path),
        "--output-json",
        str(output),
    )
    target = json.loads(output.read_text(encoding="utf-8"))
    assert target["entry_script"] == "train.py"
    assert target["model_hub_id"] == "Qwen/Qwen3-0.6B"
    assert target["dataset_hub_id"] == "karthiksagarn/astro_horoscope"
    assert target["dataset_split"] == "train"


def test_discover_execution_target_defaults_training_framework_to_pta_without_explicit_mindspore(tmp_path: Path):
    output = tmp_path / "target.json"

    run_script(
        "discover_execution_target.py",
        "--working-dir",
        str(tmp_path),
        "--target",
        "training",
        "--output-json",
        str(output),
    )
    target = json.loads(output.read_text(encoding="utf-8"))
    assert target["framework_path"] == "pta"
    assert any("training target defaulted to PTA" in item for item in target["evidence"])


def test_discover_execution_target_respects_explicit_mindspore_for_training(tmp_path: Path):
    output = tmp_path / "target.json"

    run_script(
        "discover_execution_target.py",
        "--working-dir",
        str(tmp_path),
        "--target",
        "training",
        "--framework-hint",
        "mindspore",
        "--output-json",
        str(output),
    )
    target = json.loads(output.read_text(encoding="utf-8"))
    assert target["framework_path"] == "mindspore"


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


def test_normalize_blockers_preserves_package_names(tmp_path: Path):
    checks_path = tmp_path / "checks.json"
    checks_path.write_text(
        json.dumps(
            [
                {
                    "id": "runtime-importability",
                    "status": "block",
                    "summary": "Ascend hidden runtime imports are unavailable.",
                    "category_hint": "env",
                    "package_names": ["decorator", "scipy", "attrs"],
                }
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
    assert normalized["blockers_detailed"][0]["package_names"] == ["decorator", "scipy", "attrs"]


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


def test_build_dependency_closure_defaults_training_framework_to_pta_when_unknown(tmp_path: Path):
    target_path = tmp_path / "target.json"
    closure_path = tmp_path / "closure.json"
    target_path.write_text(
        json.dumps(
            {
                "working_dir": str(tmp_path),
                "target_type": "training",
                "framework_path": "unknown",
                "framework_hint": "auto",
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
    framework = closure["layers"]["framework"]
    assert framework["framework_path"] == "pta"
    assert framework["required_packages"] == ["torch", "torch_npu"]


def test_build_dependency_closure_uses_recipe_runtime_profile_without_entry_script(tmp_path: Path):
    target_path = tmp_path / "target.json"
    closure_path = tmp_path / "closure.json"
    target_path.write_text(
        json.dumps(
            {
                "working_dir": str(tmp_path),
                "target_type": "training",
                "entry_script": "workspace-assets/examples/train_qwen3_0_6b.py",
                "framework_path": "pta",
                "selected_python": sys.executable,
                "model_path": "workspace-assets/models/Qwen__Qwen3-0.6B",
                "model_hub_id": "Qwen/Qwen3-0.6B",
                "dataset_path": "workspace-assets/datasets/karthiksagarn__astro_horoscope",
                "dataset_hub_id": "karthiksagarn/astro_horoscope",
                "dataset_split": "train",
                "expected_runtime_profile": [
                    {
                        "import_name": "datasets",
                        "package_name": "datasets",
                        "required_for": "bundled-example",
                        "reason": "example dataset loader",
                    },
                    {
                        "import_name": "transformers",
                        "package_name": "transformers==4.57.6",
                        "required_for": "bundled-example",
                        "reason": "example transformers pin",
                    },
                    {
                        "import_name": "sentencepiece",
                        "package_name": "sentencepiece",
                        "required_for": "bundled-example",
                        "reason": "qwen tokenizer dependency",
                    },
                ],
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
    runtime_layer = closure["layers"]["runtime_dependencies"]
    required_imports = runtime_layer["required_imports"]
    implicit_profile = runtime_layer["implicit_dependency_profile"]
    workspace = closure["layers"]["workspace_assets"]

    assert "datasets" in required_imports
    assert "transformers" in required_imports
    assert "sentencepiece" in required_imports
    assert "accelerate" in required_imports
    assert any(item["package_name"] == "transformers==4.57.6" for item in implicit_profile)
    assert workspace["entry_script"]["source"] == "bundled-example"
    assert workspace["model_path"]["asset_provider"] == "huggingface"
    assert workspace["dataset_path"]["asset_provider"] == "huggingface"


def test_build_dependency_closure_prefers_explicit_hf_cache_envs(tmp_path: Path):
    target_path = tmp_path / "target.json"
    closure_path = tmp_path / "closure.json"
    hub_cache = tmp_path / "custom-hf-cache" / "hub"
    datasets_cache = tmp_path / "custom-hf-cache" / "datasets"
    hf_home = tmp_path / "ignored-hf-home"
    target_path.write_text(
        json.dumps(
            {
                "working_dir": str(tmp_path),
                "target_type": "training",
                "entry_script": "workspace-assets/examples/train_qwen3_0_6b.py",
                "framework_path": "pta",
                "selected_python": sys.executable,
                "model_hub_id": "Qwen/Qwen3-0.6B",
                "dataset_hub_id": "karthiksagarn/astro_horoscope",
                "dataset_split": "train",
            }
        ),
        encoding="utf-8",
    )

    env = dict(os.environ)
    env["HUGGINGFACE_HUB_CACHE"] = str(hub_cache)
    env["HF_DATASETS_CACHE"] = str(datasets_cache)
    env["HF_HOME"] = str(hf_home)
    env["HF_ENDPOINT"] = "http://127.0.0.1:9"

    run_script(
        "build_dependency_closure.py",
        "--target-json",
        str(target_path),
        "--output-json",
        str(closure_path),
        env=env,
    )
    closure = json.loads(closure_path.read_text(encoding="utf-8"))
    remote_assets = closure["layers"]["remote_assets"]
    cache_layout = remote_assets["cache_layout"]

    assert cache_layout["source"] == "explicit_cache_env"
    assert cache_layout["hf_home"] == str(hf_home.resolve())
    assert cache_layout["hub_cache"] == str(hub_cache.resolve())
    assert cache_layout["datasets_cache"] == str(datasets_cache.resolve())


def test_build_dependency_closure_uses_custom_hf_endpoint_and_working_dir_cache_default(tmp_path: Path):
    target_path = tmp_path / "target.json"
    closure_path = tmp_path / "closure.json"
    target_path.write_text(
        json.dumps(
            {
                "working_dir": str(tmp_path),
                "target_type": "training",
                "entry_script": "workspace-assets/examples/train_qwen3_0_6b.py",
                "framework_path": "pta",
                "selected_python": sys.executable,
                "model_hub_id": "Qwen/Qwen3-0.6B",
                "dataset_hub_id": "karthiksagarn/astro_horoscope",
                "dataset_split": "train",
            }
        ),
        encoding="utf-8",
    )

    env = dict(os.environ)
    env.pop("HUGGINGFACE_HUB_CACHE", None)
    env.pop("HF_DATASETS_CACHE", None)
    env.pop("HF_HOME", None)
    env["HF_ENDPOINT"] = "http://127.0.0.1:10/"

    run_script(
        "build_dependency_closure.py",
        "--target-json",
        str(target_path),
        "--output-json",
        str(closure_path),
        env=env,
    )
    closure = json.loads(closure_path.read_text(encoding="utf-8"))
    remote_assets = closure["layers"]["remote_assets"]
    cache_layout = remote_assets["cache_layout"]

    assert remote_assets["hf_endpoint"] == "http://127.0.0.1:10"
    assert cache_layout["source"] == "working_dir_default"
    assert cache_layout["hf_home"] == str((tmp_path / "huggingface-cache").resolve())


def test_build_dependency_closure_adds_ascend_hidden_runtime_profile_for_mindspore(tmp_path: Path):
    (tmp_path / "train.py").write_text(
        "import mindspore as ms\n",
        encoding="utf-8",
    )
    (tmp_path / "model").mkdir()
    ascend_home = tmp_path / "ascend"
    ascend_home.mkdir()
    (ascend_home / "set_env.sh").write_text("export FAKE_ASCEND_READY=1\n", encoding="utf-8")
    target_path = tmp_path / "target.json"
    closure_path = tmp_path / "closure.json"
    target_path.write_text(
        json.dumps(
            {
                "working_dir": str(tmp_path),
                "target_type": "training",
                "entry_script": "train.py",
                "framework_path": "mindspore",
                "selected_python": sys.executable,
                "model_path": "model",
                "launch_cmd": "python train.py",
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
    runtime_layer = closure["layers"]["runtime_dependencies"]
    required_imports = runtime_layer["required_imports"]
    implicit_profile = runtime_layer["implicit_dependency_profile"]

    assert "mindspore" in required_imports
    assert "decorator" in required_imports
    assert "scipy" in required_imports
    assert "attr" in required_imports
    assert any(item["package_name"] == "attrs" for item in implicit_profile)


def test_build_dependency_closure_adds_accelerate_for_transformers_runtime(tmp_path: Path):
    (tmp_path / "infer.py").write_text(
        "\n".join(
            [
                "import torch",
                "import torch_npu",
                "from transformers import AutoModelForCausalLM",
                "model = AutoModelForCausalLM.from_pretrained('demo')",
            ]
        ),
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
    runtime_layer = closure["layers"]["runtime_dependencies"]
    required_imports = runtime_layer["required_imports"]
    implicit_profile = runtime_layer["implicit_dependency_profile"]

    assert "transformers" in required_imports
    assert "accelerate" in required_imports
    assert any(item["package_name"] == "accelerate" for item in implicit_profile)


def test_build_dependency_closure_collects_common_training_and_inference_import_candidates(tmp_path: Path):
    (tmp_path / "infer.py").write_text(
        "\n".join(
            [
                "import torch",
                "import torch_npu",
                "from transformers import AutoTokenizer",
                "import peft",
                "import trl",
                "import evaluate",
                "import sentencepiece",
            ]
        ),
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
    runtime_layer = closure["layers"]["runtime_dependencies"]
    required_imports = runtime_layer["required_imports"]

    assert "transformers" in required_imports
    assert "accelerate" in required_imports
    assert "peft" in required_imports
    assert "trl" in required_imports
    assert "evaluate" in required_imports
    assert "sentencepiece" in required_imports


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


def test_build_dependency_closure_marks_broken_detected_ascend_env_script(tmp_path: Path):
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
import subprocess
import sys

if len(sys.argv) >= 3 and sys.argv[1] == "-c":
    code = sys.argv[2]
    if "platform.python_version" in code and "version_info" in code:
        print(json.dumps({"version_info": [3, 10, 0], "version": "3.10.0"}))
        raise SystemExit(0)
    mode = sys.argv[3]
    payload = json.loads(sys.argv[4])
    if mode == "import":
        packages = payload.get("packages", [])
        available = {"torch"}
        print(json.dumps({name: name in available for name in packages}))
        raise SystemExit(0)
    if mode == "framework_smoke":
        print(json.dumps({"success": False, "details": [], "error": "missing runtime"}))
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
        "source /definitely/missing/setenv.bash\n",
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

    assert system["ascend_env_script_present"] is True
    assert system["probe_env_source"] == "sourced_script_failed"
    assert system["probe_env_error"]


def test_build_dependency_closure_prefers_explicit_cann_path_for_custom_install(tmp_path: Path):
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
    runtime_ready = bool(os.environ.get("ASCEND_HOME_PATH")) and bool(os.environ.get("ASCEND_OPP_PATH"))
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

    cann_root = tmp_path / "cann_custom_path" / "8.5.0"
    toolkit_root = cann_root / "ascend-toolkit"
    (toolkit_root / "opp").mkdir(parents=True)
    (toolkit_root / "runtime" / "lib64").mkdir(parents=True)
    (toolkit_root / "set_env.sh").write_text(
        "\n".join(
            [
                f'export ASCEND_HOME_PATH="{toolkit_root}"',
                f'export ASCEND_OPP_PATH="{toolkit_root / "opp"}"',
                f'export LD_LIBRARY_PATH="{toolkit_root / "runtime" / "lib64"}:$LD_LIBRARY_PATH"',
            ]
        )
        + "\n",
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
                "cann_path": str(cann_root),
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
    system = closure["layers"]["system"]
    framework = closure["layers"]["framework"]

    assert system["ascend_env_script_path"] == str(toolkit_root / "set_env.sh")
    assert system["ascend_env_selection_source"] == "bounded_search"
    assert system["cann_path_input"] == str(cann_root)
    assert system["probe_env_source"] == "sourced_script"
    assert framework["import_probes"]["torch_npu"] is True


def test_build_dependency_closure_uses_bounded_search_for_custom_home_install(tmp_path: Path):
    if os.name == "nt" or shutil.which("bash") is None:
        return

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "infer.py").write_text(
        "import torch\nimport torch_npu\n",
        encoding="utf-8",
    )
    (workspace / "model").mkdir()

    fake_python = workspace / ".venv" / "bin" / "python"
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
    runtime_ready = bool(os.environ.get("ASCEND_HOME_PATH")) and bool(os.environ.get("ASCEND_OPP_PATH"))
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

    home_root = tmp_path / "home"
    toolkit_root = home_root / "cann_custom_path" / "8.5.0" / "ascend-toolkit"
    (toolkit_root / "opp").mkdir(parents=True)
    (toolkit_root / "runtime" / "lib64").mkdir(parents=True)
    (toolkit_root / "set_env.sh").write_text(
        "\n".join(
            [
                f'export ASCEND_HOME_PATH="{toolkit_root}"',
                f'export ASCEND_OPP_PATH="{toolkit_root / "opp"}"',
                f'export LD_LIBRARY_PATH="{toolkit_root / "runtime" / "lib64"}:$LD_LIBRARY_PATH"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    target_path = tmp_path / "target.json"
    closure_path = tmp_path / "closure.json"
    target_path.write_text(
        json.dumps(
            {
                "working_dir": str(workspace),
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
    env["HOME"] = str(home_root)

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

    assert system["ascend_env_selection_source"] == "bounded_search"
    assert system["ascend_env_script_path"] == str(toolkit_root / "set_env.sh")
    assert system["probe_env_source"] == "sourced_script"


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
                        },
                        "selected_env_root": str(tmp_path / ".venv"),
                        "probe_source": "selected_env",
                        "probe_python_path": str(tmp_path / ".venv" / "bin" / "python"),
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


def test_collect_readiness_checks_accepts_remote_huggingface_assets(tmp_path: Path):
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
                            "uv_available": True,
                            "uv_path": "/usr/bin/uv",
                        },
                        "selected_env_root": str(tmp_path / ".venv"),
                        "selected_python": str(tmp_path / ".venv" / "bin" / "python"),
                        "selection_status": "selected",
                        "probe_source": "workspace_env",
                        "probe_python_path": str(tmp_path / ".venv" / "bin" / "python"),
                    },
                    "framework": {
                        "framework_path": "pta",
                        "required_packages": [],
                        "import_probes": {},
                    },
                    "runtime_dependencies": {
                        "required_imports": [],
                        "import_probes": {},
                    },
                    "remote_assets": {
                        "hf_endpoint": "https://hf-mirror.com",
                        "hf_endpoint_source": "default",
                        "endpoint_reachable": True,
                        "cache_layout": {
                            "source": "working_dir_default",
                            "hub_cache": str(tmp_path / "huggingface-cache" / "hub"),
                            "hub_cache_writable": True,
                            "datasets_cache": str(tmp_path / "huggingface-cache" / "datasets"),
                            "datasets_cache_writable": True,
                        },
                        "assets": {
                            "model_path": {
                                "repo_id": "Qwen/Qwen3-0.6B",
                                "repo_type": "model",
                                "cache_path": str(tmp_path / "huggingface-cache" / "hub"),
                                "ready": True,
                            },
                            "dataset_path": {
                                "repo_id": "karthiksagarn/astro_horoscope",
                                "repo_type": "dataset",
                                "dataset_split": "train",
                                "cache_path": str(tmp_path / "huggingface-cache" / "datasets"),
                                "ready": True,
                            },
                        },
                    },
                    "workspace_assets": {
                        "entry_script": {
                            "path": "workspace-assets/examples/train_qwen3_0_6b.py",
                            "exists": False,
                            "required": True,
                            "source": "bundled-example",
                            "template_path": str(ROOT / "examples" / "qwen3_0_6b_training_example.py"),
                            "example_recipe_id": "qwen3-0.6b-hf-training",
                        },
                        "model_path": {
                            "path": None,
                            "exists": False,
                            "required": True,
                            "satisfied": True,
                            "asset_provider": "huggingface",
                            "repo_id": "Qwen/Qwen3-0.6B",
                            "repo_type": "model",
                            "resolution_mode": "remote-huggingface",
                        },
                        "dataset_path": {
                            "path": None,
                            "exists": False,
                            "required": True,
                            "satisfied": True,
                            "asset_provider": "huggingface",
                            "repo_id": "karthiksagarn/astro_horoscope",
                            "repo_type": "dataset",
                            "dataset_split": "train",
                            "resolution_mode": "remote-huggingface",
                        },
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
    assert by_id["workspace-entry_script"]["category_hint"] == "asset"
    assert by_id["workspace-entry_script"]["template_path"].endswith("qwen3_0_6b_training_example.py")
    assert by_id["remote-huggingface-endpoint"]["status"] == "ok"
    assert by_id["workspace-model_path"]["status"] == "ok"
    assert by_id["workspace-model_path"]["asset_repo_id"] == "Qwen/Qwen3-0.6B"
    assert by_id["workspace-dataset_path"]["status"] == "ok"
    assert by_id["workspace-dataset_path"]["dataset_split"] == "train"


def test_collect_readiness_checks_blocks_broken_ascend_env_script(tmp_path: Path):
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
                        "requires_ascend": True,
                        "device_paths_present": True,
                        "ascend_env_script_present": True,
                        "ascend_env_script_path": "/usr/local/Ascend/cann-8.5.0-bak/set_env.sh",
                        "ascend_env_active": False,
                        "probe_env_source": "sourced_script_failed",
                        "probe_env_error": "No such file or directory",
                    },
                    "python_environment": {
                        "tooling": {
                            "uv_available": True,
                            "uv_path": "/usr/bin/uv",
                        },
                        "selected_env_root": str(tmp_path / ".venv"),
                        "selected_python": str(tmp_path / ".venv" / "bin" / "python"),
                        "selection_status": "selected",
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
                    },
                    "runtime_dependencies": {
                        "required_imports": [],
                        "import_probes": {},
                    },
                    "workspace_assets": {
                        "entry_script": {"required": True, "exists": True},
                        "dataset_path": {"required": True, "exists": True},
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
    assert by_id["system-device"]["status"] == "ok"
    assert by_id["system-ascend-env"]["status"] == "block"
    assert by_id["system-ascend-env"]["category_hint"] == "system"
    assert any("probe_env_source=sourced_script_failed" in item for item in by_id["system-ascend-env"]["evidence"])
    assert any("probe_env_error=No such file or directory" in item for item in by_id["system-ascend-env"]["evidence"])


def test_collect_readiness_checks_uses_structured_hidden_runtime_package_names(tmp_path: Path):
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
                        "requires_ascend": True,
                    },
                    "python_environment": {
                        "tooling": {
                            "uv_available": True,
                            "uv_path": "/usr/bin/uv",
                        }
                    },
                    "framework": {
                        "framework_path": "pta",
                        "required_packages": ["torch", "torch_npu"],
                        "import_probes": {
                            "torch": True,
                            "torch_npu": True,
                        },
                        "probe_source": "workspace_env",
                    },
                    "runtime_dependencies": {
                        "required_imports": ["decorator", "scipy", "attr"],
                        "import_probes": {
                            "decorator": False,
                            "scipy": False,
                            "attr": False,
                        },
                        "probe_source": "workspace_env",
                        "implicit_dependency_profile": [
                            {
                                "import_name": "decorator",
                                "package_name": "decorator",
                                "required_for": "ascend-compiler",
                                "reason": "Ascend compiler adapters import decorator.",
                            },
                            {
                                "import_name": "scipy",
                                "package_name": "scipy",
                                "required_for": "ascend-compiler",
                                "reason": "Ascend compiler adapters import scipy.",
                            },
                            {
                                "import_name": "attr",
                                "package_name": "attrs",
                                "required_for": "ascend-compiler",
                                "reason": "Ascend compiler adapters import attr.",
                            },
                        ],
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
    assert by_id["runtime-importability"]["status"] == "block"
    assert by_id["runtime-importability"]["package_names"] == ["decorator", "scipy", "attrs"]
    assert "package_name[attr]=attrs" in by_id["runtime-importability"]["evidence"]


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


def test_plan_env_fix_prefers_structured_runtime_package_names(tmp_path: Path):
    blockers_path = tmp_path / "blockers.json"
    closure_path = tmp_path / "closure.json"
    output = tmp_path / "plan.json"

    blockers_path.write_text(
        json.dumps(
            {
                "blockers_detailed": [
                    {
                        "id": "runtime-importability",
                        "category": "env_remediable",
                        "summary": "Required runtime imports are unavailable in the selected environment: decorator, scipy, attr.",
                        "evidence": [
                            "probe_source=workspace_env",
                            "decorator",
                            "scipy",
                            "attr",
                        ],
                        "package_names": ["decorator", "scipy", "attrs"],
                        "remediable": True,
                        "revalidation_scope": ["runtime-dependencies", "framework"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    closure_path.write_text(json.dumps({"layers": {}}, indent=2), encoding="utf-8")

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
    assert plan["actions"][0]["action_type"] == "install_runtime_dependency"
    assert plan["actions"][0]["package_names"] == ["decorator", "scipy", "attrs"]


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
    assert by_id["python-selected-env"]["status"] == "block"
    assert by_id["python-selected-env"]["category_hint"] == "env"
    assert by_id["framework-importability"]["status"] == "warn"
    assert "do not install framework packages into system python" in by_id["framework-importability"]["summary"].lower()
    assert by_id["runtime-importability"]["status"] == "warn"
    assert "do not install runtime packages into system python" in by_id["runtime-importability"]["summary"].lower()


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


def test_plan_env_fix_defaults_training_framework_repair_to_pta(tmp_path: Path):
    blockers_path = tmp_path / "normalized.json"
    closure_path = tmp_path / "closure.json"
    plan_path = tmp_path / "plan.json"

    blockers_path.write_text(
        json.dumps(
            {
                "blockers_detailed": [
                    {
                        "id": "framework-missing",
                        "category": "framework_remediable",
                        "summary": "Framework packages are missing.",
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
                "target_type": "training",
                "layers": {
                    "framework": {
                        "framework_path": "unknown",
                        "required_packages": [],
                    }
                },
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
    assert plan["actions"][0]["action_type"] == "repair_pta_framework"
    assert plan["actions"][0]["package_names"] == ["torch", "torch_npu"]


def test_plan_env_fix_plans_huggingface_downloads_and_example_scaffold(tmp_path: Path):
    blockers_path = tmp_path / "normalized.json"
    closure_path = tmp_path / "closure.json"
    plan_path = tmp_path / "plan.json"

    blockers_path.write_text(
        json.dumps(
            {
                "blockers_detailed": [
                    {
                        "id": "workspace-entry_script",
                        "category": "asset_remediable",
                        "summary": "Required training entry script is missing but can be scaffolded from the bundled training example.",
                        "remediable": True,
                        "template_path": str(ROOT / "examples" / "qwen3_0_6b_training_example.py"),
                        "asset_kind": "entry_script",
                        "asset_local_path": "workspace-assets/examples/train_qwen3_0_6b.py",
                        "revalidation_scope": ["workspace-assets", "target", "runtime-dependencies"],
                    },
                    {
                        "id": "workspace-model_path",
                        "category": "asset_remediable",
                        "summary": "Required asset model_path is missing locally but can be downloaded from Hugging Face.",
                        "remediable": True,
                        "asset_kind": "model_path",
                        "asset_provider": "huggingface",
                        "asset_repo_id": "Qwen/Qwen3-0.6B",
                        "asset_local_path": "workspace-assets/models/Qwen__Qwen3-0.6B",
                        "revalidation_scope": ["workspace-assets", "task-smoke"],
                    },
                    {
                        "id": "workspace-dataset_path",
                        "category": "asset_remediable",
                        "summary": "Required dataset asset is missing locally but can be downloaded from Hugging Face.",
                        "remediable": True,
                        "asset_kind": "dataset_path",
                        "asset_provider": "huggingface",
                        "asset_repo_id": "karthiksagarn/astro_horoscope",
                        "asset_local_path": "workspace-assets/datasets/karthiksagarn__astro_horoscope",
                        "dataset_split": "train",
                        "revalidation_scope": ["workspace-assets", "task-smoke"],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    closure_path.write_text(json.dumps({"layers": {}}, indent=2), encoding="utf-8")

    run_script(
        "plan_env_fix.py",
        "--blockers-json",
        str(blockers_path),
        "--closure-json",
        str(closure_path),
        "--output-json",
        str(plan_path),
        "--allow-network",
    )
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    action_types = [item["action_type"] for item in plan["actions"]]
    assert action_types == [
        "scaffold_example_entry_script",
        "download_huggingface_model_asset",
        "download_huggingface_dataset_asset",
    ]
    assert plan["actions"][1]["allowed"] is True
    assert plan["actions"][2]["dataset_split"] == "train"


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


def test_execute_env_fix_scaffolds_example_entry_script(tmp_path: Path):
    plan_path = tmp_path / "plan.json"
    output_path = tmp_path / "result.json"
    template_path = tmp_path / "template.py"
    destination_path = tmp_path / "workspace-assets" / "examples" / "train_qwen3_0_6b.py"
    template_path.write_text("print('example')\n", encoding="utf-8")
    plan_path.write_text(
        json.dumps(
            {
                "actions": [
                    {
                        "id": "action-1",
                        "action_type": "scaffold_example_entry_script",
                        "allowed": True,
                        "requires_confirmation": True,
                        "reason": "scaffold bundled example",
                        "revalidation_scope": ["workspace-assets", "target"],
                        "template_path": str(template_path),
                        "destination_path": str(destination_path),
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
        "--execute",
        "--confirm-asset-repair",
    )
    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["results"][0]["status"] == "executed"
    assert destination_path.read_text(encoding="utf-8") == "print('example')\n"


def test_execute_env_fix_finds_uv_in_user_local_bin_without_path_refresh(tmp_path: Path):
    plan_path = tmp_path / "plan.json"
    output_path = tmp_path / "result.json"
    home_path = tmp_path / "home"
    local_bin = home_path / ".local" / "bin"
    runner_path = local_bin / "uv_runner.py"
    env_root = tmp_path / ".venv"

    plan_path.write_text(
        json.dumps(
            {
                "actions": [
                    {
                        "id": "action-1",
                        "action_type": "create_or_select_env",
                        "allowed": True,
                        "requires_confirmation": True,
                        "reason": "create workspace environment",
                        "revalidation_scope": ["python-environment"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    local_bin.mkdir(parents=True)
    runner_path.write_text(
        f"""#!/usr/bin/env python3
import shutil
import sys
from pathlib import Path

REAL_PYTHON = r"{sys.executable}"


def main() -> int:
    args = sys.argv[1:]
    if len(args) >= 2 and args[0] == "venv":
        env_root = Path(args[1])
        if sys.platform == "win32":
            target = env_root / "Scripts" / "python.exe"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(REAL_PYTHON, target)
        else:
            target = env_root / "bin" / "python"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("#!/usr/bin/env python3\\n", encoding="utf-8")
            target.chmod(target.stat().st_mode | 0o111)
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
""",
        encoding="utf-8",
    )

    env = dict(os.environ)
    env["HOME"] = str(home_path)
    env["USERPROFILE"] = str(home_path)
    env["PATH"] = ""

    if os.name == "nt":
        uv_path = local_bin / "uv.cmd"
        uv_path.write_text(
            f'@echo off\r\n"{sys.executable}" "{runner_path}" %*\r\n',
            encoding="utf-8",
        )
    else:
        uv_path = local_bin / "uv"
        uv_path.write_text(
            f"#!{sys.executable}\nfrom uv_runner import main\nraise SystemExit(main())\n",
            encoding="utf-8",
        )
        uv_path.chmod(uv_path.stat().st_mode | 0o111)
    runner_path.chmod(runner_path.stat().st_mode | 0o111)

    run_script(
        "execute_env_fix.py",
        "--plan-json",
        str(plan_path),
        "--output-json",
        str(output_path),
        "--execute",
        "--selected-env-root",
        str(env_root),
        "--confirm-create-env",
        env=env,
    )
    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["results"][0]["status"] == "executed"
    assert env_root.exists()
    if os.name == "nt":
        assert (env_root / "Scripts" / "python.exe").exists()
    else:
        assert (env_root / "bin" / "python").exists()


def test_execute_env_fix_downloads_huggingface_assets(tmp_path: Path):
    plan_path = tmp_path / "plan.json"
    output_path = tmp_path / "result.json"
    env_root = tmp_path / ".venv"
    modules_dir = tmp_path / "fake-modules"
    uv_dir = tmp_path / "fake-uv"
    model_dest = tmp_path / "workspace-assets" / "models" / "Qwen__Qwen3-0.6B"
    dataset_dest = tmp_path / "workspace-assets" / "datasets" / "karthiksagarn__astro_horoscope"

    python_path = env_root / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    python_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(sys.executable, python_path)

    (modules_dir / "huggingface_hub").mkdir(parents=True)
    (modules_dir / "huggingface_hub" / "__init__.py").write_text(
        """import json\nimport os\nfrom pathlib import Path\n\ndef snapshot_download(repo_id, local_dir):\n    path = Path(local_dir)\n    path.mkdir(parents=True, exist_ok=True)\n    (path / 'config.json').write_text(json.dumps({'repo_id': repo_id, 'hf_endpoint': os.environ.get('HF_ENDPOINT'), 'hf_home': os.environ.get('HF_HOME')}), encoding='utf-8')\n    return str(path)\n""",
        encoding="utf-8",
    )
    (modules_dir / "datasets").mkdir(parents=True)
    (modules_dir / "datasets" / "__init__.py").write_text(
        """import json\nimport os\nfrom pathlib import Path\n\nclass _Dataset:\n    def __init__(self, repo_id, split):\n        self.repo_id = repo_id\n        self.split = split\n\n    def save_to_disk(self, local_dir):\n        path = Path(local_dir)\n        path.mkdir(parents=True, exist_ok=True)\n        (path / 'dataset_info.json').write_text(json.dumps({'repo_id': self.repo_id, 'split': self.split, 'hf_endpoint': os.environ.get('HF_ENDPOINT'), 'hf_home': os.environ.get('HF_HOME')}), encoding='utf-8')\n\n\ndef load_dataset(repo_id, split=None):\n    return _Dataset(repo_id, split)\n""",
        encoding="utf-8",
    )

    uv_dir.mkdir()
    uv_py = uv_dir / "uv"
    uv_py.write_text(
        """#!/usr/bin/env python3\nraise SystemExit(0)\n""",
        encoding="utf-8",
    )
    uv_py.chmod(uv_py.stat().st_mode | 0o111)
    uv_cmd = uv_dir / "uv.cmd"
    uv_cmd.write_text(f'@echo off\r\n"{sys.executable}" "%~dp0uv" %*\r\n', encoding="utf-8")

    plan_path.write_text(
        json.dumps(
            {
                "actions": [
                    {
                        "id": "action-1",
                        "action_type": "download_huggingface_model_asset",
                        "allowed": True,
                        "requires_confirmation": True,
                        "reason": "download model asset",
                        "revalidation_scope": ["workspace-assets", "task-smoke"],
                        "repo_id": "Qwen/Qwen3-0.6B",
                        "destination_path": str(model_dest),
                    },
                    {
                        "id": "action-2",
                        "action_type": "download_huggingface_dataset_asset",
                        "allowed": True,
                        "requires_confirmation": True,
                        "reason": "download dataset asset",
                        "revalidation_scope": ["workspace-assets", "task-smoke"],
                        "repo_id": "karthiksagarn/astro_horoscope",
                        "destination_path": str(dataset_dest),
                        "dataset_split": "train",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    env = dict(os.environ)
    env["PATH"] = str(uv_dir) + os.pathsep + env.get("PATH", "")
    env["PYTHONPATH"] = str(modules_dir)

    run_script(
        "execute_env_fix.py",
        "--plan-json",
        str(plan_path),
        "--output-json",
        str(output_path),
        "--execute",
        "--working-dir",
        str(tmp_path),
        "--selected-env-root",
        str(env_root),
        "--confirm-asset-repair",
        env=env,
    )
    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert [item["status"] for item in result["results"]] == ["executed", "executed"]
    model_info = json.loads((model_dest / "config.json").read_text(encoding="utf-8"))
    dataset_info = json.loads((dataset_dest / "dataset_info.json").read_text(encoding="utf-8"))
    assert model_info["hf_endpoint"] == "https://hf-mirror.com"
    assert dataset_info["hf_endpoint"] == "https://hf-mirror.com"
    assert model_info["hf_home"] == str((tmp_path / "huggingface-cache").resolve())
    assert dataset_info["hf_home"] == str((tmp_path / "huggingface-cache").resolve())


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


def test_build_readiness_report_includes_remote_asset_guidance_and_revalidation_scope(tmp_path: Path):
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
                },
                {
                    "id": "remote-huggingface-endpoint",
                    "status": "ok",
                    "summary": "remote endpoint reachable",
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
                        "framework_path": "pta",
                    },
                    "remote_assets": {
                        "hf_endpoint": "https://hf-mirror.com",
                        "hf_endpoint_source": "default",
                        "endpoint_reachable": True,
                        "cache_layout": {
                            "source": "working_dir_default",
                            "hf_home": str(tmp_path / "huggingface-cache"),
                            "hub_cache": str(tmp_path / "huggingface-cache" / "hub"),
                            "datasets_cache": str(tmp_path / "huggingface-cache" / "datasets"),
                            "hub_cache_writable": True,
                            "datasets_cache_writable": True,
                        },
                        "assets": {
                            "model_path": {"repo_id": "Qwen/Qwen3-0.6B"},
                            "dataset_path": {"repo_id": "karthiksagarn/astro_horoscope"},
                        },
                    },
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
                "needs_revalidation": ["workspace-assets"],
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
    _, verdict = load_report_pair(report_json)
    markdown = report_md.read_text(encoding="utf-8")

    assert verdict["remote_asset_guidance"]["mode"] == "remote-assets"
    assert verdict["remote_asset_guidance"]["hf_endpoint"] == "https://hf-mirror.com"
    assert verdict["revalidated"] is True
    assert verdict["revalidation_covered_scopes"] == ["framework", "workspace-assets"]
    assert "## Remote Asset Guidance" in markdown
    assert "huggingface-cache" in markdown


def test_build_readiness_report_guides_against_system_python_when_workspace_env_missing(tmp_path: Path):
    target_path = tmp_path / "target.json"
    normalized_path = tmp_path / "normalized.json"
    checks_path = tmp_path / "checks.json"
    closure_path = tmp_path / "closure.json"
    report_json = tmp_path / "report.json"
    report_md = tmp_path / "report.md"

    target_path.write_text(
        json.dumps(
            {
                "target_type": "inference",
                "entry_script": "inference.py",
                "framework_path": "pta",
            }
        ),
        encoding="utf-8",
    )
    normalized_path.write_text(
        json.dumps(
            {
                "blockers": [
                    "No usable workspace-local Python environment is available for readiness checks. Do not use system python or pip for remediation."
                ],
                "warnings": [],
                "blockers_detailed": [
                    {
                        "id": "python-selected-env",
                        "category": "env_remediable",
                        "severity": "high",
                        "summary": "No usable workspace-local Python environment is available for readiness checks. Do not use system python or pip for remediation.",
                        "evidence": ["system_python_fallback=forbidden"],
                        "remediable": True,
                        "remediation_owner": "readiness-agent",
                        "revalidation_scope": ["python-environment", "framework"],
                    }
                ],
                "warnings_detailed": [],
            }
        ),
        encoding="utf-8",
    )
    checks_path.write_text(
        json.dumps(
            [
                {
                    "id": "python-selected-env",
                    "status": "block",
                    "summary": "No usable workspace-local Python environment is available for readiness checks. Do not use system python or pip for remediation.",
                }
            ]
        ),
        encoding="utf-8",
    )
    closure_path.write_text(
        json.dumps(
            {
                "layers": {
                    "python_environment": {
                        "selected_env_root": None,
                        "probe_python_path": None,
                    }
                }
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
        "--output-json",
        str(report_json),
        "--output-md",
        str(report_md),
    )
    envelope, verdict = load_report_pair(report_json)
    markdown = report_md.read_text(encoding="utf-8")
    assert envelope["status"] == "partial"
    assert verdict["status"] == "BLOCKED"
    assert "do not use system python or pip" in verdict["next_action"].lower()
    assert verdict["selected_environment_guidance"]["system_python_allowed"] is False
    assert "Environment Guidance" in markdown
    assert "system_python_allowed: `false`" in markdown
    assert "Do not use system python or pip" in markdown


def test_build_readiness_report_separates_manual_and_auto_blockers(tmp_path: Path):
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
                "framework_path": "pta",
            }
        ),
        encoding="utf-8",
    )
    normalized_path.write_text(
        json.dumps(
            {
                "blockers": [
                    "Dataset path is missing.",
                    "uv is missing from the selected execution path.",
                ],
                "warnings": [],
                "blockers_detailed": [
                    {
                        "id": "workspace-dataset-path",
                        "category": "workspace_manual",
                        "summary": "Dataset path is missing.",
                    },
                    {
                        "id": "python-uv",
                        "category": "env_remediable",
                        "summary": "uv is missing from the selected execution path.",
                    },
                ],
                "warnings_detailed": [],
            }
        ),
        encoding="utf-8",
    )
    checks_path.write_text(
        json.dumps(
            [
                {
                    "id": "workspace-dataset-path",
                    "status": "block",
                    "summary": "Dataset path is missing.",
                },
                {
                    "id": "python-uv",
                    "status": "block",
                    "summary": "uv is missing from the selected execution path.",
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
    markdown = report_md.read_text(encoding="utf-8")
    assert envelope["status"] == "partial"
    assert verdict["status"] == "BLOCKED"
    assert "manual workspace blockers remain" in verdict["summary"].lower()
    assert "dataset or config paths first" in verdict["next_action"].lower()
    assert "## Manual Blockers" in markdown
    assert "## Auto-Remediable Blockers" in markdown


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
    assert "https://pypi.tuna.tsinghua.edu.cn/simple" in calls[0]
    assert "torch" in calls[0]
    assert "torch_npu" not in calls[0]
    assert "--index-url" in calls[1]
    assert "https://pypi.tuna.tsinghua.edu.cn/simple" in calls[1]
    assert "torch_npu" in calls[1]


def test_execute_env_fix_prefers_tsinghua_mirror_for_runtime_dependency(tmp_path: Path):
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
                        "action_type": "install_runtime_dependency",
                        "allowed": True,
                        "requires_confirmation": False,
                        "reason": "Missing runtime dependency.",
                        "revalidation_scope": ["runtime-dependencies"],
                        "package_names": ["datasets"],
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
        env=env,
    )

    calls = [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    result = json.loads(result_path.read_text(encoding="utf-8"))

    assert len(calls) == 1
    assert calls[0][:4] == ["pip", "install", "--python", str(python_path)]
    assert "--index-url" in calls[0]
    assert "https://pypi.tuna.tsinghua.edu.cn/simple" in calls[0]
    assert result["results"][0]["status"] == "executed"


def test_execute_env_fix_falls_back_after_default_tsinghua_mirror_failure(tmp_path: Path):
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
                        "action_type": "install_runtime_dependency",
                        "allowed": True,
                        "requires_confirmation": False,
                        "reason": "Missing runtime dependency.",
                        "revalidation_scope": ["runtime-dependencies"],
                        "package_names": ["sentencepiece"],
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
argv = sys.argv[1:]
log_path.open("a", encoding="utf-8").write(json.dumps(argv) + "\\n")
if "--index-url" in argv and "https://pypi.tuna.tsinghua.edu.cn/simple" in argv:
    sys.stderr.write("mirror failed\\n")
    raise SystemExit(1)
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
        env=env,
    )

    calls = [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    result = json.loads(result_path.read_text(encoding="utf-8"))

    assert len(calls) == 2
    assert "--index-url" in calls[0]
    assert "https://pypi.tuna.tsinghua.edu.cn/simple" in calls[0]
    assert "--index-url" in calls[1]
    assert "https://mirrors.aliyun.com/pypi/simple/" in calls[1]
    assert result["results"][0]["status"] == "executed"
    assert "fell back to the mirror https://mirrors.aliyun.com/pypi/simple/" in result["results"][0]["reason"]


def test_execute_env_fix_ignores_non_mirror_pip_index_override(tmp_path: Path):
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
                        "action_type": "install_runtime_dependency",
                        "allowed": True,
                        "requires_confirmation": False,
                        "reason": "Missing runtime dependency.",
                        "revalidation_scope": ["runtime-dependencies"],
                        "package_names": ["datasets"],
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
    env["PIP_INDEX_URL"] = "https://pypi.org/simple"

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
        env=env,
    )

    calls = [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert len(calls) == 1
    assert "--index-url" in calls[0]
    assert "https://pypi.tuna.tsinghua.edu.cn/simple" in calls[0]
    assert "https://pypi.org/simple" not in calls[0]
