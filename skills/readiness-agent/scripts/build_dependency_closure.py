#!/usr/bin/env python3
import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from python_selection import resolve_selected_python
from runtime_env import detect_ascend_runtime, resolve_runtime_environment


RUNTIME_IMPORT_CANDIDATES = {
    "mindspore",
    "torch",
    "torch_npu",
    "transformers",
    "datasets",
    "tokenizers",
    "accelerate",
    "safetensors",
    "diffusers",
}

PROBE_CODE = """
import importlib.util
import json
import sys

mode = sys.argv[1]
payload = json.loads(sys.argv[2])

if mode == "import":
    packages = payload.get("packages", [])
    print(json.dumps({name: importlib.util.find_spec(name) is not None for name in packages}))
elif mode == "framework_smoke":
    framework_path = payload.get("framework_path")
    result = {"success": False, "details": [], "error": None}
    try:
        if framework_path == "mindspore":
            import mindspore as ms
            _ = getattr(ms, "Tensor", None)
            result["details"].append("mindspore import ok")
            result["success"] = True
        elif framework_path == "pta":
            import torch
            import torch_npu
            _ = getattr(torch, "Tensor", None)
            result["details"].extend(["torch import ok", "torch_npu import ok"])
            result["success"] = True
        elif framework_path == "mixed":
            import mindspore as ms
            import torch
            import torch_npu
            _ = getattr(ms, "Tensor", None)
            _ = getattr(torch, "Tensor", None)
            result["details"].extend(["mindspore import ok", "torch import ok", "torch_npu import ok"])
            result["success"] = True
        else:
            result["error"] = f"unsupported framework path: {framework_path}"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    print(json.dumps(result))
else:
    print(json.dumps({"error": f"unknown mode: {mode}"}))
"""


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def extract_runtime_imports(entry_script: Optional[Path]) -> List[str]:
    if not entry_script or not entry_script.exists():
        return []
    text = read_text(entry_script)
    found = []
    for name in sorted(RUNTIME_IMPORT_CANDIDATES):
        if f"import {name}" in text or f"from {name}" in text:
            found.append(name)
    return found


def detect_output_path(target: dict, root: Path, entry_script: Optional[Path]) -> Optional[str]:
    if target.get("output_path"):
        return str(target["output_path"])
    if entry_script:
        text = read_text(entry_script)
        for token in ("output_dir", "save_dir", "ckpt_dir"):
            if token in text:
                return "./outputs"
    return None


def run_json_probe_with_python(
    python_path: Path,
    mode: str,
    payload: dict,
    probe_env: Optional[Dict[str, str]] = None,
) -> Tuple[dict, Optional[str]]:
    try:
        completed = subprocess.run(
            [str(python_path), "-c", PROBE_CODE, mode, json.dumps(payload)],
            check=True,
            text=True,
            capture_output=True,
            timeout=10,
            env=probe_env,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {}, str(exc)

    try:
        result = json.loads(completed.stdout.strip() or "{}")
    except json.JSONDecodeError:
        return {}, "probe returned non-JSON output"

    if not isinstance(result, dict):
        return {}, "probe returned a non-object payload"
    return result, None


def run_import_probe_with_python(
    python_path: Path,
    packages: List[str],
    probe_env: Optional[Dict[str, str]] = None,
) -> Tuple[Dict[str, bool], Optional[str]]:
    if not packages:
        return {}, None

    result, error = run_json_probe_with_python(
        python_path,
        "import",
        {"packages": packages},
        probe_env,
    )
    if error:
        return {package: False for package in packages}, error

    return {package: bool(result.get(package, False)) for package in packages}, None


def run_framework_smoke_with_python(
    python_path: Path,
    framework_path: str,
    probe_env: Optional[Dict[str, str]] = None,
) -> Tuple[dict, Optional[str]]:
    result, error = run_json_probe_with_python(
        python_path,
        "framework_smoke",
        {"framework_path": framework_path},
        probe_env,
    )
    if error:
        return {
            "status": "failed",
            "details": [],
            "error": error,
        }, error

    return {
        "status": "passed" if result.get("success") else "failed",
        "details": result.get("details") or [],
        "error": result.get("error"),
    }, None


def probe_imports(
    packages: List[str],
    python_layer: dict,
    probe_env: Optional[Dict[str, str]] = None,
) -> Tuple[Dict[str, bool], Optional[str]]:
    if not packages:
        return {}, None

    probe_python_path = python_layer.get("probe_python_path")
    if probe_python_path:
        return run_import_probe_with_python(Path(probe_python_path), packages, probe_env)

    return {package: False for package in packages}, "probe python path is unavailable"


def probe_framework_smoke(
    framework_path: str,
    python_layer: dict,
    import_probes: dict[str, bool],
    probe_env: Optional[Dict[str, str]] = None,
) -> dict:
    if framework_path not in {"mindspore", "pta", "mixed"}:
        return {
            "status": "unsupported",
            "details": [],
            "error": None,
        }

    if not import_probes or not all(import_probes.values()):
        return {
            "status": "skipped",
            "details": [],
            "error": "framework imports are incomplete",
        }

    probe_python_path = python_layer.get("probe_python_path")
    if not probe_python_path:
        return {
            "status": "failed",
            "details": [],
            "error": "probe python path is unavailable",
        }

    result, _ = run_framework_smoke_with_python(Path(probe_python_path), framework_path, probe_env)
    return result


def build_python_layer(target: dict, root: Path, system_layer: dict) -> dict:
    uv_path = shutil.which("uv")
    selection = resolve_selected_python(
        root=root,
        selected_python=target.get("selected_python"),
        selected_env_root=target.get("selected_env_root"),
    )
    selection_status = str(selection.get("selection_status") or "missing")
    probe_python_path = selection.get("selected_python") if selection_status == "selected" else None
    return {
        "tooling": {
            "uv_path": uv_path,
            "uv_available": bool(uv_path),
        },
        "selected_env_root": selection.get("selected_env_root"),
        "selected_python": selection.get("selected_python"),
        "selection_status": selection_status,
        "selection_reason": selection.get("selection_reason"),
        "selection_source": selection.get("selection_source"),
        "python_version": selection.get("python_version"),
        "helper_python_compatible": selection.get("helper_python_compatible"),
        "probe_python_path": probe_python_path,
        "probe_source": selection.get("selection_source"),
        "python_path": probe_python_path,
        "runtime_env_source": system_layer.get("probe_env_source"),
        "runtime_env_error": system_layer.get("probe_env_error"),
    }


def build_framework_layer(
    target: dict,
    python_layer: dict,
    probe_env: Optional[Dict[str, str]] = None,
) -> dict:
    framework_path = target.get("framework_path") or "unknown"
    required_packages: List[str] = []
    if framework_path == "mindspore":
        required_packages = ["mindspore"]
    elif framework_path == "pta":
        required_packages = ["torch", "torch_npu"]
    elif framework_path == "mixed":
        required_packages = ["mindspore", "torch", "torch_npu"]
    import_probes, probe_error = probe_imports(required_packages, python_layer, probe_env)
    smoke_prerequisite = probe_framework_smoke(framework_path, python_layer, import_probes, probe_env)
    return {
        "framework_path": framework_path,
        "required_packages": required_packages,
        "import_probes": import_probes,
        "probe_source": python_layer.get("probe_source"),
        "probe_python_path": python_layer.get("probe_python_path"),
        "probe_error": probe_error,
        "smoke_prerequisite": smoke_prerequisite,
        "compatibility_status": "unknown",
    }


def build_runtime_layer(
    entry_script: Optional[Path],
    python_layer: dict,
    probe_env: Optional[Dict[str, str]] = None,
) -> dict:
    required_imports = extract_runtime_imports(entry_script)
    import_probes, probe_error = probe_imports(required_imports, python_layer, probe_env)
    return {
        "required_imports": required_imports,
        "import_probes": import_probes,
        "probe_source": python_layer.get("probe_source"),
        "probe_python_path": python_layer.get("probe_python_path"),
        "probe_error": probe_error,
    }


def build_workspace_layer(target: dict, root: Path, target_type: str, entry_script: Optional[Path]) -> dict:
    def file_state(value: Optional[str]) -> dict:
        if not value:
            return {"path": None, "exists": False, "required": False}
        path = Path(value)
        path = path if path.is_absolute() else (root / path)
        return {"path": str(path.relative_to(root) if path.exists() or path.is_relative_to(root) else path), "exists": path.exists(), "required": True}

    entry_state = file_state(target.get("entry_script"))
    config_state = file_state(target.get("config_path"))
    model_state = file_state(target.get("model_path"))
    dataset_state = file_state(target.get("dataset_path"))
    checkpoint_state = file_state(target.get("checkpoint_path"))
    output_path = detect_output_path(target, root, entry_script)
    output_state = file_state(output_path)
    if output_path:
        output_state["required"] = target_type == "training"

    dataset_state["required"] = target_type == "training"
    model_state["required"] = True

    return {
        "entry_script": entry_state,
        "config_path": config_state,
        "model_path": model_state,
        "dataset_path": dataset_state,
        "checkpoint_path": checkpoint_state,
        "output_path": output_state,
    }


def build_task_layer(target: dict) -> dict:
    target_type = target.get("target_type") or "unknown"
    if target_type == "training":
        smoke_path = [
            "config parse",
            "dataset openability",
            "model construction",
            "train-step smoke",
        ]
    elif target_type == "inference":
        smoke_path = [
            "model load",
            "tokenizer load",
            "forward or generation smoke",
        ]
    else:
        smoke_path = []
    return {
        "target_type": target_type,
        "minimum_smoke_path": smoke_path,
        "launch_cmd": target.get("launch_cmd"),
    }


def build_dependency_closure(target: dict, root: Path) -> dict:
    entry_script = None
    if target.get("entry_script"):
        entry_script = Path(target["entry_script"])
        if not entry_script.is_absolute():
            entry_script = root / entry_script

    target_type = target.get("target_type") or "unknown"
    system_layer = detect_ascend_runtime()
    probe_env, probe_env_source, probe_env_error = resolve_runtime_environment(system_layer)
    system_layer["probe_env_source"] = probe_env_source
    system_layer["probe_env_error"] = probe_env_error
    python_layer = build_python_layer(target, root, system_layer)
    layers = {
        "system": system_layer,
        "python_environment": python_layer,
        "framework": build_framework_layer(target, python_layer, probe_env),
        "runtime_dependencies": build_runtime_layer(entry_script, python_layer, probe_env),
        "workspace_assets": build_workspace_layer(target, root, target_type, entry_script),
        "task_execution": build_task_layer(target),
    }

    missing_required = []
    workspace_assets = layers["workspace_assets"]
    for key, item in workspace_assets.items():
        if item["required"] and not item["exists"]:
            missing_required.append(key)

    return {
        "working_dir": str(root),
        "target_type": target_type,
        "layers": layers,
        "missing_required": missing_required,
        "complete_for_static_validation": not missing_required,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a dependency closure for readiness-agent")
    parser.add_argument("--target-json", required=True, help="path to execution target JSON")
    parser.add_argument("--output-json", required=True, help="path to output dependency closure JSON")
    args = parser.parse_args()

    target = json.loads(Path(args.target_json).read_text(encoding="utf-8"))
    root = Path(target["working_dir"]).resolve()
    closure = build_dependency_closure(target, root)
    Path(args.output_json).write_text(json.dumps(closure, indent=2), encoding="utf-8")
    print(json.dumps({"target_type": closure["target_type"], "missing_required": closure["missing_required"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
