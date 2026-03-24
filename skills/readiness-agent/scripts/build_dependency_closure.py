#!/usr/bin/env python3
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


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

WORKSPACE_ENV_CANDIDATES = (
    ".venv",
    "venv",
    ".env",
    "env",
)

PYTHON_RELATIVE_CANDIDATES = (
    Path("bin/python"),
    Path("bin/python3"),
    Path("Scripts/python.exe"),
    Path("Scripts/python"),
)

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


def resolve_optional_path(value: str | None, root: Path) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    return path


def extract_runtime_imports(entry_script: Path | None) -> list[str]:
    if not entry_script or not entry_script.exists():
        return []
    text = read_text(entry_script)
    found = []
    for name in sorted(RUNTIME_IMPORT_CANDIDATES):
        if f"import {name}" in text or f"from {name}" in text:
            found.append(name)
    return found


def detect_output_path(target: dict, root: Path, entry_script: Path | None) -> str | None:
    if target.get("output_path"):
        return str(target["output_path"])
    if entry_script:
        text = read_text(entry_script)
        for token in ("output_dir", "save_dir", "ckpt_dir"):
            if token in text:
                return "./outputs"
    return None


def build_system_layer() -> dict:
    ascend_env = Path("/usr/local/Ascend/ascend-toolkit/set_env.sh")
    return {
        "requires_ascend": True,
        "device_paths_present": any(Path("/dev").glob("davinci*")),
        "ascend_env_script_present": ascend_env.exists(),
    }


def discover_selected_env_root(target: dict, root: Path) -> tuple[Path | None, str | None]:
    explicit = resolve_optional_path(target.get("selected_env_root"), root)
    if explicit:
        return explicit, "explicit_input"

    for candidate in WORKSPACE_ENV_CANDIDATES:
        path = root / candidate
        if path.exists() and path.is_dir():
            return path, "workspace_inference"
    return None, None


def resolve_probe_python(
    target: dict,
    root: Path,
    selected_env_root: Path | None,
) -> tuple[Path | None, str]:
    explicit_python = resolve_optional_path(target.get("selected_python"), root)
    if explicit_python:
        return explicit_python, "explicit_python"

    if selected_env_root:
        for candidate in PYTHON_RELATIVE_CANDIDATES:
            python_path = selected_env_root / candidate
            if python_path.exists() and python_path.is_file():
                return python_path, "selected_env"
        return None, "selected_env_missing_python"

    current_python = Path(sys.executable).resolve()
    return current_python, "current_interpreter"


def run_json_probe_with_python(
    python_path: Path,
    mode: str,
    payload: dict,
) -> tuple[dict, str | None]:
    try:
        completed = subprocess.run(
            [str(python_path), "-c", PROBE_CODE, mode, json.dumps(payload)],
            check=True,
            text=True,
            capture_output=True,
            timeout=10,
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


def run_import_probe_with_python(python_path: Path, packages: list[str]) -> tuple[dict[str, bool], str | None]:
    if not packages:
        return {}, None

    result, error = run_json_probe_with_python(
        python_path,
        "import",
        {"packages": packages},
    )
    if error:
        return {package: False for package in packages}, error

    return {package: bool(result.get(package, False)) for package in packages}, None


def run_framework_smoke_with_python(
    python_path: Path,
    framework_path: str,
) -> tuple[dict, str | None]:
    result, error = run_json_probe_with_python(
        python_path,
        "framework_smoke",
        {"framework_path": framework_path},
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


def probe_imports(packages: list[str], python_layer: dict) -> tuple[dict[str, bool], str | None]:
    if not packages:
        return {}, None

    probe_python_path = python_layer.get("probe_python_path")
    if probe_python_path:
        return run_import_probe_with_python(Path(probe_python_path), packages)

    return {package: False for package in packages}, "probe python path is unavailable"


def probe_framework_smoke(framework_path: str, python_layer: dict, import_probes: dict[str, bool]) -> dict:
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

    result, _ = run_framework_smoke_with_python(Path(probe_python_path), framework_path)
    return result


def build_python_layer(target: dict, root: Path) -> dict:
    uv_path = shutil.which("uv")
    selected_env_root, selected_env_source = discover_selected_env_root(target, root)
    probe_python_path, probe_source = resolve_probe_python(target, root, selected_env_root)
    return {
        "tooling": {
            "uv_path": uv_path,
            "uv_available": bool(uv_path),
        },
        "selected_env_root": str(selected_env_root) if selected_env_root else None,
        "selected_env_source": selected_env_source,
        "probe_python_path": str(probe_python_path) if probe_python_path else None,
        "probe_source": probe_source,
        "python_path": str(probe_python_path) if probe_python_path else (shutil.which("python3") or shutil.which("python")),
    }


def build_framework_layer(target: dict, python_layer: dict) -> dict:
    framework_path = target.get("framework_path") or "unknown"
    required_packages: list[str] = []
    if framework_path == "mindspore":
        required_packages = ["mindspore"]
    elif framework_path == "pta":
        required_packages = ["torch", "torch_npu"]
    elif framework_path == "mixed":
        required_packages = ["mindspore", "torch", "torch_npu"]
    import_probes, probe_error = probe_imports(required_packages, python_layer)
    smoke_prerequisite = probe_framework_smoke(framework_path, python_layer, import_probes)
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


def build_runtime_layer(entry_script: Path | None, python_layer: dict) -> dict:
    required_imports = extract_runtime_imports(entry_script)
    import_probes, probe_error = probe_imports(required_imports, python_layer)
    return {
        "required_imports": required_imports,
        "import_probes": import_probes,
        "probe_source": python_layer.get("probe_source"),
        "probe_python_path": python_layer.get("probe_python_path"),
        "probe_error": probe_error,
    }


def build_workspace_layer(target: dict, root: Path, target_type: str, entry_script: Path | None) -> dict:
    def file_state(value: str | None) -> dict:
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
    python_layer = build_python_layer(target, root)
    layers = {
        "system": build_system_layer(),
        "python_environment": python_layer,
        "framework": build_framework_layer(target, python_layer),
        "runtime_dependencies": build_runtime_layer(entry_script, python_layer),
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
