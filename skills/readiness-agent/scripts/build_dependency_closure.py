#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import subprocess
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ascend_compat import assess_installed_framework_compatibility, resolve_framework_compatibility
from python_selection import resolve_selected_python
from runtime_env import detect_ascend_runtime, detect_cann_version, resolve_runtime_environment


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
    "peft",
    "trl",
    "evaluate",
    "sentencepiece",
}

ASCEND_HIDDEN_RUNTIME_DEPENDENCIES = [
    {
        "import_name": "decorator",
        "package_name": "decorator",
        "framework_paths": ["mindspore", "pta", "mixed"],
        "required_for": "ascend-compiler",
        "reason": "Ascend compiler adapters import decorator during graph and operator compilation.",
    },
    {
        "import_name": "scipy",
        "package_name": "scipy",
        "framework_paths": ["mindspore", "pta", "mixed"],
        "required_for": "ascend-compiler",
        "reason": "Ascend compiler adapters import scipy during TBE or AOE initialization.",
    },
    {
        "import_name": "attr",
        "package_name": "attrs",
        "framework_paths": ["mindspore", "pta", "mixed"],
        "required_for": "ascend-compiler",
        "reason": "Ascend compiler adapters import attr from the attrs package during TBE or AOE initialization.",
    },
]

TRANSFORMERS_COMMON_RUNTIME_DEPENDENCIES = [
    {
        "import_name": "accelerate",
        "package_name": "accelerate",
        "required_for": "transformers-runtime",
        "reason": "Complete Transformers engineering workflows commonly rely on Accelerate for loading, placement, and training orchestration.",
    },
]

DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"

PROBE_CODE = """
import importlib.util
import json
import sys

mode = sys.argv[1]
payload = json.loads(sys.argv[2])

if mode == "import":
    packages = payload.get("packages", [])
    print(json.dumps({name: importlib.util.find_spec(name) is not None for name in packages}))
elif mode == "package_versions":
    try:
        from importlib import metadata as importlib_metadata
    except ImportError:
        import importlib_metadata
    packages = payload.get("packages", [])
    result = {"versions": {}, "errors": {}}
    for name in packages:
        try:
            module = __import__(name)
            version = getattr(module, "__version__", None)
            if version is None:
                candidates = [name]
                dashed_name = name.replace("_", "-")
                if dashed_name not in candidates:
                    candidates.append(dashed_name)
                for candidate in candidates:
                    try:
                        version = importlib_metadata.version(candidate)
                        break
                    except Exception:
                        continue
            result["versions"][name] = version
        except Exception as exc:
            result["versions"][name] = None
            result["errors"][name] = f"{type(exc).__name__}: {exc}"
    print(json.dumps(result))
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


def target_runtime_profile(target: dict) -> List[dict]:
    profile = target.get("expected_runtime_profile") or []
    if not isinstance(profile, list):
        return []
    result: List[dict] = []
    for item in profile:
        if not isinstance(item, dict):
            continue
        import_name = str(item.get("import_name") or "").strip()
        package_name = str(item.get("package_name") or "").strip()
        if not import_name or not package_name:
            continue
        result.append(
            {
                "import_name": import_name,
                "package_name": package_name,
                "required_for": str(item.get("required_for") or "target-hint").strip() or "target-hint",
                "reason": str(item.get("reason") or "Runtime dependency derived from target metadata.").strip(),
            }
        )
    return result


def transformers_common_runtime_profile(
    entry_script: Optional[Path],
    explicit_imports: List[str],
) -> List[dict]:
    if "transformers" not in explicit_imports:
        return []

    return [
        {
            "import_name": item["import_name"],
            "package_name": item["package_name"],
            "required_for": item["required_for"],
            "reason": item["reason"],
        }
        for item in TRANSFORMERS_COMMON_RUNTIME_DEPENDENCIES
    ]


def ascend_hidden_runtime_profile(framework_path: str, system_layer: dict) -> List[dict]:
    ascend_evidence_present = bool(
        system_layer.get("device_paths_present")
        or system_layer.get("ascend_env_script_present")
        or system_layer.get("ascend_env_active")
    )
    if not ascend_evidence_present:
        return []
    if framework_path not in {"mindspore", "pta", "mixed"}:
        return []

    return [
        {
            "import_name": item["import_name"],
            "package_name": item["package_name"],
            "required_for": item["required_for"],
            "reason": item["reason"],
        }
        for item in ASCEND_HIDDEN_RUNTIME_DEPENDENCIES
        if framework_path in item["framework_paths"]
    ]


def normalize_hf_endpoint(value: Optional[str]) -> str:
    endpoint = (value or DEFAULT_HF_ENDPOINT).strip()
    if "://" not in endpoint:
        endpoint = f"https://{endpoint}"
    return endpoint.rstrip("/")


def nearest_existing_parent(path: Path) -> Path:
    current = path
    while not current.exists() and current != current.parent:
        current = current.parent
    return current


def path_is_writable(path: Path) -> bool:
    candidate = path if path.exists() else nearest_existing_parent(path)
    return os.access(candidate, os.W_OK)


def resolve_hf_cache_layout(root: Path) -> dict:
    hub_cache_env = os.environ.get("HUGGINGFACE_HUB_CACHE")
    datasets_cache_env = os.environ.get("HF_DATASETS_CACHE")
    hf_home_env = os.environ.get("HF_HOME")

    if hub_cache_env or datasets_cache_env:
        source = "explicit_cache_env"
        hf_home = Path(hf_home_env).resolve() if hf_home_env else None
        hub_cache = Path(hub_cache_env).resolve() if hub_cache_env else ((hf_home / "hub") if hf_home else (root / "huggingface-cache" / "hub"))
        datasets_cache = (
            Path(datasets_cache_env).resolve()
            if datasets_cache_env
            else ((hf_home / "datasets") if hf_home else (root / "huggingface-cache" / "datasets"))
        )
    elif hf_home_env:
        source = "hf_home"
        hf_home = Path(hf_home_env).resolve()
        hub_cache = hf_home / "hub"
        datasets_cache = hf_home / "datasets"
    else:
        source = "working_dir_default"
        hf_home = (root / "huggingface-cache").resolve()
        hub_cache = hf_home / "hub"
        datasets_cache = hf_home / "datasets"

    return {
        "source": source,
        "hf_home": str(hf_home) if hf_home else None,
        "hub_cache": str(hub_cache),
        "datasets_cache": str(datasets_cache),
        "hub_cache_writable": path_is_writable(hub_cache),
        "datasets_cache_writable": path_is_writable(datasets_cache),
    }


def probe_hf_endpoint(endpoint: str) -> Tuple[bool, Optional[str]]:
    parsed = urlparse(endpoint)
    if not parsed.scheme or not parsed.netloc:
        return False, "HF endpoint is missing a host"

    probe_urls = [
        endpoint,
        urljoin(endpoint + "/", "api/models/Qwen/Qwen3-0.6B"),
    ]
    try:
        for probe_url in probe_urls:
            request = Request(probe_url, method="HEAD", headers={"User-Agent": "readiness-agent/0.1"})
            with urlopen(request, timeout=5) as response:
                status = getattr(response, "status", None) or response.getcode()
                if status and int(status) < 500:
                    return True, None
    except Exception as exc:
        return False, str(exc)
    return False, "HF endpoint probe did not return a successful HTTP response"
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


def run_package_version_probe_with_python(
    python_path: Path,
    packages: List[str],
    probe_env: Optional[Dict[str, str]] = None,
) -> Tuple[Dict[str, Optional[str]], Dict[str, str], Optional[str]]:
    if not packages:
        return {}, {}, None

    result, error = run_json_probe_with_python(
        python_path,
        "package_versions",
        {"packages": packages},
        probe_env,
    )
    if error:
        return {package: None for package in packages}, {}, error

    raw_versions = result.get("versions") or {}
    raw_errors = result.get("errors") or {}
    versions: Dict[str, Optional[str]] = {}
    errors: Dict[str, str] = {}
    for package in packages:
        version = raw_versions.get(package)
        versions[package] = str(version).strip() if isinstance(version, str) and str(version).strip() else None
        probe_error = raw_errors.get(package)
        if probe_error:
            errors[package] = str(probe_error)
    return versions, errors, None


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


def probe_package_versions(
    packages: List[str],
    python_layer: dict,
    import_probes: Dict[str, bool],
    probe_env: Optional[Dict[str, str]] = None,
) -> Tuple[Dict[str, Optional[str]], Dict[str, str], Optional[str]]:
    if not packages or not import_probes or not all(import_probes.get(package, False) for package in packages):
        return {}, {}, None

    probe_python_path = python_layer.get("probe_python_path")
    if not probe_python_path:
        return {}, {}, "probe python path is unavailable"

    return run_package_version_probe_with_python(Path(probe_python_path), packages, probe_env)


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
        "system_python_fallback_allowed": False,
    }


def build_framework_layer(
    target: dict,
    python_layer: dict,
    system_layer: dict,
    probe_env: Optional[Dict[str, str]] = None,
) -> dict:
    framework_path = str(target.get("framework_path") or "unknown")
    if framework_path == "unknown":
        target_type = str(target.get("target_type") or "").strip().lower()
        framework_hint = str(target.get("framework_hint") or "").strip().lower()
        if target_type == "training" and framework_hint != "mindspore":
            framework_path = "pta"
    required_packages: List[str] = []
    if framework_path == "mindspore":
        required_packages = ["mindspore"]
    elif framework_path == "pta":
        required_packages = ["torch", "torch_npu"]
    elif framework_path == "mixed":
        required_packages = ["mindspore", "torch", "torch_npu"]
    compatibility = resolve_framework_compatibility(
        framework_path,
        system_layer.get("cann_version"),
        python_layer.get("python_version"),
    )
    import_probes, probe_error = probe_imports(required_packages, python_layer, probe_env)
    smoke_prerequisite = probe_framework_smoke(framework_path, python_layer, import_probes, probe_env)
    installed_package_versions, version_probe_errors, version_probe_error = probe_package_versions(
        required_packages,
        python_layer,
        import_probes,
        probe_env,
    )
    installed_compatibility = assess_installed_framework_compatibility(
        framework_path,
        system_layer.get("cann_version"),
        python_layer.get("python_version"),
        installed_package_versions,
    )
    return {
        "framework_path": framework_path,
        "required_packages": required_packages,
        "resolved_package_specs": compatibility.get("package_specs", []),
        "import_probes": import_probes,
        "probe_source": python_layer.get("probe_source"),
        "probe_python_path": python_layer.get("probe_python_path"),
        "probe_error": probe_error,
        "smoke_prerequisite": smoke_prerequisite,
        "installed_package_versions": installed_package_versions,
        "version_probe_errors": version_probe_errors,
        "version_probe_error": version_probe_error,
        "compatibility_status": compatibility.get("status", "unknown"),
        "compatibility_reference": compatibility,
        "installed_compatibility_status": installed_compatibility.get("status", "unknown"),
        "installed_compatibility_reference": installed_compatibility,
    }


def build_runtime_layer(
    target: dict,
    entry_script: Optional[Path],
    framework_path: str,
    system_layer: dict,
    python_layer: dict,
    probe_env: Optional[Dict[str, str]] = None,
) -> dict:
    explicit_imports = extract_runtime_imports(entry_script)
    for item in target_runtime_profile(target):
        import_name = str(item.get("import_name") or "").strip()
        if import_name and import_name not in explicit_imports:
            explicit_imports.append(import_name)

    implicit_profile = list(target_runtime_profile(target))
    for item in ascend_hidden_runtime_profile(framework_path, system_layer):
        import_name = str(item.get("import_name") or "").strip()
        if import_name and all(existing.get("import_name") != import_name for existing in implicit_profile):
            implicit_profile.append(item)
    for item in transformers_common_runtime_profile(entry_script, explicit_imports):
        import_name = str(item.get("import_name") or "").strip()
        if import_name and all(existing.get("import_name") != import_name for existing in implicit_profile):
            implicit_profile.append(item)
    required_imports: List[str] = list(explicit_imports)
    for item in implicit_profile:
        import_name = str(item.get("import_name") or "").strip()
        if import_name and import_name not in required_imports:
            required_imports.append(import_name)
    import_probes, probe_error = probe_imports(required_imports, python_layer, probe_env)
    return {
        "explicit_imports": explicit_imports,
        "implicit_dependency_profile": implicit_profile,
        "required_imports": required_imports,
        "import_probes": import_probes,
        "probe_source": python_layer.get("probe_source"),
        "probe_python_path": python_layer.get("probe_python_path"),
        "probe_error": probe_error,
    }


def build_remote_asset_layer(target: dict, root: Path, target_type: str) -> dict:
    cache_layout = resolve_hf_cache_layout(root)
    hf_endpoint = normalize_hf_endpoint(os.environ.get("HF_ENDPOINT"))
    assets: Dict[str, dict] = {}
    if target.get("model_hub_id"):
        assets["model_path"] = {
            "repo_id": target.get("model_hub_id"),
            "repo_type": "model",
        }
    if target_type == "training" and target.get("dataset_hub_id"):
        assets["dataset_path"] = {
            "repo_id": target.get("dataset_hub_id"),
            "repo_type": "dataset",
            "dataset_split": target.get("dataset_split"),
        }

    endpoint_reachable = None
    endpoint_error = None
    if assets:
        endpoint_reachable, endpoint_error = probe_hf_endpoint(hf_endpoint)
        if "model_path" in assets:
            assets["model_path"]["cache_path"] = cache_layout["hub_cache"]
            assets["model_path"]["ready"] = bool(
                endpoint_reachable and cache_layout.get("hub_cache_writable")
            )
        if "dataset_path" in assets:
            assets["dataset_path"]["cache_path"] = cache_layout["datasets_cache"]
            assets["dataset_path"]["ready"] = bool(
                endpoint_reachable and cache_layout.get("datasets_cache_writable")
            )

    return {
        "hf_endpoint": hf_endpoint,
        "hf_endpoint_source": "env" if os.environ.get("HF_ENDPOINT") else "default",
        "cache_layout": cache_layout,
        "endpoint_reachable": endpoint_reachable,
        "endpoint_error": endpoint_error,
        "assets": assets,
    }


def build_workspace_layer(
    target: dict,
    root: Path,
    target_type: str,
    entry_script: Optional[Path],
    remote_assets_layer: dict,
) -> dict:
    def file_state(value: Optional[str]) -> dict:
        if not value:
            return {"path": None, "exists": False, "required": False, "satisfied": False}
        path = Path(value)
        path = path if path.is_absolute() else (root / path)
        exists = path.exists()
        return {
            "path": str(path.relative_to(root) if exists or path.is_relative_to(root) else path),
            "exists": exists,
            "required": True,
            "satisfied": exists,
        }

    entry_state = file_state(target.get("entry_script"))
    config_state = file_state(target.get("config_path"))
    model_state = file_state(target.get("model_path"))
    dataset_state = file_state(target.get("dataset_path"))
    checkpoint_state = file_state(target.get("checkpoint_path"))

    dataset_state["required"] = target_type == "training"
    model_state["required"] = True
    entry_state["satisfied"] = entry_state["exists"] or not entry_state["required"]
    config_state["satisfied"] = config_state["exists"] or not config_state["required"]
    model_state["satisfied"] = model_state["exists"] or not model_state["required"]
    dataset_state["satisfied"] = dataset_state["exists"] or not dataset_state["required"]
    checkpoint_state["satisfied"] = checkpoint_state["exists"] or not checkpoint_state["required"]
    if target.get("example_recipe_id"):
        entry_state["source"] = "bundled-example"
        entry_state["template_path"] = target.get("example_template_path")
        entry_state["example_recipe_id"] = target.get("example_recipe_id")
        entry_state["reference_transformers_version"] = target.get("reference_transformers_version")
    if target.get("model_hub_id"):
        model_state["source"] = "huggingface"
        model_state["asset_provider"] = "huggingface"
        model_state["repo_id"] = target.get("model_hub_id")
        model_state["repo_type"] = "model"
        if not target.get("model_path"):
            model_state["resolution_mode"] = "remote-huggingface"
            model_state["satisfied"] = "model_path" in (remote_assets_layer.get("assets") or {})
    if target.get("dataset_hub_id"):
        dataset_state["source"] = "huggingface"
        dataset_state["asset_provider"] = "huggingface"
        dataset_state["repo_id"] = target.get("dataset_hub_id")
        dataset_state["repo_type"] = "dataset"
        dataset_state["dataset_split"] = target.get("dataset_split")
        if not target.get("dataset_path"):
            dataset_state["resolution_mode"] = "remote-huggingface"
            dataset_state["satisfied"] = "dataset_path" in (remote_assets_layer.get("assets") or {})

    return {
        "entry_script": entry_state,
        "config_path": config_state,
        "model_path": model_state,
        "dataset_path": dataset_state,
        "checkpoint_path": checkpoint_state,
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
    system_layer = detect_ascend_runtime(target)
    probe_env, probe_env_source, probe_env_error = resolve_runtime_environment(system_layer)
    system_layer["probe_env_source"] = probe_env_source
    system_layer["probe_env_error"] = probe_env_error
    system_layer.update(
        detect_cann_version(
            cann_path=target.get("cann_path"),
            script_path=system_layer.get("ascend_env_script_path"),
            environ=probe_env,
        )
    )
    python_layer = build_python_layer(target, root, system_layer)
    framework_layer = build_framework_layer(target, python_layer, system_layer, probe_env)
    remote_assets_layer = build_remote_asset_layer(target, root, target_type)
    layers = {
        "system": system_layer,
        "python_environment": python_layer,
        "framework": framework_layer,
        "runtime_dependencies": build_runtime_layer(
            target,
            entry_script,
            framework_layer.get("framework_path") or "unknown",
            system_layer,
            python_layer,
            probe_env,
        ),
        "remote_assets": remote_assets_layer,
        "workspace_assets": build_workspace_layer(target, root, target_type, entry_script, remote_assets_layer),
        "task_execution": build_task_layer(target),
    }

    missing_required = []
    workspace_assets = layers["workspace_assets"]
    for key, item in workspace_assets.items():
        if item["required"] and not item.get("satisfied", item.get("exists", False)):
            missing_required.append(key)

    return {
        "working_dir": str(root),
        "target_type": target_type,
        "layers": layers,
        "missing_required": missing_required,
        "complete_for_static_validation": not missing_required,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a dependency closure for readiness-agent",
        epilog=(
            "Internal helper. Use the top-level readiness workflow entrypoint instead of "
            "calling leaf helpers directly. Do not repair system Python when the workspace "
            "environment is unresolved."
        ),
    )
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
