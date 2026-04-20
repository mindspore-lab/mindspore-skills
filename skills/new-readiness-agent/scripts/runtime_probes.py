#!/usr/bin/env python3
import json
import shutil
import subprocess
from typing import Dict, List, Optional, Tuple


PROBE_CODE = """
import importlib.util
import json
import sys

mode = sys.argv[1]
payload = json.loads(sys.argv[2])

if mode == "import":
    packages = payload.get("packages", [])
    result = {"imports": {}, "errors": {}}
    for name in packages:
        try:
            __import__(name)
            result["imports"][name] = True
        except Exception as exc:
            result["imports"][name] = False
            result["errors"][name] = f"{type(exc).__name__}: {exc}"
    print(json.dumps(result))
elif mode == "package_versions":
    try:
        from importlib import metadata as importlib_metadata
    except ImportError:
        import importlib_metadata
    packages = payload.get("packages", [])
    result = {"versions": {}, "errors": {}}
    for name in packages:
        candidates = [name]
        dashed_name = name.replace("_", "-")
        if dashed_name not in candidates:
            candidates.append(dashed_name)
        version = None
        for candidate in candidates:
            try:
                version = importlib_metadata.version(candidate)
                break
            except Exception:
                continue
        try:
            if version is None:
                module = __import__(name)
                version = getattr(module, "__version__", None)
            result["versions"][name] = version
        except Exception as exc:
            result["versions"][name] = None
            result["errors"][name] = f"{type(exc).__name__}: {exc}"
    print(json.dumps(result))
else:
    print(json.dumps({"error": f"unknown mode: {mode}"}))
"""


def run_json_probe_with_python(
    python_path: Optional[str],
    mode: str,
    payload: Dict[str, object],
    probe_env: Optional[Dict[str, str]],
) -> Tuple[Dict[str, object], Optional[str]]:
    if not python_path:
        return {}, "python path is unavailable"
    launcher = f"exec({PROBE_CODE!r})"
    command = [str(python_path), "-c", launcher, mode, json.dumps(payload)]
    try:
        completed = subprocess.run(
            command,
            check=True,
            text=True,
            capture_output=True,
            timeout=20,
            env=probe_env,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {}, str(exc)
    stdout = completed.stdout.strip()
    if not stdout:
        return {}, "probe returned empty output"
    try:
        result = json.loads(stdout)
    except json.JSONDecodeError:
        return {}, "probe returned non-JSON output"
    if not isinstance(result, dict):
        return {}, "probe returned a non-object payload"
    if result.get("error"):
        return {}, str(result.get("error"))
    return result, None


def probe_imports(
    packages: List[str],
    python_path: Optional[str],
    probe_env: Optional[Dict[str, str]],
) -> Tuple[Dict[str, bool], Dict[str, str], Optional[str]]:
    normalized_imports: Dict[str, bool] = {}
    normalized_errors: Dict[str, str] = {}
    first_error: Optional[str] = None
    for name in packages:
        result, error = run_json_probe_with_python(python_path, "import", {"packages": [name]}, probe_env)
        imports = result.get("imports") if isinstance(result.get("imports"), dict) else {}
        errors = result.get("errors") if isinstance(result.get("errors"), dict) else {}
        normalized_imports[name] = bool(imports.get(name))
        if errors.get(name):
            normalized_errors[name] = str(errors[name])
        elif error:
            normalized_errors[name] = str(error)
            if first_error is None:
                first_error = error
    return normalized_imports, normalized_errors, first_error


def probe_package_versions(
    packages: List[str],
    python_path: Optional[str],
    probe_env: Optional[Dict[str, str]],
) -> Tuple[Dict[str, Optional[str]], Dict[str, str], Optional[str]]:
    normalized_versions: Dict[str, Optional[str]] = {}
    normalized_errors: Dict[str, str] = {}
    first_error: Optional[str] = None
    for name in packages:
        result, error = run_json_probe_with_python(python_path, "package_versions", {"packages": [name]}, probe_env)
        versions = result.get("versions") if isinstance(result.get("versions"), dict) else {}
        errors = result.get("errors") if isinstance(result.get("errors"), dict) else {}
        value = versions.get(name)
        normalized_versions[name] = None if value is None else str(value)
        if errors.get(name):
            normalized_errors[name] = str(errors[name])
        elif error:
            normalized_errors[name] = str(error)
            if first_error is None:
                first_error = error
    return normalized_versions, normalized_errors, first_error


def make_check(
    check_id: str,
    status: str,
    summary: str,
    evidence: Optional[List[str]] = None,
    **extra: object,
) -> Dict[str, object]:
    payload: Dict[str, object] = {
        "id": check_id,
        "status": status,
        "summary": summary,
        "evidence": list(evidence or []),
    }
    payload.update(extra)
    return payload


def summarize_framework_compatibility(compatibility: Dict[str, object]) -> Tuple[str, List[str], Dict[str, object]]:
    status = str(compatibility.get("status") or "unknown")
    installed_versions = compatibility.get("installed_versions") if isinstance(compatibility.get("installed_versions"), dict) else {}
    recommended_specs = compatibility.get("recommended_package_specs") if isinstance(compatibility.get("recommended_package_specs"), list) else []
    matched_row = compatibility.get("matched_row") if isinstance(compatibility.get("matched_row"), dict) else None
    compatible_rows = compatibility.get("compatible_rows") if isinstance(compatibility.get("compatible_rows"), list) else []

    installed_tokens = [f"{name}={value}" for name, value in installed_versions.items() if value]
    compatible_tokens = []
    for row in compatible_rows:
        if not isinstance(row, dict):
            continue
        cann_version = str(row.get("cann_version") or "").strip()
        python_range = str(row.get("python_range") or "").strip()
        package_specs = row.get("package_specs") if isinstance(row.get("package_specs"), list) else []
        compatible_tokens.append(f"CANN {cann_version}, Python {python_range}: {', '.join(str(item) for item in package_specs)}")

    reason = str(compatibility.get("reason") or "").strip()
    if status == "compatible" and matched_row:
        matched_specs = matched_row.get("package_specs") if isinstance(matched_row.get("package_specs"), list) else []
        summary = f"installed framework packages match a local compatibility row: {', '.join(str(item) for item in matched_specs)}"
    elif status == "incompatible":
        summary = reason or "installed framework packages do not match the local compatibility reference"
    else:
        installed_suffix = f" Installed versions: {', '.join(installed_tokens)}." if installed_tokens else ""
        recommended_suffix = f" Recommended packages: {', '.join(str(item) for item in recommended_specs)}." if recommended_specs else ""
        summary = f"{reason}{installed_suffix}{recommended_suffix}".strip()

    evidence = []
    if installed_tokens:
        evidence.append("installed " + ", ".join(installed_tokens))
    if recommended_specs:
        evidence.append("recommended " + ", ".join(recommended_specs))
    if compatible_tokens:
        evidence.append("compatible rows " + "; ".join(compatible_tokens[:3]))

    details: Dict[str, object] = {
        "compatibility_status": status,
        "reference_status": compatibility.get("reference_status"),
        "installed_versions": installed_versions,
        "recommended_package_specs": recommended_specs,
        "matched_row": matched_row,
        "compatible_rows": compatible_rows,
        "reason": compatibility.get("reason"),
    }
    return summary or f"framework compatibility status: {status}", evidence, details


def summarize_import_failures(package_names: List[str], import_errors: Dict[str, str]) -> str:
    details = []
    for name in package_names:
        error = str(import_errors.get(name) or "").strip()
        details.append(f"{name} ({error})" if error else name)
    return ", ".join(details)


def summarize_ascend_runtime(
    system_layer: Dict[str, object],
    cann_input: Optional[str],
    probe_env_source: Optional[str],
    probe_env_error: Optional[str],
) -> Tuple[str, List[str], Dict[str, object]]:
    script_path = str(system_layer.get("ascend_env_script_path") or "").strip()
    candidate_paths = [str(item) for item in (system_layer.get("ascend_env_candidate_paths") or []) if str(item).strip()]
    selection_source = str(system_layer.get("ascend_env_selection_source") or probe_env_source or "").strip()

    if system_layer.get("ascend_env_active"):
        summary = "Ascend runtime variables are already active in the current environment."
    elif script_path:
        summary = f"Ascend runtime can be sourced from {script_path}."
    elif cann_input:
        summary = f"Ascend runtime evidence comes from explicit CANN path {cann_input}."
    elif candidate_paths:
        summary = f"Ascend runtime candidate script found at {candidate_paths[0]}."
    else:
        summary = "Ascend runtime evidence is weak or unresolved."

    evidence: List[str] = []
    if selection_source:
        evidence.append(f"selection_source={selection_source}")
    if cann_input:
        evidence.append(f"cann_path={cann_input}")
    if script_path:
        evidence.append(f"ascend_env_script={script_path}")
    evidence.extend(candidate_paths[:3])
    if probe_env_error:
        evidence.append(f"probe_error={probe_env_error}")

    details = {
        "ascend_env_active": bool(system_layer.get("ascend_env_active")),
        "ascend_env_script_path": script_path or None,
        "ascend_env_candidate_paths": candidate_paths,
        "ascend_env_selection_source": selection_source or None,
        "cann_path_input": cann_input,
        "probe_environment_source": probe_env_source,
        "probe_environment_error": probe_env_error,
    }
    return summary, evidence, details


def summarize_cann_version(
    cann_version_info: Dict[str, object],
    system_layer: Dict[str, object],
    cann_input: Optional[str],
) -> Tuple[str, List[str], Dict[str, object]]:
    version = str(cann_version_info.get("cann_version") or "").strip()
    source = str(cann_version_info.get("cann_version_source") or "").strip()
    version_file = str(cann_version_info.get("cann_version_file") or "").strip()
    script_path = str(system_layer.get("ascend_env_script_path") or "").strip()

    if version:
        if version_file:
            summary = f"CANN version detected: {version} from {version_file}."
        elif source == "ascend_env_script" and script_path:
            summary = f"CANN version detected: {version} from Ascend env script {script_path}."
        elif source == "cann_path" and cann_input:
            summary = f"CANN version detected: {version} from CANN path {cann_input}."
        elif cann_input:
            summary = f"CANN version detected: {version} from CANN path {cann_input}."
        else:
            summary = f"CANN version detected: {version}."
    elif script_path:
        summary = f"CANN version is unresolved; inspected Ascend env script {script_path}."
    elif cann_input:
        summary = f"CANN version is unresolved for CANN path {cann_input}."
    else:
        summary = "CANN version is unresolved."

    evidence: List[str] = []
    if version_file:
        evidence.append(version_file)
    if cann_input:
        evidence.append(f"cann_path={cann_input}")
    if script_path:
        evidence.append(f"ascend_env_script={script_path}")
    if source:
        evidence.append(f"source={source}")

    details = {
        "cann_version": version or None,
        "cann_version_source": source or None,
        "cann_version_file": version_file or None,
        "cann_path_input": cann_input,
        "ascend_env_script_path": script_path or None,
    }
    return summary, evidence, details


def executable_exists(command_name: str) -> bool:
    return bool(shutil.which(command_name))


def launcher_ready(
    launcher_value: Optional[str],
    selected_candidate: Optional[Dict[str, object]],
    import_probes: Dict[str, bool],
) -> Tuple[str, str]:
    if not launcher_value:
        return "block", "launcher is unresolved"
    if launcher_value == "python":
        if selected_candidate and selected_candidate.get("status") == "selected":
            return "ok", "runtime python is available"
        return "block", "selected runtime python is unavailable"
    if launcher_value == "bash":
        return ("ok", "bash launcher is available") if executable_exists("bash") else ("block", "bash launcher is unavailable")
    if launcher_value == "make":
        return ("ok", "make launcher is available") if executable_exists("make") else ("warn", "make launcher is unavailable in PATH")
    if launcher_value == "msrun":
        return ("ok", "msrun launcher is available") if executable_exists("msrun") else ("warn", "msrun launcher is not visible in PATH")
    if launcher_value == "torchrun":
        return ("ok", "torchrun launcher requirements are present") if import_probes.get("torch") else ("block", "torchrun requires torch in the selected environment")
    if launcher_value == "accelerate":
        return ("ok", "accelerate launcher requirements are present") if import_probes.get("accelerate") else ("block", "accelerate is missing in the selected environment")
    if launcher_value == "deepspeed":
        return ("ok", "deepspeed launcher requirements are present") if import_probes.get("deepspeed") else ("warn", "deepspeed is not importable in the selected environment")
    if launcher_value == "llamafactory-cli":
        if import_probes.get("llamafactory"):
            return "ok", "llamafactory launcher requirements are present"
        if executable_exists("llamafactory-cli"):
            return "ok", "llamafactory-cli executable is available"
        return "block", "llamafactory-cli is unresolved in the selected environment"
    return "warn", f"launcher {launcher_value} has no specialized readiness probe"
