#!/usr/bin/env python3
import argparse
import json
import os
import re
import shutil
import subprocess
import textwrap
from pathlib import Path
from typing import List, Optional, Tuple

from python_selection import derive_env_root_from_python, python_in_env


UV_INSTALL_CMD = "curl -LsSf https://astral.sh/uv/install.sh | sh"
DEFAULT_PIP_INDEX_URL = "https://pypi.tuna.tsinghua.edu.cn/simple"
FALLBACK_PIP_INDEX_URL = "https://mirrors.aliyun.com/pypi/simple/"
SUPPORTED_PIP_INDEX_URLS = (
    DEFAULT_PIP_INDEX_URL,
    FALLBACK_PIP_INDEX_URL,
)
PTA_CPU_TORCH_PACKAGES = {"torch", "torchvision", "torchaudio"}
DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"
HF_MODEL_DOWNLOAD_CODE = textwrap.dedent(
    """
    import sys
    from pathlib import Path
    from huggingface_hub import snapshot_download

    repo_id = sys.argv[1]
    destination = Path(sys.argv[2])
    destination.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=repo_id, local_dir=str(destination))
    print(destination)
    """
)
HF_DATASET_DOWNLOAD_CODE = textwrap.dedent(
    """
    import sys
    from pathlib import Path
    from datasets import load_dataset

    repo_id = sys.argv[1]
    destination = Path(sys.argv[2])
    split = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] else None
    destination.parent.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(repo_id, split=split) if split else load_dataset(repo_id)
    dataset.save_to_disk(str(destination))
    print(destination)
    """
)


def load_actions(path: Path) -> List[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("actions", [])


def default_uv_bin_dir() -> Path:
    return Path.home() / ".local" / "bin"


def uv_path_export_fragment() -> str:
    return f'{default_uv_bin_dir()}:$PATH'


def resolve_uv_executable() -> Optional[Path]:
    uv_path = shutil.which("uv")
    if uv_path:
        return Path(uv_path)

    candidates = [default_uv_bin_dir() / "uv"]
    if os.name == "nt":
        candidates.extend([default_uv_bin_dir() / "uv.exe", default_uv_bin_dir() / "uv.cmd"])

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def append_path_export(profile_path: Path) -> None:
    line = f'export PATH="{uv_path_export_fragment()}"'
    existing = profile_path.read_text(encoding="utf-8", errors="replace") if profile_path.exists() else ""
    if line in existing:
        return
    prefix = "" if not existing or existing.endswith("\n") else "\n"
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.write_text(existing + prefix + line + "\n", encoding="utf-8")


def ensure_uv_env(selected_env_root: Path, python_version: Optional[str]) -> Tuple[bool, str]:
    uv_path = resolve_uv_executable()
    if not uv_path:
        return False, "uv is not directly resolvable from PATH or ~/.local/bin"
    cmd = [str(uv_path), "venv", str(selected_env_root)]
    if python_version:
        cmd.extend(["--python", python_version])
    try:
        subprocess.run(cmd, check=True, text=True, capture_output=True)
    except subprocess.CalledProcessError as exc:
        return False, exc.stderr.strip() or exc.stdout.strip() or "uv venv failed"
    return True, "environment created"


def selected_python_path(env_root: Path) -> Path:
    python_path = python_in_env(env_root)
    if python_path:
        return python_path
    return env_root / "bin" / "python"


def resolve_env_root(args: argparse.Namespace) -> Optional[Path]:
    if args.selected_env_root:
        return Path(args.selected_env_root)

    if args.selected_python:
        derived = derive_env_root_from_python(Path(args.selected_python))
        if derived:
            return derived

    if args.working_dir:
        return Path(args.working_dir).resolve() / ".venv"

    return None


def package_base_name(package_name: str) -> str:
    token = package_name.strip()
    token = token.split("[", 1)[0]
    parts = re.split(r"(==|>=|<=|~=|!=|>|<)", token, maxsplit=1)
    return parts[0].strip().replace("_", "-").lower()


def normalize_index_url(index_url: str) -> str:
    return index_url.rstrip("/")


def is_supported_pip_index_url(index_url: str) -> bool:
    normalized = normalize_index_url(index_url)
    return any(normalized == normalize_index_url(item) for item in SUPPORTED_PIP_INDEX_URLS)


def preferred_pip_index_urls() -> List[str]:
    explicit_index = os.environ.get("READINESS_PIP_INDEX_URL") or os.environ.get("PIP_INDEX_URL")
    if explicit_index and is_supported_pip_index_url(explicit_index):
        primary = explicit_index
    else:
        primary = DEFAULT_PIP_INDEX_URL

    ordered_indexes = [primary]
    ordered_indexes.extend(
        index_url
        for index_url in SUPPORTED_PIP_INDEX_URLS
        if normalize_index_url(index_url) != normalize_index_url(primary)
    )
    return ordered_indexes


def run_install_command(cmd: List[str]) -> Tuple[bool, str]:
    try:
        subprocess.run(cmd, check=True, text=True, capture_output=True)
    except subprocess.CalledProcessError as exc:
        return False, exc.stderr.strip() or exc.stdout.strip() or "uv pip install failed"
    return True, ""


def build_uv_pip_install_command(
    uv_path: Path,
    python_path: Path,
    package_names: List[str],
    *,
    index_url: Optional[str] = None,
) -> List[str]:
    cmd = [str(uv_path), "pip", "install", "--python", str(python_path)]
    if index_url:
        cmd.extend(["--index-url", index_url])
    cmd.extend(package_names)
    return cmd


def install_packages(env_root: Path, package_names: List[str], index_url: Optional[str] = None) -> Tuple[bool, str]:
    uv_path = resolve_uv_executable()
    if not uv_path:
        return False, "uv is not directly resolvable from PATH or ~/.local/bin"
    python_path = selected_python_path(env_root)
    if not python_path.exists():
        return False, "selected environment python is missing"
    if not package_names:
        return False, "no package names were provided"

    if index_url:
        if not is_supported_pip_index_url(index_url):
            return False, f"non-mirror package index is not allowed: {index_url}"
        index_urls = [index_url]
    else:
        index_urls = preferred_pip_index_urls()

    failures = []
    for current_index_url in index_urls:
        cmd = build_uv_pip_install_command(
            uv_path,
            python_path,
            package_names,
            index_url=current_index_url,
        )
        ok, message = run_install_command(cmd)
        if ok:
            if current_index_url == index_urls[0]:
                return True, f"installed {' '.join(package_names)} from {current_index_url}"
            return True, (
                f"installed {' '.join(package_names)} after {index_urls[0]} failed; "
                f"fell back to the mirror {current_index_url}"
            )
        failures.append(f"{current_index_url} failed: {message}")

    return False, "; ".join(failures)


def install_runtime_dependency(env_root: Path, package_name: str) -> Tuple[bool, str]:
    return install_packages(env_root, [package_name])


def install_pta_framework_packages(env_root: Path, package_names: List[str]) -> Tuple[bool, str]:
    cpu_torch_packages = [
        package_name
        for package_name in package_names
        if package_base_name(package_name) in PTA_CPU_TORCH_PACKAGES
    ]
    remaining_packages = [
        package_name
        for package_name in package_names
        if package_name not in cpu_torch_packages
    ]

    messages: List[str] = []
    if cpu_torch_packages:
        ok, message = install_packages(env_root, cpu_torch_packages)
        if not ok:
            return False, message
        messages.append(message)

    if remaining_packages:
        ok, message = install_packages(env_root, remaining_packages)
        if not ok:
            return False, message
        messages.append(message)

    if not messages:
        return False, "no PTA framework package names were provided"

    return True, "; ".join(messages)


def preferred_hf_endpoint() -> str:
    return os.environ.get("HF_ENDPOINT") or DEFAULT_HF_ENDPOINT


def huggingface_download_env(working_dir: Optional[Path] = None) -> dict:
    env = dict(os.environ)
    env["HF_ENDPOINT"] = preferred_hf_endpoint()
    if not env.get("HUGGINGFACE_HUB_CACHE") and not env.get("HF_DATASETS_CACHE") and not env.get("HF_HOME") and working_dir:
        env["HF_HOME"] = str((working_dir / "huggingface-cache").resolve())
    return env


def run_selected_python(
    env_root: Path,
    code: str,
    arguments: List[str],
    *,
    env: Optional[dict] = None,
) -> Tuple[bool, str]:
    python_path = selected_python_path(env_root)
    if not python_path.exists():
        return False, "selected environment python is missing"
    try:
        completed = subprocess.run(
            [str(python_path), "-c", code, *arguments],
            check=True,
            text=True,
            capture_output=True,
            env=env,
        )
    except subprocess.CalledProcessError as exc:
        return False, exc.stderr.strip() or exc.stdout.strip() or "selected python execution failed"
    return True, completed.stdout.strip() or "selected python execution completed"


def scaffold_example_entry_script(template_path: Path, destination_path: Path) -> Tuple[bool, str]:
    if not template_path.exists():
        return False, f"template path is missing: {template_path}"
    if destination_path.exists():
        return True, f"reused existing entry script at {destination_path}"
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(template_path, destination_path)
    return True, f"scaffolded entry script at {destination_path}"


def download_huggingface_model_asset(
    env_root: Path,
    repo_id: str,
    destination_path: Path,
    working_dir: Optional[Path],
) -> Tuple[bool, str]:
    ok, message = install_packages(env_root, ["huggingface_hub"])
    if not ok:
        return False, message
    return run_selected_python(
        env_root,
        HF_MODEL_DOWNLOAD_CODE,
        [repo_id, str(destination_path)],
        env=huggingface_download_env(working_dir),
    )


def download_huggingface_dataset_asset(
    env_root: Path,
    repo_id: str,
    destination_path: Path,
    dataset_split: Optional[str],
    working_dir: Optional[Path],
) -> Tuple[bool, str]:
    ok, message = install_packages(env_root, ["datasets"])
    if not ok:
        return False, message
    arguments = [repo_id, str(destination_path)]
    if dataset_split:
        arguments.append(dataset_split)
    return run_selected_python(
        env_root,
        HF_DATASET_DOWNLOAD_CODE,
        arguments,
        env=huggingface_download_env(working_dir),
    )


def execute_action(action: dict, args: argparse.Namespace) -> dict:
    action_type = action.get("action_type")
    result = {
        "action_id": action.get("id"),
        "action_type": action_type,
        "requires_confirmation": bool(action.get("requires_confirmation")),
        "revalidation_scope": action.get("revalidation_scope") or [],
        "status": "planned",
        "reason": action.get("reason") or "",
        "command_preview": None,
    }

    if not action.get("allowed", False):
        result["status"] = "skipped"
        result["reason"] = action.get("reason") or "action is not allowed"
        return result

    if action_type == "install_uv":
        result["command_preview"] = UV_INSTALL_CMD
        if not args.execute:
            return result
        if not args.confirm_install_uv:
            result["status"] = "skipped"
            result["reason"] = "confirmation for uv installation was not provided"
            return result
        try:
            subprocess.run(UV_INSTALL_CMD, shell=True, check=True, text=True, capture_output=True)
        except subprocess.CalledProcessError as exc:
            result["status"] = "failed"
            result["reason"] = exc.stderr.strip() or exc.stdout.strip() or "uv installer failed"
            return result
        uv_path = resolve_uv_executable()
        if not uv_path:
            result["status"] = "failed"
            result["reason"] = "uv installation command completed but uv was not found in PATH or ~/.local/bin"
            return result
        result["status"] = "executed"
        result["reason"] = f"uv installation command completed and resolved at {uv_path}"
        return result

    if action_type == "repair_uv_path":
        profile = Path(args.path_profile) if args.path_profile else None
        result["command_preview"] = f'append `export PATH="{uv_path_export_fragment()}"` to shell profile'
        if not args.execute:
            return result
        if not args.confirm_path_edit:
            result["status"] = "skipped"
            result["reason"] = "confirmation for PATH edit was not provided"
            return result
        if not profile:
            result["status"] = "failed"
            result["reason"] = "path_profile is required for PATH repair execution"
            return result
        append_path_export(profile)
        result["status"] = "executed"
        result["reason"] = f"updated PATH export in {profile}"
        return result

    if action_type == "create_or_select_env":
        env_root = resolve_env_root(args)
        result["command_preview"] = "uv venv <selected_env_root> [--python <python_version>]"
        if not args.execute:
            return result
        if not env_root:
            result["status"] = "failed"
            result["reason"] = "selected_env_root is required for environment execution"
            return result
        if env_root.exists() and selected_python_path(env_root).exists():
            result["status"] = "reused"
            result["reason"] = f"reused existing environment at {env_root}"
            return result
        if not args.confirm_create_env:
            result["status"] = "skipped"
            result["reason"] = "confirmation for environment creation was not provided"
            return result
        ok, message = ensure_uv_env(env_root, args.python_version)
        result["status"] = "executed" if ok else "failed"
        result["reason"] = message
        return result

    if action_type == "install_runtime_dependency":
        env_root = resolve_env_root(args)
        package_names = action.get("package_names") or []
        if not package_names and action.get("package_name"):
            package_names = [action["package_name"]]
        result["command_preview"] = (
            "uv pip install --python <selected_env_root>/bin/python "
            f"--index-url {DEFAULT_PIP_INDEX_URL} <package_names...> "
            f"(fallback: {FALLBACK_PIP_INDEX_URL})"
        )
        if not args.execute:
            return result
        if not env_root:
            result["status"] = "failed"
            result["reason"] = "selected_env_root is required for dependency installation"
            return result
        if not package_names:
            result["status"] = "failed"
            result["reason"] = "package_name or package_names is required for safe dependency installation"
            return result
        ok, message = install_packages(env_root, package_names)
        result["status"] = "executed" if ok else "failed"
        result["reason"] = message
        return result

    if action_type in {"repair_mindspore_framework", "repair_pta_framework", "repair_framework"}:
        env_root = resolve_env_root(args)
        package_names = action.get("package_names") or []
        if not package_names and action.get("package_name"):
            package_names = [action["package_name"]]
        if action_type == "repair_pta_framework":
            result["command_preview"] = (
                "uv pip install --python <selected_env_root>/bin/python --index-url "
                f"{DEFAULT_PIP_INDEX_URL} <torch...> (fallback: {FALLBACK_PIP_INDEX_URL}); "
                "uv pip install --python "
                f"<selected_env_root>/bin/python --index-url {DEFAULT_PIP_INDEX_URL} <torch_npu...> "
                f"(fallback: {FALLBACK_PIP_INDEX_URL})"
            )
        else:
            result["command_preview"] = (
                "uv pip install --python <selected_env_root>/bin/python "
                f"--index-url {DEFAULT_PIP_INDEX_URL} <framework package_names...> "
                f"(fallback: {FALLBACK_PIP_INDEX_URL})"
            )
        if not args.execute:
            return result
        if not args.confirm_framework_repair:
            result["status"] = "skipped"
            result["reason"] = "confirmation for framework repair was not provided"
            return result
        if not env_root:
            result["status"] = "failed"
            result["reason"] = "selected_env_root is required for framework repair"
            return result
        if not package_names:
            result["status"] = "failed"
            result["reason"] = "package_names are required for framework repair"
            return result
        if action_type == "repair_pta_framework":
            ok, message = install_pta_framework_packages(env_root, package_names)
        else:
            ok, message = install_packages(env_root, package_names)
        result["status"] = "executed" if ok else "failed"
        result["reason"] = message
        return result

    if action_type == "scaffold_example_entry_script":
        template_value = str(action.get("template_path") or "")
        destination_value = str(action.get("destination_path") or "")
        template_path = Path(template_value) if template_value else Path()
        destination_path = Path(destination_value) if destination_value else Path()
        result["command_preview"] = f"copy bundled example template to {destination_path}"
        if not args.execute:
            return result
        if not args.confirm_asset_repair:
            result["status"] = "skipped"
            result["reason"] = "confirmation for asset repair was not provided"
            return result
        if not template_value or not destination_value:
            result["status"] = "failed"
            result["reason"] = "template_path and destination_path are required for entry script scaffolding"
            return result
        ok, message = scaffold_example_entry_script(template_path, destination_path)
        result["status"] = "executed" if ok else "failed"
        result["reason"] = message
        return result

    if action_type == "download_huggingface_model_asset":
        env_root = resolve_env_root(args)
        working_dir = Path(args.working_dir).resolve() if args.working_dir else None
        repo_id = str(action.get("repo_id") or "")
        destination_value = str(action.get("destination_path") or "")
        destination_path = Path(destination_value) if destination_value else Path()
        result["command_preview"] = (
            f"HF_ENDPOINT={preferred_hf_endpoint()} download Hugging Face model {repo_id} to {destination_path}"
        )
        if not args.execute:
            return result
        if not args.confirm_asset_repair:
            result["status"] = "skipped"
            result["reason"] = "confirmation for asset repair was not provided"
            return result
        if not env_root:
            result["status"] = "failed"
            result["reason"] = "selected_env_root is required for model asset download"
            return result
        if not repo_id or not destination_value:
            result["status"] = "failed"
            result["reason"] = "repo_id and destination_path are required for model asset download"
            return result
        ok, message = download_huggingface_model_asset(env_root, repo_id, destination_path, working_dir)
        result["status"] = "executed" if ok else "failed"
        result["reason"] = message
        return result

    if action_type == "download_huggingface_dataset_asset":
        env_root = resolve_env_root(args)
        working_dir = Path(args.working_dir).resolve() if args.working_dir else None
        repo_id = str(action.get("repo_id") or "")
        destination_value = str(action.get("destination_path") or "")
        destination_path = Path(destination_value) if destination_value else Path()
        dataset_split = str(action.get("dataset_split") or "").strip() or None
        result["command_preview"] = (
            f"HF_ENDPOINT={preferred_hf_endpoint()} download Hugging Face dataset {repo_id} to {destination_path}"
        )
        if not args.execute:
            return result
        if not args.confirm_asset_repair:
            result["status"] = "skipped"
            result["reason"] = "confirmation for asset repair was not provided"
            return result
        if not env_root:
            result["status"] = "failed"
            result["reason"] = "selected_env_root is required for dataset asset download"
            return result
        if not repo_id or not destination_value:
            result["status"] = "failed"
            result["reason"] = "repo_id and destination_path are required for dataset asset download"
            return result
        ok, message = download_huggingface_dataset_asset(
            env_root,
            repo_id,
            destination_path,
            dataset_split,
            working_dir,
        )
        result["status"] = "executed" if ok else "failed"
        result["reason"] = message
        return result

    result["status"] = "skipped"
    result["reason"] = f"unsupported action_type {action_type!r}"
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Execute controlled env-fix actions for readiness-agent")
    parser.add_argument("--plan-json", required=True, help="path to remediation plan JSON")
    parser.add_argument("--output-json", required=True, help="path to output execution result JSON")
    parser.add_argument("--execute", action="store_true", help="execute actions instead of dry-run planning")
    parser.add_argument("--working-dir", help="workspace root for default environment creation")
    parser.add_argument("--selected-env-root", help="selected environment root for env actions")
    parser.add_argument("--selected-python", help="selected Python interpreter for env-root derivation")
    parser.add_argument("--python-version", help="python version for environment creation")
    parser.add_argument("--path-profile", help="shell profile path for PATH repair")
    parser.add_argument("--confirm-install-uv", action="store_true", help="confirm uv installation")
    parser.add_argument("--confirm-path-edit", action="store_true", help="confirm PATH edit")
    parser.add_argument("--confirm-create-env", action="store_true", help="confirm environment creation")
    parser.add_argument("--confirm-framework-repair", action="store_true", help="confirm framework repair")
    parser.add_argument("--confirm-asset-repair", action="store_true", help="confirm asset download or scaffold actions")
    args = parser.parse_args()

    results = [execute_action(action, args) for action in load_actions(Path(args.plan_json))]
    output = {
        "execute": args.execute,
        "results": results,
        "executed_actions": [item["action_id"] for item in results if item["status"] in {"executed", "reused"}],
        "failed_actions": [item["action_id"] for item in results if item["status"] == "failed"],
        "needs_revalidation": sorted(
            {
                scope
                for item in results
                if item["status"] in {"executed", "reused"}
                for scope in item.get("revalidation_scope", [])
            }
        ),
    }
    Path(args.output_json).write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"executed": len(output["executed_actions"]), "failed": len(output["failed_actions"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
