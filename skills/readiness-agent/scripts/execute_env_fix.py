#!/usr/bin/env python3
import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

from python_selection import derive_env_root_from_python, python_in_env


UV_INSTALL_CMD = "curl -LsSf https://astral.sh/uv/install.sh | sh"
UV_BIN_DIR = "$HOME/.local/bin"
TORCH_CPU_INDEX_URL = "https://download.pytorch.org/whl/cpu"
PTA_CPU_TORCH_PACKAGES = {"torch", "torchvision", "torchaudio"}


def load_actions(path: Path) -> List[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("actions", [])


def append_path_export(profile_path: Path) -> None:
    line = f'export PATH="{UV_BIN_DIR}:$PATH"'
    existing = profile_path.read_text(encoding="utf-8", errors="replace") if profile_path.exists() else ""
    if line in existing:
        return
    prefix = "" if not existing or existing.endswith("\n") else "\n"
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.write_text(existing + prefix + line + "\n", encoding="utf-8")


def ensure_uv_env(selected_env_root: Path, python_version: Optional[str]) -> Tuple[bool, str]:
    uv_path = shutil.which("uv")
    if not uv_path:
        return False, "uv is not directly resolvable"
    cmd = [uv_path, "venv", str(selected_env_root)]
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


def install_packages(env_root: Path, package_names: List[str], index_url: Optional[str] = None) -> Tuple[bool, str]:
    uv_path = shutil.which("uv")
    if not uv_path:
        return False, "uv is not directly resolvable"
    python_path = selected_python_path(env_root)
    if not python_path.exists():
        return False, "selected environment python is missing"
    if not package_names:
        return False, "no package names were provided"
    cmd = [uv_path, "pip", "install", "--python", str(python_path)]
    if index_url:
        cmd.extend(["--index-url", index_url])
    cmd.extend(package_names)
    try:
        subprocess.run(cmd, check=True, text=True, capture_output=True)
    except subprocess.CalledProcessError as exc:
        return False, exc.stderr.strip() or exc.stdout.strip() or "uv pip install failed"
    return True, f"installed {' '.join(package_names)}"


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
        ok, message = install_packages(env_root, cpu_torch_packages, index_url=TORCH_CPU_INDEX_URL)
        if not ok:
            return False, message
        messages.append(f"{message} from cpu index")

    if remaining_packages:
        ok, message = install_packages(env_root, remaining_packages)
        if not ok:
            return False, message
        messages.append(message)

    if not messages:
        return False, "no PTA framework package names were provided"

    return True, "; ".join(messages)


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
        result["status"] = "executed"
        result["reason"] = "uv installation command completed"
        return result

    if action_type == "repair_uv_path":
        profile = Path(args.path_profile) if args.path_profile else None
        result["command_preview"] = f'append `export PATH="{UV_BIN_DIR}:$PATH"` to shell profile'
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
        result["command_preview"] = "uv pip install --python <selected_env_root>/bin/python <package_names...>"
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
                f"{TORCH_CPU_INDEX_URL} <torch...>; uv pip install --python <selected_env_root>/bin/python <torch_npu...>"
            )
        else:
            result["command_preview"] = "uv pip install --python <selected_env_root>/bin/python <framework package_names...>"
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
