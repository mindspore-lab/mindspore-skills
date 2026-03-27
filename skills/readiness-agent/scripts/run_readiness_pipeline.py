#!/usr/bin/env python3
import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from python_selection import derive_env_root_from_python


SCRIPT_DIR = Path(__file__).resolve().parent
VALUE_FLAGS = {
    "--working-dir",
    "--output-dir",
    "--target",
    "--framework-hint",
    "--cann-path",
    "--mode",
    "--entry-script",
    "--selected-python",
    "--selected-env-root",
    "--config-path",
    "--model-path",
    "--model-hub-id",
    "--dataset-path",
    "--dataset-hub-id",
    "--dataset-split",
    "--checkpoint-path",
    "--task-smoke-cmd",
    "--fix-scope",
    "--python-version",
    "--path-profile",
    "--timeout-seconds",
}
BOOL_FLAGS = {
    "--check",
    "--fix",
    "--auto",
    "--allow-network",
    "--verbose",
}
HELP_FLAGS = {"-h", "--help"}


def maybe_add(arguments: List[str], flag: str, value: Optional[str]) -> None:
    if value:
        arguments.extend([flag, value])


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def shell_export(name: str, value: Optional[str]) -> Optional[str]:
    if value in {None, ""}:
        return None
    return f"export {name}={shlex.quote(str(value))}"


def resolve_optional_path(value: Optional[str], root: Path) -> Optional[Path]:
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def runner_for_selection(selection: dict) -> str:
    if selection.get("selection_status") == "selected" and selection.get("selected_python"):
        return str(selection["selected_python"])
    return sys.executable


def run_helper(script_name: str, runner: str, working_dir: Path, arguments: List[str]) -> subprocess.CompletedProcess[str]:
    script_path = SCRIPT_DIR / script_name
    return subprocess.run(
        [runner, str(script_path), *arguments],
        cwd=str(working_dir),
        check=True,
        text=True,
        capture_output=True,
    )


def build_paths(output_dir: Path) -> Dict[str, Path]:
    meta_dir = output_dir / "meta"
    return {
        "output_dir": output_dir,
        "meta_dir": meta_dir,
        "inputs_json": meta_dir / "inputs.json",
        "env_json": meta_dir / "env.json",
        "selected_python_json": meta_dir / "selected-python.json",
        "target_json": meta_dir / "execution-target.json",
        "closure_json": meta_dir / "dependency-closure.json",
        "task_smoke_json": meta_dir / "task-smoke.json",
        "checks_json": meta_dir / "checks.json",
        "normalized_json": meta_dir / "blockers.json",
        "plan_json": meta_dir / "remediation.json",
        "fix_applied_json": meta_dir / "fix-applied.json",
        "report_json": output_dir / "report.json",
        "report_md": output_dir / "report.md",
        "verdict_json": meta_dir / "readiness-verdict.json",
    }


def sanitize_cli_args(raw_args: List[str]) -> Tuple[List[str], List[dict]]:
    sanitized: List[str] = []
    ignored: List[dict] = []
    index = 0

    while index < len(raw_args):
        token = raw_args[index]
        if token in HELP_FLAGS:
            sanitized.append(token)
            index += 1
            continue

        if token.startswith("--"):
            flag, has_inline_value, inline_value = token.partition("=")

            if flag in BOOL_FLAGS:
                sanitized.append(flag)
                if has_inline_value:
                    ignored.append(
                        {
                            "token": token,
                            "reason": "bool_flag_inline_value_ignored",
                        }
                    )
                index += 1
                continue

            if flag in VALUE_FLAGS:
                if has_inline_value:
                    if inline_value:
                        sanitized.extend([flag, inline_value])
                    else:
                        ignored.append({"token": token, "reason": "missing_value"})
                    index += 1
                    continue

                if index + 1 < len(raw_args) and not raw_args[index + 1].startswith("-"):
                    sanitized.extend([flag, raw_args[index + 1]])
                    index += 2
                    continue

                ignored.append({"token": token, "reason": "missing_value"})
                index += 1
                continue

            ignored.append({"token": token, "reason": "unknown_flag"})
            if not has_inline_value and index + 1 < len(raw_args) and not raw_args[index + 1].startswith("-"):
                ignored.append({"token": raw_args[index + 1], "reason": "unknown_flag_value"})
                index += 2
                continue

            index += 1
            continue

        if token.startswith("-"):
            ignored.append({"token": token, "reason": "unknown_short_flag"})
            if index + 1 < len(raw_args) and not raw_args[index + 1].startswith("-"):
                ignored.append({"token": raw_args[index + 1], "reason": "unknown_short_flag_value"})
                index += 2
                continue

            index += 1
            continue

        ignored.append({"token": token, "reason": "unsupported_positional_argument"})
        index += 1

    return sanitized, ignored


def write_inputs_snapshot(
    args: argparse.Namespace,
    working_dir: Path,
    output_dir: Path,
    path: Path,
    raw_cli_args: List[str],
    ignored_cli_args: List[dict],
) -> None:
    payload = {
        "working_dir": str(working_dir),
        "output_dir": str(output_dir),
        "target": args.target,
        "framework_hint": args.framework_hint,
        "cann_path": args.cann_path,
        "mode": args.mode,
        "verbose": bool(getattr(args, "verbose", False)),
        "entry_script": args.entry_script,
        "selected_python": args.selected_python,
        "selected_env_root": args.selected_env_root,
        "config_path": args.config_path,
        "model_path": args.model_path,
        "model_hub_id": args.model_hub_id,
        "dataset_path": args.dataset_path,
        "dataset_hub_id": args.dataset_hub_id,
        "dataset_split": args.dataset_split,
        "checkpoint_path": args.checkpoint_path,
        "task_smoke_cmd": args.task_smoke_cmd,
        "allow_network": args.allow_network,
        "fix_scope": args.fix_scope,
        "python_version": args.python_version,
        "timeout_seconds": args.timeout_seconds,
        "path_profile": args.path_profile,
        "raw_cli_args": raw_cli_args,
        "ignored_cli_args": ignored_cli_args,
    }
    write_json(path, payload)


def resolve_env_root_for_fix(
    working_dir: Path,
    selected_python: Optional[str],
    selected_env_root: Optional[str],
    selection: dict,
) -> Path:
    explicit_env_root = resolve_optional_path(selected_env_root, working_dir)
    if explicit_env_root:
        return explicit_env_root

    explicit_python = resolve_optional_path(selected_python, working_dir)
    if explicit_python:
        derived = derive_env_root_from_python(explicit_python)
        if derived:
            return derived.resolve()

    current_env_root = selection.get("selected_env_root")
    if current_env_root:
        return Path(str(current_env_root)).resolve()

    return (working_dir / ".venv").resolve()


def run_selected_python_resolution(
    working_dir: Path,
    paths: Dict[str, Path],
    selected_python: Optional[str],
    selected_env_root: Optional[str],
) -> dict:
    arguments = [
        "--working-dir",
        str(working_dir),
        "--output-json",
        str(paths["selected_python_json"]),
    ]
    maybe_add(arguments, "--selected-python", selected_python)
    maybe_add(arguments, "--selected-env-root", selected_env_root)
    run_helper("resolve_selected_python.py", sys.executable, working_dir, arguments)
    return load_json(paths["selected_python_json"])


def run_pipeline_pass(
    args: argparse.Namespace,
    working_dir: Path,
    paths: Dict[str, Path],
    selected_python: Optional[str],
    selected_env_root: Optional[str],
) -> dict:
    selection = run_selected_python_resolution(
        working_dir,
        paths,
        selected_python,
        selected_env_root,
    )
    helper_runner = runner_for_selection(selection)

    discover_args = [
        "--working-dir",
        str(working_dir),
        "--target",
        args.target,
        "--output-json",
        str(paths["target_json"]),
    ]
    maybe_add(discover_args, "--framework-hint", args.framework_hint)
    maybe_add(discover_args, "--cann-path", args.cann_path)
    maybe_add(discover_args, "--entry-script", args.entry_script)
    maybe_add(discover_args, "--config-path", args.config_path)
    maybe_add(discover_args, "--model-path", args.model_path)
    maybe_add(discover_args, "--model-hub-id", args.model_hub_id)
    maybe_add(discover_args, "--dataset-path", args.dataset_path)
    maybe_add(discover_args, "--dataset-hub-id", args.dataset_hub_id)
    maybe_add(discover_args, "--dataset-split", args.dataset_split)
    maybe_add(discover_args, "--checkpoint-path", args.checkpoint_path)
    maybe_add(discover_args, "--selected-python", selected_python)
    maybe_add(discover_args, "--selected-env-root", selected_env_root)
    maybe_add(discover_args, "--task-smoke-cmd", args.task_smoke_cmd)
    run_helper("discover_execution_target.py", helper_runner, working_dir, discover_args)
    target = load_json(paths["target_json"])

    run_helper(
        "build_dependency_closure.py",
        helper_runner,
        working_dir,
        [
            "--target-json",
            str(paths["target_json"]),
            "--output-json",
            str(paths["closure_json"]),
        ],
    )
    closure = load_json(paths["closure_json"])

    task_smoke_requested = bool(target.get("task_smoke_cmd"))
    if task_smoke_requested:
        run_helper(
            "run_task_smoke.py",
            helper_runner,
            working_dir,
            [
                "--target-json",
                str(paths["target_json"]),
                "--closure-json",
                str(paths["closure_json"]),
                "--output-json",
                str(paths["task_smoke_json"]),
                "--timeout-seconds",
                str(args.timeout_seconds),
            ],
        )
    elif paths["task_smoke_json"].exists():
        paths["task_smoke_json"].unlink()

    collect_args = [
        "--target-json",
        str(paths["target_json"]),
        "--closure-json",
        str(paths["closure_json"]),
        "--output-json",
        str(paths["checks_json"]),
    ]
    if task_smoke_requested:
        collect_args.extend(["--task-smoke-json", str(paths["task_smoke_json"])])
    run_helper("collect_readiness_checks.py", helper_runner, working_dir, collect_args)
    checks = load_json(paths["checks_json"])

    run_helper(
        "normalize_blockers.py",
        helper_runner,
        working_dir,
        [
            "--input-json",
            str(paths["checks_json"]),
            "--output-json",
            str(paths["normalized_json"]),
        ],
    )
    normalized = load_json(paths["normalized_json"])

    return {
        "selection": selection,
        "helper_runner": helper_runner,
        "target": target,
        "closure": closure,
        "checks": checks,
        "normalized": normalized,
    }


def run_fix_plan_and_execution(
    args: argparse.Namespace,
    working_dir: Path,
    paths: Dict[str, Path],
    pipeline_state: dict,
) -> dict:
    helper_runner = pipeline_state["helper_runner"]
    run_helper(
        "plan_env_fix.py",
        helper_runner,
        working_dir,
        [
            "--blockers-json",
            str(paths["normalized_json"]),
            "--closure-json",
            str(paths["closure_json"]),
            "--output-json",
            str(paths["plan_json"]),
            "--fix-scope",
            args.fix_scope,
            *(["--allow-network"] if args.allow_network else []),
        ],
    )

    selection = pipeline_state["selection"]
    env_root = resolve_env_root_for_fix(
        working_dir,
        args.selected_python,
        args.selected_env_root,
        selection,
    )
    selected_python = args.selected_python or selection.get("selected_python")

    execute_args = [
        "--plan-json",
        str(paths["plan_json"]),
        "--output-json",
        str(paths["fix_applied_json"]),
        "--working-dir",
        str(working_dir),
        "--selected-env-root",
        str(env_root),
    ]
    maybe_add(execute_args, "--selected-python", selected_python)
    maybe_add(execute_args, "--python-version", args.python_version)
    maybe_add(execute_args, "--path-profile", args.path_profile)

    if args.mode in {"fix", "auto"}:
        execute_args.extend(
            [
                "--execute",
                "--confirm-install-uv",
                "--confirm-path-edit",
                "--confirm-create-env",
                "--confirm-framework-repair",
                "--confirm-asset-repair",
            ]
        )

    run_helper("execute_env_fix.py", helper_runner, working_dir, execute_args)
    return load_json(paths["fix_applied_json"])

def write_default_fix_outputs(paths: Dict[str, Path]) -> dict:
    plan = {
        "actions": [],
        "skipped": [],
    }
    fix_applied = {
        "execute": False,
        "results": [],
        "executed_actions": [],
        "failed_actions": [],
        "needs_revalidation": [],
    }
    write_json(paths["plan_json"], plan)
    write_json(paths["fix_applied_json"], fix_applied)
    return fix_applied


def build_readiness_env_payload(working_dir: Path, pipeline_state: dict) -> dict:
    closure = pipeline_state.get("closure", {})
    layers = closure.get("layers", {})
    system_layer = layers.get("system", {})
    python_layer = layers.get("python_environment", {})
    remote_assets = layers.get("remote_assets", {})
    cache_layout = remote_assets.get("cache_layout") or {}

    payload = {
        "working_dir": str(working_dir),
        "ascend_env_script": system_layer.get("ascend_env_script_path"),
        "hf_endpoint": remote_assets.get("hf_endpoint") or os.environ.get("HF_ENDPOINT") or "https://hf-mirror.com",
        "selected_python": python_layer.get("probe_python_path") or python_layer.get("selected_python"),
        "selected_env_root": python_layer.get("selected_env_root"),
    }

    hub_cache = cache_layout.get("hub_cache")
    datasets_cache = cache_layout.get("datasets_cache")
    hf_home = cache_layout.get("hf_home") or str((working_dir / "huggingface-cache").resolve())
    explicit_hub = os.environ.get("HUGGINGFACE_HUB_CACHE")
    explicit_datasets = os.environ.get("HF_DATASETS_CACHE")
    explicit_hf_home = os.environ.get("HF_HOME")

    if explicit_hub or explicit_datasets:
        payload["huggingface_hub_cache"] = explicit_hub or hub_cache
        payload["hf_datasets_cache"] = explicit_datasets or datasets_cache
    elif explicit_hf_home:
        payload["hf_home"] = explicit_hf_home
    else:
        payload["hf_home"] = hf_home

    return payload


def write_readiness_env_file(path: Path, working_dir: Path, pipeline_state: dict) -> dict:
    payload = build_readiness_env_payload(working_dir, pipeline_state)
    lines = [
        "#!/usr/bin/env bash",
        "# Generated by readiness-agent. Source this file before running training or inference in this workspace.",
        "",
    ]
    ascend_env_script = payload.get("ascend_env_script")
    if ascend_env_script:
        lines.append(f"source {shlex.quote(str(ascend_env_script))}")
        lines.append("")

    export_lines = [
        shell_export("READINESS_WORKING_DIR", payload.get("working_dir")),
        shell_export("HF_ENDPOINT", payload.get("hf_endpoint")),
        shell_export("HF_HOME", payload.get("hf_home")),
        shell_export("HUGGINGFACE_HUB_CACHE", payload.get("huggingface_hub_cache")),
        shell_export("HF_DATASETS_CACHE", payload.get("hf_datasets_cache")),
        shell_export("READINESS_SELECTED_ENV_ROOT", payload.get("selected_env_root")),
        shell_export("READINESS_SELECTED_PYTHON", payload.get("selected_python")),
    ]
    lines.extend([line for line in export_lines if line])
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    return payload


def write_env_snapshot(
    path: Path,
    mode: str,
    passes: int,
    initial_selection: dict,
    final_state: dict,
    fix_applied: dict,
    readiness_env_path: Optional[Path] = None,
) -> None:
    payload = {
        "mode": mode,
        "pipeline_passes": passes,
        "control_python": sys.executable,
        "initial_selection": initial_selection,
        "final_helper_runner": final_state["helper_runner"],
        "final_selection": final_state["selection"],
        "fix_execute": bool(fix_applied.get("execute")),
        "executed_actions": fix_applied.get("executed_actions", []),
        "failed_actions": fix_applied.get("failed_actions", []),
        "needs_revalidation": fix_applied.get("needs_revalidation", []),
        "readiness_env_path": str(readiness_env_path) if readiness_env_path else None,
    }
    write_json(path, payload)


def build_report(
    working_dir: Path,
    paths: Dict[str, Path],
    helper_runner: str,
) -> None:
    run_helper(
        "build_readiness_report.py",
        helper_runner,
        working_dir,
        [
            "--target-json",
            str(paths["target_json"]),
            "--normalized-json",
            str(paths["normalized_json"]),
            "--checks-json",
            str(paths["checks_json"]),
            "--closure-json",
            str(paths["closure_json"]),
            "--fix-applied-json",
            str(paths["fix_applied_json"]),
            "--output-json",
            str(paths["report_json"]),
            "--output-md",
            str(paths["report_md"]),
            "--output-verdict-json",
            str(paths["verdict_json"]),
        ],
    )


def normalize_mode_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> str:
    alias_modes = [
        mode
        for mode, enabled in (
            ("check", getattr(args, "check", False)),
            ("fix", getattr(args, "fix", False)),
            ("auto", getattr(args, "auto", False)),
        )
        if enabled
    ]
    if len(alias_modes) > 1:
        parser.error("use at most one of --check, --fix, or --auto")

    alias_mode = alias_modes[0] if alias_modes else None
    explicit_mode = args.mode
    if alias_mode and explicit_mode and explicit_mode != alias_mode:
        parser.error("--mode conflicts with the requested alias flag")

    if alias_mode:
        return alias_mode
    if explicit_mode:
        return explicit_mode
    return "check"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the full readiness-agent helper pipeline with optional env re-entry",
        allow_abbrev=False,
    )
    parser.add_argument("--working-dir", help="workspace root (defaults to the current shell path)")
    parser.add_argument("--output-dir", help="output directory for readiness artifacts (defaults to <working_dir>/readiness-output)")
    parser.add_argument("--target", default="auto", help="training, inference, or auto")
    parser.add_argument("--framework-hint", help="explicit framework preference such as mindspore or pta")
    parser.add_argument("--cann-path", help="explicit CANN root or set_env.sh path")
    parser.add_argument("--mode", choices=("check", "fix", "auto"), help="check, fix, or auto")
    parser.add_argument("--check", action="store_true", help="alias for --mode check")
    parser.add_argument("--fix", action="store_true", help="alias for --mode fix")
    parser.add_argument("--auto", action="store_true", help="alias for --mode auto")
    parser.add_argument("--verbose", action="store_true", help="accepted for caller compatibility; currently no-op")
    parser.add_argument("--entry-script", help="explicit entry script path")
    parser.add_argument("--selected-python", help="explicit Python interpreter for the workspace")
    parser.add_argument("--selected-env-root", help="explicit environment root for the workspace")
    parser.add_argument("--config-path", help="explicit config path")
    parser.add_argument("--model-path", help="explicit model path")
    parser.add_argument("--model-hub-id", help="explicit Hugging Face model repo ID")
    parser.add_argument("--dataset-path", help="explicit dataset path")
    parser.add_argument("--dataset-hub-id", help="explicit Hugging Face dataset repo ID")
    parser.add_argument("--dataset-split", help="explicit dataset split for remote dataset download")
    parser.add_argument("--checkpoint-path", help="explicit checkpoint path")
    parser.add_argument("--task-smoke-cmd", help="explicit minimal task smoke command")
    parser.add_argument("--allow-network", action="store_true", help="allow network-dependent remediation planning")
    parser.add_argument("--fix-scope", default="safe-user-space", help="active fix scope")
    parser.add_argument("--python-version", help="Python version hint for environment creation")
    parser.add_argument("--path-profile", help="shell profile path for PATH repair")
    parser.add_argument("--timeout-seconds", type=int, default=10, help="timeout for explicit task smoke execution")
    raw_cli_args = sys.argv[1:]
    sanitized_cli_args, ignored_cli_args = sanitize_cli_args(raw_cli_args)
    args = parser.parse_args(sanitized_cli_args)
    args.mode = normalize_mode_args(parser, args)

    working_dir = Path(args.working_dir).resolve() if args.working_dir else Path.cwd().resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else (working_dir / "readiness-output").resolve()
    paths = build_paths(output_dir)
    paths["meta_dir"].mkdir(parents=True, exist_ok=True)
    write_inputs_snapshot(
        args,
        working_dir,
        output_dir,
        paths["inputs_json"],
        raw_cli_args,
        ignored_cli_args,
    )

    initial_state = run_pipeline_pass(
        args,
        working_dir,
        paths,
        args.selected_python,
        args.selected_env_root,
    )
    fix_applied = run_fix_plan_and_execution(args, working_dir, paths, initial_state)

    final_state = initial_state
    passes = 1
    if fix_applied.get("executed_actions"):
        rerun_selected_env_root = args.selected_env_root or str(
            resolve_env_root_for_fix(
                working_dir,
                args.selected_python,
                args.selected_env_root,
                initial_state["selection"],
            )
        )
        final_state = run_pipeline_pass(
            args,
            working_dir,
            paths,
            args.selected_python,
            rerun_selected_env_root,
        )
        passes = 2

    readiness_env_path = (working_dir / ".readiness.env").resolve()
    write_readiness_env_file(readiness_env_path, working_dir, final_state)

    write_env_snapshot(
        paths["env_json"],
        args.mode,
        passes,
        initial_state["selection"],
        final_state,
        fix_applied,
        readiness_env_path,
    )
    build_report(working_dir, paths, final_state["helper_runner"])

    verdict = load_json(paths["verdict_json"])
    print(
        json.dumps(
            {
                "status": verdict.get("status"),
                "target": verdict.get("target"),
                "selected_python": final_state["selection"].get("selected_python"),
                "pipeline_passes": passes,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
