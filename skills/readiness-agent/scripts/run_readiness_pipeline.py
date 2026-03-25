#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

from python_selection import derive_env_root_from_python


SCRIPT_DIR = Path(__file__).resolve().parent


def maybe_add(arguments: List[str], flag: str, value: Optional[str]) -> None:
    if value:
        arguments.extend([flag, value])


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


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


def write_inputs_snapshot(args: argparse.Namespace, working_dir: Path, output_dir: Path, path: Path) -> None:
    payload = {
        "working_dir": str(working_dir),
        "output_dir": str(output_dir),
        "target": args.target,
        "mode": args.mode,
        "entry_script": args.entry_script,
        "selected_python": args.selected_python,
        "selected_env_root": args.selected_env_root,
        "config_path": args.config_path,
        "model_path": args.model_path,
        "dataset_path": args.dataset_path,
        "checkpoint_path": args.checkpoint_path,
        "task_smoke_cmd": args.task_smoke_cmd,
        "allow_network": args.allow_network,
        "fix_scope": args.fix_scope,
        "python_version": args.python_version,
        "timeout_seconds": args.timeout_seconds,
        "path_profile": args.path_profile,
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
    maybe_add(discover_args, "--entry-script", args.entry_script)
    maybe_add(discover_args, "--config-path", args.config_path)
    maybe_add(discover_args, "--model-path", args.model_path)
    maybe_add(discover_args, "--dataset-path", args.dataset_path)
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
            ]
        )

    run_helper("execute_env_fix.py", helper_runner, working_dir, execute_args)
    return load_json(paths["fix_applied_json"])


def write_env_snapshot(
    path: Path,
    mode: str,
    passes: int,
    initial_selection: dict,
    final_state: dict,
    fix_applied: dict,
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
    parser = argparse.ArgumentParser(description="Run the full readiness-agent helper pipeline with optional env re-entry")
    parser.add_argument("--working-dir", required=True, help="workspace root")
    parser.add_argument("--output-dir", help="output directory for readiness artifacts (defaults to <working_dir>/out)")
    parser.add_argument("--target", default="auto", help="training, inference, or auto")
    parser.add_argument("--mode", choices=("check", "fix", "auto"), help="check, fix, or auto")
    parser.add_argument("--check", action="store_true", help="alias for --mode check")
    parser.add_argument("--fix", action="store_true", help="alias for --mode fix")
    parser.add_argument("--auto", action="store_true", help="alias for --mode auto")
    parser.add_argument("--entry-script", help="explicit entry script path")
    parser.add_argument("--selected-python", help="explicit Python interpreter for the workspace")
    parser.add_argument("--selected-env-root", help="explicit environment root for the workspace")
    parser.add_argument("--config-path", help="explicit config path")
    parser.add_argument("--model-path", help="explicit model path")
    parser.add_argument("--dataset-path", help="explicit dataset path")
    parser.add_argument("--checkpoint-path", help="explicit checkpoint path")
    parser.add_argument("--task-smoke-cmd", help="explicit minimal task smoke command")
    parser.add_argument("--allow-network", action="store_true", help="allow network-dependent remediation planning")
    parser.add_argument("--fix-scope", default="safe-user-space", help="active fix scope")
    parser.add_argument("--python-version", help="Python version hint for environment creation")
    parser.add_argument("--path-profile", help="shell profile path for PATH repair")
    parser.add_argument("--timeout-seconds", type=int, default=10, help="timeout for explicit task smoke execution")
    args = parser.parse_args()
    args.mode = normalize_mode_args(parser, args)

    working_dir = Path(args.working_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else (working_dir / "out").resolve()
    paths = build_paths(output_dir)
    paths["meta_dir"].mkdir(parents=True, exist_ok=True)
    write_inputs_snapshot(args, working_dir, output_dir, paths["inputs_json"])

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

    write_env_snapshot(
        paths["env_json"],
        args.mode,
        passes,
        initial_state["selection"],
        final_state,
        fix_applied,
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
