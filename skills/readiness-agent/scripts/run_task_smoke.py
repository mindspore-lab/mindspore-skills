#!/usr/bin/env python3
import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path


def make_check(
    check_id: str,
    status: str,
    summary: str,
    evidence: list[str] | None = None,
    *,
    category_hint: str | None = None,
    severity: str | None = None,
    remediable: bool | None = None,
    remediation_owner: str | None = None,
    revalidation_scope: list[str] | None = None,
    command_preview: str | None = None,
    exit_code: int | None = None,
    stdout_head: str | None = None,
    stderr_head: str | None = None,
    timed_out: bool | None = None,
) -> dict:
    payload = {
        "id": check_id,
        "status": status,
        "summary": summary,
        "evidence": evidence or [],
    }
    if category_hint is not None:
        payload["category_hint"] = category_hint
    if severity is not None:
        payload["severity"] = severity
    if remediable is not None:
        payload["remediable"] = remediable
    if remediation_owner is not None:
        payload["remediation_owner"] = remediation_owner
    if revalidation_scope is not None:
        payload["revalidation_scope"] = revalidation_scope
    if command_preview is not None:
        payload["command_preview"] = command_preview
    if exit_code is not None:
        payload["exit_code"] = exit_code
    if stdout_head is not None:
        payload["stdout_head"] = stdout_head
    if stderr_head is not None:
        payload["stderr_head"] = stderr_head
    if timed_out is not None:
        payload["timed_out"] = timed_out
    return payload


def head_line(text: str | None) -> str | None:
    if not text:
        return None
    stripped = text.strip()
    if not stripped:
        return None
    return stripped.splitlines()[0]


def format_command(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def resolve_entry_script(target: dict, root: Path) -> Path | None:
    entry_script = target.get("entry_script")
    if not entry_script:
        return None
    path = Path(entry_script)
    if not path.is_absolute():
        path = root / path
    return path


def resolve_probe_python(closure: dict) -> str:
    python_env = closure.get("layers", {}).get("python_environment", {})
    return python_env.get("probe_python_path") or sys.executable


def run_script_parse_smoke(entry_script: Path | None, probe_python: str, root: Path) -> dict:
    if not entry_script or not entry_script.exists():
        return make_check(
            "task-smoke-script-parse",
            "skipped",
            "Task smoke script-parse step is skipped because the entry script is unavailable.",
        )

    if entry_script.suffix.lower() != ".py":
        return make_check(
            "task-smoke-script-parse",
            "skipped",
            "Task smoke script-parse step is skipped because the entry script is not a Python script.",
            evidence=[f"entry_script={entry_script.relative_to(root)}"],
        )

    try:
        command = [probe_python, "-m", "py_compile", str(entry_script)]
        completed = subprocess.run(
            command,
            cwd=str(root),
            text=True,
            capture_output=True,
            timeout=10,
        )
    except subprocess.TimeoutExpired as exc:
        return make_check(
            "task-smoke-script-parse",
            "block",
            "Task smoke script-parse step timed out in the selected environment.",
            evidence=[f"probe_python={probe_python}", f"entry_script={entry_script.relative_to(root)}"],
            category_hint="workspace",
            severity="high",
            remediable=False,
            remediation_owner="workspace",
            revalidation_scope=["workspace-assets", "task-smoke"],
            command_preview=format_command(command),
            timed_out=True,
            stdout_head=head_line(exc.stdout),
            stderr_head=head_line(exc.stderr),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return make_check(
            "task-smoke-script-parse",
            "block",
            "Task smoke script-parse step failed to start in the selected environment.",
            evidence=[f"probe_python={probe_python}", f"error={exc}"],
            category_hint="workspace",
            severity="high",
            remediable=False,
            remediation_owner="workspace",
            revalidation_scope=["workspace-assets", "task-smoke"],
            command_preview=format_command(command),
            timed_out=False,
        )

    if completed.returncode == 0:
        return make_check(
            "task-smoke-script-parse",
            "ok",
            "Task smoke script-parse step passed in the selected environment.",
            evidence=[f"probe_python={probe_python}", f"entry_script={entry_script.relative_to(root)}"],
            command_preview=format_command(command),
            exit_code=completed.returncode,
            stdout_head=head_line(completed.stdout),
            stderr_head=head_line(completed.stderr),
            timed_out=False,
        )

    error = (completed.stderr or completed.stdout or "").strip()
    return make_check(
        "task-smoke-script-parse",
        "block",
        "Task smoke script-parse step failed in the selected environment.",
        evidence=[
            f"probe_python={probe_python}",
            f"entry_script={entry_script.relative_to(root)}",
            f"error={error}",
        ],
        category_hint="workspace",
        severity="high",
        remediable=False,
        remediation_owner="workspace",
        revalidation_scope=["workspace-assets", "task-smoke"],
        command_preview=format_command(command),
        exit_code=completed.returncode,
        stdout_head=head_line(completed.stdout),
        stderr_head=head_line(completed.stderr),
        timed_out=False,
    )


def build_smoke_command(smoke_cmd: str, probe_python: str) -> list[str]:
    parts = shlex.split(smoke_cmd)
    if not parts:
        return []
    if parts[0] in {"python", "python3"}:
        parts[0] = probe_python
    return parts


def run_explicit_task_smoke(target: dict, closure: dict, root: Path, timeout_seconds: int) -> dict:
    smoke_cmd = target.get("task_smoke_cmd")
    if not smoke_cmd:
        return make_check(
            "task-smoke-executed",
            "skipped",
            "No explicit task smoke command is available for this execution target.",
        )

    probe_python = resolve_probe_python(closure)
    command = build_smoke_command(str(smoke_cmd), probe_python)
    if not command:
        return make_check(
            "task-smoke-executed",
            "block",
            "Explicit task smoke command is empty after parsing.",
            evidence=[f"task_smoke_cmd={smoke_cmd}"],
            category_hint="workspace",
            severity="high",
            remediable=False,
            remediation_owner="workspace",
            revalidation_scope=["task-smoke", "target"],
        )

    try:
        completed = subprocess.run(
            command,
            cwd=str(root),
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        return make_check(
            "task-smoke-executed",
            "block",
            "Explicit task smoke command timed out.",
            evidence=[f"command={command}"],
            category_hint="workspace",
            severity="high",
            remediable=False,
            remediation_owner="workspace",
            revalidation_scope=["task-smoke", "target"],
            command_preview=format_command(command),
            timed_out=True,
            stdout_head=head_line(exc.stdout),
            stderr_head=head_line(exc.stderr),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return make_check(
            "task-smoke-executed",
            "block",
            "Explicit task smoke command failed to start.",
            evidence=[f"command={command}", f"error={exc}"],
            category_hint="workspace",
            severity="high",
            remediable=False,
            remediation_owner="workspace",
            revalidation_scope=["task-smoke", "target"],
            command_preview=format_command(command),
            timed_out=False,
        )

    if completed.returncode == 0:
        evidence = [f"command={command}"]
        stdout = (completed.stdout or "").strip()
        if stdout:
            evidence.append(f"stdout={stdout.splitlines()[0]}")
        return make_check(
            "task-smoke-executed",
            "ok",
            "Explicit task smoke command completed successfully.",
            evidence=evidence,
            command_preview=format_command(command),
            exit_code=completed.returncode,
            stdout_head=head_line(completed.stdout),
            stderr_head=head_line(completed.stderr),
            timed_out=False,
        )

    error = (completed.stderr or completed.stdout or "").strip()
    return make_check(
        "task-smoke-executed",
        "block",
        "Explicit task smoke command failed.",
        evidence=[f"command={command}", f"error={error}"],
        category_hint="workspace",
        severity="high",
        remediable=False,
        remediation_owner="workspace",
        revalidation_scope=["task-smoke", "target"],
        command_preview=format_command(command),
        exit_code=completed.returncode,
        stdout_head=head_line(completed.stdout),
        stderr_head=head_line(completed.stderr),
        timed_out=False,
    )


def run_task_smoke(target: dict, closure: dict, timeout_seconds: int) -> list[dict]:
    root = Path(target["working_dir"]).resolve()
    probe_python = resolve_probe_python(closure)
    entry_script = resolve_entry_script(target, root)
    checks = [
        run_script_parse_smoke(entry_script, probe_python, root),
        run_explicit_task_smoke(target, closure, root, timeout_seconds),
    ]
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description="Run minimal task smoke checks for readiness-agent")
    parser.add_argument("--target-json", required=True, help="path to execution target JSON")
    parser.add_argument("--closure-json", required=True, help="path to dependency closure JSON")
    parser.add_argument("--output-json", required=True, help="path to output task smoke checks JSON")
    parser.add_argument("--timeout-seconds", type=int, default=10, help="timeout for explicit smoke execution")
    args = parser.parse_args()

    target = json.loads(Path(args.target_json).read_text(encoding="utf-8"))
    closure = json.loads(Path(args.closure_json).read_text(encoding="utf-8"))
    checks = run_task_smoke(target, closure, args.timeout_seconds)
    Path(args.output_json).write_text(json.dumps(checks, indent=2), encoding="utf-8")
    print(json.dumps({"checks": len(checks)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
