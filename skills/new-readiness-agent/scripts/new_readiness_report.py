#!/usr/bin/env python3
import json
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional
from uuid import uuid4


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def shared_status(verdict_status: str) -> str:
    if verdict_status == "READY":
        return "success"
    if verdict_status in {"WARN", "NEEDS_CONFIRMATION"}:
        return "partial"
    return "failed"


def artifact_ref(path: Path, root: Path) -> str:
    return str(path.relative_to(root))


def git_commit(root: Path) -> Optional[str]:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            check=True,
            text=True,
            capture_output=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() or None


def build_env_snapshot(root: Path, verdict: Dict[str, object]) -> Dict[str, object]:
    evidence_summary = verdict.get("evidence_summary") if isinstance(verdict.get("evidence_summary"), dict) else {}
    package_versions = evidence_summary.get("package_versions") if isinstance(evidence_summary.get("package_versions"), dict) else {}
    selected_runtime = evidence_summary.get("selected_runtime_environment") if isinstance(evidence_summary.get("selected_runtime_environment"), dict) else {}
    return {
        "mindspore_version": package_versions.get("mindspore"),
        "cann_version": evidence_summary.get("cann_version"),
        "driver_version": None,
        "python_version": selected_runtime.get("python_version"),
        "platform": platform.platform(),
        "git_commit": git_commit(root),
    }


def build_report_envelope(
    *,
    run_id: str,
    verdict: Dict[str, object],
    output_dir: Path,
    verdict_path: Path,
    lock_path: Path,
    form_path: Path,
    run_log_path: Path,
) -> Dict[str, object]:
    report_status = shared_status(str(verdict.get("status") or ""))
    envelope = {
        "schema_version": "1.0.0",
        "skill": "new-readiness-agent",
        "run_id": run_id,
        "status": report_status,
        "start_time": now_utc_iso(),
        "end_time": now_utc_iso(),
        "duration_sec": 0,
        "steps": [
            {
                "name": "workspace-analyzer",
                "status": "success",
            },
            {
                "name": "compatibility-validator",
                "status": report_status,
                "message": verdict.get("summary"),
            },
            {
                "name": "snapshot-builder",
                "status": "success",
            },
            {
                "name": "report-builder",
                "status": report_status,
            },
        ],
        "logs": [
            artifact_ref(run_log_path, output_dir),
        ],
        "artifacts": [
            artifact_ref(verdict_path, output_dir),
            artifact_ref(lock_path, output_dir),
            artifact_ref(form_path, output_dir),
        ],
        "env_ref": "meta/env.json",
        "inputs_ref": "meta/inputs.json",
    }
    if report_status == "failed":
        envelope["error"] = {
            "code": "E_VERIFY",
            "message": verdict.get("summary") or "readiness verification failed",
        }
    return envelope


def render_markdown(report: Dict[str, object], lock_ref: str, form_ref: str) -> str:
    checks = report.get("checks") if isinstance(report.get("checks"), list) else []
    pending_fields = report.get("pending_confirmation_fields") if isinstance(report.get("pending_confirmation_fields"), list) else []
    lines = [
        "# New Readiness Report",
        "",
        "## Summary",
        "",
        f"- phase: `{report.get('phase')}`",
        f"- status: `{report.get('status')}`",
        f"- can_run: `{str(report.get('can_run')).lower()}`",
        f"- target: `{report.get('target')}`",
        f"- summary: {report.get('summary')}",
        "",
        "## What",
        "",
        f"- launcher: `{(report.get('launcher') or {}).get('value')}`",
        f"- framework: `{(report.get('framework') or {}).get('value')}`",
        f"- selected_python: `{report.get('selected_python')}`",
        f"- selected_env_root: `{report.get('selected_env_root')}`",
        "",
        "## How",
        "",
        "- workspace scan only",
        "- near-launch probes only",
        "- no environment mutation",
        "",
        "## Confirm",
        "",
        f"- confirmation_required: `{str(report.get('confirmation_required')).lower()}`",
        f"- pending_fields: `{', '.join(pending_fields) if pending_fields else 'none'}`",
        f"- confirmation_form: `{form_ref}`",
        "",
        "## Verify",
        "",
    ]
    for item in checks:
        lines.append(f"- `{item.get('id')}`: `{item.get('status')}` {item.get('summary')}")
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- readiness_lock: `{lock_ref}`",
            "",
            "## Environment",
            "",
            f"- cann_version: `{(report.get('evidence_summary') or {}).get('cann_version')}`",
            f"- uses_llamafactory: `{str((report.get('evidence_summary') or {}).get('uses_llamafactory')).lower()}`",
            "",
            "## Logs",
            "",
            "- see `logs/run.log`",
            "",
            "## Next",
            "",
            f"- {report.get('next_action')}",
            "",
        ]
    )
    return "\n".join(lines)


def build_verdict(run_id: str, root: Path, state: Dict[str, object]) -> Dict[str, object]:
    profile = state["profile"]
    validation = state["validation"]
    confirmation = state.get("confirmation") if isinstance(state.get("confirmation"), dict) else {}
    selected_env = profile.get("selected_environment") or {}
    launcher_candidate = profile.get("selected_launcher_candidate") or {}

    return {
        "schema_version": "new-readiness-agent/0.1",
        "skill": "new-readiness-agent",
        "run_id": run_id,
        "phase": "awaiting_confirmation" if validation["status"] == "NEEDS_CONFIRMATION" else "validated",
        "status": validation["status"],
        "confirmation_required": bool(confirmation.get("required")),
        "pending_confirmation_fields": list(confirmation.get("gate_pending_fields") or []),
        "can_run": validation["can_run"],
        "target": profile.get("target"),
        "summary": validation["summary"],
        "missing_items": validation["missing_items"],
        "warnings": validation["warnings"],
        "next_action": validation["next_action"],
        "launcher": {
            "value": profile.get("launcher"),
            "command_template": profile.get("launch_command"),
            "candidate": launcher_candidate,
        },
        "framework": {
            "value": profile.get("framework"),
        },
        "selected_python": selected_env.get("python_path"),
        "selected_env_root": selected_env.get("env_root"),
        "environment_candidates": state["scan"]["environment"]["candidates"],
        "checks": validation["checks"],
        "confirmation_form": profile["confirmation_form"],
        "confirmed_fields": profile["confirmed_fields"],
        "evidence_summary": validation["evidence_summary"],
        "lock_ref": "artifacts/workspace-readiness.lock.json",
        "latest_cache_ref": {
            "root": "runs/latest/new-readiness-agent",
            "lock": "runs/latest/new-readiness-agent/workspace-readiness.lock.json",
            "confirmation": "runs/latest/new-readiness-agent/confirmation-latest.json",
            "run_ref": "runs/latest/new-readiness-agent/run-ref.json",
        },
    }


def build_workspace_lock(verdict: Dict[str, object]) -> Dict[str, object]:
    confirmed_fields = verdict.get("confirmed_fields") if isinstance(verdict.get("confirmed_fields"), dict) else {}
    launcher = verdict.get("launcher") if isinstance(verdict.get("launcher"), dict) else {}
    framework = verdict.get("framework") if isinstance(verdict.get("framework"), dict) else {}
    evidence_summary = verdict.get("evidence_summary") if isinstance(verdict.get("evidence_summary"), dict) else {}
    required_packages = evidence_summary.get("required_packages") or []
    return {
        "schema_version": "new-readiness-lock/0.1",
        "skill": "new-readiness-agent",
        "phase": verdict.get("phase"),
        "status": verdict.get("status"),
        "confirmation_required": verdict.get("confirmation_required"),
        "pending_confirmation_fields": verdict.get("pending_confirmation_fields"),
        "can_run": verdict.get("can_run"),
        "target": verdict.get("target"),
        "launcher": launcher.get("value"),
        "framework": framework.get("value"),
        "backend": "ascend-npu",
        "cann": evidence_summary.get("cann_version"),
        "selected_python": verdict.get("selected_python"),
        "selected_env_root": verdict.get("selected_env_root"),
        "entry_script": (confirmed_fields.get("entry_script") or {}).get("value"),
        "launch_command": launcher.get("command_template"),
        "config_path": (confirmed_fields.get("config_path") or {}).get("value"),
        "model_path": (confirmed_fields.get("model_path") or {}).get("value"),
        "dataset_path": (confirmed_fields.get("dataset_path") or {}).get("value"),
        "checkpoint_path": (confirmed_fields.get("checkpoint_path") or {}).get("value"),
        "required_packages": required_packages,
        "missing_items": verdict.get("missing_items"),
        "warnings": verdict.get("warnings"),
        "confirmed_fields": confirmed_fields,
        "environment_candidates": verdict.get("environment_candidates"),
        "evidence_summary": evidence_summary,
        "updated_at": now_utc_iso(),
    }


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_latest_cache(root: Path, run_id: str, lock_payload: Dict[str, object], confirmation_form: Dict[str, object], confirmed_fields: Dict[str, object], output_dir: Path) -> Dict[str, str]:
    latest_root = root / "runs" / "latest" / "new-readiness-agent"
    latest_root.mkdir(parents=True, exist_ok=True)
    lock_path = latest_root / "workspace-readiness.lock.json"
    confirmation_path = latest_root / "confirmation-latest.json"
    run_ref_path = latest_root / "run-ref.json"

    write_json(lock_path, lock_payload)
    write_json(
        confirmation_path,
        {
            "schema_version": "new-readiness-agent/confirmation/0.1",
            "confirmed_fields": confirmed_fields,
            "confirmation_form": confirmation_form,
            "updated_at": now_utc_iso(),
        },
    )
    write_json(
        run_ref_path,
        {
            "schema_version": "new-readiness-agent/run-ref/0.1",
            "run_id": run_id,
            "output_dir": str(output_dir),
            "updated_at": now_utc_iso(),
        },
    )
    return {
        "lock": str(lock_path),
        "confirmation": str(confirmation_path),
        "run_ref": str(run_ref_path),
    }


def write_report_bundle(
    *,
    root: Path,
    run_id: str,
    output_dir: Path,
    inputs_snapshot: Dict[str, object],
    state: Dict[str, object],
) -> Dict[str, object]:
    logs_dir = output_dir / "logs"
    meta_dir = output_dir / "meta"
    artifacts_dir = output_dir / "artifacts"
    logs_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    verdict = build_verdict(run_id, root, state)
    lock_payload = build_workspace_lock(verdict)

    report_path = output_dir / "report.json"
    markdown_path = output_dir / "report.md"
    run_log_path = logs_dir / "run.log"
    env_path = meta_dir / "env.json"
    inputs_path = meta_dir / "inputs.json"
    verdict_path = meta_dir / "readiness-verdict.json"
    lock_path = artifacts_dir / "workspace-readiness.lock.json"
    form_path = artifacts_dir / "confirmation-options.json"

    write_json(env_path, build_env_snapshot(root, verdict))
    write_json(inputs_path, inputs_snapshot)
    write_json(verdict_path, verdict)
    write_json(lock_path, lock_payload)
    write_json(form_path, verdict["confirmation_form"])

    latest_cache = write_latest_cache(root, run_id, lock_payload, verdict["confirmation_form"], verdict["confirmed_fields"], output_dir)
    verdict["latest_cache_ref"] = latest_cache
    write_json(verdict_path, verdict)

    run_log_path.write_text(
        "\n".join(
            [
                f"run_id={run_id}",
                f"status={verdict['status']}",
                f"can_run={str(verdict['can_run']).lower()}",
                f"target={verdict['target']}",
                f"launcher={(verdict['launcher'] or {}).get('value')}",
                f"framework={(verdict['framework'] or {}).get('value')}",
            ]
        ),
        encoding="utf-8",
    )

    envelope = build_report_envelope(
        run_id=run_id,
        verdict=verdict,
        output_dir=output_dir,
        verdict_path=verdict_path,
        lock_path=lock_path,
        form_path=form_path,
        run_log_path=run_log_path,
    )
    write_json(report_path, envelope)
    markdown_path.write_text(
        render_markdown(
            verdict,
            lock_ref=artifact_ref(lock_path, output_dir),
            form_ref=artifact_ref(form_path, output_dir),
        ),
        encoding="utf-8",
    )
    return {
        "envelope": envelope,
        "verdict": verdict,
        "lock_payload": lock_payload,
    }
