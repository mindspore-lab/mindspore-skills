#!/usr/bin/env python3
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from asset_registry import ASSET_KINDS
from asset_schema import asset_locator_summary


LATEST_CACHE_REF = {
    "root": "readiness-output/latest/new-readiness-agent",
    "lock": "readiness-output/latest/new-readiness-agent/workspace-readiness.lock.json",
    "confirmation": "readiness-output/latest/new-readiness-agent/confirmation-latest.json",
    "run_ref": "readiness-output/latest/new-readiness-agent/run-ref.json",
}

LIGHTWEIGHT_RUN_ARTIFACTS = {
    "verdict": "meta/readiness-verdict.json",
    "lock": "artifacts/workspace-readiness.lock.json",
    "confirmation": "artifacts/confirmation-step.json",
}

FULL_RUN_ARTIFACTS = {
    **LIGHTWEIGHT_RUN_ARTIFACTS,
    "report": "report.json",
    "markdown": "report.md",
    "env": "meta/env.json",
    "inputs": "meta/inputs.json",
    "run_log": "logs/run.log",
}


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


def include_full_bundle(verdict: Dict[str, object]) -> bool:
    return str(verdict.get("status") or "") != "NEEDS_CONFIRMATION"


def current_run_artifact_refs(full_bundle: bool) -> Dict[str, str]:
    return dict(FULL_RUN_ARTIFACTS if full_bundle else LIGHTWEIGHT_RUN_ARTIFACTS)


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
    compatibility = evidence_summary.get("compatibility") if isinstance(evidence_summary.get("compatibility"), dict) else {}
    return {
        "mindspore_version": package_versions.get("mindspore"),
        "torch_version": package_versions.get("torch"),
        "torch_npu_version": package_versions.get("torch_npu"),
        "cann_version": evidence_summary.get("cann_version"),
        "cann_path": verdict.get("cann_path"),
        "ascend_env_script_path": verdict.get("ascend_env_script_path"),
        "cann_version_file": verdict.get("cann_version_file"),
        "cann_candidate_paths": verdict.get("cann_candidate_paths"),
        "driver_version": None,
        "python_version": selected_runtime.get("python_version"),
        "compatibility": compatibility,
        "package_versions": package_versions,
        "platform": platform.platform(),
        "git_commit": git_commit(root),
    }


def build_report_envelope(
    *,
    run_id: str,
    verdict: Dict[str, object],
    artifact_refs: List[str],
    log_refs: List[str],
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
        "logs": log_refs,
        "artifacts": artifact_refs,
        "env_ref": "meta/env.json",
        "inputs_ref": "meta/inputs.json",
    }
    if report_status == "failed":
        envelope["error"] = {
            "code": "E_VERIFY",
            "message": verdict.get("summary") or "readiness verification failed",
        }
    return envelope


def compatibility_lines(report: Dict[str, object]) -> List[str]:
    evidence_summary = report.get("evidence_summary") if isinstance(report.get("evidence_summary"), dict) else {}
    compatibility = evidence_summary.get("compatibility") if isinstance(evidence_summary.get("compatibility"), dict) else {}
    if not compatibility:
        return []

    installed_versions = compatibility.get("installed_versions") if isinstance(compatibility.get("installed_versions"), dict) else {}
    recommended_specs = [str(item) for item in (compatibility.get("recommended_package_specs") or []) if str(item).strip()]
    matched_row = compatibility.get("matched_row") if isinstance(compatibility.get("matched_row"), dict) else None

    lines = [
        "## Compatibility",
        "",
        f"- status: `{compatibility.get('status')}`",
        f"- reference_status: `{compatibility.get('reference_status')}`",
        f"- reason: {compatibility.get('reason')}",
    ]
    if installed_versions:
        installed_summary = ", ".join(f"{name}={value}" for name, value in installed_versions.items() if value)
        if installed_summary:
            lines.append(f"- installed_versions: `{installed_summary}`")
    if recommended_specs:
        lines.append(f"- recommended_packages: `{', '.join(recommended_specs)}`")
    if matched_row:
        matched_summary = ", ".join(f"{key}={value}" for key, value in matched_row.items() if value)
        if matched_summary:
            lines.append(f"- matched_row: `{matched_summary}`")
    lines.append("")
    return lines


def render_markdown(report: Dict[str, object], artifact_refs: Dict[str, str]) -> str:
    checks = report.get("checks") if isinstance(report.get("checks"), list) else []
    pending_fields = report.get("pending_confirmation_fields") if isinstance(report.get("pending_confirmation_fields"), list) else []
    current_confirmation = report.get("current_confirmation") if isinstance(report.get("current_confirmation"), dict) else {}
    assets = report.get("assets") if isinstance(report.get("assets"), dict) else {}
    latest_cache_ref = report.get("latest_cache_ref") if isinstance(report.get("latest_cache_ref"), dict) else {}
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
        "## Assets",
        "",
    ]
    for asset_name in ASSET_KINDS:
        asset_bundle = assets.get(asset_name) if isinstance(assets.get(asset_name), dict) else {}
        selected = asset_bundle.get("selected") if isinstance(asset_bundle.get("selected"), dict) else {}
        lines.append(f"- {asset_name}: `{selected.get('source_type')}` `{asset_locator_summary(selected) or 'unresolved'}`")
    lines.extend(
        [
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
            f"- confirmation_step: `{artifact_refs.get('confirmation')}`",
            "",
        ]
    )
    if current_confirmation:
        lines.extend(
            [
                "### Current Step",
                "",
                f"- field: `{current_confirmation.get('field')}`",
                f"- label: `{current_confirmation.get('label')}`",
                f"- prompt: {current_confirmation.get('prompt')}",
                "",
                "### Options",
                "",
            ]
        )
        for option in current_confirmation.get("options") or []:
            lines.append(f"{option.get('index')}. {option.get('label')}")
        lines.append("")
    lines.extend(["## Verify", ""])
    for item in checks:
        lines.append(f"- `{item.get('id')}`: `{item.get('status')}` {item.get('summary')}")
    lines.extend([""] + compatibility_lines(report))
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- readiness_verdict: `{artifact_refs.get('verdict')}`",
            f"- readiness_lock: `{artifact_refs.get('lock')}`",
            f"- confirmation_step: `{artifact_refs.get('confirmation')}`",
        ]
    )
    if artifact_refs.get("report"):
        lines.extend(
            [
                f"- report_json: `{artifact_refs.get('report')}`",
                f"- report_markdown: `{artifact_refs.get('markdown')}`",
                f"- env_snapshot: `{artifact_refs.get('env')}`",
                f"- inputs_snapshot: `{artifact_refs.get('inputs')}`",
                f"- run_log: `{artifact_refs.get('run_log')}`",
            ]
        )
    if latest_cache_ref:
        lines.extend(
            [
                f"- latest_lock: `{latest_cache_ref.get('lock')}`",
                f"- latest_confirmation: `{latest_cache_ref.get('confirmation')}`",
                f"- latest_run_ref: `{latest_cache_ref.get('run_ref')}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Environment",
            "",
            f"- cann_version: `{(report.get('evidence_summary') or {}).get('cann_version')}`",
            f"- cann_path: `{report.get('cann_path')}`",
            f"- ascend_env_script_path: `{report.get('ascend_env_script_path')}`",
            f"- cann_version_file: `{report.get('cann_version_file')}`",
            f"- torch_version: `{((report.get('evidence_summary') or {}).get('package_versions') or {}).get('torch')}`",
            f"- torch_npu_version: `{((report.get('evidence_summary') or {}).get('package_versions') or {}).get('torch_npu')}`",
            f"- mindspore_version: `{((report.get('evidence_summary') or {}).get('package_versions') or {}).get('mindspore')}`",
            f"- uses_llamafactory: `{str((report.get('evidence_summary') or {}).get('uses_llamafactory')).lower()}`",
            "",
            "## Logs",
            "",
            f"- see `{artifact_refs.get('run_log', 'logs/run.log')}`",
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
    system_layer = validation.get("system_layer") if isinstance(validation.get("system_layer"), dict) else {}
    cann_version_info = validation.get("cann_version_info") if isinstance(validation.get("cann_version_info"), dict) else {}
    resolved_cann_path = profile.get("cann_path") or system_layer.get("cann_path_input")

    return {
        "schema_version": "new-readiness-agent/0.1",
        "skill": "new-readiness-agent",
        "run_id": run_id,
        "phase": "awaiting_confirmation" if validation["status"] == "NEEDS_CONFIRMATION" else "validated",
        "status": validation["status"],
        "confirmation_required": bool(confirmation.get("required")),
        "pending_confirmation_fields": list(confirmation.get("pending_fields") or []),
        "current_confirmation": confirmation.get("current_confirmation"),
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
        "assets": profile.get("assets"),
        "selected_python": selected_env.get("python_path"),
        "selected_env_root": selected_env.get("env_root"),
        "cann_path": resolved_cann_path,
        "ascend_env_script_path": system_layer.get("ascend_env_script_path"),
        "cann_version_file": cann_version_info.get("cann_version_file"),
        "cann_candidate_paths": system_layer.get("ascend_env_candidate_paths"),
        "environment_candidates": state["scan"]["environment"]["candidates"],
        "checks": validation["checks"],
        "confirmed_fields": profile["confirmed_fields"],
        "evidence_summary": validation["evidence_summary"],
        "lock_ref": "artifacts/workspace-readiness.lock.json",
        "latest_cache_ref": dict(LATEST_CACHE_REF),
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
        "current_confirmation": verdict.get("current_confirmation"),
        "can_run": verdict.get("can_run"),
        "target": verdict.get("target"),
        "launcher": launcher.get("value"),
        "framework": framework.get("value"),
        "backend": "ascend-npu",
        "cann": evidence_summary.get("cann_version"),
        "cann_path": verdict.get("cann_path"),
        "ascend_env_script_path": verdict.get("ascend_env_script_path"),
        "cann_version_file": verdict.get("cann_version_file"),
        "cann_candidate_paths": verdict.get("cann_candidate_paths"),
        "selected_python": verdict.get("selected_python"),
        "selected_env_root": verdict.get("selected_env_root"),
        "entry_script": (confirmed_fields.get("entry_script") or {}).get("value"),
        "launch_command": launcher.get("command_template"),
        "assets": verdict.get("assets"),
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


def write_latest_cache(root: Path, run_id: str, lock_payload: Dict[str, object], current_confirmation: Optional[Dict[str, object]], pending_confirmation_fields: List[str], confirmed_fields: Dict[str, object], output_dir: Path) -> Dict[str, str]:
    latest_root = root / "readiness-output" / "latest" / "new-readiness-agent"
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
            "current_confirmation": current_confirmation,
            "pending_confirmation_fields": pending_confirmation_fields,
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
    return dict(LATEST_CACHE_REF)


def write_report_bundle(
    *,
    root: Path,
    run_id: str,
    output_dir: Path,
    inputs_snapshot: Dict[str, object],
    state: Dict[str, object],
) -> Dict[str, object]:
    meta_dir = output_dir / "meta"
    artifacts_dir = output_dir / "artifacts"
    meta_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    verdict = build_verdict(run_id, root, state)
    full_bundle = include_full_bundle(verdict)
    artifact_refs = current_run_artifact_refs(full_bundle)
    lock_payload = build_workspace_lock(verdict)

    report_path = output_dir / "report.json"
    markdown_path = output_dir / "report.md"
    logs_dir = output_dir / "logs"
    run_log_path = logs_dir / "run.log"
    env_path = meta_dir / "env.json"
    inputs_path = meta_dir / "inputs.json"
    verdict_path = meta_dir / "readiness-verdict.json"
    lock_path = artifacts_dir / "workspace-readiness.lock.json"
    confirmation_path = artifacts_dir / "confirmation-step.json"

    write_json(verdict_path, verdict)
    write_json(lock_path, lock_payload)
    write_json(confirmation_path, verdict["current_confirmation"] or {})

    latest_cache = write_latest_cache(
        root,
        run_id,
        lock_payload,
        verdict.get("current_confirmation") if isinstance(verdict.get("current_confirmation"), dict) else None,
        list(verdict.get("pending_confirmation_fields") or []),
        verdict["confirmed_fields"],
        output_dir,
    )
    verdict["latest_cache_ref"] = latest_cache
    write_json(verdict_path, verdict)

    envelope = None
    if full_bundle:
        logs_dir.mkdir(parents=True, exist_ok=True)
        write_json(env_path, build_env_snapshot(root, verdict))
        write_json(inputs_path, inputs_snapshot)
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
            artifact_refs=[
                artifact_refs["markdown"],
                artifact_refs["env"],
                artifact_refs["inputs"],
                artifact_refs["verdict"],
                artifact_refs["lock"],
                artifact_refs["confirmation"],
            ],
            log_refs=[artifact_refs["run_log"]],
        )
        write_json(report_path, envelope)
        markdown_path.write_text(
            render_markdown(
                verdict,
                artifact_refs=artifact_refs,
            ),
            encoding="utf-8",
        )
    return {
        "envelope": envelope,
        "verdict": verdict,
        "lock_payload": lock_payload,
        "artifact_refs": artifact_refs,
    }
