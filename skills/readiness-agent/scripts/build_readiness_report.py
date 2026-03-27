#!/usr/bin/env python3
import argparse
import json
import shlex
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Set, Tuple
from uuid import uuid4


READY_LEVELS = {"runtime_smoke", "task_smoke"}
AUTO_REMEDIABLE_CATEGORIES = {"env_remediable", "framework_remediable", "asset_remediable"}
DEFAULT_PIP_INDEX_URL = "https://pypi.tuna.tsinghua.edu.cn/simple"
FALLBACK_PIP_INDEX_URL = "https://mirrors.aliyun.com/pypi/simple/"


def derive_evidence_level(checks: List[dict]) -> str:
    ok_ids = {
        str(item.get("id"))
        for item in checks
        if (item.get("status") or "").strip().lower() == "ok"
    }

    if "task-smoke-executed" in ok_ids:
        return "task_smoke"
    if "framework-smoke-prerequisite" in ok_ids:
        return "runtime_smoke"
    if "framework-importability" in ok_ids:
        return "import"
    return "structural"


def check_by_id(checks: List[dict], check_id: str) -> Optional[dict]:
    for item in checks:
        if str(item.get("id")) == check_id:
            return item
    return None


def interpret_task_smoke_state(target: dict, checks: List[dict]) -> str:
    if not target.get("task_smoke_cmd"):
        return "not_requested"

    task_smoke = check_by_id(checks, "task-smoke-executed")
    if not task_smoke:
        return "missing_result"

    status = (task_smoke.get("status") or "").strip().lower()
    if status == "ok":
        return "passed"
    if status == "block":
        return "failed"
    if status == "skipped":
        return "skipped"
    return "unknown"


def scopes_for_check(check_id: str) -> Set[str]:
    if check_id in {"system-device", "system-ascend-env"}:
        return {"system"}
    if check_id == "python-uv":
        return {"tool-resolution", "python-environment"}
    if check_id in {"python-selected-env", "python-selected-python"}:
        return {"python-environment"}
    if check_id in {"framework-path", "framework-importability", "framework-smoke-prerequisite", "framework-compatibility"}:
        return {"framework"}
    if check_id == "runtime-importability":
        return {"runtime-dependencies"}
    if check_id.startswith("remote-huggingface-"):
        return {"workspace-assets"}
    if check_id == "target-stability":
        return {"target"}
    if check_id.startswith("workspace-"):
        return {"workspace-assets"}
    if check_id.startswith("task-smoke-"):
        return {"task-smoke"}
    return set()


def derive_revalidation_state(fix_applied: dict, checks: List[dict]) -> Tuple[bool, List[str], List[str]]:
    executed_actions = fix_applied.get("executed_actions", [])
    if not executed_actions:
        return True, [], []

    required_scopes = sorted({scope for scope in fix_applied.get("needs_revalidation", []) if scope})
    if not required_scopes:
        return True, [], []

    covered_scopes = sorted(
        {
            scope
            for item in checks
            for scope in scopes_for_check(str(item.get("id") or ""))
        }
    )
    missing_scopes = [scope for scope in required_scopes if scope not in covered_scopes]
    return not missing_scopes, required_scopes, covered_scopes


def synthesize_user_result(
    target: dict,
    normalized: dict,
    evidence_level: str,
    checks: List[dict],
    revalidated: bool,
) -> Tuple[str, bool, str, str]:
    blockers = normalized.get("blockers_detailed", [])
    warnings = normalized.get("warnings_detailed", [])
    target_type = target.get("target_type") or "unknown"
    task_smoke_state = interpret_task_smoke_state(target, checks)
    has_workspace_manual = any(item.get("category") == "workspace_manual" for item in blockers)
    has_auto_remediable = any(item.get("category") in AUTO_REMEDIABLE_CATEGORIES for item in blockers)
    has_unknown = any(item.get("category") == "unknown" for item in blockers)

    if any(item.get("category") == "system_fatal" for item in blockers):
        return (
            "BLOCKED",
            False,
            f"{target_type.capitalize()} is blocked by a system-layer readiness failure.",
            "Resolve system-layer blockers and rerun readiness.",
        )
    if blockers:
        next_action = "Resolve blockers and rerun readiness."
        summary = f"{target_type.capitalize()} is blocked because required readiness prerequisites remain unresolved."
        if any(str(item.get("id")) == "python-selected-env" for item in blockers):
            next_action = (
                "Create or select a workspace-local Python environment first, then rerun readiness. "
                "Do not use system python or pip for this target."
            )
            summary = f"{target_type.capitalize()} is blocked because no usable workspace-local Python environment is selected yet."
        if has_workspace_manual and has_auto_remediable:
            next_action = (
                "Resolve manual workspace blockers such as missing dataset or config paths first, "
                "then rerun readiness to clear any remaining environment blockers."
            )
            summary = (
                f"{target_type.capitalize()} is blocked because manual workspace blockers remain, "
                "and additional environment remediation is still required."
            )
        elif has_workspace_manual:
            next_action = "Resolve workspace blockers such as dataset, config, or checkpoint paths and rerun readiness."
            summary = f"{target_type.capitalize()} is blocked because manual workspace inputs are still missing."
        elif has_auto_remediable:
            next_action = (
                "Repair the selected workspace environment or other remediable target-scoped inputs in fix/auto mode, then rerun readiness."
            )
            summary = f"{target_type.capitalize()} is blocked because remediable target-scoped environment or input issues still remain."
        elif has_unknown:
            next_action = "Inspect unresolved blockers, confirm the intended target, and rerun readiness."
            summary = f"{target_type.capitalize()} is blocked because one or more readiness blockers remain unresolved."
        return (
            "BLOCKED",
            False,
            summary,
            next_action,
        )
    if target_type not in {"training", "inference"}:
        return (
            "WARN",
            False,
            "Readiness target remains ambiguous and cannot yet be certified.",
            "Confirm the intended execution target and rerun readiness.",
        )
    if not revalidated:
        return (
            "WARN",
            False,
            f"{target_type.capitalize()} changed through remediation, but required revalidation is still incomplete.",
            "Rerun the required readiness checks before certification.",
        )
    if task_smoke_state == "failed":
        return (
            "BLOCKED",
            False,
            f"{target_type.capitalize()} is blocked because the explicit task smoke command failed.",
            "Inspect the task smoke failure, fix the target path, and rerun readiness.",
        )
    if task_smoke_state in {"skipped", "missing_result", "unknown"}:
        return (
            "WARN",
            False,
            f"{target_type.capitalize()} requires task-smoke revalidation before certification can be completed.",
            "Run the explicit task smoke command and rerun readiness.",
        )
    if warnings:
        if evidence_level in READY_LEVELS:
            return (
                "WARN",
                True,
                f"{target_type.capitalize()} may run, but warnings still reduce certification confidence.",
                "Inspect warnings before proceeding with the intended task.",
            )
        return (
            "WARN",
            False,
            f"{target_type.capitalize()} may run, but readiness evidence is still incomplete.",
            "Resolve warnings or gather stronger runtime evidence, then rerun readiness.",
        )
    if evidence_level in READY_LEVELS:
        return (
            "READY",
            True,
            f"Current environment is ready for {target_type}.",
            "Start the intended task with the discovered entry script.",
        )
    return (
        "WARN",
        False,
        f"{target_type.capitalize()} appears plausible, but evidence is not strong enough for READY.",
        "Gather stronger runtime evidence and rerun readiness.",
    )


def build_report(
    target: dict,
    normalized: dict,
    evidence_level: str,
    fix_applied: dict,
    checks: List[dict],
    dependency_closure: dict,
) -> dict:
    effective_evidence_level = derive_evidence_level(checks) if evidence_level == "auto" else evidence_level
    task_smoke_state = interpret_task_smoke_state(target, checks)
    executed_actions = fix_applied.get("executed_actions", [])
    revalidated, revalidation_required, revalidation_covered = derive_revalidation_state(fix_applied, checks)
    status, can_run, summary, next_action = synthesize_user_result(
        target,
        normalized,
        effective_evidence_level,
        checks,
        revalidated,
    )
    target_type = target.get("target_type")
    if target_type not in {"training", "inference"}:
        target_type = "training" if target.get("entry_script", "").lower().find("train") >= 0 else "inference"

    return {
        "schema_version": "readiness-agent/0.1",
        "skill": "readiness-agent",
        "status": status,
        "can_run": can_run,
        "target": target_type,
        "summary": summary,
        "blockers": normalized.get("blockers", []),
        "warnings": normalized.get("warnings", []),
        "next_action": next_action,
        "execution_target": target,
        "evidence_level": effective_evidence_level,
        "task_smoke_state": task_smoke_state,
        "dependency_closure": dependency_closure,
        "checks": checks,
        "blockers_detailed": normalized.get("blockers_detailed", []),
        "warnings_detailed": normalized.get("warnings_detailed", []),
        "fix_applied": fix_applied,
        "revalidated": revalidated,
        "revalidation_required_scopes": revalidation_required,
        "revalidation_covered_scopes": revalidation_covered,
        "selected_environment_guidance": build_selected_environment_guidance(target, dependency_closure),
        "remote_asset_guidance": build_remote_asset_guidance(dependency_closure),
    }


def build_selected_environment_guidance(target: dict, dependency_closure: dict) -> dict:
    python_env = dependency_closure.get("layers", {}).get("python_environment", {})
    selected_env_root = python_env.get("selected_env_root")
    probe_python_path = python_env.get("probe_python_path")
    working_dir = target.get("working_dir") or dependency_closure.get("working_dir")
    readiness_env_path = str((Path(working_dir).resolve() / ".readiness.env")) if working_dir else None
    guidance = {
        "selected_env_root": selected_env_root,
        "selected_python": probe_python_path,
        "system_python_allowed": False,
        "readiness_env_path": readiness_env_path,
    }
    if readiness_env_path:
        guidance["source_command"] = f"source {shlex.quote(readiness_env_path)}"
    if probe_python_path:
        guidance["verification_command"] = f"{probe_python_path} <script.py>"
        if readiness_env_path:
            guidance["run_command"] = (
                f"source {shlex.quote(readiness_env_path)} && "
                f"{shlex.quote(probe_python_path)} <script.py>"
            )
        guidance["install_command"] = (
            f"uv pip install --python {probe_python_path} --index-url "
            f"{DEFAULT_PIP_INDEX_URL} <package>"
        )
        guidance["install_command_fallback"] = (
            f"uv pip install --python {probe_python_path} --index-url "
            f"{FALLBACK_PIP_INDEX_URL} <package>"
        )
    else:
        guidance["verification_command"] = None
        guidance["install_command"] = None
        guidance["message"] = (
            "Workspace-local Python is unresolved. Do not use system python or pip. "
            "Create or select a workspace-local environment first."
        )
    return guidance


def build_remote_asset_guidance(dependency_closure: dict) -> Optional[dict]:
    remote_assets = dependency_closure.get("layers", {}).get("remote_assets", {})
    assets = remote_assets.get("assets") or {}
    if not assets:
        return None

    cache_layout = remote_assets.get("cache_layout") or {}
    return {
        "mode": "remote-assets",
        "asset_kinds": sorted(assets.keys()),
        "hf_endpoint": remote_assets.get("hf_endpoint"),
        "hf_endpoint_source": remote_assets.get("hf_endpoint_source"),
        "endpoint_reachable": remote_assets.get("endpoint_reachable"),
        "cache_source": cache_layout.get("source"),
        "hf_home": cache_layout.get("hf_home"),
        "hub_cache": cache_layout.get("hub_cache"),
        "datasets_cache": cache_layout.get("datasets_cache"),
        "hub_cache_writable": cache_layout.get("hub_cache_writable"),
        "datasets_cache_writable": cache_layout.get("datasets_cache_writable"),
        "notes": [
            "Remote Hugging Face assets resolve at runtime by default when repo IDs are known.",
            "Existing HUGGINGFACE_HUB_CACHE, HF_DATASETS_CACHE, or HF_HOME settings take precedence over the working-directory default cache.",
        ],
    }


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def resolve_verdict_output_path(output_json: Path, explicit: Optional[str]) -> Path:
    if explicit:
        return Path(explicit)
    return output_json.parent / "meta" / "readiness-verdict.json"


def artifact_ref(path: Path, root: Path) -> str:
    return str(path.relative_to(root))


def map_shared_status(verdict_status: str) -> str:
    if verdict_status == "READY":
        return "success"
    if verdict_status in {"WARN", "BLOCKED"}:
        return "partial"
    return "failed"


def build_shared_envelope(
    verdict: dict,
    output_json: Path,
    output_md: Path,
    output_verdict_json: Path,
) -> dict:
    run_root = output_json.parent
    timestamp = now_utc_iso()
    shared_status = map_shared_status(str(verdict.get("status") or ""))
    return {
        "schema_version": "1.0.0",
        "skill": "readiness-agent",
        "run_id": f"readiness-agent-{uuid4().hex[:12]}",
        "status": shared_status,
        "start_time": timestamp,
        "end_time": timestamp,
        "duration_sec": 0,
        "steps": [
            {
                "name": "readiness-certification",
                "status": shared_status,
                "message": verdict.get("summary", ""),
            }
        ],
        "logs": [],
        "artifacts": [
            artifact_ref(output_md, run_root),
            artifact_ref(output_verdict_json, run_root),
        ],
        "env_ref": "meta/env.json",
        "inputs_ref": "meta/inputs.json",
    }


def render_markdown(report: dict) -> str:
    lines = [
        "# Readiness Report",
        "",
        f"- status: `{report['status']}`",
        f"- can_run: `{str(report['can_run']).lower()}`",
        f"- target: `{report['target']}`",
        f"- summary: {report['summary']}",
        f"- next_action: {report['next_action']}",
        "",
    ]
    if report["blockers"]:
        lines.extend(["## Blockers", ""])
        for item in report["blockers"]:
            lines.append(f"- {item}")
        lines.append("")
    blocker_groups = {
        "manual": [
            item for item in report.get("blockers_detailed", [])
            if item.get("category") == "workspace_manual"
        ],
        "auto": [
            item for item in report.get("blockers_detailed", [])
            if item.get("category") in AUTO_REMEDIABLE_CATEGORIES
        ],
    }
    if blocker_groups["manual"]:
        lines.extend(["## Manual Blockers", ""])
        for item in blocker_groups["manual"]:
            lines.append(f"- {item.get('summary')}")
        lines.append("")
    if blocker_groups["auto"]:
        lines.extend(["## Auto-Remediable Blockers", ""])
        for item in blocker_groups["auto"]:
            lines.append(f"- {item.get('summary')}")
        lines.append("")
    if report["warnings"]:
        lines.extend(["## Warnings", ""])
        for item in report["warnings"]:
            lines.append(f"- {item}")
        lines.append("")
    guidance = report.get("selected_environment_guidance") or {}
    lines.extend(["## Environment Guidance", ""])
    lines.append("- system_python_allowed: `false`")
    if guidance.get("readiness_env_path"):
        lines.append(f"- readiness_env_path: `{guidance['readiness_env_path']}`")
    if guidance.get("source_command"):
        lines.append(f"- source_command: `{guidance['source_command']}`")
    if guidance.get("selected_env_root"):
        lines.append(f"- selected_env_root: `{guidance['selected_env_root']}`")
    if guidance.get("selected_python"):
        lines.append(f"- selected_python: `{guidance['selected_python']}`")
    if guidance.get("run_command"):
        lines.append(f"- run_command: `{guidance['run_command']}`")
    if guidance.get("verification_command"):
        lines.append(f"- verification_command: `{guidance['verification_command']}`")
    if guidance.get("install_command"):
        lines.append(f"- install_command: `{guidance['install_command']}`")
    if guidance.get("install_command_fallback"):
        lines.append(f"- install_command_fallback: `{guidance['install_command_fallback']}`")
    if guidance.get("message"):
        lines.append(f"- message: {guidance['message']}")
    lines.append("")
    remote_guidance = report.get("remote_asset_guidance") or {}
    if remote_guidance:
        lines.extend(["## Remote Asset Guidance", ""])
        lines.append(f"- mode: `{remote_guidance.get('mode')}`")
        if remote_guidance.get("asset_kinds"):
            lines.append(f"- asset_kinds: `{', '.join(remote_guidance['asset_kinds'])}`")
        if remote_guidance.get("hf_endpoint"):
            lines.append(f"- hf_endpoint: `{remote_guidance['hf_endpoint']}`")
        if remote_guidance.get("hf_endpoint_source"):
            lines.append(f"- hf_endpoint_source: `{remote_guidance['hf_endpoint_source']}`")
        if remote_guidance.get("cache_source"):
            lines.append(f"- cache_source: `{remote_guidance['cache_source']}`")
        if remote_guidance.get("hf_home"):
            lines.append(f"- hf_home: `{remote_guidance['hf_home']}`")
        if remote_guidance.get("hub_cache"):
            lines.append(f"- hub_cache: `{remote_guidance['hub_cache']}`")
        if remote_guidance.get("datasets_cache"):
            lines.append(f"- datasets_cache: `{remote_guidance['datasets_cache']}`")
        if remote_guidance.get("notes"):
            for note in remote_guidance["notes"]:
                lines.append(f"- note: {note}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a minimal readiness report")
    parser.add_argument("--target-json", required=True, help="path to execution target JSON")
    parser.add_argument("--normalized-json", required=True, help="path to normalized blockers JSON")
    parser.add_argument("--output-json", required=True, help="path to output report JSON")
    parser.add_argument("--output-md", required=True, help="path to output report Markdown")
    parser.add_argument("--output-verdict-json", help="optional path to output readiness verdict JSON")
    parser.add_argument("--checks-json", help="optional path to checks JSON for auto evidence derivation")
    parser.add_argument("--closure-json", help="optional path to dependency closure JSON")
    parser.add_argument("--evidence-level", default="auto", help="internal evidence level or 'auto'")
    parser.add_argument("--fix-applied-json", help="optional path to fix actions JSON")
    args = parser.parse_args()

    target = json.loads(Path(args.target_json).read_text(encoding="utf-8"))
    normalized = json.loads(Path(args.normalized_json).read_text(encoding="utf-8"))
    checks = []
    if args.checks_json:
        checks = json.loads(Path(args.checks_json).read_text(encoding="utf-8"))
    dependency_closure = {}
    if args.closure_json:
        dependency_closure = json.loads(Path(args.closure_json).read_text(encoding="utf-8"))
    fix_applied = {
        "execute": False,
        "results": [],
        "executed_actions": [],
        "failed_actions": [],
        "needs_revalidation": [],
    }
    if args.fix_applied_json:
        fix_applied = json.loads(Path(args.fix_applied_json).read_text(encoding="utf-8"))

    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_verdict_json = resolve_verdict_output_path(output_json, args.output_verdict_json)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_verdict_json.parent.mkdir(parents=True, exist_ok=True)

    verdict = build_report(target, normalized, args.evidence_level, fix_applied, checks, dependency_closure)
    shared_report = build_shared_envelope(verdict, output_json, output_md, output_verdict_json)

    output_json.write_text(json.dumps(shared_report, indent=2), encoding="utf-8")
    output_verdict_json.write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    output_md.write_text(render_markdown(verdict), encoding="utf-8")
    print(json.dumps({"status": verdict["status"], "target": verdict["target"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
