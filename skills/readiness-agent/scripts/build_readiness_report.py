#!/usr/bin/env python3
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


READY_LEVELS = {"runtime_smoke", "task_smoke"}


def derive_evidence_level(checks: list[dict]) -> str:
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


def check_by_id(checks: list[dict], check_id: str) -> dict | None:
    for item in checks:
        if str(item.get("id")) == check_id:
            return item
    return None


def interpret_task_smoke_state(target: dict, checks: list[dict]) -> str:
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


def scopes_for_check(check_id: str) -> set[str]:
    if check_id in {"system-device", "system-ascend-env"}:
        return {"system"}
    if check_id == "python-uv":
        return {"tool-resolution", "python-environment"}
    if check_id == "python-selected-env":
        return {"python-environment"}
    if check_id in {"framework-path", "framework-importability", "framework-smoke-prerequisite"}:
        return {"framework"}
    if check_id == "runtime-importability":
        return {"runtime-dependencies"}
    if check_id == "target-stability":
        return {"target"}
    if check_id.startswith("workspace-"):
        return {"workspace-assets"}
    if check_id.startswith("task-smoke-"):
        return {"task-smoke"}
    return set()


def derive_revalidation_state(fix_applied: dict, checks: list[dict]) -> tuple[bool, list[str], list[str]]:
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
    checks: list[dict],
    revalidated: bool,
) -> tuple[str, bool, str, str]:
    blockers = normalized.get("blockers_detailed", [])
    warnings = normalized.get("warnings_detailed", [])
    target_type = target.get("target_type") or "unknown"
    task_smoke_state = interpret_task_smoke_state(target, checks)

    if any(item.get("category") == "system_fatal" for item in blockers):
        return (
            "BLOCKED",
            False,
            f"{target_type.capitalize()} is blocked by a system-layer readiness failure.",
            "Resolve system-layer blockers and rerun readiness.",
        )
    if blockers:
        next_action = "Resolve blockers and rerun readiness."
        if any(item.get("category") == "workspace_manual" for item in blockers):
            next_action = "Resolve workspace blockers and rerun readiness."
        return (
            "BLOCKED",
            False,
            f"{target_type.capitalize()} is blocked because required readiness prerequisites remain unresolved.",
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
    checks: list[dict],
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
    }


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def resolve_verdict_output_path(output_json: Path, explicit: str | None) -> Path:
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
    if report["warnings"]:
        lines.extend(["## Warnings", ""])
        for item in report["warnings"]:
            lines.append(f"- {item}")
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
