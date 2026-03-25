#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import List, Optional


def describe_probe_source(probe_source: Optional[str]) -> str:
    mapping = {
        "selected_env": "selected environment",
        "explicit_env": "selected environment",
        "workspace_env": "selected environment",
        "explicit_python": "selected Python interpreter",
    }
    return mapping.get(probe_source or "", "selected Python interpreter")


def make_check(
    check_id: str,
    status: str,
    summary: str,
    *,
    category_hint: Optional[str] = None,
    severity: Optional[str] = None,
    remediable: Optional[bool] = None,
    remediation_owner: Optional[str] = None,
    revalidation_scope: Optional[List[str]] = None,
    evidence: Optional[List[str]] = None,
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
    return payload


def collect_checks(target: dict, closure: dict) -> List[dict]:
    checks: List[dict] = []
    target_type = target.get("target_type") or "unknown"
    framework_layer = closure.get("layers", {}).get("framework", {})
    framework_path = framework_layer.get("framework_path", "unknown")
    runtime_layer = closure.get("layers", {}).get("runtime_dependencies", {})
    system = closure.get("layers", {}).get("system", {})
    python_env = closure.get("layers", {}).get("python_environment", {})
    workspace = closure.get("layers", {}).get("workspace_assets", {})
    selected_env_root = python_env.get("selected_env_root")
    selected_python = python_env.get("selected_python")
    selection_status = python_env.get("selection_status")
    selection_reason = python_env.get("selection_reason")
    probe_source = python_env.get("probe_source")
    probe_python_path = python_env.get("probe_python_path")

    if target_type in {"training", "inference"}:
        checks.append(
            make_check(
                "target-stability",
                "ok",
                f"Execution target is resolved as {target_type}.",
                evidence=[f"target_type={target_type}"],
            )
        )
    else:
        checks.append(
            make_check(
                "target-stability",
                "warn",
                "Execution target remains ambiguous.",
                category_hint="unknown",
                severity="medium",
                evidence=["target_type is unresolved"],
            )
        )

    if system.get("requires_ascend"):
        ascend_env_script_path = system.get("ascend_env_script_path") or "/usr/local/Ascend/ascend-toolkit/set_env.sh"
        probe_env_source = system.get("probe_env_source")
        probe_env_error = system.get("probe_env_error")
        if not system.get("device_paths_present"):
            checks.append(
                make_check(
                    "system-device",
                    "block",
                    "Ascend device visibility is missing for the selected target.",
                    category_hint="system",
                    severity="fatal",
                    remediable=False,
                    remediation_owner="manual-system",
                    revalidation_scope=["system"],
                    evidence=["/dev/davinci* not found"],
                )
            )
        else:
            checks.append(
                make_check(
                    "system-device",
                    "ok",
                    "Ascend device visibility evidence is present.",
                    evidence=["/dev/davinci* exists"],
                )
            )

        if not system.get("ascend_env_script_present"):
            checks.append(
                make_check(
                    "system-ascend-env",
                    "block",
                    "Ascend environment sourcing script is missing.",
                    category_hint="system",
                    severity="fatal",
                    remediable=False,
                    remediation_owner="manual-system",
                    revalidation_scope=["system"],
                    evidence=[f"{ascend_env_script_path} missing"],
                )
            )
        else:
            evidence = [f"set_env.sh={ascend_env_script_path}"]
            if probe_env_source:
                evidence.append(f"probe_env_source={probe_env_source}")
            if probe_env_error:
                evidence.append(f"probe_env_error={probe_env_error}")
            checks.append(
                make_check(
                    "system-ascend-env",
                    "ok",
                    "Ascend environment sourcing script is present.",
                    evidence=evidence,
                )
            )

    uv_available = python_env.get("tooling", {}).get("uv_available", False)
    if uv_available:
        checks.append(
            make_check(
                "python-uv",
                "ok",
                "uv is directly resolvable.",
                evidence=[str(python_env.get("tooling", {}).get("uv_path"))],
            )
        )
    else:
        checks.append(
            make_check(
                "python-uv",
                "block",
                "uv is missing from the selected execution path.",
                category_hint="env",
                severity="high",
                remediable=True,
                remediation_owner="readiness-agent",
                revalidation_scope=["tool-resolution", "python-environment"],
                evidence=["uv_path not found"],
            )
        )

    if selection_status == "selected" and probe_python_path:
        checks.append(
            make_check(
                "python-selected-python",
                "ok",
                "Selected Python is resolved and probeable.",
                evidence=[
                    f"selected_python={selected_python}",
                    f"probe_python_path={probe_python_path}",
                    f"probe_source={probe_source}",
                ],
            )
        )
    else:
        evidence = []
        if selected_env_root:
            evidence.append(f"selected_env_root={selected_env_root}")
        if selected_python:
            evidence.append(f"selected_python={selected_python}")
        if probe_source:
            evidence.append(f"probe_source={probe_source}")
        if selection_reason:
            evidence.append(f"selection_reason={selection_reason}")
        checks.append(
            make_check(
                "python-selected-python",
                "block",
                "Selected Python is unavailable or unusable for readiness checks.",
                category_hint="env",
                severity="high",
                remediable=True,
                remediation_owner="readiness-agent",
                revalidation_scope=["python-environment", "framework"],
                evidence=evidence or ["selected_python is unresolved"],
            )
        )

    if framework_path in {"mindspore", "pta"}:
        checks.append(
            make_check(
                "framework-path",
                "ok",
                f"Framework path is resolved as {framework_path}.",
                evidence=[f"framework_path={framework_path}"],
            )
        )
    elif framework_path == "mixed":
        checks.append(
            make_check(
                "framework-path",
                "warn",
                "Framework evidence is mixed and requires manual confirmation.",
                category_hint="unknown",
                severity="medium",
                evidence=["framework_path=mixed"],
            )
        )
    else:
        checks.append(
            make_check(
                "framework-path",
                "warn",
                "Framework path is not yet resolved.",
                category_hint="unknown",
                severity="medium",
                evidence=["framework_path=unknown"],
            )
        )

    required_framework = framework_layer.get("required_packages", [])
    framework_probes = framework_layer.get("import_probes", {})
    framework_probe_source = framework_layer.get("probe_source") or "current_interpreter"
    framework_probe_label = describe_probe_source(framework_probe_source)
    framework_probe_error = framework_layer.get("probe_error")
    framework_smoke = framework_layer.get("smoke_prerequisite") or {}
    missing_framework = [pkg for pkg in required_framework if not framework_probes.get(pkg, False)]
    if required_framework and not missing_framework:
        checks.append(
            make_check(
                "framework-importability",
                "ok",
                f"Required framework packages are importable in the {framework_probe_label}.",
                evidence=[f"probe_source={framework_probe_source}", *required_framework],
            )
        )
    elif missing_framework:
        evidence = [f"probe_source={framework_probe_source}", *missing_framework]
        if framework_probe_error:
            evidence.append(f"probe_error={framework_probe_error}")
        checks.append(
            make_check(
                "framework-importability",
                "block",
                f"Required framework packages are unavailable in the {framework_probe_label}: {', '.join(missing_framework)}.",
                category_hint="framework",
                severity="high",
                remediable=True,
                remediation_owner="readiness-agent",
                revalidation_scope=["framework", "task-smoke"],
                evidence=evidence,
            )
        )

    smoke_status = framework_smoke.get("status")
    smoke_details = framework_smoke.get("details") or []
    smoke_error = framework_smoke.get("error")
    if smoke_status == "passed":
        checks.append(
            make_check(
                "framework-smoke-prerequisite",
                "ok",
                f"Framework smoke prerequisite passed in the {framework_probe_label}.",
                evidence=[f"probe_source={framework_probe_source}", *smoke_details],
            )
        )
    elif smoke_status == "failed":
        evidence = [f"probe_source={framework_probe_source}", *smoke_details]
        if smoke_error:
            evidence.append(f"smoke_error={smoke_error}")
        checks.append(
            make_check(
                "framework-smoke-prerequisite",
                "block",
                f"Framework smoke prerequisite failed in the {framework_probe_label}.",
                category_hint="framework",
                severity="high",
                remediable=True,
                remediation_owner="readiness-agent",
                revalidation_scope=["framework", "task-smoke"],
                evidence=evidence,
            )
        )

    required_runtime = runtime_layer.get("required_imports", [])
    runtime_probes = runtime_layer.get("import_probes", {})
    runtime_probe_source = runtime_layer.get("probe_source") or framework_probe_source
    runtime_probe_label = describe_probe_source(runtime_probe_source)
    runtime_probe_error = runtime_layer.get("probe_error")
    missing_runtime = [
        pkg for pkg in required_runtime
        if pkg not in required_framework and not runtime_probes.get(pkg, False)
    ]
    if required_runtime and not missing_runtime:
        checks.append(
            make_check(
                "runtime-importability",
                "ok",
                f"Required runtime imports are available in the {runtime_probe_label}.",
                evidence=[f"probe_source={runtime_probe_source}", *required_runtime],
            )
        )
    elif missing_runtime:
        evidence = [f"probe_source={runtime_probe_source}", *missing_runtime]
        if runtime_probe_error:
            evidence.append(f"probe_error={runtime_probe_error}")
        checks.append(
            make_check(
                "runtime-importability",
                "block",
                f"Required runtime imports are unavailable in the {runtime_probe_label}: {', '.join(missing_runtime)}.",
                category_hint="env",
                severity="high",
                remediable=True,
                remediation_owner="readiness-agent",
                revalidation_scope=["runtime-dependencies", "framework"],
                evidence=evidence,
            )
        )

    for key in ("entry_script", "model_path", "dataset_path", "checkpoint_path", "output_path"):
        asset = workspace.get(key, {})
        if not asset:
            continue
        required = asset.get("required", False)
        exists = asset.get("exists", False)
        if required and not exists:
            if key == "entry_script":
                checks.append(
                    make_check(
                        f"workspace-{key}",
                        "block",
                        "Required entry script is missing.",
                        category_hint="workspace",
                        severity="high",
                        remediable=False,
                        remediation_owner="workspace",
                        revalidation_scope=["workspace-assets", "target"],
                        evidence=[f"{key} missing"],
                    )
                )
            elif key == "dataset_path":
                checks.append(
                    make_check(
                        f"workspace-{key}",
                        "block",
                        "Required dataset path is missing for training.",
                        category_hint="workspace",
                        severity="high",
                        remediable=False,
                        remediation_owner="workspace",
                        revalidation_scope=["workspace-assets", "task-smoke"],
                        evidence=[f"{key} missing"],
                    )
                )
            else:
                checks.append(
                    make_check(
                        f"workspace-{key}",
                        "block",
                        f"Required asset {key} is missing.",
                        category_hint="asset",
                        severity="high",
                        remediable=True,
                        remediation_owner="readiness-agent",
                        revalidation_scope=["workspace-assets", "task-smoke"],
                        evidence=[f"{key} missing"],
                    )
                )
        elif required and exists:
            checks.append(
                make_check(
                    f"workspace-{key}",
                    "ok",
                    f"Required asset {key} is present.",
                    evidence=[f"{key} exists"],
                )
            )

    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect deterministic readiness checks from target and closure")
    parser.add_argument("--target-json", required=True, help="path to execution target JSON")
    parser.add_argument("--closure-json", required=True, help="path to dependency closure JSON")
    parser.add_argument("--task-smoke-json", help="optional path to task smoke checks JSON")
    parser.add_argument("--output-json", required=True, help="path to output checks JSON")
    args = parser.parse_args()

    target = json.loads(Path(args.target_json).read_text(encoding="utf-8"))
    closure = json.loads(Path(args.closure_json).read_text(encoding="utf-8"))
    checks = collect_checks(target, closure)
    if args.task_smoke_json:
        checks.extend(json.loads(Path(args.task_smoke_json).read_text(encoding="utf-8")))
    Path(args.output_json).write_text(json.dumps(checks, indent=2), encoding="utf-8")
    print(json.dumps({"checks": len(checks)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
