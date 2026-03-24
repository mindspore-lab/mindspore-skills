#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def build_action(
    action_id: str,
    blocker: dict,
    action_type: str,
    summary: str,
    requires_confirmation: bool,
    allowed: bool,
    reason: str,
    revalidation_scope: list[str],
    **extra: object,
) -> dict:
    payload = {
        "id": action_id,
        "blocker_id": blocker.get("id"),
        "category": blocker.get("category"),
        "action_type": action_type,
        "summary": summary,
        "requires_confirmation": requires_confirmation,
        "allowed": allowed,
        "reason": reason,
        "revalidation_scope": revalidation_scope,
    }
    payload.update(extra)
    return payload


def infer_framework_action(summary: str, closure: dict) -> tuple[str, str]:
    text = summary.lower()
    framework_path = closure.get("layers", {}).get("framework", {}).get("framework_path", "unknown")
    if "mindspore" in text or framework_path == "mindspore":
        return "repair_mindspore_framework", "MindSpore framework path requires repair inside the selected environment."
    if "torch_npu" in text or "torch" in text or framework_path == "pta":
        return "repair_pta_framework", "PTA framework path requires repair inside the selected environment."
    return "repair_framework", "Framework path requires repair inside the selected environment."


def package_hints(blocker: dict, closure: dict, *, framework_fallback: bool = False) -> list[str]:
    evidence = blocker.get("evidence") or []
    packages = [item for item in evidence if isinstance(item, str) and item]
    if packages:
        return packages
    if framework_fallback:
        return closure.get("layers", {}).get("framework", {}).get("required_packages", [])
    return []


def plan_actions(blockers: list[dict], closure: dict, allow_network: bool, fix_scope: str) -> dict:
    actions: list[dict] = []
    skipped: list[dict] = []

    for index, blocker in enumerate(blockers, 1):
        category = blocker.get("category")
        summary = (blocker.get("summary") or "").strip()
        remediable = bool(blocker.get("remediable"))
        if not remediable or fix_scope == "none":
            skipped.append(
                {
                    "blocker_id": blocker.get("id"),
                    "reason": "blocker is outside active fix scope or not marked remediable",
                }
            )
            continue

        revalidation_scope = blocker.get("revalidation_scope") or []
        lower = summary.lower()

        if category == "env_remediable":
            if "uv" in lower and ("missing" in lower or "not" in lower):
                actions.append(
                    build_action(
                        f"action-{index}",
                        blocker,
                        "install_uv",
                        "Install uv and verify direct shell resolution.",
                        True,
                        True,
                        "uv is missing from the selected execution path.",
                        revalidation_scope or ["tool-resolution", "python-environment"],
                    )
                )
            elif "path" in lower and "uv" in lower:
                actions.append(
                    build_action(
                        f"action-{index}",
                        blocker,
                        "repair_uv_path",
                        "Repair PATH so uv is directly resolvable.",
                        True,
                        True,
                        "uv exists but direct shell resolution is incomplete.",
                        revalidation_scope or ["tool-resolution"],
                    )
                )
            elif "environment" in lower or "env" in lower:
                actions.append(
                    build_action(
                        f"action-{index}",
                        blocker,
                        "create_or_select_env",
                        "Create or confirm the selected Python environment.",
                        True,
                        True,
                        "The selected execution target requires a usable Python environment.",
                        revalidation_scope or ["python-environment", "framework"],
                    )
                )
            else:
                packages = package_hints(blocker, closure)
                actions.append(
                    build_action(
                        f"action-{index}",
                        blocker,
                        "install_runtime_dependency",
                        "Install the missing runtime dependency in the selected environment.",
                        False,
                        True,
                        "The blocker is inside the Python environment layer and has explicit dependency scope.",
                        revalidation_scope or ["runtime-dependencies", "framework"],
                        package_name=packages[0] if len(packages) == 1 else None,
                        package_names=packages,
                    )
                )
            continue

        if category == "framework_remediable":
            action_type, reason = infer_framework_action(summary, closure)
            packages = package_hints(blocker, closure, framework_fallback=True)
            actions.append(
                build_action(
                    f"action-{index}",
                    blocker,
                    action_type,
                    "Repair the framework package set inside the selected environment.",
                    True,
                    True,
                    reason,
                    revalidation_scope or ["framework", "runtime-dependencies", "task-smoke"],
                    package_name=packages[0] if len(packages) == 1 else None,
                    package_names=packages,
                )
            )
            continue

        if category == "asset_remediable":
            if "model" in lower or "tokenizer" in lower:
                actions.append(
                    build_action(
                        f"action-{index}",
                        blocker,
                        "download_or_resolve_model_asset",
                        "Acquire the required model asset for the selected target.",
                        True,
                        allow_network,
                        "Model assets can be repaired only when network use is allowed and explicitly confirmed.",
                        revalidation_scope or ["workspace-assets", "task-smoke"],
                    )
                )
            else:
                skipped.append(
                    {
                        "blocker_id": blocker.get("id"),
                        "reason": "asset blocker is not yet covered by native env-fix planner",
                    }
                )
            continue

        skipped.append(
            {
                "blocker_id": blocker.get("id"),
                "reason": f"category {category!r} is not handled by native env-fix planner",
            }
        )

    return {
        "actions": actions,
        "skipped": skipped,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Plan native env-fix actions for readiness-agent")
    parser.add_argument("--blockers-json", required=True, help="path to normalized blockers JSON")
    parser.add_argument("--closure-json", required=True, help="path to dependency closure JSON")
    parser.add_argument("--output-json", required=True, help="path to output remediation plan JSON")
    parser.add_argument("--allow-network", action="store_true", help="allow network-dependent remediation planning")
    parser.add_argument("--fix-scope", default="safe-user-space", help="active fix scope")
    args = parser.parse_args()

    blockers = json.loads(Path(args.blockers_json).read_text(encoding="utf-8")).get("blockers_detailed", [])
    closure = json.loads(Path(args.closure_json).read_text(encoding="utf-8"))
    result = plan_actions(blockers, closure, args.allow_network, args.fix_scope)
    Path(args.output_json).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"actions": len(result["actions"]), "skipped": len(result["skipped"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
