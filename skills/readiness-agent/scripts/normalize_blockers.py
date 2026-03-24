#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


BLOCKER_CATEGORY_DEFAULTS = {
    "system": ("system_fatal", False, "manual-system"),
    "env": ("env_remediable", True, "readiness-agent"),
    "framework": ("framework_remediable", True, "readiness-agent"),
    "asset": ("asset_remediable", True, "readiness-agent"),
    "workspace": ("workspace_manual", False, "workspace"),
    "unknown": ("unknown", False, "unknown"),
}


def normalize_category(category_hint: str | None) -> tuple[str, bool, str]:
    if not category_hint:
        return BLOCKER_CATEGORY_DEFAULTS["unknown"]
    key = category_hint.strip().lower()
    return BLOCKER_CATEGORY_DEFAULTS.get(key, BLOCKER_CATEGORY_DEFAULTS["unknown"])


def normalize_checks(checks: list[dict]) -> dict:
    blockers_detailed: list[dict] = []
    warnings_detailed: list[dict] = []

    for idx, check in enumerate(checks, 1):
        status = (check.get("status") or "").strip().lower()
        summary = (check.get("summary") or "").strip() or f"check-{idx}"
        evidence = check.get("evidence") or []

        if status == "warn":
            warnings_detailed.append(
                {
                    "id": check.get("id") or f"warn-{idx}",
                    "summary": summary,
                    "evidence": evidence,
                    "lowers_ready_confidence": True,
                }
            )
            continue

        if status != "block":
            continue

        category, remediable, owner = normalize_category(check.get("category_hint"))
        if "remediable" in check:
            remediable = bool(check["remediable"])
        if check.get("remediation_owner"):
            owner = str(check["remediation_owner"])

        blockers_detailed.append(
            {
                "id": check.get("id") or f"block-{idx}",
                "category": category,
                "severity": check.get("severity") or ("fatal" if category == "system_fatal" else "high"),
                "summary": summary,
                "evidence": evidence,
                "remediable": remediable,
                "remediation_owner": owner,
                "revalidation_scope": check.get("revalidation_scope") or [],
            }
        )

    return {
        "blockers": [item["summary"] for item in blockers_detailed],
        "warnings": [item["summary"] for item in warnings_detailed],
        "blockers_detailed": blockers_detailed,
        "warnings_detailed": warnings_detailed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Normalize readiness checks into blockers and warnings")
    parser.add_argument("--input-json", required=True, help="path to input checks JSON")
    parser.add_argument("--output-json", required=True, help="path to output normalized blocker JSON")
    args = parser.parse_args()

    checks = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
    result = normalize_checks(checks)
    Path(args.output_json).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"blockers": len(result["blockers"]), "warnings": len(result["warnings"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
