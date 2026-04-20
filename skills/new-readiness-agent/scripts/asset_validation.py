#!/usr/bin/env python3
from pathlib import Path
from typing import Dict, List, Optional

from environment_selection import resolve_optional_path


def make_asset_check(check_id: str, status: str, summary: str, evidence: Optional[List[str]] = None, **extra: object) -> Dict[str, object]:
    payload: Dict[str, object] = {
        "id": check_id,
        "status": status,
        "summary": summary,
        "evidence": evidence or [],
    }
    payload.update(extra)
    return payload


def _find_related_cache_candidate(asset_bundle: Dict[str, object], repo_id: str) -> Optional[Dict[str, object]]:
    for candidate in asset_bundle.get("candidates") or []:
        if candidate.get("source_type") != "hf_cache":
            continue
        locator = candidate.get("locator") if isinstance(candidate.get("locator"), dict) else {}
        if locator.get("repo_id") == repo_id:
            return candidate
    return None


def validate_asset_selection(
    kind: str,
    asset_bundle: Dict[str, object],
    root: Path,
    *,
    launcher: Optional[str] = None,
) -> Dict[str, object]:
    requirement = asset_bundle.get("requirement") if isinstance(asset_bundle.get("requirement"), dict) else {}
    required = bool(requirement.get("required"))
    selected = asset_bundle.get("selected") if isinstance(asset_bundle.get("selected"), dict) else {}
    source_type = str(selected.get("source_type") or "")
    locator = selected.get("locator") if isinstance(selected.get("locator"), dict) else {}
    check_id = f"workspace-{kind}-asset"

    if kind == "config" and launcher == "llamafactory-cli":
        required = True

    if source_type in {"", "unknown"}:
        if required:
            return make_asset_check(check_id, "block", f"{kind} asset is required but unresolved.")
        return make_asset_check(check_id, "skipped", f"{kind} asset is optional and unresolved.")

    if source_type == "none":
        if required:
            return make_asset_check(check_id, "block", f"{kind} asset is required but was explicitly marked as not provided.")
        return make_asset_check(check_id, "skipped", f"{kind} asset is not needed for this workflow.")

    if source_type == "inline_config":
        return make_asset_check(
            check_id,
            "ok",
            "config is satisfied by inline script configuration.",
            evidence=[str(locator.get("entry_script") or ""), str(locator.get("callsite") or "")],
        )

    if source_type == "local_path":
        raw_path = str(locator.get("path") or "")
        resolved = resolve_optional_path(raw_path, root)
        if resolved and resolved.exists():
            return make_asset_check(check_id, "ok", f"{kind} local path exists.", evidence=[str(resolved)])
        if required:
            return make_asset_check(check_id, "block", f"{kind} local path does not exist.", evidence=[str(resolved or raw_path)])
        return make_asset_check(check_id, "warn", f"{kind} local path does not exist locally.", evidence=[str(resolved or raw_path)])

    if source_type == "hf_cache":
        cache_path = Path(str(locator.get("cache_path") or "")).expanduser()
        if cache_path.exists():
            return make_asset_check(
                check_id,
                "ok",
                f"{kind} is satisfied by an existing Hugging Face cache entry.",
                evidence=[str(locator.get("repo_id") or ""), str(cache_path)],
            )
        if required:
            return make_asset_check(check_id, "block", f"{kind} Hugging Face cache path does not exist.", evidence=[str(cache_path)])
        return make_asset_check(check_id, "warn", f"{kind} Hugging Face cache path was selected but is missing.", evidence=[str(cache_path)])

    if source_type in {"hf_hub", "script_managed_remote"}:
        repo_id = str(locator.get("repo_id") or "")
        if not repo_id:
            if required:
                return make_asset_check(check_id, "block", f"{kind} remote asset is missing a repo identifier.")
            return make_asset_check(check_id, "warn", f"{kind} remote asset is missing a repo identifier.")

        cache_candidate = _find_related_cache_candidate(asset_bundle, repo_id)
        if cache_candidate:
            cache_locator = cache_candidate.get("locator") if isinstance(cache_candidate.get("locator"), dict) else {}
            cache_path = Path(str(cache_locator.get("cache_path") or "")).expanduser()
            if cache_path.exists():
                return make_asset_check(
                    check_id,
                    "ok",
                    f"{kind} remote asset is backed by a local Hugging Face cache.",
                    evidence=[repo_id, str(cache_path)],
                )

        summary = f"{kind} relies on a remote Hugging Face asset at runtime: {repo_id}"
        if source_type == "script_managed_remote":
            summary = f"{kind} is managed by the entry script and may download or reuse cache at runtime: {repo_id}"
        return make_asset_check(
            check_id,
            "warn",
            summary,
            evidence=[repo_id, str(locator.get("entry_script") or ""), str(locator.get("callsite") or "")],
        )

    if required:
        return make_asset_check(check_id, "block", f"{kind} asset uses an unsupported source type: {source_type}")
    return make_asset_check(check_id, "warn", f"{kind} asset uses an unsupported source type: {source_type}")
