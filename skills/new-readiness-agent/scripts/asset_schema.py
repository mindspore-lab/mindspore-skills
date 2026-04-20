#!/usr/bin/env python3
import hashlib
import json
from typing import Dict, Iterable, List, Optional


ASSET_SOURCE_TYPES = {
    "local_path",
    "hf_cache",
    "hf_hub",
    "script_managed_remote",
    "inline_config",
    "none",
    "unknown",
}


def _candidate_identity_locator(source_type: str, locator: Dict[str, object]) -> Dict[str, object]:
    if source_type == "script_managed_remote":
        return {
            key: locator.get(key)
            for key in ("repo_id", "entry_script", "split")
            if locator.get(key) not in {None, ""}
        }
    if source_type == "inline_config":
        return {
            key: locator.get(key)
            for key in ("entry_script", "path")
            if locator.get(key) not in {None, ""}
        }
    return locator


def _stable_locator_token(locator: Dict[str, object]) -> str:
    return json.dumps(locator, sort_keys=True, ensure_ascii=True)


def stable_asset_candidate_id(kind: str, source_type: str, locator: Dict[str, object]) -> str:
    identity_locator = _candidate_identity_locator(source_type, locator)
    payload = f"{kind}|{source_type}|{_stable_locator_token(identity_locator)}"
    return f"{kind}-{hashlib.sha1(payload.encode('utf-8')).hexdigest()[:10]}"


def make_asset_requirement(kind: str, *, required: bool, reason: str) -> Dict[str, object]:
    return {
        "kind": kind,
        "required": required,
        "reason": reason,
    }


def make_asset_candidate(
    kind: str,
    source_type: str,
    *,
    label: str,
    locator: Dict[str, object],
    confidence: float,
    selection_source: str,
    evidence: Optional[Iterable[str]] = None,
    **extra: object,
) -> Dict[str, object]:
    if source_type not in ASSET_SOURCE_TYPES:
        raise ValueError(f"unsupported asset source_type: {source_type}")
    payload: Dict[str, object] = {
        "id": stable_asset_candidate_id(kind, source_type, locator),
        "kind": kind,
        "source_type": source_type,
        "label": label,
        "locator": locator,
        "confidence": round(max(0.0, min(float(confidence), 0.99)), 2),
        "selection_source": selection_source,
        "evidence": [str(item) for item in (evidence or []) if str(item).strip()],
    }
    payload.update(extra)
    return payload


def dedupe_asset_candidates(candidates: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
    result: List[Dict[str, object]] = []
    index_by_id: Dict[str, int] = {}
    for candidate in candidates:
        candidate_id = str(candidate.get("id") or "")
        if candidate_id not in index_by_id:
            index_by_id[candidate_id] = len(result)
            result.append(dict(candidate))
            continue

        existing = result[index_by_id[candidate_id]]
        existing_evidence = [str(item) for item in existing.get("evidence") or [] if str(item).strip()]
        incoming_evidence = [str(item) for item in candidate.get("evidence") or [] if str(item).strip()]
        merged_evidence: List[str] = []
        for item in existing_evidence + incoming_evidence:
            if item in merged_evidence:
                continue
            merged_evidence.append(item)

        existing_confidence = float(existing.get("confidence") or 0.0)
        incoming_confidence = float(candidate.get("confidence") or 0.0)
        base = dict(candidate) if incoming_confidence > existing_confidence else dict(existing)
        base["evidence"] = merged_evidence
        if bool(existing.get("exists")) or bool(candidate.get("exists")):
            base["exists"] = True
        result[index_by_id[candidate_id]] = base
    return result


def rank_asset_candidates(candidates: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
    return sorted(
        dedupe_asset_candidates(candidates),
        key=lambda item: (
            float(item.get("confidence") or 0.0),
            str(item.get("label") or ""),
        ),
        reverse=True,
    )


def make_selected_asset(
    kind: str,
    requirement: Dict[str, object],
    *,
    candidate: Optional[Dict[str, object]] = None,
    source_type: Optional[str] = None,
    locator: Optional[Dict[str, object]] = None,
    selection_source: Optional[str] = None,
    evidence: Optional[Iterable[str]] = None,
) -> Dict[str, object]:
    if candidate is not None:
        return {
            "kind": kind,
            "required": bool(requirement.get("required")),
            "selected_candidate_id": candidate.get("id"),
            "source_type": candidate.get("source_type"),
            "label": candidate.get("label"),
            "locator": candidate.get("locator") or {},
            "selection_source": candidate.get("selection_source"),
            "confidence": candidate.get("confidence"),
            "evidence": list(candidate.get("evidence") or []),
        }
    resolved_source = source_type or "unknown"
    return {
        "kind": kind,
        "required": bool(requirement.get("required")),
        "selected_candidate_id": None,
        "source_type": resolved_source,
        "label": resolved_source.replace("_", " "),
        "locator": locator or {},
        "selection_source": selection_source or "manual_confirmation",
        "confidence": 0.0,
        "evidence": [str(item) for item in (evidence or []) if str(item).strip()],
    }


def asset_locator_summary(asset: Dict[str, object]) -> str:
    locator = asset.get("locator") if isinstance(asset.get("locator"), dict) else {}
    source_type = str(asset.get("source_type") or "")
    if source_type == "local_path":
        return str(locator.get("path") or "")
    if source_type == "hf_cache":
        repo_id = str(locator.get("repo_id") or "")
        cache_path = str(locator.get("cache_path") or "")
        return f"{repo_id} -> {cache_path}".strip(" ->")
    if source_type == "hf_hub":
        repo_id = str(locator.get("repo_id") or "")
        split = str(locator.get("split") or "")
        return f"{repo_id}{':' + split if split else ''}"
    if source_type == "script_managed_remote":
        repo_id = str(locator.get("repo_id") or "")
        entry_script = str(locator.get("entry_script") or "")
        return f"{repo_id} via {entry_script}".strip()
    if source_type == "inline_config":
        return str(locator.get("entry_script") or "inline config")
    return ""
