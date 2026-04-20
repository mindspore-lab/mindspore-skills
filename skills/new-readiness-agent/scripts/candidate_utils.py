#!/usr/bin/env python3
from typing import Dict, List, Optional


COMMON_LOCAL_SUFFIXES = (".py", ".sh", ".yaml", ".yml", ".json", ".ckpt", ".pt", ".bin", ".txt")


CATALOG_FIELD_OPTIONS = {
    "target": [
        ("training", "training"),
        ("inference", "inference"),
    ],
    "launcher": [
        ("python", "python"),
        ("bash", "bash"),
        ("torchrun", "torchrun"),
        ("accelerate", "accelerate"),
        ("deepspeed", "deepspeed"),
        ("msrun", "msrun"),
        ("llamafactory-cli", "llamafactory-cli"),
        ("make", "make"),
    ],
    "framework": [
        ("mindspore", "mindspore"),
        ("pta", "pta"),
        ("mixed", "mixed"),
    ],
}


def choose_top_candidate(items: List[Dict[str, object]]) -> Optional[Dict[str, object]]:
    if not items:
        return None
    return max(items, key=lambda item: float(item.get("confidence") or 0.0))


def merge_catalog_candidates(field_name: str, detected_candidates: List[Dict[str, object]]) -> List[Dict[str, object]]:
    seen_values = {item.get("value") for item in detected_candidates}
    results = list(detected_candidates)
    for value, label in CATALOG_FIELD_OPTIONS.get(field_name, []):
        if value in seen_values:
            continue
        results.append(
            {
                "value": value,
                "label": label,
                "confidence": 0.18,
                "selection_source": "catalog",
            }
        )
    return results


def _looks_like_hf_repo_shape(value: str) -> bool:
    token = str(value or "").strip()
    if not token or token.count("/") != 1:
        return False
    owner, repo = token.split("/", 1)
    if not owner or not repo:
        return False
    if any(part.strip() != part or " " in part for part in (owner, repo)):
        return False
    return True


def looks_like_local_path(value: str) -> bool:
    token = str(value or "").strip()
    if not token:
        return False
    if token.startswith(".") or token.startswith("/") or token.startswith("\\") or token.startswith("~"):
        return True
    if "\\" in token:
        return True
    if "/" in token and not _looks_like_hf_repo_shape(token):
        return True
    if ":" in token and not token.startswith(("hf_hub:", "hf_cache:", "script_managed_remote:", "local:")):
        return True
    if token.endswith(COMMON_LOCAL_SUFFIXES):
        return True
    return False


def looks_like_hf_repo_id(value: str) -> bool:
    token = str(value or "").strip()
    if not _looks_like_hf_repo_shape(token):
        return False
    if token.startswith((".", "~")) or "\\" in token:
        return False
    if token.endswith(COMMON_LOCAL_SUFFIXES):
        return False
    return True


def ranked_candidates(items: List[Dict[str, object]]) -> List[Dict[str, object]]:
    return sorted(items, key=lambda item: float(item.get("confidence") or 0.0), reverse=True)
