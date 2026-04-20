#!/usr/bin/env python3
import json
from pathlib import Path
from typing import Dict, List, Optional

from asset_registry import (
    ASSET_KINDS,
    asset_confirmation_sequence,
    asset_field_name,
    asset_kinds_with_validation_gate,
    asset_portable_headers,
    asset_should_confirm,
)
from asset_schema import make_selected_asset, rank_asset_candidates
from candidate_utils import (
    choose_top_candidate,
    looks_like_hf_repo_id,
    looks_like_local_path,
    merge_catalog_candidates,
    ranked_candidates,
)
from environment_selection import resolve_optional_path


BASE_VALIDATION_GATE_FIELDS = (
    "target",
    "launcher",
    "framework",
    "runtime_environment",
    "entry_script",
    "cann_path",
)
VALIDATION_GATE_FIELDS = BASE_VALIDATION_GATE_FIELDS[:-1] + tuple(asset_field_name(kind) for kind in asset_kinds_with_validation_gate()) + ("cann_path",)

PORTABLE_QUESTION_MAX_OPTIONS = 4
SKIP_FOR_NOW_LABEL = "Skip for now"
PORTABLE_HEADER_BY_FIELD = {
    "target": "Target",
    "launcher": "Launcher",
    "framework": "Framework",
    "runtime_environment": "Runtime Env",
    "entry_script": "Entry Script",
    **asset_portable_headers(),
    "cann_path": "CANN Path",
}

BASE_CONFIRMATION_SEQUENCE = (
    {
        "field": "target",
        "label": "Target",
        "candidate_key": "target_candidates",
        "catalog_key": "target",
        "allow_free_text": False,
        "prompt": "Confirm the intended workflow target before continuing the readiness scan.",
    },
    {
        "field": "launcher",
        "label": "Launcher",
        "candidate_key": "launcher_candidates",
        "catalog_key": "launcher",
        "allow_free_text": False,
        "prompt": "Confirm which launcher this workspace should use.",
    },
    {
        "field": "framework",
        "label": "Framework",
        "candidate_key": "framework_candidates",
        "catalog_key": "framework",
        "allow_free_text": False,
        "prompt": "Confirm the framework stack that should run on this workspace.",
    },
    {
        "field": "runtime_environment",
        "label": "Python / Environment",
        "candidate_key": "environment",
        "allow_free_text": True,
        "manual_hint": "If none of the detected environments fit, provide the intended selected_python and selected_env_root on the next run.",
        "prompt": "Confirm the runtime Python environment that should be used for readiness checks.",
    },
    {
        "field": "entry_script",
        "label": "Entry Script",
        "candidate_key": "entry_candidates",
        "allow_free_text": True,
        "manual_hint": "Provide the local training or inference script path if it is not listed.",
        "prompt": "Confirm the entry script path for the workload.",
    },
)
CONFIRMATION_SEQUENCE = BASE_CONFIRMATION_SEQUENCE + asset_confirmation_sequence() + (
    {
        "field": "cann_path",
        "label": "CANN / set_env.sh",
        "candidate_key": "cann_candidates",
        "allow_free_text": True,
        "manual_hint": "Provide a CANN root or set_env.sh path if you already know it.",
        "prompt": "Confirm the CANN or Ascend environment path for this workspace.",
    },
)


def parse_confirmation_overrides(raw_items: Optional[List[str]]) -> Dict[str, str]:
    overrides: Dict[str, str] = {}
    if not raw_items:
        return overrides
    for item in raw_items:
        if "=" not in str(item):
            continue
        field_name, raw_value = str(item).split("=", 1)
        field_name = field_name.strip()
        if field_name:
            overrides[field_name] = raw_value.strip()
    return overrides


def build_numbered_options(
    field_candidates: List[Dict[str, object]],
    *,
    allow_free_text: bool,
    include_unknown: bool = True,
) -> List[Dict[str, object]]:
    options: List[Dict[str, object]] = []
    for item in field_candidates:
        options.append(
            {
                "value": item.get("value"),
                "label": item.get("label"),
                "confidence": item.get("confidence"),
                "selection_source": item.get("selection_source"),
            }
        )
    if allow_free_text:
        options.append(
            {
                "value": "__manual__",
                "label": "enter a custom value manually",
                "confidence": 0.0,
                "selection_source": "manual",
            }
        )
    if include_unknown:
        options.append(
            {
                "value": "__unknown__",
                "label": SKIP_FOR_NOW_LABEL,
                "confidence": 0.0,
                "selection_source": "manual",
            }
        )
    for index, option in enumerate(options, start=1):
        option["index"] = index
    return options


def is_manual_option(option: Dict[str, object]) -> bool:
    return str(option.get("value") or "") == "__manual__"


def is_unknown_option(option: Dict[str, object]) -> bool:
    return str(option.get("value") or "") == "__unknown__"


def portable_option_label(option: Dict[str, object], *, recommended: bool) -> str:
    if is_unknown_option(option):
        return SKIP_FOR_NOW_LABEL
    label = str(option.get("label") or "").strip()
    if recommended and label and not label.endswith("(Recommended)"):
        return f"{label} (Recommended)"
    return label


def portable_option_detail(option: Dict[str, object]) -> str:
    locator = option.get("locator") if isinstance(option.get("locator"), dict) else {}
    if locator.get("path"):
        return f"Path: {locator.get('path')}."
    if locator.get("cache_path"):
        return f"Cache path: {locator.get('cache_path')}."
    if locator.get("repo_id"):
        return f"Repo ID: {locator.get('repo_id')}."
    if option.get("python_path"):
        return f"Python: {option.get('python_path')}."
    if option.get("env_root"):
        return f"Environment root: {option.get('env_root')}."
    source_type = str(option.get("source_type") or "").strip()
    if source_type:
        return f"Source type: {source_type}."
    return ""


def _script_analysis_base_description(kind: str, source_type: str, locator: Dict[str, object]) -> Optional[str]:
    repo_id = str(locator.get("repo_id") or "").strip()
    if source_type == "hf_hub":
        if kind == "dataset":
            return "Detected from the entry script. Runtime will load or download this dataset from the HF Hub repo used by the script."
        if kind == "tokenizer":
            return "Detected from the entry script. Runtime will load or download this tokenizer from the HF Hub repo used by the script."
        return "Detected from the entry script. Runtime will load or download this asset from the HF Hub repo used by the script." if repo_id else "Detected from the entry script."

    if source_type == "hf_cache":
        if kind == "dataset":
            return "Detected from the entry script. A matching local HF cache already exists for the dataset repo the script uses."
        if kind == "tokenizer":
            return "Detected from the entry script. A matching local HF cache already exists for the tokenizer repo the script uses."
        return "Detected from the entry script. A matching local HF cache already exists for the repo the script uses." if repo_id else "Detected from the entry script."

    if source_type == "local_path":
        if kind == "config":
            return "Detected from the entry script. Runtime uses this local path as the config asset."
        if kind == "checkpoint":
            return "Detected from the entry script. Runtime references this local path for checkpoint resume or load behavior."
        return "Detected from the entry script and resolved as a local workspace path."

    if source_type == "inline_config":
        return "Detected from the entry script. Runtime configuration is defined inline in the script."

    return None


def portable_option_description(option: Dict[str, object], *, recommended: bool) -> str:
    if is_unknown_option(option):
        return "Skip confirming this field for now and continue with a lower-confidence readiness result."
    if is_manual_option(option):
        return "Use the host's manual-input path to provide a custom value for this field."
    kind = str(option.get("kind") or "").strip()
    selection_source = str(option.get("selection_source") or "").strip()
    source_type = str(option.get("source_type") or "").strip()
    locator = option.get("locator") if isinstance(option.get("locator"), dict) else {}
    detail = portable_option_detail(option)

    if selection_source == "script_analysis":
        base = _script_analysis_base_description(kind, source_type, locator)
        if base:
            return f"{base} {detail}".strip()

    if recommended:
        base = "Recommended based on current workspace evidence."
    elif selection_source in {"catalog", "catalog_default"}:
        base = "Available canonical choice from the built-in readiness catalog."
    elif selection_source in {"cached_confirmation", "explicit_input"}:
        base = "Previously selected or explicitly supplied runtime value."
    else:
        base = "Detected candidate from the current workspace evidence."
    if detail:
        return f"{base} {detail}"
    return base


def uniquify_portable_option_labels(options: List[Dict[str, object]]) -> List[Dict[str, object]]:
    seen: Dict[str, int] = {}
    for option in options:
        label = str(option.get("label") or "").strip()
        count = seen.get(label, 0)
        if count > 0:
            option["label"] = f"{label} [{count + 1}]"
        seen[label] = count + 1
    return options


def portable_short_header(field_name: str, label: object) -> str:
    header = PORTABLE_HEADER_BY_FIELD.get(field_name)
    if header:
        return header
    return str(label or field_name).strip()[:12] or "Confirm"


def build_portable_question(
    *,
    field_name: str,
    label: object,
    prompt: object,
    options: List[Dict[str, object]],
    recommended_value: object,
    allow_free_text: bool,
    manual_hint: object,
) -> Dict[str, object]:
    recommended_text = str(recommended_value).strip() if recommended_value is not None else ""
    recommended_token = recommended_text or None
    manual_option = next((option for option in options if is_manual_option(option)), None)
    unknown_option = next((option for option in options if is_unknown_option(option)), None)
    candidate_options = [option for option in options if not is_manual_option(option) and not is_unknown_option(option)]

    max_candidate_options = PORTABLE_QUESTION_MAX_OPTIONS - (1 if unknown_option else 0)
    shortlist = list(candidate_options[:max_candidate_options])
    if recommended_token:
        recommended_option = next((option for option in candidate_options if str(option.get("value")) == recommended_token), None)
        if recommended_option is not None:
            shortlist = [recommended_option] + [option for option in shortlist if option is not recommended_option]
            shortlist = shortlist[:max_candidate_options]

    portable_options: List[Dict[str, object]] = []
    for option in shortlist:
        recommended = recommended_token is not None and str(option.get("value")) == recommended_token
        portable_options.append(
            {
                "value": option.get("value"),
                "label": portable_option_label(option, recommended=recommended),
                "description": portable_option_description(option, recommended=recommended),
                "recommended": recommended,
                "source_option_index": option.get("index"),
            }
        )
    if unknown_option is not None:
        portable_options.append(
            {
                "value": unknown_option.get("value"),
                "label": portable_option_label(unknown_option, recommended=False),
                "description": portable_option_description(unknown_option, recommended=False),
                "recommended": False,
                "source_option_index": unknown_option.get("index"),
            }
        )
    if allow_free_text and manual_option is not None and len(portable_options) < 2:
        portable_options.insert(
            0,
            {
                "value": manual_option.get("value"),
                "label": "Use manual input",
                "description": str(manual_hint or portable_option_description(manual_option, recommended=False)).strip(),
                "recommended": False,
                "source_option_index": manual_option.get("index"),
            },
        )

    portable_options = uniquify_portable_option_labels(portable_options)
    selection_strategy = "full_projection" if len(options) <= PORTABLE_QUESTION_MAX_OPTIONS else "recommended_first_shortlist"
    if allow_free_text and manual_option is not None and any(is_manual_option(option) for option in portable_options):
        selection_strategy = "manual_fallback_projection"
    return {
        "header": portable_short_header(field_name, label),
        "question": str(prompt or "").strip(),
        "multi_select": False,
        "options": portable_options,
        "selection_strategy": selection_strategy,
        "full_option_count": len(options),
        "response_binding": {
            "field": field_name,
            "cli_flag": "--confirm",
            "format": f"{field_name}=<value>",
        },
    }


def load_cached_confirmation(root: Path) -> Dict[str, object]:
    confirmation_path = root / "readiness-output" / "latest" / "new-readiness-agent" / "confirmation-latest.json"
    if not confirmation_path.exists():
        return {}
    try:
        payload = json.loads(confirmation_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def asset_bundle(scan: Dict[str, object], kind: str) -> Dict[str, object]:
    catalog = scan.get("asset_catalog") if isinstance(scan.get("asset_catalog"), dict) else {}
    assets = catalog.get("assets") if isinstance(catalog.get("assets"), dict) else {}
    bundle = assets.get(kind)
    if isinstance(bundle, dict):
        return bundle
    return {"requirement": {"kind": kind, "required": False, "reason": ""}, "candidates": []}


def find_asset_candidate(asset_candidates: List[Dict[str, object]], raw_value: str) -> Optional[Dict[str, object]]:
    token = str(raw_value or "").strip()
    if not token:
        return None
    for candidate_item in asset_candidates:
        if token == candidate_item.get("id"):
            return candidate_item
        locator = candidate_item.get("locator") if isinstance(candidate_item.get("locator"), dict) else {}
        if token in {
            str(locator.get("path") or ""),
            str(locator.get("cache_path") or ""),
            str(locator.get("repo_id") or ""),
        }:
            return candidate_item
    return None


def build_asset_confirmation_options(bundle: Dict[str, object], *, allow_free_text: bool) -> List[Dict[str, object]]:
    options: List[Dict[str, object]] = []
    for candidate_item in rank_asset_candidates(bundle.get("candidates") or []):
        options.append(
            {
                "value": candidate_item.get("id"),
                "kind": candidate_item.get("kind"),
                "label": candidate_item.get("label"),
                "confidence": candidate_item.get("confidence"),
                "selection_source": candidate_item.get("selection_source"),
                "source_type": candidate_item.get("source_type"),
                "locator": candidate_item.get("locator"),
            }
        )
    if allow_free_text:
        options.append(
            {
                "value": "__manual__",
                "label": "enter a custom value manually",
                "confidence": 0.0,
                "selection_source": "manual",
            }
        )
    options.append(
        {
            "value": "__unknown__",
            "label": SKIP_FOR_NOW_LABEL,
            "confidence": 0.0,
            "selection_source": "manual",
        }
    )
    for index, option in enumerate(options, start=1):
        option["index"] = index
    return options


def infer_manual_asset(kind: str, requirement: Dict[str, object], raw_value: str) -> Dict[str, object]:
    value = str(raw_value or "").strip()
    lowered = value.lower()
    if lowered in {"none", "__none__"}:
        return make_selected_asset(kind, requirement, source_type="none", locator={}, selection_source="manual_confirmation")
    if lowered in {"inline_config", "inline"}:
        return make_selected_asset(kind, requirement, source_type="inline_config", locator={}, selection_source="manual_confirmation")
    if value.startswith("local:"):
        return make_selected_asset(kind, requirement, source_type="local_path", locator={"path": value.split(":", 1)[1].strip()}, selection_source="manual_confirmation")
    if value.startswith("hf_cache:"):
        return make_selected_asset(kind, requirement, source_type="hf_cache", locator={"cache_path": value.split(":", 1)[1].strip()}, selection_source="manual_confirmation")
    if value.startswith("hf_hub:"):
        locator: Dict[str, object] = {"repo_id": value.split(":", 1)[1].strip()}
        if kind == "dataset":
            locator["split"] = "train"
        return make_selected_asset(kind, requirement, source_type="hf_hub", locator=locator, selection_source="manual_confirmation")
    if value.startswith("script_managed_remote:"):
        locator = {"repo_id": value.split(":", 1)[1].strip()}
        if kind == "dataset":
            locator["split"] = "train"
        return make_selected_asset(kind, requirement, source_type="script_managed_remote", locator=locator, selection_source="manual_confirmation")
    if looks_like_local_path(value):
        return make_selected_asset(kind, requirement, source_type="local_path", locator={"path": value}, selection_source="manual_confirmation")
    if kind in {"model", "tokenizer", "dataset"} and looks_like_hf_repo_id(value):
        locator = {"repo_id": value}
        if kind == "dataset":
            locator["split"] = "train"
        return make_selected_asset(kind, requirement, source_type="hf_hub", locator=locator, selection_source="manual_confirmation")
    return make_selected_asset(kind, requirement, source_type="unknown", locator={"raw": value}, selection_source="manual_confirmation")


def choose_asset(
    kind: str,
    cached_confirmation: Dict[str, object],
    bundle: Dict[str, object],
    confirmation_override: Optional[str] = None,
) -> Dict[str, object]:
    field_name = f"{kind}_asset"
    requirement = bundle.get("requirement") if isinstance(bundle.get("requirement"), dict) else {"kind": kind, "required": False, "reason": ""}
    asset_candidates = list(bundle.get("candidates") or [])
    cached_fields = cached_confirmation.get("confirmed_fields") if isinstance(cached_confirmation.get("confirmed_fields"), dict) else {}
    cached_item = cached_fields.get(field_name) if isinstance(cached_fields.get(field_name), dict) else None

    if confirmation_override is not None:
        if confirmation_override == "__unknown__":
            return {
                "value": None,
                "source": "manual_confirmation",
                "confirmed": True,
                "asset": make_selected_asset(kind, requirement, source_type="unknown", locator={}, selection_source="manual_confirmation"),
            }
        candidate_item = find_asset_candidate(asset_candidates, confirmation_override)
        if candidate_item:
            return {
                "value": candidate_item.get("id"),
                "source": "manual_confirmation",
                "confirmed": True,
                "asset": make_selected_asset(kind, requirement, candidate=candidate_item),
            }
        return {
            "value": confirmation_override,
            "source": "manual_confirmation",
            "confirmed": True,
            "asset": infer_manual_asset(kind, requirement, confirmation_override),
        }

    explicit_candidate_id = bundle.get("explicit_candidate_id")
    if explicit_candidate_id:
        explicit_candidate = find_asset_candidate(asset_candidates, str(explicit_candidate_id))
        if explicit_candidate:
            return {
                "value": explicit_candidate.get("id"),
                "source": "explicit_input",
                "confirmed": True,
                "asset": make_selected_asset(kind, requirement, candidate=explicit_candidate),
            }

    if isinstance(cached_item, dict):
        cached_asset = cached_item.get("asset") if isinstance(cached_item.get("asset"), dict) else None
        cached_value = cached_item.get("value")
        cached_confirmed = bool(cached_item.get("confirmed", False))
        if cached_asset:
            candidate_item = find_asset_candidate(asset_candidates, str(cached_value or "")) if cached_value else None
            if candidate_item:
                return {
                    "value": candidate_item.get("id"),
                    "source": "cached_confirmation",
                    "confirmed": cached_confirmed,
                    "asset": make_selected_asset(kind, requirement, candidate=candidate_item),
                }
            return {
                "value": cached_value,
                "source": "cached_confirmation",
                "confirmed": cached_confirmed,
                "asset": cached_asset,
            }

    top_candidate = asset_candidates[0] if asset_candidates else None
    if top_candidate:
        return {
            "value": top_candidate.get("id"),
            "source": "auto_recommended",
            "confirmed": False,
            "asset": make_selected_asset(kind, requirement, candidate=top_candidate),
        }

    return {
        "value": None,
        "source": "missing",
        "confirmed": False,
        "asset": make_selected_asset(kind, requirement, source_type="unknown", locator={}, selection_source="missing"),
    }


def choose_value(
    field_name: str,
    explicit_value: Optional[str],
    cached_confirmation: Dict[str, object],
    field_candidates: List[Dict[str, object]],
    confirmation_override: Optional[str] = None,
) -> Dict[str, object]:
    cached_fields = cached_confirmation.get("confirmed_fields") if isinstance(cached_confirmation.get("confirmed_fields"), dict) else {}
    cached_item = cached_fields.get(field_name) if isinstance(cached_fields.get(field_name), dict) else None

    if confirmation_override is not None:
        if confirmation_override == "__unknown__":
            return {"value": None, "source": "manual_confirmation", "confirmed": True}
        return {"value": confirmation_override, "source": "manual_confirmation", "confirmed": True}
    if explicit_value not in {None, ""}:
        return {"value": explicit_value, "source": "explicit_input", "confirmed": True}
    if isinstance(cached_item, dict) and bool(cached_item.get("confirmed", False)):
        return {"value": cached_item.get("value"), "source": "cached_confirmation", "confirmed": True}
    if isinstance(cached_item, dict) and cached_item.get("value") not in {None, ""}:
        return {
            "value": cached_item.get("value"),
            "source": "cached_confirmation",
            "confirmed": bool(cached_item.get("confirmed", False)),
        }
    top = choose_top_candidate(list(field_candidates))
    if not top:
        return {"value": None, "source": "missing", "confirmed": False}
    return {"value": top.get("value"), "source": "auto_recommended", "confirmed": False}


def choose_environment(
    root: Path,
    args: object,
    cached_confirmation: Dict[str, object],
    environment_result: Dict[str, object],
    confirmation_override: Optional[str] = None,
) -> Dict[str, object]:
    candidates = environment_result.get("candidates") or []
    explicit_python = getattr(args, "selected_python", None)
    explicit_env_root = getattr(args, "selected_env_root", None)
    cached_fields = cached_confirmation.get("confirmed_fields") if isinstance(cached_confirmation.get("confirmed_fields"), dict) else {}

    def with_runtime_env_fields(
        candidate_item: Dict[str, object],
        *,
        python_value: Optional[str] = None,
        env_value: Optional[str] = None,
    ) -> Dict[str, object]:
        normalized = dict(candidate_item)
        if python_value and not normalized.get("python_path"):
            normalized["python_path"] = python_value
        if env_value and not normalized.get("env_root"):
            normalized["env_root"] = env_value
        return normalized

    if confirmation_override is not None:
        if confirmation_override == "__unknown__":
            return {"candidate": None, "source": "manual_confirmation", "confirmed": True}
        for candidate_item in candidates:
            if confirmation_override in {candidate_item.get("id"), candidate_item.get("python_path"), candidate_item.get("env_root")}:
                return {"candidate": candidate_item, "source": "manual_confirmation", "confirmed": True}

    if explicit_python or explicit_env_root:
        explicit_python_value = str(resolve_optional_path(explicit_python, root) or explicit_python) if explicit_python else None
        explicit_env_value = str(resolve_optional_path(explicit_env_root, root) or explicit_env_root) if explicit_env_root else None
        for candidate_item in candidates:
            if explicit_python_value and candidate_item.get("python_path") == explicit_python_value:
                return {
                    "candidate": with_runtime_env_fields(candidate_item, python_value=explicit_python_value, env_value=explicit_env_value),
                    "source": "explicit_input",
                    "confirmed": True,
                }
            if explicit_env_value and candidate_item.get("env_root") == explicit_env_value:
                return {
                    "candidate": with_runtime_env_fields(candidate_item, python_value=explicit_python_value, env_value=explicit_env_value),
                    "source": "explicit_input",
                    "confirmed": True,
                }

    cached_env = cached_fields.get("selected_env_root") if isinstance(cached_fields, dict) else None
    cached_python = cached_fields.get("selected_python") if isinstance(cached_fields, dict) else None
    cached_env_value = cached_env.get("value") if isinstance(cached_env, dict) else None
    cached_python_value = cached_python.get("value") if isinstance(cached_python, dict) else None
    cached_confirmed = bool(
        (isinstance(cached_env, dict) and cached_env.get("confirmed", False))
        and (isinstance(cached_python, dict) and cached_python.get("confirmed", False))
    )
    if cached_confirmed and cached_env_value in {None, ""} and cached_python_value in {None, ""}:
        return {"candidate": None, "source": "cached_confirmation", "confirmed": True}
    if cached_env_value or cached_python_value:
        for candidate_item in candidates:
            if cached_env_value and candidate_item.get("env_root") == cached_env_value:
                return {
                    "candidate": with_runtime_env_fields(candidate_item, python_value=cached_python_value, env_value=cached_env_value),
                    "source": "cached_confirmation",
                    "confirmed": cached_confirmed,
                }
            if cached_python_value and candidate_item.get("python_path") == cached_python_value:
                return {
                    "candidate": with_runtime_env_fields(candidate_item, python_value=cached_python_value, env_value=cached_env_value),
                    "source": "cached_confirmation",
                    "confirmed": cached_confirmed,
                }
        synthetic_candidate = {
            "id": "env-cached",
            "kind": "cached-confirmation",
            "selection_source": "cached_confirmation",
            "label": "cached runtime environment",
            "python_path": cached_python_value,
            "env_root": cached_env_value,
            "status": "unresolved",
            "reason": "cached runtime environment was not rediscovered in the current scan",
            "confidence": 0.51,
            "recommended": True,
        }
        candidates.insert(0, synthetic_candidate)
        return {"candidate": synthetic_candidate, "source": "cached_confirmation", "confirmed": cached_confirmed}

    recommended = choose_top_candidate(list(candidates))
    if not recommended:
        return {"candidate": None, "source": "missing", "confirmed": False}
    return {"candidate": recommended, "source": "auto_recommended", "confirmed": False}


def confirmation_definition(field_name: str) -> Dict[str, object]:
    for item in CONFIRMATION_SEQUENCE:
        if item["field"] == field_name:
            return dict(item)
    raise KeyError(field_name)


def confirmation_field_is_confirmed(field_name: str, confirmed_fields: Dict[str, object]) -> bool:
    item = confirmed_fields.get(field_name)
    return isinstance(item, dict) and bool(item.get("confirmed", False))


def build_runtime_environment_options(scan: Dict[str, object]) -> List[Dict[str, object]]:
    options: List[Dict[str, object]] = []
    for item in ranked_candidates(list(scan["environment"]["candidates"])):
        options.append(
            {
                "value": item.get("id"),
                "label": item.get("label"),
                "confidence": item.get("confidence"),
                "selection_source": item.get("selection_source"),
                "python_path": item.get("python_path"),
                "env_root": item.get("env_root"),
            }
        )
    options.append(
        {
            "value": "__manual__",
            "label": "enter a custom environment manually",
            "confidence": 0.0,
            "selection_source": "manual",
        }
    )
    options.append(
        {
            "value": "__unknown__",
            "label": SKIP_FOR_NOW_LABEL,
            "confidence": 0.0,
            "selection_source": "manual",
        }
    )
    for index, option in enumerate(options, start=1):
        option["index"] = index
    return options


def active_confirmation_sequence(scan: Dict[str, object], profile: Dict[str, object]) -> List[Dict[str, object]]:
    launcher_value = str(profile.get("launcher") or "")
    assets = profile.get("assets") if isinstance(profile.get("assets"), dict) else {}
    sequence: List[Dict[str, object]] = []
    asset_fields = {asset_field_name(kind): kind for kind in ASSET_KINDS}
    for item in CONFIRMATION_SEQUENCE:
        field_name = str(item.get("field"))
        if field_name in asset_fields:
            asset_kind = asset_fields[field_name]
            bundle = assets.get(asset_kind) if isinstance(assets.get(asset_kind), dict) else {}
            if not asset_should_confirm(asset_kind, bundle, launcher_value):
                continue
        sequence.append(dict(item))
    return sequence


def build_field_confirmation_step(
    scan: Dict[str, object],
    profile: Dict[str, object],
    field_name: str,
    step_number: int,
    total_steps: int,
) -> Dict[str, object]:
    definition = confirmation_definition(field_name)
    confirmed_fields = profile.get("confirmed_fields") if isinstance(profile.get("confirmed_fields"), dict) else {}
    field_item = confirmed_fields.get(field_name) if isinstance(confirmed_fields.get(field_name), dict) else {}
    recommended_value = field_item.get("value")

    if field_name == "runtime_environment":
        selected_environment = profile.get("selected_environment") or {}
        options = build_runtime_environment_options(scan)
        if isinstance(field_item, dict):
            recommended_value = field_item.get("value")
        else:
            recommended_value = selected_environment.get("id") if selected_environment else None
    else:
        candidate_key = str(definition.get("candidate_key"))
        if candidate_key == "cann_candidates":
            candidates = list(scan["cann"]["candidates"])
            options = build_numbered_options(ranked_candidates(candidates), allow_free_text=bool(definition.get("allow_free_text", True)))
        elif candidate_key.startswith("asset:"):
            asset_kind = candidate_key.split(":", 1)[1]
            options = build_asset_confirmation_options(asset_bundle(scan, asset_kind), allow_free_text=bool(definition.get("allow_free_text", True)))
        else:
            candidates = list(scan.get(candidate_key) or [])
            catalog_key = definition.get("catalog_key")
            if isinstance(catalog_key, str):
                candidates = merge_catalog_candidates(catalog_key, candidates)
            options = build_numbered_options(ranked_candidates(candidates), allow_free_text=bool(definition.get("allow_free_text", True)))

    for option in options:
        option["recommended"] = option.get("value") == recommended_value

    return {
        "field": field_name,
        "interaction_mode": "single-field-confirmation",
        "label": definition.get("label"),
        "prompt": definition.get("prompt"),
        "step_number": step_number,
        "total_steps": total_steps,
        "recommended_value": recommended_value,
        "allow_free_text": bool(definition.get("allow_free_text", True)),
        "manual_hint": definition.get("manual_hint"),
        "options": options,
        "portable_question": build_portable_question(
            field_name=field_name,
            label=definition.get("label"),
            prompt=definition.get("prompt"),
            options=options,
            recommended_value=recommended_value,
            allow_free_text=bool(definition.get("allow_free_text", True)),
            manual_hint=definition.get("manual_hint"),
        ),
    }


def build_confirmation_state(scan: Dict[str, object], profile: Dict[str, object]) -> Dict[str, object]:
    confirmed_fields = profile.get("confirmed_fields") if isinstance(profile.get("confirmed_fields"), dict) else {}
    sequence = active_confirmation_sequence(scan, profile)
    pending_fields = [item["field"] for item in sequence if not confirmation_field_is_confirmed(item["field"], confirmed_fields)]
    current_step_number = len(sequence) - len(pending_fields) + 1 if pending_fields else len(sequence)
    current_confirmation = build_field_confirmation_step(scan, profile, pending_fields[0], current_step_number, len(sequence)) if pending_fields else None
    return {
        "required": bool(pending_fields),
        "ready_for_validation": not pending_fields,
        "pending_fields": pending_fields,
        "gate_pending_fields": [field_name for field_name in VALIDATION_GATE_FIELDS if field_name in pending_fields],
        "current_confirmation": current_confirmation,
    }
