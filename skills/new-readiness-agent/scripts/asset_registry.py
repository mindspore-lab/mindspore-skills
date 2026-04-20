#!/usr/bin/env python3
from typing import Dict, Tuple


ENTRY_PATTERNS = (
    "train.py",
    "training.py",
    "finetune.py",
    "main.py",
    "infer.py",
    "inference.py",
    "predict.py",
    "run.py",
)
CONFIG_SUFFIXES = {".yaml", ".yml", ".json"}

ASSET_KINDS = ("config", "model", "tokenizer", "dataset", "checkpoint")

ASSET_REGISTRY: Dict[str, Dict[str, object]] = {
    "config": {
        "label": "Config Asset",
        "portable_header": "Config",
        "candidate_key": "asset:config",
        "allow_free_text": True,
        "manual_hint": "Use a detected asset option, or enter local:/path, inline_config, none, or unknown.",
        "prompt": "Confirm how this workspace satisfies its runtime configuration.",
        "confirmation_mode": "required_or_candidates",
        "force_launchers": ("llamafactory-cli",),
        "validation_gate": True,
    },
    "model": {
        "label": "Model Asset",
        "portable_header": "Model",
        "candidate_key": "asset:model",
        "allow_free_text": True,
        "manual_hint": "Use a detected asset option, or enter local:/path, hf_hub:repo_id, hf_cache:/path, script_managed_remote:repo_id, or unknown.",
        "prompt": "Confirm how this workspace satisfies the model requirement.",
        "confirmation_mode": "required_or_candidates",
        "force_launchers": (),
        "validation_gate": True,
    },
    "tokenizer": {
        "label": "Tokenizer Asset",
        "portable_header": "Tokenizer",
        "candidate_key": "asset:tokenizer",
        "allow_free_text": True,
        "manual_hint": "Use a detected asset option, or enter local:/path, hf_hub:repo_id, hf_cache:/path, script_managed_remote:repo_id, or unknown.",
        "prompt": "Confirm how this workspace satisfies the tokenizer requirement.",
        "confirmation_mode": "required_or_candidates",
        "force_launchers": (),
        "validation_gate": True,
    },
    "dataset": {
        "label": "Dataset Asset",
        "portable_header": "Dataset",
        "candidate_key": "asset:dataset",
        "allow_free_text": True,
        "manual_hint": "Use a detected asset option, or enter local:/path, hf_hub:repo_id, hf_cache:/path, script_managed_remote:repo_id, or unknown.",
        "prompt": "Confirm how this workspace satisfies the dataset requirement.",
        "confirmation_mode": "required_or_candidates",
        "force_launchers": (),
        "validation_gate": True,
    },
    "checkpoint": {
        "label": "Checkpoint Asset",
        "portable_header": "Checkpoint",
        "candidate_key": "asset:checkpoint",
        "allow_free_text": True,
        "manual_hint": "Use a detected checkpoint option, or enter local:/path, none, or unknown.",
        "prompt": "Confirm whether this workspace depends on a checkpoint before launch.",
        "confirmation_mode": "candidates_only",
        "force_launchers": (),
        "validation_gate": False,
    },
}


def asset_field_name(kind: str) -> str:
    return f"{kind}_asset"


def asset_confirmation_entry(kind: str) -> Dict[str, object]:
    config = dict(ASSET_REGISTRY[kind])
    return {
        "field": asset_field_name(kind),
        "label": config["label"],
        "candidate_key": config["candidate_key"],
        "allow_free_text": config["allow_free_text"],
        "manual_hint": config["manual_hint"],
        "prompt": config["prompt"],
    }


def asset_confirmation_sequence() -> Tuple[Dict[str, object], ...]:
    return tuple(asset_confirmation_entry(kind) for kind in ASSET_KINDS)


def asset_portable_headers() -> Dict[str, str]:
    return {
        asset_field_name(kind): str(ASSET_REGISTRY[kind]["portable_header"])
        for kind in ASSET_KINDS
    }


def asset_kinds_with_validation_gate() -> Tuple[str, ...]:
    return tuple(kind for kind in ASSET_KINDS if bool(ASSET_REGISTRY[kind]["validation_gate"]))


def asset_should_confirm(kind: str, bundle: Dict[str, object], launcher_value: str) -> bool:
    config = ASSET_REGISTRY[kind]
    required = bool(((bundle.get("requirement") or {}).get("required")))
    has_candidates = bool(list(bundle.get("candidates") or []))
    if launcher_value in set(config.get("force_launchers") or ()):
        return True
    mode = str(config.get("confirmation_mode") or "required_or_candidates")
    if mode == "candidates_only":
        return has_candidates
    return required or has_candidates
