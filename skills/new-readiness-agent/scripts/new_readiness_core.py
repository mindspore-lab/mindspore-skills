#!/usr/bin/env python3
import json
import os
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Tuple

from ascend_compat import assess_installed_framework_compatibility
from asset_discovery import discover_asset_catalog
from asset_registry import ASSET_KINDS, CONFIG_SUFFIXES, ENTRY_PATTERNS, asset_kinds_with_validation_gate
from asset_schema import asset_locator_summary
from asset_validation import validate_asset_selection
from candidate_utils import choose_top_candidate, looks_like_local_path, merge_catalog_candidates
from confirmation_flow import (
    asset_bundle,
    build_confirmation_state,
    choose_asset,
    choose_environment,
    choose_value,
    load_cached_confirmation,
    parse_confirmation_overrides,
)
from environment_selection import (
    build_environment_candidates,
    resolve_optional_path,
    split_command,
)
from runtime_env import detect_ascend_runtime, detect_cann_version, resolve_runtime_environment
from runtime_env import build_selected_runtime_environment
from runtime_probes import (
    launcher_ready,
    make_check,
    probe_imports,
    probe_package_versions,
    summarize_ascend_runtime,
    summarize_cann_version,
    summarize_framework_compatibility,
    summarize_import_failures,
)


SKIP_DIRS = {
    "__pycache__",
    ".git",
    ".hg",
    ".svn",
    ".pytest_cache",
    "node_modules",
    "dist",
    "build",
    "readiness-output",
    "runs",
    "venv",
    ".venv",
    ".env",
    "env",
}
FRAMEWORK_PACKAGES = {
    "mindspore": ["mindspore"],
    "pta": ["torch", "torch_npu"],
    "mixed": ["mindspore", "torch", "torch_npu"],
}
LAUNCHER_PACKAGES = {
    "torchrun": ["torch"],
    "accelerate": ["accelerate"],
    "deepspeed": ["deepspeed"],
    "llamafactory-cli": ["llamafactory", "transformers"],
}
RUNTIME_IMPORT_CANDIDATES = {
    "mindspore",
    "torch",
    "torch_npu",
    "transformers",
    "datasets",
    "tokenizers",
    "accelerate",
    "safetensors",
    "diffusers",
    "peft",
    "trl",
    "evaluate",
    "sentencepiece",
    "huggingface_hub",
    "llamafactory",
    "deepspeed",
}


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def should_skip_dirname(name: str) -> bool:
    return name.startswith(".") or name in SKIP_DIRS


def list_files(root: Path, max_depth: int = 3) -> List[Path]:
    files: List[Path] = []
    if not root.exists():
        return files
    root_depth = len(root.resolve().parts)
    for current_root, dirnames, filenames in os.walk(root):
        current_path = Path(current_root)
        try:
            depth = len(current_path.resolve().parts) - root_depth
        except OSError:
            continue
        if depth > max_depth:
            dirnames[:] = []
            continue
        dirnames[:] = [name for name in dirnames if not should_skip_dirname(name)]
        for name in filenames:
            files.append(current_path / name)
    return files


def candidate(
    value: Optional[str],
    label: str,
    source: str,
    confidence: float,
    **extra: object,
) -> Dict[str, object]:
    payload: Dict[str, object] = {
        "value": value,
        "label": label,
        "selection_source": source,
        "confidence": round(max(0.0, min(confidence, 0.99)), 2),
    }
    payload.update(extra)
    return payload


def dedupe_candidates(items: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
    seen = set()
    result: List[Dict[str, object]] = []
    for item in items:
        key = (
            item.get("value"),
            item.get("label"),
            item.get("selection_source"),
            item.get("command_template"),
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def parse_command_candidate(command: str, root: Path, source: str, confidence: float, label: str) -> Optional[Dict[str, object]]:
    tokens = split_command(command)
    if not tokens:
        return None

    launcher = None
    entry_script = None
    config_path = None
    make_target = None
    uses_llamafactory = any("llamafactory" in token.lower() for token in tokens)
    probe_tokens = list(tokens)

    if tokens[0] == "make":
        launcher = "make"
        for token in tokens[1:]:
            if not token.startswith("-"):
                make_target = token
                break
    elif tokens[0] in {"bash", "sh"}:
        launcher = "bash"
        if len(tokens) > 1:
            entry_script = str(resolve_optional_path(tokens[1], root) or tokens[1])
    else:
        if len(tokens) >= 2 and tokens[0] == "uv" and tokens[1] == "run":
            probe_tokens = tokens[2:]
        elif len(tokens) >= 2 and tokens[0] == "conda" and tokens[1] == "run":
            index = 2
            while index < len(tokens):
                token = tokens[index]
                if token in {"-n", "--name", "-p", "--prefix"}:
                    index += 2
                    continue
                if token.startswith("-"):
                    index += 1
                    continue
                break
            probe_tokens = tokens[index:]

        if not probe_tokens:
            launcher = "python"
        elif "llamafactory-cli" in probe_tokens[0].lower() or any("llamafactory-cli" in item.lower() for item in probe_tokens):
            launcher = "llamafactory-cli"
        elif probe_tokens[0] == "torchrun" or (
            len(probe_tokens) >= 3 and probe_tokens[0].startswith("python") and probe_tokens[1] == "-m" and probe_tokens[2] == "torch.distributed.run"
        ):
            launcher = "torchrun"
        elif probe_tokens[0] == "accelerate" or (
            len(probe_tokens) >= 3 and probe_tokens[0].startswith("python") and probe_tokens[1] == "-m" and probe_tokens[2].startswith("accelerate")
        ):
            launcher = "accelerate"
        elif probe_tokens[0] == "deepspeed" or (
            len(probe_tokens) >= 3 and probe_tokens[0].startswith("python") and probe_tokens[1] == "-m" and probe_tokens[2].startswith("deepspeed")
        ):
            launcher = "deepspeed"
        elif probe_tokens[0] == "msrun":
            launcher = "msrun"
        elif probe_tokens[0].startswith("python") or probe_tokens[0].endswith("python.exe") or probe_tokens[0].endswith("/python"):
            launcher = "python"
        else:
            launcher = "python"

        for token in probe_tokens:
            lowered = token.lower()
            if token.endswith((".py", ".sh")):
                entry_script = str(resolve_optional_path(token, root) or token)
                break
            if lowered.endswith(("train", "infer", "predict")) and not entry_script:
                entry_script = token

        config_flags = {"--config", "--config_file", "--config-path", "--yaml_path", "--cfg"}
        for index, token in enumerate(probe_tokens):
            lowered = token.lower()
            if token in config_flags and index + 1 < len(probe_tokens):
                config_path = str(resolve_optional_path(probe_tokens[index + 1], root) or probe_tokens[index + 1])
                break
            if lowered.endswith((".yaml", ".yml", ".json")) and config_path is None:
                config_path = str(resolve_optional_path(token, root) or token)

    return candidate(
        launcher,
        label,
        source,
        confidence,
        command_template=command,
        entry_script=entry_script,
        config_path=config_path,
        make_target=make_target,
        uses_llamafactory=uses_llamafactory,
        evidence=[command],
    )


def build_launcher_candidates(root: Path, args: object, files: List[Path]) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []
    explicit_command = getattr(args, "launch_command", None)
    explicit_hint = getattr(args, "launcher_hint", None)

    if explicit_command:
        parsed = parse_command_candidate(str(explicit_command), root, "explicit_input", 0.99, "explicit launch command")
        if parsed:
            results.append(parsed)

    if explicit_hint and explicit_hint != "auto":
        results.append(candidate(str(explicit_hint), f"explicit launcher_hint={explicit_hint}", "explicit_input", 0.98))

    makefile = next((path for path in files if path.name == "Makefile"), None)
    if makefile:
        current_target = None
        for line in read_text(makefile).splitlines():
            stripped = line.rstrip()
            if not stripped:
                continue
            if not stripped.startswith(("\t", " ")):
                if ":" in stripped and not stripped.startswith("#"):
                    current_target = stripped.split(":", 1)[0].strip()
                continue
            command_text = stripped.strip()
            if not any(token in command_text for token in ("torchrun", "accelerate", "deepspeed", "msrun", "llamafactory-cli", "python", "bash", "sh")):
                continue
            make_target = current_target or "default"
            results.append(
                candidate(
                    "make",
                    f"Makefile target {make_target}",
                    "workspace_scan",
                    0.78,
                    command_template=f"make {make_target}",
                    make_target=make_target,
                    underlying_command=command_text,
                    uses_llamafactory="llamafactory" in command_text.lower(),
                    evidence=[command_text],
                )
            )

    for path in files:
        if path.suffix.lower() not in {".sh", ".bash", ".cmd", ".bat"}:
            continue
        if path.parent != root:
            continue
        for line in read_text(path).splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if not any(token in stripped for token in ("torchrun", "accelerate", "deepspeed", "msrun", "llamafactory-cli", "python ", "bash ", "sh ")):
                continue
            parsed = parse_command_candidate(stripped, root, "workspace_scan", 0.83, f"wrapper script {path.name}")
            if parsed:
                parsed["wrapper_script"] = str(path)
                if not parsed.get("entry_script"):
                    parsed["entry_script"] = str(path)
                results.append(parsed)
            break

    explicit_entry = resolve_optional_path(getattr(args, "entry_script", None), root)
    if explicit_entry:
        results.append(
            candidate(
                "python" if explicit_entry.suffix.lower() == ".py" else "bash",
                f"entry script {explicit_entry.name}",
                "explicit_input",
                0.9,
                command_template=f"{'python' if explicit_entry.suffix.lower() == '.py' else 'bash'} {explicit_entry}",
                entry_script=str(explicit_entry),
            )
        )

    return dedupe_candidates(results)


def build_entry_candidates(root: Path, args: object, files: List[Path], launcher_candidates: List[Dict[str, object]]) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []
    explicit_entry = resolve_optional_path(getattr(args, "entry_script", None), root)
    if explicit_entry:
        results.append(candidate(str(explicit_entry), f"explicit entry_script={explicit_entry.name}", "explicit_input", 0.99, exists=explicit_entry.exists()))

    for item in launcher_candidates:
        entry_script = item.get("entry_script")
        if entry_script:
            entry_path = resolve_optional_path(str(entry_script), root)
            results.append(
                candidate(
                    str(entry_path or entry_script),
                    f"launcher candidate from {item.get('label')}",
                    str(item.get("selection_source")),
                    float(item.get("confidence") or 0.75) - 0.02,
                    exists=bool(entry_path and entry_path.exists()),
                )
            )

    for name in ENTRY_PATTERNS:
        path = root / name
        if path.exists():
            score = 0.86 if "train" in name or "infer" in name else 0.75
            results.append(candidate(str(path), f"workspace entry {name}", "workspace_scan", score, exists=True))

    for path in files:
        if path.suffix.lower() not in {".py", ".sh"}:
            continue
        if path.parent != root:
            continue
        if path.name in ENTRY_PATTERNS:
            continue
        lowered = path.name.lower()
        if any(token in lowered for token in ("train", "infer", "predict", "run", "launch", "finetune")):
            results.append(candidate(str(path), f"workspace entry {path.name}", "workspace_scan", 0.7, exists=True))

    return dedupe_candidates(results)


def build_config_candidates(root: Path, args: object, files: List[Path], launcher_candidates: List[Dict[str, object]]) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []
    explicit_config = resolve_optional_path(getattr(args, "config_path", None), root)
    if explicit_config:
        results.append(candidate(str(explicit_config), f"explicit config_path={explicit_config.name}", "explicit_input", 0.99, exists=explicit_config.exists()))

    for item in launcher_candidates:
        config_path = item.get("config_path")
        if config_path:
            path = resolve_optional_path(str(config_path), root)
            results.append(
                candidate(
                    str(path or config_path),
                    f"launcher config from {item.get('label')}",
                    str(item.get("selection_source")),
                    float(item.get("confidence") or 0.75) - 0.01,
                    exists=bool(path and path.exists()),
                )
            )

    for path in files:
        if path.suffix.lower() not in CONFIG_SUFFIXES:
            continue
        if path.parent != root:
            continue
        lowered = path.name.lower()
        if any(token in lowered for token in ("config", "train", "infer", "llama", "qwen", "sft", "dpo")):
            results.append(candidate(str(path), f"workspace config {path.name}", "workspace_scan", 0.76, exists=True))

    return dedupe_candidates(results)


def collect_dependency_text(files: List[Path]) -> str:
    texts: List[str] = []
    for path in files:
        if path.name in {"pyproject.toml", "requirements.txt", "requirements-dev.txt", "environment.yml", "conda.yaml"}:
            texts.append(read_text(path))
    return "\n".join(texts)


def collect_entry_runtime_imports(entry_script: Optional[str], root: Path) -> List[str]:
    if not entry_script:
        return []
    path = resolve_optional_path(entry_script, root)
    if not path or not path.exists() or path.suffix.lower() != ".py":
        return []
    text = read_text(path)
    found: List[str] = []
    lowered = text.lower()
    for name in sorted(RUNTIME_IMPORT_CANDIDATES):
        if f"import {name}" in lowered or f"from {name}" in lowered:
            found.append(name)
    return found


def build_target_candidates(
    args: object,
    entry_candidates: List[Dict[str, object]],
    launcher_candidates: List[Dict[str, object]],
    config_candidates: List[Dict[str, object]],
    asset_catalog: Dict[str, object],
    root: Path,
) -> List[Dict[str, object]]:
    explicit_target = getattr(args, "target", None)
    if explicit_target in {"training", "inference"}:
        return [candidate(explicit_target, f"explicit target={explicit_target}", "explicit_input", 0.99)]

    scores = {"training": 0, "inference": 0}
    evidence = {"training": [], "inference": []}

    dataset_bundle = ((asset_catalog.get("assets") or {}).get("dataset") or {}) if isinstance(asset_catalog.get("assets"), dict) else {}
    dataset_candidates = list(dataset_bundle.get("candidates") or [])
    script_hints = asset_catalog.get("script_hints") if isinstance(asset_catalog.get("script_hints"), dict) else {}

    if dataset_candidates or list(script_hints.get("dataset_hints") or []):
        scores["training"] += 2
        evidence["training"].append("dataset evidence suggests training")

    for item in entry_candidates:
        value = str(item.get("value") or "").lower()
        if any(token in value for token in ("train", "finetune", "sft", "dpo", "pretrain")):
            scores["training"] += 2
            evidence["training"].append(f"entry script suggests training: {Path(value).name}")
        if any(token in value for token in ("infer", "predict", "serve", "generate")):
            scores["inference"] += 2
            evidence["inference"].append(f"entry script suggests inference: {Path(value).name}")

    for item in launcher_candidates:
        command = str(item.get("command_template") or "").lower()
        if any(token in command for token in ("sft", "dpo", "pt", "rm", "ppo", "train")):
            scores["training"] += 1
            evidence["training"].append("launch command suggests training")
        if any(token in command for token in ("infer", "predict", "generate", "chat")):
            scores["inference"] += 1
            evidence["inference"].append("launch command suggests inference")

    for item in config_candidates:
        config_path = resolve_optional_path(str(item.get("value")), root)
        if not config_path or not config_path.exists():
            continue
        text = read_text(config_path).lower()
        if any(token in text for token in ("per_device_train_batch_size", "num_train_epochs", "train_file", "dataset_dir", "stage: sft", "stage: dpo")):
            scores["training"] += 2
            evidence["training"].append(f"config suggests training: {config_path.name}")
        if any(token in text for token in ("max_new_tokens", "top_p", "temperature", "do_sample", "generation")):
            scores["inference"] += 2
            evidence["inference"].append(f"config suggests inference: {config_path.name}")

    for hint in script_hints.get("inline_config") or []:
        scores["training"] += 1
        evidence["training"].append(f"inline TrainingArguments detected in {Path(str(hint.get('entry_script') or '')).name}")

    results: List[Dict[str, object]] = []
    if scores["training"] > 0:
        results.append(candidate("training", "auto-detected training", "workspace_inference", 0.83 if scores["training"] > scores["inference"] else 0.62, evidence=evidence["training"]))
    if scores["inference"] > 0:
        results.append(candidate("inference", "auto-detected inference", "workspace_inference", 0.83 if scores["inference"] > scores["training"] else 0.62, evidence=evidence["inference"]))
    return dedupe_candidates(results)


def build_framework_candidates(
    args: object,
    entry_candidates: List[Dict[str, object]],
    launcher_candidates: List[Dict[str, object]],
    config_candidates: List[Dict[str, object]],
    dependency_text: str,
    root: Path,
) -> List[Dict[str, object]]:
    explicit_framework = getattr(args, "framework_hint", None)
    if explicit_framework in {"mindspore", "pta", "mixed"}:
        return [candidate(explicit_framework, f"explicit framework_hint={explicit_framework}", "explicit_input", 0.99)]

    text_chunks = [dependency_text.lower()]
    for item in entry_candidates + config_candidates:
        path = resolve_optional_path(str(item.get("value")), root)
        if path and path.exists():
            text_chunks.append(read_text(path).lower())
    for item in launcher_candidates:
        text_chunks.append(str(item.get("command_template") or "").lower())
        text_chunks.append(str(item.get("underlying_command") or "").lower())
    combined = "\n".join(text_chunks)

    has_mindspore = any(token in combined for token in ("mindspore", "msrun"))
    has_torch = "torch" in combined or any(str(item.get("value")) in {"torchrun", "accelerate", "deepspeed", "llamafactory-cli"} for item in launcher_candidates)
    has_torch_npu = "torch_npu" in combined or "llamafactory" in combined

    results: List[Dict[str, object]] = []
    if has_mindspore and (has_torch or has_torch_npu):
        results.append(candidate("mixed", "mixed framework evidence", "workspace_inference", 0.72, evidence=["mindspore and PTA evidence both detected"]))
    elif has_mindspore:
        results.append(candidate("mindspore", "mindspore evidence", "workspace_inference", 0.84, evidence=["mindspore-related evidence detected"]))
    elif has_torch or has_torch_npu:
        results.append(candidate("pta", "PTA evidence", "workspace_inference", 0.86 if has_torch_npu else 0.78, evidence=["torch / torch_npu / launcher evidence detected"]))
    return results


def build_cann_candidates(root: Path, args: object) -> Dict[str, object]:
    results: List[Dict[str, object]] = []
    explicit_cann = getattr(args, "cann_path", None)
    if explicit_cann:
        results.append(candidate(str(resolve_optional_path(explicit_cann, root) or explicit_cann), "explicit cann_path", "explicit_input", 0.99))

    system_layer = detect_ascend_runtime({"cann_path": explicit_cann})
    if system_layer.get("cann_path_input"):
        results.append(candidate(str(system_layer["cann_path_input"]), "cann path input", "explicit_input", 0.99))
    if system_layer.get("ascend_env_script_path"):
        results.append(candidate(str(system_layer["ascend_env_script_path"]), "selected Ascend env script", "runtime_detection", 0.84))
    for path in system_layer.get("ascend_env_candidate_paths") or []:
        results.append(candidate(str(path), "Ascend env script candidate", "runtime_detection", 0.62))

    results = dedupe_candidates(results)
    cann_version = detect_cann_version(explicit_cann, system_layer.get("ascend_env_script_path"))
    return {
        "candidates": results,
        "recommended": choose_top_candidate(results),
        "system_layer": system_layer,
        "version": cann_version,
    }


def shell_quote(value: Optional[str]) -> str:
    return json.dumps(str(value or ""))


def build_base_launch_command(
    launcher_value: Optional[str],
    selected_launcher_candidate: Optional[Dict[str, object]],
    selected_python: Optional[str],
    entry_script: Optional[str],
) -> Optional[str]:
    launcher = str(launcher_value or "").strip()
    entry = str(entry_script or "").strip()
    python_path = str(selected_python or "").strip()
    command_template = str((selected_launcher_candidate or {}).get("command_template") or "").strip()

    if launcher == "python":
        if entry:
            executable = python_path or "python"
            return f"{shell_quote(executable)} {shell_quote(entry)}"
        return command_template or None
    if launcher == "bash":
        if entry:
            return f"bash {shell_quote(entry)}"
        return command_template or None
    if launcher in {"torchrun", "accelerate", "deepspeed", "msrun", "llamafactory-cli", "make"}:
        if command_template:
            return command_template
        if launcher == "torchrun" and entry:
            return f"torchrun {shell_quote(entry)}"
        if launcher == "accelerate" and entry:
            return f"accelerate launch {shell_quote(entry)}"
        if launcher == "deepspeed" and entry:
            return f"deepspeed {shell_quote(entry)}"
        if launcher == "msrun" and entry:
            return f"msrun {shell_quote(entry)}"
        if launcher == "llamafactory-cli":
            return "llamafactory-cli train"
        if launcher == "make":
            return "make"
    return command_template or None


def maybe_prefix_cann_environment(cann_value: Optional[str], command: Optional[str]) -> Optional[str]:
    base_command = str(command or "").strip()
    if not base_command:
        return None
    if not cann_value:
        return base_command
    system_layer = detect_ascend_runtime({"cann_path": cann_value})
    script_path = str(system_layer.get("ascend_env_script_path") or "").strip()
    if not script_path:
        return base_command
    return f"source {shell_quote(script_path)} && {base_command}"


def derive_launch_command(
    *,
    root: Path,
    args: object,
    cached_confirmation: Dict[str, object],
    confirmation_override: Optional[str],
    field_candidates: List[Dict[str, object]],
    launcher_value: Optional[str],
    selected_launcher_candidate: Optional[Dict[str, object]],
    selected_python: Optional[str],
    entry_script: Optional[str],
    cann_path: Optional[str],
) -> Dict[str, object]:
    explicit_value = getattr(args, "launch_command", None)
    if confirmation_override not in {None, ""}:
        return {"value": confirmation_override, "source": "manual_confirmation", "confirmed": True}
    if explicit_value not in {None, ""}:
        return {"value": explicit_value, "source": "explicit_input", "confirmed": True}

    cached_fields = cached_confirmation.get("confirmed_fields") if isinstance(cached_confirmation.get("confirmed_fields"), dict) else {}
    cached_item = cached_fields.get("launch_command") if isinstance(cached_fields.get("launch_command"), dict) else None
    cached_value = cached_item.get("value") if isinstance(cached_item, dict) else None
    if cached_value not in {None, ""}:
        return {"value": cached_value, "source": "cached_confirmation", "confirmed": True}

    derived_command = maybe_prefix_cann_environment(
        cann_path,
        build_base_launch_command(
            launcher_value,
            selected_launcher_candidate,
            selected_python,
            entry_script,
        ),
    )
    if derived_command:
        return {"value": derived_command, "source": "derived", "confirmed": True}

    top = choose_top_candidate(list(field_candidates))
    if not top:
        return {"value": None, "source": "missing", "confirmed": True}
    return {"value": top.get("command_template") or top.get("value"), "source": "auto_recommended", "confirmed": True}


def build_required_packages(
    framework_value: Optional[str],
    launcher_value: Optional[str],
    runtime_imports: List[str],
    uses_llamafactory: bool,
) -> List[str]:
    packages = set(FRAMEWORK_PACKAGES.get(str(framework_value), []))
    packages.update(LAUNCHER_PACKAGES.get(str(launcher_value), []))
    packages.update(runtime_imports)
    if uses_llamafactory:
        packages.update({"llamafactory", "transformers", "datasets", "accelerate"})
    return sorted(packages)


def merge_selected_scan_candidate(
    field_candidates: List[Dict[str, object]],
    selected_value: Optional[str],
    *,
    label: str,
    root: Path,
) -> List[Dict[str, object]]:
    normalized_value = str(selected_value or "").strip()
    if not normalized_value:
        return list(field_candidates)
    resolved_path = resolve_optional_path(normalized_value, root)
    candidate_value = str(resolved_path or normalized_value)
    synthesized = candidate(
        candidate_value,
        label,
        "profile_selection",
        0.99,
        exists=bool(resolved_path and resolved_path.exists()),
    )
    return dedupe_candidates([synthesized, *list(field_candidates)])


def selected_local_asset_path(asset_choice: Dict[str, object]) -> Optional[str]:
    selected_asset = asset_choice.get("asset") if isinstance(asset_choice.get("asset"), dict) else {}
    if str(selected_asset.get("source_type") or "") != "local_path":
        return None
    locator = selected_asset.get("locator") if isinstance(selected_asset.get("locator"), dict) else {}
    raw_path = str(locator.get("path") or "").strip()
    return raw_path or None


def asset_bundle_from_catalog(asset_catalog: Dict[str, object], kind: str) -> Dict[str, object]:
    assets = asset_catalog.get("assets") if isinstance(asset_catalog.get("assets"), dict) else {}
    bundle = assets.get(kind)
    if isinstance(bundle, dict):
        return bundle
    return {"requirement": {"kind": kind, "required": False, "reason": ""}, "candidates": []}


def refresh_asset_catalog(
    scan: Dict[str, object],
    root: Path,
    args: object,
    *,
    target_value: Optional[str],
    entry_value: Optional[str],
    config_value: Optional[str] = None,
) -> Dict[str, object]:
    files = [Path(item) for item in list(scan.get("files") or [])]
    entry_candidates = merge_selected_scan_candidate(
        list(scan.get("entry_candidates") or []),
        entry_value,
        label=f"selected entry {Path(str(entry_value)).name}" if entry_value else "selected entry",
        root=root,
    )
    config_candidates = merge_selected_scan_candidate(
        list(scan.get("config_candidates") or []),
        config_value,
        label=f"selected config {Path(str(config_value)).name}" if config_value else "selected config",
        root=root,
    )
    asset_args = SimpleNamespace(**vars(args))
    return discover_asset_catalog(
        root=root,
        files=files,
        entry_candidates=entry_candidates,
        config_candidates=config_candidates,
        args=asset_args,
        target_hint=str(target_value) if target_value not in {None, "", "auto"} else None,
    )


def resolve_profile_assets(
    scan: Dict[str, object],
    root: Path,
    args: object,
    *,
    cached_confirmation: Dict[str, object],
    confirmation_overrides: Dict[str, str],
    target_value: Optional[str],
    entry_value: Optional[str],
) -> Tuple[Dict[str, object], Dict[str, Dict[str, object]]]:
    refreshed_asset_catalog = refresh_asset_catalog(
        scan,
        root,
        args,
        target_value=target_value,
        entry_value=entry_value,
        config_value=getattr(args, "config_path", None),
    )
    config_choice = choose_asset("config", cached_confirmation, asset_bundle_from_catalog(refreshed_asset_catalog, "config"), confirmation_overrides.get("config_asset"))

    selected_config_path = selected_local_asset_path(config_choice)
    if selected_config_path:
        refreshed_asset_catalog = refresh_asset_catalog(
            scan,
            root,
            args,
            target_value=target_value,
            entry_value=entry_value,
            config_value=selected_config_path,
        )
        config_choice = choose_asset("config", cached_confirmation, asset_bundle_from_catalog(refreshed_asset_catalog, "config"), confirmation_overrides.get("config_asset"))

    asset_choices: Dict[str, Dict[str, object]] = {"config": config_choice}
    for kind in ASSET_KINDS:
        if kind == "config":
            continue
        asset_choices[kind] = choose_asset(
            kind,
            cached_confirmation,
            asset_bundle_from_catalog(refreshed_asset_catalog, kind),
            confirmation_overrides.get(f"{kind}_asset"),
        )
    return refreshed_asset_catalog, asset_choices


def analyze_workspace(root: Path, args: object) -> Dict[str, object]:
    files = list_files(root)
    launcher_candidates = build_launcher_candidates(root, args, files)
    entry_candidates = build_entry_candidates(root, args, files, launcher_candidates)
    config_candidates = build_config_candidates(root, args, files, launcher_candidates)
    initial_target_hint = getattr(args, "target", None)
    if initial_target_hint not in {"training", "inference"}:
        initial_target_hint = None
    asset_catalog = discover_asset_catalog(
        root=root,
        files=files,
        entry_candidates=entry_candidates,
        config_candidates=config_candidates,
        args=args,
        target_hint=initial_target_hint,
    )
    dependency_text = collect_dependency_text(files)
    target_candidates = build_target_candidates(args, entry_candidates, launcher_candidates, config_candidates, asset_catalog, root)
    framework_candidates = build_framework_candidates(args, entry_candidates, launcher_candidates, config_candidates, dependency_text, root)
    launch_command_candidates = [item for item in launcher_candidates if item.get("command_template")]
    recommended_launcher = choose_top_candidate(list(launcher_candidates))
    environment_result = build_environment_candidates(
        root,
        launch_command=str(getattr(args, "launch_command", None) or (recommended_launcher.get("command_template") if recommended_launcher else "") or ""),
        selected_python=getattr(args, "selected_python", None),
        selected_env_root=getattr(args, "selected_env_root", None),
    )
    cann_result = build_cann_candidates(root, args)

    return {
        "working_dir": str(root),
        "files": [str(path) for path in files],
        "launcher_candidates": launcher_candidates,
        "entry_candidates": entry_candidates,
        "config_candidates": config_candidates,
        "asset_catalog": asset_catalog,
        "target_candidates": target_candidates,
        "framework_candidates": framework_candidates,
        "launch_command_candidates": launch_command_candidates,
        "environment": environment_result,
        "cann": cann_result,
        "dependency_text": dependency_text,
    }


def finalize_profile(scan: Dict[str, object], root: Path, args: object) -> Dict[str, object]:
    cached_confirmation = load_cached_confirmation(root)
    confirmation_overrides = parse_confirmation_overrides(getattr(args, "confirm", None))

    target_choice = choose_value("target", getattr(args, "target", None) if getattr(args, "target", None) != "auto" else None, cached_confirmation, list(scan["target_candidates"]), confirmation_overrides.get("target"))
    framework_choice = choose_value("framework", getattr(args, "framework_hint", None) if getattr(args, "framework_hint", None) != "auto" else None, cached_confirmation, list(scan["framework_candidates"]), confirmation_overrides.get("framework"))
    launcher_choice = choose_value("launcher", getattr(args, "launcher_hint", None) if getattr(args, "launcher_hint", None) != "auto" else None, cached_confirmation, list(scan["launcher_candidates"]), confirmation_overrides.get("launcher"))
    entry_choice = choose_value("entry_script", getattr(args, "entry_script", None), cached_confirmation, list(scan["entry_candidates"]), confirmation_overrides.get("entry_script"))
    cann_choice = choose_value("cann_path", getattr(args, "cann_path", None), cached_confirmation, list(scan["cann"]["candidates"]), confirmation_overrides.get("cann_path"))
    extra_context_choice = choose_value("extra_context", getattr(args, "extra_context", None), cached_confirmation, [], confirmation_overrides.get("extra_context"))
    environment_choice = choose_environment(root, args, cached_confirmation, dict(scan["environment"]), confirmation_overrides.get("runtime_environment"))

    selected_launcher_candidate = next((item for item in scan["launcher_candidates"] if item.get("value") == launcher_choice["value"]), None)
    selected_environment_candidate = environment_choice.get("candidate")
    command_choice = derive_launch_command(
        root=root,
        args=args,
        cached_confirmation=cached_confirmation,
        confirmation_override=confirmation_overrides.get("launch_command"),
        field_candidates=list(scan["launch_command_candidates"]),
        launcher_value=launcher_choice["value"],
        selected_launcher_candidate=selected_launcher_candidate,
        selected_python=selected_environment_candidate.get("python_path") if selected_environment_candidate else getattr(args, "selected_python", None),
        entry_script=entry_choice["value"],
        cann_path=cann_choice["value"],
    )
    runtime_imports = collect_entry_runtime_imports(str(entry_choice["value"]) if entry_choice.get("value") else None, root)
    uses_llamafactory = bool(
        (selected_launcher_candidate and selected_launcher_candidate.get("uses_llamafactory"))
        or str(command_choice.get("value") or "").lower().find("llamafactory") >= 0
    )

    refreshed_asset_catalog, asset_choices = resolve_profile_assets(
        scan,
        root,
        args,
        cached_confirmation=cached_confirmation,
        confirmation_overrides=confirmation_overrides,
        target_value=target_choice["value"],
        entry_value=entry_choice["value"],
    )
    scan["asset_catalog"] = refreshed_asset_catalog

    required_packages = build_required_packages(
        str(framework_choice.get("value") or ""),
        str(launcher_choice.get("value") or ""),
        runtime_imports,
        uses_llamafactory,
    )

    confirmed_fields = {
        "target": target_choice,
        "framework": framework_choice,
        "launcher": launcher_choice,
        "entry_script": entry_choice,
        "cann_path": cann_choice,
        "launch_command": command_choice,
        "extra_context": extra_context_choice,
        "runtime_environment": {
            "value": selected_environment_candidate.get("id") if selected_environment_candidate else None,
            "source": environment_choice["source"],
            "confirmed": environment_choice["confirmed"],
        },
        "selected_python": {
            "value": selected_environment_candidate.get("python_path") if selected_environment_candidate else getattr(args, "selected_python", None),
            "source": environment_choice["source"],
            "confirmed": environment_choice["confirmed"],
        },
        "selected_env_root": {
            "value": selected_environment_candidate.get("env_root") if selected_environment_candidate else getattr(args, "selected_env_root", None),
            "source": environment_choice["source"],
            "confirmed": environment_choice["confirmed"],
        },
    }
    for kind, choice in asset_choices.items():
        confirmed_fields[f"{kind}_asset"] = choice

    assets = {
        kind: {
            **asset_bundle(scan, kind),
            "selected": choice["asset"],
        }
        for kind, choice in asset_choices.items()
    }

    confirmation_state = build_confirmation_state(
        scan,
        {
            "confirmed_fields": confirmed_fields,
            "selected_environment": selected_environment_candidate,
            "assets": assets,
            "launcher": launcher_choice["value"],
        },
    )
    return {
        "target": target_choice["value"],
        "framework": framework_choice["value"],
        "launcher": launcher_choice["value"],
        "entry_script": entry_choice["value"],
        "cann_path": cann_choice["value"],
        "launch_command": command_choice["value"] or (selected_launcher_candidate.get("command_template") if selected_launcher_candidate else None),
        "extra_context": extra_context_choice["value"],
        "selected_environment": selected_environment_candidate,
        "assets": assets,
        "confirmed_fields": confirmed_fields,
        "required_packages": required_packages,
        "runtime_imports": runtime_imports,
        "uses_llamafactory": uses_llamafactory,
        "selected_launcher_candidate": selected_launcher_candidate,
        "cached_confirmation": cached_confirmation,
        "confirmation_overrides": confirmation_overrides,
        "confirmation_state": confirmation_state,
    }


def build_pending_validation(scan: Dict[str, object], profile: Dict[str, object], root: Path) -> Dict[str, object]:
    checks: List[Dict[str, object]] = []
    selected_env = profile.get("selected_environment")
    confirmation_state = profile.get("confirmation_state") if isinstance(profile.get("confirmation_state"), dict) else {}
    confirmed_fields = profile.get("confirmed_fields") if isinstance(profile.get("confirmed_fields"), dict) else {}

    def pending_check(field_name: str, check_id: str, *, fallback_label: str) -> None:
        item = confirmed_fields.get(field_name) if isinstance(confirmed_fields.get(field_name), dict) else {}
        value = item.get("value")
        confirmed = bool(item.get("confirmed", False))
        display_value = value
        if field_name.endswith("_asset") and isinstance(item.get("asset"), dict):
            display_value = asset_locator_summary(item["asset"]) or item["asset"].get("source_type")
        if confirmed and value not in {None, ""}:
            checks.append(make_check(check_id, "ok", f"{fallback_label} confirmed: {display_value}"))
            return
        if value in {None, ""}:
            checks.append(make_check(check_id, "warn", f"{fallback_label} still needs a user selection."))
            return
        checks.append(make_check(check_id, "warn", f"{fallback_label} recommendation is ready, but still needs user confirmation: {display_value}"))

    pending_check("target", "target-selection", fallback_label="target")
    pending_check("launcher", "launcher-selection", fallback_label="launcher")
    pending_check("framework", "framework-selection", fallback_label="framework")
    pending_check("entry_script", "workspace-entry-script", fallback_label="entry script")
    for kind in ASSET_KINDS:
        field_name = f"{kind}_asset"
        should_include = field_name in list(confirmation_state.get("pending_fields") or []) or isinstance(confirmed_fields.get(field_name), dict)
        if kind != "checkpoint":
            should_include = should_include or bool(((profile.get("assets") or {}).get(kind) or {}).get("candidates")) or bool((((profile.get("assets") or {}).get(kind) or {}).get("requirement") or {}).get("required"))
        if should_include:
            pending_check(field_name, f"workspace-{kind}-asset", fallback_label=f"{kind} asset")

    if selected_env:
        env_status = "ok" if confirmed_fields.get("selected_python", {}).get("confirmed") and confirmed_fields.get("selected_env_root", {}).get("confirmed") else "warn"
        summary = "runtime environment is confirmed" if env_status == "ok" else "runtime environment recommendation is ready, but still needs user confirmation"
        checks.append(
            make_check(
                "python-environment",
                env_status,
                summary,
                evidence=[str(selected_env.get("python_path") or selected_env.get("env_root") or "")],
            )
        )
    else:
        checks.append(make_check("python-environment", "warn", "runtime environment is still unresolved"))

    cann_input = profile.get("cann_path")
    system_layer = detect_ascend_runtime({"cann_path": cann_input})
    selected_runtime_env = build_selected_runtime_environment(selected_env)
    probe_env, probe_env_source, probe_env_error = resolve_runtime_environment(system_layer, selected_runtime_env)
    cann_version_info = detect_cann_version(cann_input, system_layer.get("ascend_env_script_path"), probe_env)
    ascend_summary, ascend_evidence, ascend_details = summarize_ascend_runtime(system_layer, cann_input, probe_env_source, probe_env_error)
    cann_summary, cann_evidence, cann_details = summarize_cann_version(cann_version_info, system_layer, cann_input)
    checks.append(
        make_check(
            "ascend-runtime",
            "ok" if (system_layer.get("ascend_env_active") or system_layer.get("ascend_env_script_present") or cann_input) else "warn",
            ascend_summary,
            evidence=ascend_evidence,
            details=ascend_details,
        )
    )
    checks.append(
        make_check(
            "cann-version",
            "ok" if cann_version_info.get("cann_version") else "warn",
            cann_summary,
            evidence=cann_evidence,
            details=cann_details,
        )
    )

    current_confirmation = confirmation_state.get("current_confirmation") if isinstance(confirmation_state.get("current_confirmation"), dict) else {}
    current_label = str(current_confirmation.get("label") or "the next readiness field")
    gate_pending = list(confirmation_state.get("gate_pending_fields") or [])
    checks.append(
        make_check(
            "confirmation-needed",
            "warn",
            f"final readiness verification is waiting for confirmation of: {current_label}",
        )
    )
    checks.append(
        make_check(
            "runtime-smoke",
            "skipped",
            f"near-launch readiness validation is deferred until {current_label} is confirmed",
        )
    )

    warnings = [item["summary"] for item in checks if item["status"] == "warn"]
    evidence_summary = {
        "target_candidates": scan.get("target_candidates"),
        "framework_candidates": scan.get("framework_candidates"),
        "launcher_candidates": scan.get("launcher_candidates"),
        "selected_runtime_environment": selected_env,
        "assets": profile.get("assets"),
        "hf_cache_layout": ((scan.get("asset_catalog") or {}).get("cache_layout") if isinstance(scan.get("asset_catalog"), dict) else {}),
        "cann_version": cann_version_info.get("cann_version"),
        "cann_source": cann_version_info.get("cann_version_source"),
        "cann_path": profile.get("cann_path") or system_layer.get("cann_path_input"),
        "ascend_env_script_path": system_layer.get("ascend_env_script_path"),
        "ascend_env_candidate_paths": system_layer.get("ascend_env_candidate_paths"),
        "ascend_env_selection_source": system_layer.get("ascend_env_selection_source"),
        "cann_version_file": cann_version_info.get("cann_version_file"),
        "uses_llamafactory": profile.get("uses_llamafactory"),
        "required_packages": profile.get("required_packages"),
        "package_versions": {},
        "package_errors": {},
        "import_errors": {},
        "package_version_probe_error": None,
        "compatibility": None,
    }
    return {
        "status": "NEEDS_CONFIRMATION",
        "can_run": False,
        "summary": f"Workspace scan is complete. Confirm {current_label} before the readiness scan can continue.",
        "next_action": f"Choose one numbered option for {current_label}, apply that confirmation on the next run, and continue to the next readiness check.",
        "checks": checks,
        "missing_items": [],
        "warnings": warnings,
        "evidence_summary": evidence_summary,
        "probe_environment_source": probe_env_source,
        "probe_environment_error": probe_env_error,
        "cann_version_info": cann_version_info,
        "system_layer": system_layer,
    }


def validate_profile(scan: Dict[str, object], profile: Dict[str, object], root: Path) -> Dict[str, object]:
    checks: List[Dict[str, object]] = []
    selected_env = profile.get("selected_environment")
    selected_python = selected_env.get("python_path") if selected_env else None
    assets = profile.get("assets") if isinstance(profile.get("assets"), dict) else {}

    cann_input = profile.get("cann_path")
    system_layer = detect_ascend_runtime({"cann_path": cann_input})
    selected_runtime_env = build_selected_runtime_environment(selected_env)
    probe_env, probe_env_source, probe_env_error = resolve_runtime_environment(system_layer, selected_runtime_env)
    cann_version_info = detect_cann_version(cann_input, system_layer.get("ascend_env_script_path"), probe_env)
    ascend_summary, ascend_evidence, ascend_details = summarize_ascend_runtime(system_layer, cann_input, probe_env_source, probe_env_error)
    cann_summary, cann_evidence, cann_details = summarize_cann_version(cann_version_info, system_layer, cann_input)

    checks.append(make_check("target-selection", "ok" if profile.get("target") else "block", f"target selected: {profile.get('target')}" if profile.get("target") else "target is unresolved"))
    checks.append(make_check("launcher-selection", "ok" if profile.get("launcher") else "block", f"launcher selected: {profile.get('launcher')}" if profile.get("launcher") else "launcher is unresolved"))
    if selected_env:
        env_status = str(selected_env.get("status"))
        checks.append(make_check("python-environment", "ok" if env_status == "selected" else "block", str(selected_env.get("reason") or "environment selected"), evidence=[str(selected_env.get("python_path") or selected_env.get("env_root") or "")]))
    else:
        checks.append(make_check("python-environment", "block", "runtime environment is unresolved"))
    checks.append(make_check("framework-selection", "ok" if profile.get("framework") else "block", f"framework selected: {profile.get('framework')}" if profile.get("framework") else "framework is unresolved"))
    checks.append(make_check("ascend-runtime", "ok" if (system_layer.get("ascend_env_active") or system_layer.get("ascend_env_script_present") or cann_input) else "warn", ascend_summary, evidence=ascend_evidence, details=ascend_details))
    checks.append(make_check("cann-version", "ok" if cann_version_info.get("cann_version") else "warn", cann_summary, evidence=cann_evidence, details=cann_details))

    framework_packages = FRAMEWORK_PACKAGES.get(str(profile.get("framework") or ""), [])
    import_probes, import_errors, import_error = probe_imports(profile.get("required_packages") or [], selected_python, probe_env)
    package_versions, package_errors, version_error = probe_package_versions(profile.get("required_packages") or [], selected_python, probe_env)
    missing_framework_imports = [name for name in framework_packages if not import_probes.get(name)]
    missing_runtime_imports = [name for name in (profile.get("runtime_imports") or []) if not import_probes.get(name)]

    framework_summary = "framework packages are importable"
    if not framework_packages:
        framework_summary = "framework path has no package probe"
    elif missing_framework_imports:
        framework_summary = f"missing framework imports: {summarize_import_failures(missing_framework_imports, import_errors)}"

    runtime_summary = "runtime imports are available"
    if missing_runtime_imports:
        runtime_summary = f"missing runtime imports: {summarize_import_failures(missing_runtime_imports, import_errors)}"

    checks.append(
        make_check(
            "framework-importability",
            "ok" if not missing_framework_imports and framework_packages else ("warn" if not framework_packages else "block"),
            framework_summary,
            evidence=[f"{name}={import_probes.get(name)}" for name in framework_packages] + [f"{name}: {import_errors.get(name)}" for name in missing_framework_imports if import_errors.get(name)],
            probe_error=import_error,
            import_errors={name: import_errors.get(name) for name in missing_framework_imports if import_errors.get(name)},
        )
    )
    checks.append(
        make_check(
            "runtime-dependencies",
            "ok" if not missing_runtime_imports else "block",
            runtime_summary,
            evidence=[f"{name}={import_probes.get(name)}" for name in (profile.get("runtime_imports") or [])] + [f"{name}: {import_errors.get(name)}" for name in missing_runtime_imports if import_errors.get(name)],
            probe_error=import_error,
            import_errors={name: import_errors.get(name) for name in missing_runtime_imports if import_errors.get(name)},
        )
    )

    launcher_status, launcher_summary = launcher_ready(str(profile.get("launcher") or ""), selected_env, import_probes)
    checks.append(make_check("launcher-readiness", launcher_status, launcher_summary))

    entry_path = resolve_optional_path(str(profile.get("entry_script") or ""), root)
    if entry_path and entry_path.exists():
        checks.append(make_check("workspace-entry-script", "ok", "entry script path exists.", evidence=[str(entry_path)]))
    else:
        checks.append(make_check("workspace-entry-script", "block", "entry script path is required but unresolved.", evidence=[str(entry_path or profile.get("entry_script") or "")]))

    for kind in ASSET_KINDS:
        asset_data = assets.get(kind) if isinstance(assets.get(kind), dict) else {}
        if kind == "checkpoint" and not asset_data:
            continue
        checks.append(validate_asset_selection(kind, asset_data, root, launcher=str(profile.get("launcher") or "")))

    compatibility = None
    if str(profile.get("framework")) in {"mindspore", "pta"}:
        compatibility = assess_installed_framework_compatibility(str(profile.get("framework")), cann_version_info.get("cann_version"), selected_env.get("python_version") if selected_env else None, {name: package_versions.get(name) for name in FRAMEWORK_PACKAGES.get(str(profile.get("framework")), [])})
        compat_status = compatibility.get("status")
        compat_summary, compat_evidence, compat_details = summarize_framework_compatibility(compatibility)
        if compat_status == "compatible":
            checks.append(make_check("framework-compatibility", "ok", compat_summary, evidence=compat_evidence, details=compat_details))
        elif compat_status == "incompatible":
            checks.append(make_check("framework-compatibility", "block", compat_summary, evidence=compat_evidence, details=compat_details))
        elif compat_status:
            checks.append(make_check("framework-compatibility", "warn", compat_summary, evidence=compat_evidence, details=compat_details))

    fields_needing_confirmation = [name for name, item in (profile.get("confirmed_fields") or {}).items() if isinstance(item, dict) and item.get("value") not in {None, ""} and not item.get("confirmed", False)]
    if fields_needing_confirmation:
        checks.append(make_check("confirmation-needed", "warn", f"user confirmation is still recommended for: {', '.join(fields_needing_confirmation)}"))

    critical_blockers = {
        "target-selection",
        "launcher-selection",
        "python-environment",
        "framework-selection",
        "framework-importability",
        "framework-compatibility",
        "runtime-dependencies",
        "launcher-readiness",
        "workspace-entry-script",
    }
    critical_blockers.update({f"workspace-{kind}-asset" for kind in asset_kinds_with_validation_gate()})
    has_blocker = any(item["status"] == "block" and item["id"] in critical_blockers for item in checks)
    checks.append(make_check("runtime-smoke", "ok" if not has_blocker else "block", "near-launch readiness checks passed" if not has_blocker else "near-launch readiness checks found hard blockers"))

    blockers = [item["summary"] for item in checks if item["status"] == "block"]
    warnings = [item["summary"] for item in checks if item["status"] == "warn"]
    can_run = not blockers and any(item["id"] == "runtime-smoke" and item["status"] == "ok" for item in checks)
    if blockers:
        status = "BLOCKED"
        summary = "Workspace cannot start the selected workflow yet because required readiness checks failed."
        next_action = "Review blockers, confirm the intended runtime values, and rerun new-readiness-agent after the workspace changes."
    elif warnings:
        status = "WARN"
        summary = "Workspace is close to runnable, but confidence gaps or unresolved details remain."
        next_action = "Review warnings, confirm any remaining runtime choices, and rerun new-readiness-agent if needed."
    else:
        status = "READY"
        summary = "Workspace is ready for the selected local single-machine workflow."
        next_action = "Use the selected runtime environment and launch command when you are ready to start the real workload."

    evidence_summary = {
        "target_candidates": scan.get("target_candidates"),
        "framework_candidates": scan.get("framework_candidates"),
        "launcher_candidates": scan.get("launcher_candidates"),
        "selected_runtime_environment": selected_env,
        "assets": assets,
        "hf_cache_layout": ((scan.get("asset_catalog") or {}).get("cache_layout") if isinstance(scan.get("asset_catalog"), dict) else {}),
        "cann_version": cann_version_info.get("cann_version"),
        "cann_source": cann_version_info.get("cann_version_source"),
        "cann_path": profile.get("cann_path") or system_layer.get("cann_path_input"),
        "ascend_env_script_path": system_layer.get("ascend_env_script_path"),
        "ascend_env_candidate_paths": system_layer.get("ascend_env_candidate_paths"),
        "ascend_env_selection_source": system_layer.get("ascend_env_selection_source"),
        "cann_version_file": cann_version_info.get("cann_version_file"),
        "uses_llamafactory": profile.get("uses_llamafactory"),
        "required_packages": profile.get("required_packages"),
        "package_versions": package_versions,
        "package_errors": package_errors,
        "import_errors": import_errors,
        "package_version_probe_error": version_error,
        "compatibility": compatibility,
    }

    return {
        "status": status,
        "can_run": can_run,
        "summary": summary,
        "next_action": next_action,
        "checks": checks,
        "missing_items": blockers,
        "warnings": warnings,
        "evidence_summary": evidence_summary,
        "probe_environment_source": probe_env_source,
        "probe_environment_error": probe_env_error,
        "cann_version_info": cann_version_info,
        "system_layer": system_layer,
    }


def build_run_state(root: Path, args: object) -> Dict[str, object]:
    scan = analyze_workspace(root, args)
    profile = finalize_profile(scan, root, args)
    confirmation = profile.get("confirmation_state") if isinstance(profile.get("confirmation_state"), dict) else build_confirmation_state(scan, profile)
    validation = validate_profile(scan, profile, root) if confirmation.get("ready_for_validation") else build_pending_validation(scan, profile, root)
    return {
        "scan": scan,
        "profile": profile,
        "confirmation": confirmation,
        "validation": validation,
    }
