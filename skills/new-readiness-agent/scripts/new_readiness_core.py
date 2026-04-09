#!/usr/bin/env python3
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from ascend_compat import assess_installed_framework_compatibility
from asset_discovery import discover_asset_catalog
from asset_schema import asset_locator_summary, make_selected_asset, rank_asset_candidates
from asset_validation import validate_asset_selection
from environment_selection import (
    build_environment_candidates,
    resolve_optional_path,
    split_command,
)
from runtime_env import detect_ascend_runtime, detect_cann_version, resolve_runtime_environment


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
VALIDATION_GATE_FIELDS = (
    "target",
    "launcher",
    "framework",
    "runtime_environment",
    "entry_script",
    "config_asset",
    "model_asset",
    "dataset_asset",
    "cann_path",
    "launch_command",
)
CONFIRMATION_SEQUENCE = (
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
    {
        "field": "config_asset",
        "label": "Config Asset",
        "candidate_key": "asset:config",
        "allow_free_text": True,
        "manual_hint": "Use a detected asset option, or enter local:/path, inline_config, none, or unknown.",
        "prompt": "Confirm how this workspace satisfies its runtime configuration.",
    },
    {
        "field": "model_asset",
        "label": "Model Asset",
        "candidate_key": "asset:model",
        "allow_free_text": True,
        "manual_hint": "Use a detected asset option, or enter local:/path, hf_hub:repo_id, hf_cache:/path, or unknown.",
        "prompt": "Confirm how this workspace satisfies the model requirement.",
    },
    {
        "field": "dataset_asset",
        "label": "Dataset Asset",
        "candidate_key": "asset:dataset",
        "allow_free_text": True,
        "manual_hint": "Use a detected asset option, or enter local:/path, hf_hub:repo_id, hf_cache:/path, script_managed_remote:repo_id, or unknown.",
        "prompt": "Confirm how this workspace satisfies the dataset requirement.",
    },
    {
        "field": "checkpoint_asset",
        "label": "Checkpoint Asset",
        "candidate_key": "asset:checkpoint",
        "allow_free_text": True,
        "manual_hint": "Use a detected checkpoint option, or enter local:/path, none, or unknown.",
        "prompt": "Confirm whether this workspace depends on a checkpoint before launch.",
    },
    {
        "field": "cann_path",
        "label": "CANN / set_env.sh",
        "candidate_key": "cann_candidates",
        "allow_free_text": True,
        "manual_hint": "Provide a CANN root or set_env.sh path if you already know it.",
        "prompt": "Confirm the CANN or Ascend environment path for this workspace.",
    },
    {
        "field": "launch_command",
        "label": "Launch Command",
        "candidate_key": "launch_command_candidates",
        "allow_free_text": True,
        "manual_hint": "Provide the launch command template if the detected candidates are incomplete.",
        "prompt": "Confirm the launch command template that represents the real workload entry.",
    },
)
PROBE_CODE = """
import importlib.util
import json
import sys

mode = sys.argv[1]
payload = json.loads(sys.argv[2])

if mode == "import":
    packages = payload.get("packages", [])
    result = {"imports": {}, "errors": {}}
    for name in packages:
        try:
            __import__(name)
            result["imports"][name] = True
        except Exception as exc:
            result["imports"][name] = False
            result["errors"][name] = f"{type(exc).__name__}: {exc}"
    print(json.dumps(result))
elif mode == "package_versions":
    try:
        from importlib import metadata as importlib_metadata
    except ImportError:
        import importlib_metadata
    packages = payload.get("packages", [])
    result = {"versions": {}, "errors": {}}
    for name in packages:
        try:
            module = __import__(name)
            version = getattr(module, "__version__", None)
            if version is None:
                candidates = [name]
                dashed_name = name.replace("_", "-")
                if dashed_name not in candidates:
                    candidates.append(dashed_name)
                for candidate in candidates:
                    try:
                        version = importlib_metadata.version(candidate)
                        break
                    except Exception:
                        continue
            result["versions"][name] = version
        except Exception as exc:
            result["versions"][name] = None
            result["errors"][name] = f"{type(exc).__name__}: {exc}"
    print(json.dumps(result))
else:
    print(json.dumps({"error": f"unknown mode: {mode}"}))
"""


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


def choose_top_candidate(items: List[Dict[str, object]]) -> Optional[Dict[str, object]]:
    if not items:
        return None
    items.sort(key=lambda item: float(item.get("confidence") or 0.0), reverse=True)
    return items[0]


def merge_catalog_candidates(field_name: str, detected_candidates: List[Dict[str, object]]) -> List[Dict[str, object]]:
    merged = list(detected_candidates)
    seen_values = {item.get("value") for item in detected_candidates}
    for value, label in CATALOG_FIELD_OPTIONS.get(field_name, []):
        if value in seen_values:
            continue
        merged.append(candidate(value, label, "catalog", 0.18))
    return dedupe_candidates(merged)


def looks_like_local_path(value: str) -> bool:
    if not value:
        return False
    path = str(value)
    return (
        path.startswith(".")
        or path.startswith("/")
        or path.startswith("\\")
        or path.startswith("~")
        or "\\" in path
        or path.endswith((".py", ".sh", ".yaml", ".yml", ".json", ".ckpt", ".pt", ".bin"))
    )

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


def parse_confirmation_overrides(raw_items: Optional[List[str]]) -> Dict[str, str]:
    overrides: Dict[str, str] = {}
    if not raw_items:
        return overrides
    for item in raw_items:
        if "=" not in str(item):
            continue
        field_name, raw_value = str(item).split("=", 1)
        field_name = field_name.strip()
        if not field_name:
            continue
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
                "label": "unknown / not sure",
                "confidence": 0.0,
                "selection_source": "manual",
            }
        )
    for index, option in enumerate(options, start=1):
        option["index"] = index
    return options


def load_cached_confirmation(root: Path) -> Dict[str, object]:
    confirmation_path = root / "runs" / "latest" / "new-readiness-agent" / "confirmation-latest.json"
    if not confirmation_path.exists():
        return {}
    try:
        payload = json.loads(confirmation_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    return payload


def asset_bundle(scan: Dict[str, object], kind: str) -> Dict[str, object]:
    catalog = scan.get("asset_catalog") if isinstance(scan.get("asset_catalog"), dict) else {}
    assets = catalog.get("assets") if isinstance(catalog.get("assets"), dict) else {}
    bundle = assets.get(kind)
    return bundle if isinstance(bundle, dict) else {"requirement": {"kind": kind, "required": False, "reason": ""}, "candidates": []}


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
    asset_candidates = rank_asset_candidates(bundle.get("candidates") or [])
    for candidate_item in asset_candidates:
        options.append(
            {
                "value": candidate_item.get("id"),
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
            "label": "unknown / not sure",
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
        cache_path = value.split(":", 1)[1].strip()
        return make_selected_asset(kind, requirement, source_type="hf_cache", locator={"cache_path": cache_path}, selection_source="manual_confirmation")
    if value.startswith("hf_hub:"):
        repo_id = value.split(":", 1)[1].strip()
        locator: Dict[str, object] = {"repo_id": repo_id}
        if kind == "dataset":
            locator["split"] = "train"
        return make_selected_asset(kind, requirement, source_type="hf_hub", locator=locator, selection_source="manual_confirmation")
    if value.startswith("script_managed_remote:"):
        repo_id = value.split(":", 1)[1].strip()
        locator = {"repo_id": repo_id}
        if kind == "dataset":
            locator["split"] = "train"
        return make_selected_asset(kind, requirement, source_type="script_managed_remote", locator=locator, selection_source="manual_confirmation")
    if looks_like_local_path(value):
        return make_selected_asset(kind, requirement, source_type="local_path", locator={"path": value}, selection_source="manual_confirmation")
    if kind in {"model", "dataset"} and re.match(r"^[^/\s]+/[^/\s]+$", value):
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
        manual_asset = infer_manual_asset(kind, requirement, confirmation_override)
        return {
            "value": confirmation_override,
            "source": "manual_confirmation",
            "confirmed": True,
            "asset": manual_asset,
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
    cached_item = None
    if isinstance(cached_fields, dict):
        probe_item = cached_fields.get(field_name)
        if isinstance(probe_item, dict):
            cached_item = probe_item

    if confirmation_override is not None:
        if confirmation_override == "__unknown__":
            return {"value": None, "source": "manual_confirmation", "confirmed": True}
        return {"value": confirmation_override, "source": "manual_confirmation", "confirmed": True}
    if explicit_value not in {None, ""}:
        return {"value": explicit_value, "source": "explicit_input", "confirmed": True}
    if isinstance(cached_item, dict) and bool(cached_item.get("confirmed", False)):
        return {
            "value": cached_item.get("value"),
            "source": "cached_confirmation",
            "confirmed": True,
        }
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
                return {"candidate": candidate_item, "source": "explicit_input", "confirmed": True}
            if explicit_env_value and candidate_item.get("env_root") == explicit_env_value:
                return {"candidate": candidate_item, "source": "explicit_input", "confirmed": True}

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
                return {"candidate": candidate_item, "source": "cached_confirmation", "confirmed": cached_confirmed}
            if cached_python_value and candidate_item.get("python_path") == cached_python_value:
                return {"candidate": candidate_item, "source": "cached_confirmation", "confirmed": cached_confirmed}
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


def run_json_probe_with_python(
    python_path: Path,
    mode: str,
    payload: Dict[str, object],
    probe_env: Optional[Dict[str, str]],
) -> Tuple[Dict[str, object], Optional[str]]:
    launcher = f"exec({PROBE_CODE!r})"
    try:
        completed = subprocess.run(
            [str(python_path), "-c", launcher, mode, json.dumps(payload)],
            check=True,
            text=True,
            capture_output=True,
            timeout=10,
            env=probe_env,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {}, str(exc)
    try:
        result = json.loads(completed.stdout.strip() or "{}")
    except json.JSONDecodeError:
        return {}, "probe returned non-JSON output"
    if not isinstance(result, dict):
        return {}, "probe returned a non-object payload"
    return result, None


def probe_imports(packages: List[str], python_path: Optional[str], probe_env: Optional[Dict[str, str]]) -> Tuple[Dict[str, bool], Dict[str, str], Optional[str]]:
    if not packages:
        return {}, {}, None
    if not python_path:
        return {package: False for package in packages}, {}, "selected python is unavailable"
    result, error = run_json_probe_with_python(Path(python_path), "import", {"packages": packages}, probe_env)
    if error:
        return {package: False for package in packages}, {}, error
    imports = result.get("imports") if isinstance(result.get("imports"), dict) else result
    errors = result.get("errors") if isinstance(result.get("errors"), dict) else {}
    normalized_imports = {package: bool(imports.get(package)) for package in packages}
    normalized_errors = {str(key): str(value) for key, value in errors.items()}
    return normalized_imports, normalized_errors, None


def probe_package_versions(packages: List[str], python_path: Optional[str], probe_env: Optional[Dict[str, str]]) -> Tuple[Dict[str, Optional[str]], Dict[str, str], Optional[str]]:
    if not packages or not python_path:
        return {}, {}, None
    result, error = run_json_probe_with_python(Path(python_path), "package_versions", {"packages": packages}, probe_env)
    if error:
        return {}, {}, error
    versions = result.get("versions") if isinstance(result.get("versions"), dict) else {}
    errors = result.get("errors") if isinstance(result.get("errors"), dict) else {}
    normalized_versions = {
        package: str(versions.get(package)).strip() if isinstance(versions.get(package), str) and str(versions.get(package)).strip() else None
        for package in packages
    }
    normalized_errors = {str(key): str(value) for key, value in errors.items()}
    return normalized_versions, normalized_errors, None


def make_check(check_id: str, status: str, summary: str, evidence: Optional[List[str]] = None, **extra: object) -> Dict[str, object]:
    payload: Dict[str, object] = {
        "id": check_id,
        "status": status,
        "summary": summary,
        "evidence": evidence or [],
    }
    payload.update(extra)
    return payload


def summarize_framework_compatibility(compatibility: Dict[str, object]) -> Tuple[str, List[str], Dict[str, object]]:
    status = str(compatibility.get("status") or "unresolved")
    reason = str(compatibility.get("reason") or "").strip()
    installed_versions = compatibility.get("installed_versions") if isinstance(compatibility.get("installed_versions"), dict) else {}
    recommended_specs = [str(item) for item in (compatibility.get("recommended_package_specs") or []) if str(item).strip()]
    compatible_rows = compatibility.get("compatible_rows") if isinstance(compatibility.get("compatible_rows"), list) else []
    matched_row = compatibility.get("matched_row") if isinstance(compatibility.get("matched_row"), dict) else None

    installed_tokens = [f"{name}={value}" for name, value in installed_versions.items() if value]
    compatible_tokens: List[str] = []
    for row in compatible_rows:
        if not isinstance(row, dict):
            continue
        row_tokens = []
        for key in ("mindspore", "torch", "torch_npu"):
            value = row.get(key)
            if value:
                row_tokens.append(f"{key}=={value}")
        branch = row.get("branch")
        if branch:
            row_tokens.append(f"branch={branch}")
        if row_tokens:
            compatible_tokens.append(", ".join(row_tokens))

    if status == "compatible":
        summary = reason or "framework versions match the local compatibility table"
    elif status == "incompatible":
        expected = "; ".join(compatible_tokens[:3]) if compatible_tokens else ", ".join(recommended_specs)
        expected_suffix = f" Expected one of: {expected}." if expected else ""
        installed_suffix = f" Installed: {', '.join(installed_tokens)}." if installed_tokens else ""
        summary = f"{reason}{installed_suffix}{expected_suffix}".strip()
    else:
        installed_suffix = f" Installed: {', '.join(installed_tokens)}." if installed_tokens else ""
        recommended_suffix = f" Recommended: {', '.join(recommended_specs)}." if recommended_specs else ""
        summary = f"{reason}{installed_suffix}{recommended_suffix}".strip()

    evidence = []
    if installed_tokens:
        evidence.append("installed " + ", ".join(installed_tokens))
    if recommended_specs:
        evidence.append("recommended " + ", ".join(recommended_specs))
    if compatible_tokens:
        evidence.append("compatible rows " + "; ".join(compatible_tokens[:3]))

    details: Dict[str, object] = {
        "compatibility_status": status,
        "reference_status": compatibility.get("reference_status"),
        "installed_versions": installed_versions,
        "recommended_package_specs": recommended_specs,
        "matched_row": matched_row,
        "compatible_rows": compatible_rows,
        "reason": compatibility.get("reason"),
    }
    return summary or f"framework compatibility status: {status}", evidence, details


def summarize_import_failures(package_names: List[str], import_errors: Dict[str, str]) -> str:
    details = []
    for name in package_names:
        error = str(import_errors.get(name) or "").strip()
        if error:
            details.append(f"{name} ({error})")
        else:
            details.append(name)
    return ", ".join(details)


def executable_exists(command_name: str) -> bool:
    return bool(shutil.which(command_name))


def launcher_ready(
    launcher_value: Optional[str],
    selected_candidate: Optional[Dict[str, object]],
    import_probes: Dict[str, bool],
) -> Tuple[str, str]:
    if not launcher_value:
        return "block", "launcher is unresolved"
    if launcher_value == "python":
        if selected_candidate and selected_candidate.get("status") == "selected":
            return "ok", "runtime python is available"
        return "block", "selected runtime python is unavailable"
    if launcher_value == "bash":
        return ("ok", "bash launcher is available") if executable_exists("bash") else ("block", "bash launcher is unavailable")
    if launcher_value == "make":
        return ("ok", "make launcher is available") if executable_exists("make") else ("warn", "make launcher is unavailable in PATH")
    if launcher_value == "msrun":
        return ("ok", "msrun launcher is available") if executable_exists("msrun") else ("warn", "msrun launcher is not visible in PATH")
    if launcher_value == "torchrun":
        return ("ok", "torchrun launcher requirements are present") if import_probes.get("torch") else ("block", "torchrun requires torch in the selected environment")
    if launcher_value == "accelerate":
        return ("ok", "accelerate launcher requirements are present") if import_probes.get("accelerate") else ("block", "accelerate is missing in the selected environment")
    if launcher_value == "deepspeed":
        return ("ok", "deepspeed launcher requirements are present") if import_probes.get("deepspeed") else ("warn", "deepspeed is not importable in the selected environment")
    if launcher_value == "llamafactory-cli":
        if import_probes.get("llamafactory"):
            return "ok", "llamafactory launcher requirements are present"
        if executable_exists("llamafactory-cli"):
            return "ok", "llamafactory-cli executable is available"
        return "block", "llamafactory-cli is unresolved in the selected environment"
    return "warn", f"launcher {launcher_value} has no specialized readiness probe"

def analyze_workspace(root: Path, args: object) -> Dict[str, object]:
    files = list_files(root)
    launcher_candidates = build_launcher_candidates(root, args, files)
    entry_candidates = build_entry_candidates(root, args, files, launcher_candidates)
    config_candidates = build_config_candidates(root, args, files, launcher_candidates)
    dependency_text = collect_dependency_text(files)
    asset_catalog = discover_asset_catalog(
        root=root,
        files=files,
        entry_candidates=entry_candidates,
        config_candidates=config_candidates,
        args=args,
        target_hint=getattr(args, "target", None) if getattr(args, "target", None) != "auto" else None,
    )
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


def confirmation_definition(field_name: str) -> Dict[str, object]:
    for item in CONFIRMATION_SEQUENCE:
        if item["field"] == field_name:
            return dict(item)
    raise KeyError(field_name)


def confirmation_field_is_confirmed(field_name: str, confirmed_fields: Dict[str, object]) -> bool:
    item = confirmed_fields.get(field_name)
    return isinstance(item, dict) and bool(item.get("confirmed", False))


def ranked_candidates(items: List[Dict[str, object]]) -> List[Dict[str, object]]:
    return sorted(items, key=lambda item: float(item.get("confidence") or 0.0), reverse=True)


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
            "label": "unknown / not sure",
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
    for item in CONFIRMATION_SEQUENCE:
        field_name = str(item.get("field"))
        if field_name == "config_asset":
            bundle = assets.get("config") if isinstance(assets.get("config"), dict) else {}
            bundle_candidates = list(bundle.get("candidates") or [])
            bundle_requirement = bundle.get("requirement") if isinstance(bundle.get("requirement"), dict) else {}
            should_include = bool(bundle_candidates) or bool(bundle_requirement.get("required")) or launcher_value == "llamafactory-cli"
            if not should_include:
                continue
        if field_name == "model_asset":
            bundle = assets.get("model") if isinstance(assets.get("model"), dict) else {}
            if not (bool((bundle.get("requirement") or {}).get("required")) or list(bundle.get("candidates") or [])):
                continue
        if field_name == "dataset_asset":
            bundle = assets.get("dataset") if isinstance(assets.get("dataset"), dict) else {}
            if not (bool((bundle.get("requirement") or {}).get("required")) or list(bundle.get("candidates") or [])):
                continue
        if field_name == "checkpoint_asset":
            bundle = assets.get("checkpoint") if isinstance(assets.get("checkpoint"), dict) else {}
            if not list(bundle.get("candidates") or []):
                continue
        sequence.append(dict(item))
    return sequence


def build_field_confirmation_step(scan: Dict[str, object], profile: Dict[str, object], field_name: str, step_number: int, total_steps: int) -> Dict[str, object]:
    definition = confirmation_definition(field_name)
    confirmed_fields = profile.get("confirmed_fields") if isinstance(profile.get("confirmed_fields"), dict) else {}
    field_item = confirmed_fields.get(field_name) if isinstance(confirmed_fields.get(field_name), dict) else {}
    recommended_value = field_item.get("value")

    if field_name == "runtime_environment":
        selected_environment = profile.get("selected_environment") or {}
        options = build_runtime_environment_options(scan)
        recommended_value = field_item.get("value") if isinstance(field_item, dict) else (selected_environment.get("id") if selected_environment else None)
    else:
        candidate_key = str(definition.get("candidate_key"))
        if candidate_key == "cann_candidates":
            candidates = list(scan["cann"]["candidates"])
        elif candidate_key.startswith("asset:"):
            asset_kind = candidate_key.split(":", 1)[1]
            options = build_asset_confirmation_options(asset_bundle(scan, asset_kind), allow_free_text=bool(definition.get("allow_free_text", True)))
            candidates = []
        else:
            candidates = list(scan.get(candidate_key) or [])
        if not candidate_key.startswith("asset:"):
            catalog_key = definition.get("catalog_key")
            if isinstance(catalog_key, str):
                candidates = merge_catalog_candidates(catalog_key, candidates)
            options = build_numbered_options(
                ranked_candidates(candidates),
                allow_free_text=bool(definition.get("allow_free_text", True)),
            )

    for option in options:
        option["recommended"] = option.get("value") == recommended_value

    return {
        "field": field_name,
        "label": definition.get("label"),
        "prompt": definition.get("prompt"),
        "step_number": step_number,
        "total_steps": total_steps,
        "recommended_value": recommended_value,
        "allow_free_text": bool(definition.get("allow_free_text", True)),
        "manual_hint": definition.get("manual_hint"),
        "options": options,
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


def finalize_profile(scan: Dict[str, object], root: Path, args: object) -> Dict[str, object]:
    cached_confirmation = load_cached_confirmation(root)
    confirmation_overrides = parse_confirmation_overrides(getattr(args, "confirm", None))

    target_choice = choose_value("target", getattr(args, "target", None) if getattr(args, "target", None) != "auto" else None, cached_confirmation, list(scan["target_candidates"]), confirmation_overrides.get("target"))
    framework_choice = choose_value("framework", getattr(args, "framework_hint", None) if getattr(args, "framework_hint", None) != "auto" else None, cached_confirmation, list(scan["framework_candidates"]), confirmation_overrides.get("framework"))
    launcher_choice = choose_value("launcher", getattr(args, "launcher_hint", None) if getattr(args, "launcher_hint", None) != "auto" else None, cached_confirmation, list(scan["launcher_candidates"]), confirmation_overrides.get("launcher"))
    entry_choice = choose_value("entry_script", getattr(args, "entry_script", None), cached_confirmation, list(scan["entry_candidates"]), confirmation_overrides.get("entry_script"))
    config_choice = choose_asset("config", cached_confirmation, asset_bundle(scan, "config"), confirmation_overrides.get("config_asset"))
    model_choice = choose_asset("model", cached_confirmation, asset_bundle(scan, "model"), confirmation_overrides.get("model_asset"))
    dataset_choice = choose_asset("dataset", cached_confirmation, asset_bundle(scan, "dataset"), confirmation_overrides.get("dataset_asset"))
    checkpoint_choice = choose_asset("checkpoint", cached_confirmation, asset_bundle(scan, "checkpoint"), confirmation_overrides.get("checkpoint_asset"))
    cann_choice = choose_value("cann_path", getattr(args, "cann_path", None), cached_confirmation, list(scan["cann"]["candidates"]), confirmation_overrides.get("cann_path"))
    command_choice = choose_value("launch_command", getattr(args, "launch_command", None), cached_confirmation, list(scan["launch_command_candidates"]), confirmation_overrides.get("launch_command"))
    extra_context_choice = choose_value("extra_context", getattr(args, "extra_context", None), cached_confirmation, [], confirmation_overrides.get("extra_context"))
    environment_choice = choose_environment(root, args, cached_confirmation, dict(scan["environment"]), confirmation_overrides.get("runtime_environment"))

    selected_launcher_candidate = next((item for item in scan["launcher_candidates"] if item.get("value") == launcher_choice["value"]), None)
    selected_environment_candidate = environment_choice.get("candidate")
    runtime_imports = collect_entry_runtime_imports(str(entry_choice["value"]) if entry_choice.get("value") else None, root)
    uses_llamafactory = bool(
        (selected_launcher_candidate and selected_launcher_candidate.get("uses_llamafactory"))
        or str(command_choice.get("value") or "").lower().find("llamafactory") >= 0
    )

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
        "config_asset": config_choice,
        "model_asset": model_choice,
        "dataset_asset": dataset_choice,
        "checkpoint_asset": checkpoint_choice,
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

    assets = {
        "config": {
            **asset_bundle(scan, "config"),
            "selected": config_choice["asset"],
        },
        "model": {
            **asset_bundle(scan, "model"),
            "selected": model_choice["asset"],
        },
        "dataset": {
            **asset_bundle(scan, "dataset"),
            "selected": dataset_choice["asset"],
        },
        "checkpoint": {
            **asset_bundle(scan, "checkpoint"),
            "selected": checkpoint_choice["asset"],
        },
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
    pending_check("config_asset", "workspace-config-asset", fallback_label="config asset")
    pending_check("model_asset", "workspace-model-asset", fallback_label="model asset")
    pending_check("dataset_asset", "workspace-dataset-asset", fallback_label="dataset asset")
    if "checkpoint_asset" in list(confirmation_state.get("pending_fields") or []) or isinstance(confirmed_fields.get("checkpoint_asset"), dict):
        pending_check("checkpoint_asset", "workspace-checkpoint-asset", fallback_label="checkpoint asset")

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
    probe_env, probe_env_source, probe_env_error = resolve_runtime_environment(system_layer)
    cann_version_info = detect_cann_version(cann_input, system_layer.get("ascend_env_script_path"), probe_env)
    checks.append(
        make_check(
            "ascend-runtime",
            "ok" if (system_layer.get("ascend_env_active") or system_layer.get("ascend_env_script_present") or cann_input) else "warn",
            "Ascend runtime evidence is available." if (system_layer.get("ascend_env_active") or system_layer.get("ascend_env_script_present") or cann_input) else "Ascend runtime evidence is weak or unresolved.",
            evidence=[str(item) for item in (system_layer.get("ascend_env_candidate_paths") or [])[:3]],
            selection_source=probe_env_source,
            error=probe_env_error,
        )
    )
    checks.append(
        make_check(
            "cann-version",
            "ok" if cann_version_info.get("cann_version") else "warn",
            f"CANN version detected: {cann_version_info.get('cann_version')}" if cann_version_info.get("cann_version") else "CANN version is unresolved.",
            evidence=[str(cann_version_info.get("cann_version_file") or "")],
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
    probe_env, probe_env_source, probe_env_error = resolve_runtime_environment(system_layer)
    cann_version_info = detect_cann_version(cann_input, system_layer.get("ascend_env_script_path"), probe_env)

    checks.append(make_check("target-selection", "ok" if profile.get("target") else "block", f"target selected: {profile.get('target')}" if profile.get("target") else "target is unresolved"))
    checks.append(make_check("launcher-selection", "ok" if profile.get("launcher") else "block", f"launcher selected: {profile.get('launcher')}" if profile.get("launcher") else "launcher is unresolved"))
    if selected_env:
        env_status = str(selected_env.get("status"))
        checks.append(make_check("python-environment", "ok" if env_status == "selected" else "block", str(selected_env.get("reason") or "environment selected"), evidence=[str(selected_env.get("python_path") or selected_env.get("env_root") or "")]))
    else:
        checks.append(make_check("python-environment", "block", "runtime environment is unresolved"))
    checks.append(make_check("framework-selection", "ok" if profile.get("framework") else "block", f"framework selected: {profile.get('framework')}" if profile.get("framework") else "framework is unresolved"))
    checks.append(make_check("ascend-runtime", "ok" if (system_layer.get("ascend_env_active") or system_layer.get("ascend_env_script_present") or cann_input) else "warn", "Ascend runtime evidence is available." if (system_layer.get("ascend_env_active") or system_layer.get("ascend_env_script_present") or cann_input) else "Ascend runtime evidence is weak or unresolved.", evidence=[str(item) for item in (system_layer.get("ascend_env_candidate_paths") or [])[:3]], selection_source=probe_env_source, error=probe_env_error))
    checks.append(make_check("cann-version", "ok" if cann_version_info.get("cann_version") else "warn", f"CANN version detected: {cann_version_info.get('cann_version')}" if cann_version_info.get("cann_version") else "CANN version is unresolved.", evidence=[str(cann_version_info.get("cann_version_file") or "")]))

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

    checks.append(validate_asset_selection("config", assets.get("config") if isinstance(assets.get("config"), dict) else {}, root, launcher=str(profile.get("launcher") or "")))
    checks.append(validate_asset_selection("model", assets.get("model") if isinstance(assets.get("model"), dict) else {}, root, launcher=str(profile.get("launcher") or "")))
    checks.append(validate_asset_selection("dataset", assets.get("dataset") if isinstance(assets.get("dataset"), dict) else {}, root, launcher=str(profile.get("launcher") or "")))
    if isinstance(assets.get("checkpoint"), dict):
        checks.append(validate_asset_selection("checkpoint", assets["checkpoint"], root, launcher=str(profile.get("launcher") or "")))

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
        "workspace-config-asset",
        "workspace-model-asset",
        "workspace-dataset-asset",
    }
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
