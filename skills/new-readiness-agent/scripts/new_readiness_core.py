#!/usr/bin/env python3
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from ascend_compat import assess_installed_framework_compatibility
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
PATH_VALUE_KEYS = {
    "config": ("config", "config_file", "config_path"),
    "model": ("model_name_or_path", "model_path", "pretrained_model_name_or_path", "model_dir"),
    "dataset": ("dataset_dir", "dataset_path", "data_dir", "data_path", "train_file", "validation_file", "dataset"),
    "checkpoint": ("checkpoint_path", "resume_from_checkpoint", "load_checkpoint", "ckpt_path"),
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
    "selected_python",
    "selected_env_root",
    "entry_script",
)
PROBE_CODE = """
import importlib.util
import json
import sys

mode = sys.argv[1]
payload = json.loads(sys.argv[2])

if mode == "import":
    packages = payload.get("packages", [])
    print(json.dumps({name: importlib.util.find_spec(name) is not None for name in packages}))
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


def parse_config_values(text: str, keys: Tuple[str, ...]) -> List[str]:
    values: List[str] = []
    if not text:
        return values
    for key in keys:
        pattern = re.compile(rf"(?im)[\"']?{re.escape(key)}[\"']?\s*[:=]\s*[\"']?([^\"'\n]+)")
        for match in pattern.finditer(text):
            value = match.group(1).strip().strip(",")
            if value:
                values.append(value)
    return values


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


def resolve_local_candidate(value: str, root: Path) -> Optional[Path]:
    if not looks_like_local_path(value):
        return None
    return resolve_optional_path(value, root)


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


def build_path_candidates(
    *,
    field_name: str,
    explicit_value: Optional[str],
    root: Path,
    files: List[Path],
    config_candidates: List[Dict[str, object]],
    conventional_names: Tuple[str, ...],
    suffixes: Tuple[str, ...] = (),
) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []
    explicit_path = resolve_optional_path(explicit_value, root)
    if explicit_path:
        results.append(candidate(str(explicit_path), f"explicit {field_name}={explicit_path.name}", "explicit_input", 0.99, exists=explicit_path.exists(), is_local_path=True))

    for name in conventional_names:
        path = root / name
        if path.exists():
            results.append(candidate(str(path), f"workspace {field_name} {name}", "workspace_scan", 0.82, exists=True, is_local_path=True))

    for config_candidate in config_candidates:
        config_path = resolve_optional_path(str(config_candidate.get("value")), root)
        if not config_path or not config_path.exists():
            continue
        text = read_text(config_path)
        for raw_value in parse_config_values(text, PATH_VALUE_KEYS[field_name]):
            local = resolve_local_candidate(raw_value, root)
            if local:
                results.append(candidate(str(local), f"{field_name} from config {config_path.name}", "config_scan", 0.73, exists=local.exists(), is_local_path=True))
            else:
                results.append(candidate(raw_value, f"{field_name} reference from config {config_path.name}", "config_scan", 0.48, exists=False, is_local_path=False))

    for path in files:
        if suffixes and path.suffix.lower() in suffixes:
            results.append(candidate(str(path), f"workspace {field_name} file {path.name}", "workspace_scan", 0.7, exists=True, is_local_path=True))

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
    dataset_candidates: List[Dict[str, object]],
    root: Path,
) -> List[Dict[str, object]]:
    explicit_target = getattr(args, "target", None)
    if explicit_target in {"training", "inference"}:
        return [candidate(explicit_target, f"explicit target={explicit_target}", "explicit_input", 0.99)]

    scores = {"training": 0, "inference": 0}
    evidence = {"training": [], "inference": []}

    if dataset_candidates:
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


def build_confirmation_field(
    field_id: str,
    label: str,
    field_candidates: List[Dict[str, object]],
    *,
    recommended_value: Optional[str],
    allow_free_text: bool = True,
) -> Dict[str, object]:
    options = []
    for item in field_candidates:
        options.append(
            {
                "value": item.get("value"),
                "label": item.get("label"),
                "confidence": item.get("confidence"),
                "selection_source": item.get("selection_source"),
            }
        )
    options.append({"value": "__unknown__", "label": "unknown / not sure", "confidence": 0.0, "selection_source": "manual"})
    return {
        "field": field_id,
        "label": label,
        "recommended_value": recommended_value,
        "allow_free_text": allow_free_text,
        "options": options,
    }


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


def choose_value(
    field_name: str,
    explicit_value: Optional[str],
    cached_confirmation: Dict[str, object],
    field_candidates: List[Dict[str, object]],
) -> Dict[str, object]:
    cached_fields = cached_confirmation.get("confirmed_fields") if isinstance(cached_confirmation.get("confirmed_fields"), dict) else {}
    cached_item = None
    if isinstance(cached_fields, dict):
        probe_item = cached_fields.get(field_name)
        if isinstance(probe_item, dict):
            cached_item = probe_item

    if explicit_value not in {None, ""}:
        return {"value": explicit_value, "source": "explicit_input", "confirmed": True}
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
) -> Dict[str, object]:
    candidates = environment_result.get("candidates") or []
    explicit_python = getattr(args, "selected_python", None)
    explicit_env_root = getattr(args, "selected_env_root", None)
    cached_fields = cached_confirmation.get("confirmed_fields") if isinstance(cached_confirmation.get("confirmed_fields"), dict) else {}

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


def probe_imports(packages: List[str], python_path: Optional[str], probe_env: Optional[Dict[str, str]]) -> Tuple[Dict[str, bool], Optional[str]]:
    if not packages:
        return {}, None
    if not python_path:
        return {package: False for package in packages}, "selected python is unavailable"
    result, error = run_json_probe_with_python(Path(python_path), "import", {"packages": packages}, probe_env)
    if error:
        return {package: False for package in packages}, error
    return {package: bool(result.get(package)) for package in packages}, None


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


def config_is_readable(config_value: Optional[str], root: Path) -> Tuple[bool, Optional[str]]:
    if not config_value:
        return True, None
    config_path = resolve_optional_path(config_value, root)
    if not config_path:
        return False, "config path is unresolved"
    if not config_path.exists():
        return False, f"config file does not exist: {config_path.name}"
    text = read_text(config_path)
    if not text.strip():
        return False, f"config file is empty: {config_path.name}"
    if config_path.suffix.lower() == ".json":
        try:
            json.loads(text)
        except json.JSONDecodeError as exc:
            return False, f"config JSON is invalid: {exc}"
    return True, None


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


def needs_local_asset(field_name: str, target_value: Optional[str], selected_value: Optional[str], field_candidates: List[Dict[str, object]]) -> bool:
    explicit_or_local = any(bool(item.get("is_local_path")) for item in field_candidates) or bool(selected_value and looks_like_local_path(selected_value))
    if field_name == "entry_script":
        return True
    if field_name == "model":
        return explicit_or_local or target_value == "inference"
    if field_name == "dataset":
        return explicit_or_local or target_value == "training"
    return explicit_or_local


def asset_check(field_name: str, selected_value: Optional[str], root: Path, required: bool) -> Dict[str, object]:
    if not selected_value:
        if required:
            return make_check(f"workspace-{field_name}-path", "block", f"{field_name} path is required but unresolved.")
        return make_check(f"workspace-{field_name}-path", "skipped", f"{field_name} path is optional and unresolved.")
    local_path = resolve_optional_path(selected_value, root)
    if local_path and looks_like_local_path(selected_value):
        if local_path.exists():
            return make_check(f"workspace-{field_name}-path", "ok", f"{field_name} path exists.", evidence=[str(local_path)])
        if required:
            return make_check(f"workspace-{field_name}-path", "block", f"{field_name} path does not exist.", evidence=[str(local_path)])
        return make_check(f"workspace-{field_name}-path", "warn", f"{field_name} path does not exist locally.", evidence=[str(local_path)])
    if required:
        return make_check(f"workspace-{field_name}-path", "warn", f"{field_name} looks remote or unresolved: {selected_value}")
    return make_check(f"workspace-{field_name}-path", "skipped", f"{field_name} is not a local path candidate.")


def analyze_workspace(root: Path, args: object) -> Dict[str, object]:
    files = list_files(root)
    launcher_candidates = build_launcher_candidates(root, args, files)
    entry_candidates = build_entry_candidates(root, args, files, launcher_candidates)
    config_candidates = build_config_candidates(root, args, files, launcher_candidates)
    model_candidates = build_path_candidates(
        field_name="model",
        explicit_value=getattr(args, "model_path", None),
        root=root,
        files=files,
        config_candidates=config_candidates,
        conventional_names=("model", "models"),
    )
    dataset_candidates = build_path_candidates(
        field_name="dataset",
        explicit_value=getattr(args, "dataset_path", None),
        root=root,
        files=files,
        config_candidates=config_candidates,
        conventional_names=("dataset", "data"),
    )
    checkpoint_candidates = build_path_candidates(
        field_name="checkpoint",
        explicit_value=getattr(args, "checkpoint_path", None),
        root=root,
        files=files,
        config_candidates=config_candidates,
        conventional_names=("checkpoints",),
        suffixes=(".ckpt", ".pt", ".bin"),
    )
    dependency_text = collect_dependency_text(files)
    target_candidates = build_target_candidates(args, entry_candidates, launcher_candidates, config_candidates, dataset_candidates, root)
    framework_candidates = build_framework_candidates(args, entry_candidates, launcher_candidates, config_candidates, dependency_text, root)
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
        "model_candidates": model_candidates,
        "dataset_candidates": dataset_candidates,
        "checkpoint_candidates": checkpoint_candidates,
        "target_candidates": target_candidates,
        "framework_candidates": framework_candidates,
        "environment": environment_result,
        "cann": cann_result,
        "dependency_text": dependency_text,
    }


def build_confirmation_state(profile: Dict[str, object]) -> Dict[str, object]:
    confirmed_fields = profile.get("confirmed_fields") if isinstance(profile.get("confirmed_fields"), dict) else {}
    pending_fields: List[str] = []
    recommended_fields: List[str] = []
    unresolved_fields: List[str] = []
    gate_pending_fields: List[str] = []

    for name, item in confirmed_fields.items():
        if not isinstance(item, dict):
            continue
        if item.get("confirmed", False):
            continue
        pending_fields.append(name)
        if item.get("value") in {None, ""}:
            unresolved_fields.append(name)
        else:
            recommended_fields.append(name)

    for field_name in VALIDATION_GATE_FIELDS:
        item = confirmed_fields.get(field_name)
        if not isinstance(item, dict) or not item.get("confirmed", False):
            gate_pending_fields.append(field_name)

    return {
        "required": bool(gate_pending_fields),
        "ready_for_validation": not gate_pending_fields,
        "pending_fields": pending_fields,
        "gate_pending_fields": gate_pending_fields,
        "recommended_fields": recommended_fields,
        "unresolved_fields": unresolved_fields,
    }


def build_runtime_environment_field(
    candidates: List[Dict[str, object]],
    selected_candidate: Optional[Dict[str, object]],
) -> Dict[str, object]:
    return {
        "field": "runtime_environment",
        "label": "Python / Environment",
        "recommended_value": selected_candidate.get("id") if selected_candidate else None,
        "allow_free_text": True,
        "options": [
            {
                "value": item.get("id"),
                "label": item.get("label"),
                "confidence": item.get("confidence"),
                "selection_source": item.get("selection_source"),
                "python_path": item.get("python_path"),
                "env_root": item.get("env_root"),
            }
            for item in candidates
        ] + [{"value": "__unknown__", "label": "unknown / not sure", "confidence": 0.0, "selection_source": "manual"}],
    }


def finalize_profile(scan: Dict[str, object], root: Path, args: object) -> Dict[str, object]:
    cached_confirmation = load_cached_confirmation(root)

    target_choice = choose_value("target", getattr(args, "target", None) if getattr(args, "target", None) != "auto" else None, cached_confirmation, list(scan["target_candidates"]))
    framework_choice = choose_value("framework", getattr(args, "framework_hint", None) if getattr(args, "framework_hint", None) != "auto" else None, cached_confirmation, list(scan["framework_candidates"]))
    launcher_choice = choose_value("launcher", getattr(args, "launcher_hint", None) if getattr(args, "launcher_hint", None) != "auto" else None, cached_confirmation, list(scan["launcher_candidates"]))
    entry_choice = choose_value("entry_script", getattr(args, "entry_script", None), cached_confirmation, list(scan["entry_candidates"]))
    config_choice = choose_value("config_path", getattr(args, "config_path", None), cached_confirmation, list(scan["config_candidates"]))
    model_choice = choose_value("model_path", getattr(args, "model_path", None), cached_confirmation, list(scan["model_candidates"]))
    dataset_choice = choose_value("dataset_path", getattr(args, "dataset_path", None), cached_confirmation, list(scan["dataset_candidates"]))
    checkpoint_choice = choose_value("checkpoint_path", getattr(args, "checkpoint_path", None), cached_confirmation, list(scan["checkpoint_candidates"]))
    cann_choice = choose_value("cann_path", getattr(args, "cann_path", None), cached_confirmation, list(scan["cann"]["candidates"]))
    launch_command_candidates = [item for item in scan["launcher_candidates"] if item.get("command_template")]
    command_choice = choose_value("launch_command", getattr(args, "launch_command", None), cached_confirmation, launch_command_candidates)
    extra_context_choice = choose_value("extra_context", getattr(args, "extra_context", None), cached_confirmation, [])
    environment_choice = choose_environment(root, args, cached_confirmation, dict(scan["environment"]))

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
        "config_path": config_choice,
        "model_path": model_choice,
        "dataset_path": dataset_choice,
        "checkpoint_path": checkpoint_choice,
        "cann_path": cann_choice,
        "launch_command": command_choice,
        "extra_context": extra_context_choice,
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

    target_form_candidates = merge_catalog_candidates("target", list(scan["target_candidates"]))
    launcher_form_candidates = merge_catalog_candidates("launcher", list(scan["launcher_candidates"]))
    framework_form_candidates = merge_catalog_candidates("framework", list(scan["framework_candidates"]))
    confirmation_form = {
        "schema_version": "new-readiness-agent/form/0.1",
        "mode": "single",
        "groups": [
            {
                "id": "runtime-shape",
                "label": "Target / Launcher / Framework",
                "fields": [
                    build_confirmation_field("target", "Target", target_form_candidates, recommended_value=target_choice["value"]),
                    build_confirmation_field("launcher", "Launcher", launcher_form_candidates, recommended_value=launcher_choice["value"]),
                    build_confirmation_field("framework", "Framework", framework_form_candidates, recommended_value=framework_choice["value"]),
                ],
            },
            {
                "id": "runtime-env",
                "label": "Python / Environment / CANN",
                "fields": [
                    build_runtime_environment_field(list(scan["environment"]["candidates"]), selected_environment_candidate),
                    build_confirmation_field("cann_path", "CANN / set_env.sh", list(scan["cann"]["candidates"]), recommended_value=cann_choice["value"]),
                ],
            },
            {
                "id": "workspace-inputs",
                "label": "Entrypoint / Config / Assets",
                "fields": [
                    build_confirmation_field("entry_script", "Entry Script", list(scan["entry_candidates"]), recommended_value=entry_choice["value"]),
                    build_confirmation_field("config_path", "Config", list(scan["config_candidates"]), recommended_value=config_choice["value"]),
                    build_confirmation_field("model_path", "Model Path", list(scan["model_candidates"]), recommended_value=model_choice["value"]),
                    build_confirmation_field("dataset_path", "Dataset Path", list(scan["dataset_candidates"]), recommended_value=dataset_choice["value"]),
                    build_confirmation_field("checkpoint_path", "Checkpoint Path", list(scan["checkpoint_candidates"]), recommended_value=checkpoint_choice["value"]),
                    build_confirmation_field("launch_command", "Launch Command", launch_command_candidates, recommended_value=command_choice["value"]),
                ],
            },
        ],
        "extra_context": {
            "field": "extra_context",
            "label": "Additional context",
            "allow_free_text": True,
            "value": extra_context_choice["value"],
        },
    }

    confirmation_state = build_confirmation_state({"confirmed_fields": confirmed_fields})
    return {
        "target": target_choice["value"],
        "framework": framework_choice["value"],
        "launcher": launcher_choice["value"],
        "entry_script": entry_choice["value"],
        "config_path": config_choice["value"],
        "model_path": model_choice["value"],
        "dataset_path": dataset_choice["value"],
        "checkpoint_path": checkpoint_choice["value"],
        "cann_path": cann_choice["value"],
        "launch_command": command_choice["value"] or (selected_launcher_candidate.get("command_template") if selected_launcher_candidate else None),
        "extra_context": extra_context_choice["value"],
        "selected_environment": selected_environment_candidate,
        "confirmed_fields": confirmed_fields,
        "confirmation_form": confirmation_form,
        "required_packages": required_packages,
        "runtime_imports": runtime_imports,
        "uses_llamafactory": uses_llamafactory,
        "selected_launcher_candidate": selected_launcher_candidate,
        "cached_confirmation": cached_confirmation,
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
        if confirmed and value not in {None, ""}:
            checks.append(make_check(check_id, "ok", f"{fallback_label} confirmed: {value}"))
            return
        if value in {None, ""}:
            checks.append(make_check(check_id, "warn", f"{fallback_label} still needs a user selection."))
            return
        checks.append(make_check(check_id, "warn", f"{fallback_label} recommendation is ready, but still needs user confirmation: {value}"))

    pending_check("target", "target-selection", fallback_label="target")
    pending_check("launcher", "launcher-selection", fallback_label="launcher")
    pending_check("framework", "framework-selection", fallback_label="framework")
    pending_check("entry_script", "workspace-entry_script-path", fallback_label="entry script")

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

    gate_pending = list(confirmation_state.get("gate_pending_fields") or [])
    checks.append(
        make_check(
            "confirmation-needed",
            "warn",
            f"final readiness verification is waiting for user confirmation of: {', '.join(gate_pending)}",
        )
    )
    checks.append(
        make_check(
            "runtime-smoke",
            "skipped",
            "near-launch readiness validation is deferred until the single confirmation form is completed",
        )
    )

    warnings = [item["summary"] for item in checks if item["status"] == "warn"]
    evidence_summary = {
        "target_candidates": scan.get("target_candidates"),
        "framework_candidates": scan.get("framework_candidates"),
        "launcher_candidates": scan.get("launcher_candidates"),
        "selected_runtime_environment": selected_env,
        "cann_version": cann_version_info.get("cann_version"),
        "cann_source": cann_version_info.get("cann_version_source"),
        "uses_llamafactory": profile.get("uses_llamafactory"),
        "required_packages": profile.get("required_packages"),
        "package_versions": {},
        "package_errors": {},
        "package_version_probe_error": None,
        "compatibility": None,
    }
    return {
        "status": "NEEDS_CONFIRMATION",
        "can_run": False,
        "summary": "Workspace scan is complete, but the final readiness verdict is waiting for your confirmation of the unified runtime form.",
        "next_action": "Review the single confirmation form, choose the intended runtime values, and rerun new-readiness-agent with those confirmed selections.",
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
    import_probes, import_error = probe_imports(profile.get("required_packages") or [], selected_python, probe_env)
    package_versions, package_errors, version_error = probe_package_versions(profile.get("required_packages") or [], selected_python, probe_env)
    missing_framework_imports = [name for name in framework_packages if not import_probes.get(name)]
    missing_runtime_imports = [name for name in (profile.get("runtime_imports") or []) if not import_probes.get(name)]

    checks.append(make_check("framework-importability", "ok" if not missing_framework_imports and framework_packages else ("warn" if not framework_packages else "block"), "framework packages are importable" if framework_packages and not missing_framework_imports else ("framework path has no package probe" if not framework_packages else f"missing framework imports: {', '.join(missing_framework_imports)}"), evidence=[f"{name}={import_probes.get(name)}" for name in framework_packages], probe_error=import_error))
    checks.append(make_check("runtime-dependencies", "ok" if not missing_runtime_imports else "block", "runtime imports are available" if not missing_runtime_imports else f"missing runtime imports: {', '.join(missing_runtime_imports)}", evidence=[f"{name}={import_probes.get(name)}" for name in (profile.get("runtime_imports") or [])], probe_error=import_error))

    launcher_status, launcher_summary = launcher_ready(str(profile.get("launcher") or ""), selected_env, import_probes)
    checks.append(make_check("launcher-readiness", launcher_status, launcher_summary))

    config_ok, config_error = config_is_readable(str(profile.get("config_path") or ""), root)
    required_config = profile.get("launcher") == "llamafactory-cli"
    config_status = "ok" if config_ok else ("block" if required_config else "warn")
    checks.append(make_check("config-readability", config_status, "config is readable" if config_ok else (config_error or "config is unresolved")))

    checks.append(asset_check("entry_script", str(profile.get("entry_script") or ""), root, needs_local_asset("entry_script", profile.get("target"), str(profile.get("entry_script") or ""), list(scan["entry_candidates"]))))
    checks.append(asset_check("model", str(profile.get("model_path") or ""), root, needs_local_asset("model", profile.get("target"), str(profile.get("model_path") or ""), list(scan["model_candidates"]))))
    checks.append(asset_check("dataset", str(profile.get("dataset_path") or ""), root, needs_local_asset("dataset", profile.get("target"), str(profile.get("dataset_path") or ""), list(scan["dataset_candidates"]))))
    checks.append(asset_check("checkpoint", str(profile.get("checkpoint_path") or ""), root, False))

    compatibility = None
    if str(profile.get("framework")) in {"mindspore", "pta"}:
        compatibility = assess_installed_framework_compatibility(str(profile.get("framework")), cann_version_info.get("cann_version"), selected_env.get("python_version") if selected_env else None, {name: package_versions.get(name) for name in FRAMEWORK_PACKAGES.get(str(profile.get("framework")), [])})
        compat_status = compatibility.get("status")
        if compat_status == "compatible":
            checks.append(make_check("framework-compatibility", "ok", compatibility.get("reason") or "framework versions match the local compatibility table"))
        elif compat_status:
            checks.append(make_check("framework-compatibility", "warn", compatibility.get("reason") or f"framework compatibility status: {compat_status}"))

    fields_needing_confirmation = [name for name, item in (profile.get("confirmed_fields") or {}).items() if isinstance(item, dict) and item.get("value") not in {None, ""} and not item.get("confirmed", False)]
    if fields_needing_confirmation:
        checks.append(make_check("confirmation-needed", "warn", f"user confirmation is still recommended for: {', '.join(fields_needing_confirmation)}"))

    critical_blockers = {"target-selection", "launcher-selection", "python-environment", "framework-selection", "framework-importability", "runtime-dependencies", "launcher-readiness", "config-readability", "workspace-entry_script-path", "workspace-model-path", "workspace-dataset-path"}
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
        next_action = "Review warnings, confirm the recommended values from the single confirmation form, and rerun new-readiness-agent if needed."
    else:
        status = "READY"
        summary = "Workspace is ready for the selected local single-machine workflow."
        next_action = "Use the selected runtime environment and launch command when you are ready to start the real workload."

    evidence_summary = {
        "target_candidates": scan.get("target_candidates"),
        "framework_candidates": scan.get("framework_candidates"),
        "launcher_candidates": scan.get("launcher_candidates"),
        "selected_runtime_environment": selected_env,
        "cann_version": cann_version_info.get("cann_version"),
        "cann_source": cann_version_info.get("cann_version_source"),
        "uses_llamafactory": profile.get("uses_llamafactory"),
        "required_packages": profile.get("required_packages"),
        "package_versions": package_versions,
        "package_errors": package_errors,
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
    confirmation = profile.get("confirmation_state") if isinstance(profile.get("confirmation_state"), dict) else build_confirmation_state(profile)
    validation = validate_profile(scan, profile, root) if confirmation.get("ready_for_validation") else build_pending_validation(scan, profile, root)
    return {
        "scan": scan,
        "profile": profile,
        "confirmation": confirmation,
        "validation": validation,
    }
