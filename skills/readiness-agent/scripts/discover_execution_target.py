#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from python_selection import resolve_selected_python


SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_ROOT = SCRIPT_DIR.parent
EXAMPLES_DIR = SKILL_ROOT / "examples"
WORKSPACE_ASSET_ROOT = "workspace-assets"

TRAINING_SCRIPT_NAMES = {
    "train.py",
    "finetune.py",
    "finetune_ms.py",
}

INFERENCE_SCRIPT_NAMES = {
    "infer.py",
    "inference.py",
    "generate.py",
    "predict.py",
}

SCRIPT_SUFFIXES = {".py", ".sh", ".ipynb"}
CONFIG_SUFFIXES = {".yaml", ".yml", ".json"}

MODEL_HUB_ALIASES = {
    "qwen3-0.6b": "Qwen/Qwen3-0.6B",
    "qwen/qwen3-0.6b": "Qwen/Qwen3-0.6B",
}

DATASET_HUB_ALIASES = {
    "karthiksagarn/astro_horoscope": "karthiksagarn/astro_horoscope",
}

KNOWN_EXAMPLE_RECIPES = {
    ("training", "Qwen/Qwen3-0.6B"): {
        "recipe_id": "qwen3-0.6b-hf-training",
        "framework_path": "pta",
        "entry_script": f"{WORKSPACE_ASSET_ROOT}/examples/train_qwen3_0_6b.py",
        "template_path": str(EXAMPLES_DIR / "qwen3_0_6b_training_example.py"),
        "model_hub_id": "Qwen/Qwen3-0.6B",
        "dataset_hub_id": "karthiksagarn/astro_horoscope",
        "dataset_split": "train",
        "reference_transformers_version": "4.57.6",
        "runtime_profile": [
            {
                "import_name": "datasets",
                "package_name": "datasets",
                "required_for": "bundled-example",
                "reason": "The bundled Qwen3-0.6B training example loads a Hugging Face dataset.",
            },
            {
                "import_name": "transformers",
                "package_name": "transformers==4.57.6",
                "required_for": "bundled-example",
                "reason": "The bundled Qwen3-0.6B training example is pinned to transformers 4.57.6.",
            },
            {
                "import_name": "sentencepiece",
                "package_name": "sentencepiece",
                "required_for": "bundled-example",
                "reason": "Qwen tokenizers commonly rely on sentencepiece when restoring tokenizer assets.",
            },
        ],
    }
}

DIRECT_FROM_PRETRAINED_RE = re.compile(r"from_pretrained\(\s*(['\"])(?P<repo>[^'\"]+)\1")
DIRECT_LOAD_DATASET_RE = re.compile(r"load_dataset\(\s*(['\"])(?P<repo>[^'\"]+)\1")
DIRECT_DATASET_SPLIT_RE = re.compile(r"load_dataset\([^)]*split\s*=\s*(['\"])(?P<split>[^'\"]+)\1")
ASSIGNMENT_LITERAL_RE = re.compile(
    r"(?m)^\s*(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*(['\"])(?P<value>[^'\"]+)\2\s*$"
)
VARIABLE_FROM_PRETRAINED_RE = re.compile(r"from_pretrained\(\s*(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*[,)]")
VARIABLE_LOAD_DATASET_RE = re.compile(r"load_dataset\(\s*(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*[,)]")


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def normalize_target_hint(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    value = value.strip().lower()
    if value in {"training", "inference"}:
        return value
    return None


def normalize_framework_hint(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    value = value.strip().lower()
    if value in {"mindspore", "pta", "mixed"}:
        return value
    return None


def normalize_model_hub_id(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    raw = value.strip()
    if not raw:
        return None
    return MODEL_HUB_ALIASES.get(raw.lower(), raw)


def normalize_dataset_hub_id(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    raw = value.strip()
    if not raw:
        return None
    return DATASET_HUB_ALIASES.get(raw.lower(), raw)


def default_asset_path(kind: str, repo_id: str) -> str:
    safe = repo_id.replace("/", "__").replace("\\", "__").replace(":", "_").replace(" ", "_")
    return f"{WORKSPACE_ASSET_ROOT}/{kind}/{safe}"


def looks_like_huggingface_repo_id(value: str) -> bool:
    candidate = value.strip()
    if not candidate or "/" not in candidate:
        return False
    if candidate.startswith((".", "/", "~")):
        return False
    if "\\" in candidate or " " in candidate:
        return False
    if re.match(r"^[A-Za-z]:", candidate):
        return False
    owner, repo = candidate.split("/", 1)
    allowed = re.compile(r"^[A-Za-z0-9_.-]+$")
    return bool(owner and repo and allowed.match(owner) and allowed.match(repo))


def literal_assignments(text: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for match in ASSIGNMENT_LITERAL_RE.finditer(text):
        mapping[match.group("name")] = match.group("value").strip()
    return mapping


def extract_model_hub_id(text: str) -> Optional[str]:
    direct = DIRECT_FROM_PRETRAINED_RE.search(text)
    if direct:
        repo = direct.group("repo").strip()
        if looks_like_huggingface_repo_id(repo):
            return normalize_model_hub_id(repo)

    assignments = literal_assignments(text)
    for match in VARIABLE_FROM_PRETRAINED_RE.finditer(text):
        value = assignments.get(match.group("name"))
        if value and looks_like_huggingface_repo_id(value):
            return normalize_model_hub_id(value)
    return None


def extract_dataset_hub_id(text: str) -> Optional[str]:
    direct = DIRECT_LOAD_DATASET_RE.search(text)
    if direct:
        repo = direct.group("repo").strip()
        if looks_like_huggingface_repo_id(repo):
            return normalize_dataset_hub_id(repo)

    assignments = literal_assignments(text)
    for match in VARIABLE_LOAD_DATASET_RE.finditer(text):
        value = assignments.get(match.group("name"))
        if value and looks_like_huggingface_repo_id(value):
            return normalize_dataset_hub_id(value)
    return None


def extract_dataset_split(text: str) -> Optional[str]:
    direct = DIRECT_DATASET_SPLIT_RE.search(text)
    if direct:
        split = direct.group("split").strip()
        if split:
            return split
    return None


def default_framework_path(
    target_type: Optional[str],
    explicit_framework: Optional[str],
    local_framework: Optional[str],
    recipe_framework: Optional[str],
) -> Tuple[Optional[str], Optional[str]]:
    framework_path = explicit_framework or local_framework or recipe_framework
    if framework_path:
        return framework_path, None
    if target_type == "training":
        return "pta", "training target defaulted to PTA because no explicit MindSpore request or local framework evidence was found"
    return None, None


def resolve_example_recipe(
    target_type: Optional[str],
    model_hub_id: Optional[str],
    dataset_hub_id: Optional[str],
) -> Optional[dict]:
    if not target_type or not model_hub_id:
        return None
    recipe = KNOWN_EXAMPLE_RECIPES.get((target_type, model_hub_id))
    if not recipe:
        return None
    expected_dataset = recipe.get("dataset_hub_id")
    if dataset_hub_id and expected_dataset and dataset_hub_id != expected_dataset:
        return None
    return dict(recipe)


def infer_framework(text: str) -> Optional[str]:
    lower = text.lower()
    has_mindspore = "import mindspore" in lower or "from mindspore" in lower
    has_pta = (
        "import torch_npu" in lower
        or "from torch_npu" in lower
        or ("import torch" in lower and ".npu(" in lower)
        or "set_context(device_target='ascend')" in lower
    )
    if has_mindspore and not has_pta:
        return "mindspore"
    if has_pta and not has_mindspore:
        return "pta"
    if has_mindspore and has_pta:
        return "mixed"
    return None


def score_script(path: Path, root: Path) -> Tuple[int, List[str], Optional[str], Optional[str]]:
    score = 0
    reasons: List[str] = []
    framework = None
    target_type = None

    name = path.name.lower()
    if name in TRAINING_SCRIPT_NAMES:
        score += 40
        target_type = "training"
        reasons.append(f"script name {path.name} strongly suggests training")
    elif name in INFERENCE_SCRIPT_NAMES:
        score += 40
        target_type = "inference"
        reasons.append(f"script name {path.name} strongly suggests inference")
    elif "train" in name or "finetune" in name:
        score += 25
        target_type = "training"
        reasons.append(f"script name {path.name} suggests training")
    elif "infer" in name or "generate" in name or "predict" in name:
        score += 25
        target_type = "inference"
        reasons.append(f"script name {path.name} suggests inference")

    text = read_text(path)
    lower = text.lower()
    framework = infer_framework(text)
    if framework == "mindspore":
        score += 15
        reasons.append("imports suggest MindSpore")
    elif framework == "pta":
        score += 15
        reasons.append("imports suggest PTA")
    elif framework == "mixed":
        score += 5
        reasons.append("imports contain mixed framework evidence")

    if "optimizer" in lower or "dataloader" in lower or "dataset" in lower or "loss" in lower:
        score += 10
        target_type = target_type or "training"
        reasons.append("code contains training-oriented signals")
    if "generate(" in lower or "tokenizer" in lower or "model.generate" in lower:
        score += 10
        target_type = target_type or "inference"
        reasons.append("code contains inference-oriented signals")

    relative_parts = path.relative_to(root).parts
    if "scripts" in relative_parts or "examples" in relative_parts:
        score += 5
        reasons.append("script is located in a runnable workspace folder")

    return score, reasons, framework, target_type


def find_candidate_scripts(root: Path) -> List[Path]:
    candidates: List[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SCRIPT_SUFFIXES:
            continue
        if ".venv" in path.parts or "__pycache__" in path.parts:
            continue
        candidates.append(path)
    return candidates


def find_candidate_configs(root: Path) -> List[Path]:
    candidates: List[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in CONFIG_SUFFIXES:
            continue
        if ".venv" in path.parts or "__pycache__" in path.parts:
            continue
        candidates.append(path)
    return candidates


def find_model_markers(root: Path) -> List[str]:
    markers: List[str] = []
    names = {
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "model.safetensors",
    }
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.name in names or path.suffix == ".ckpt":
            markers.append(str(path.relative_to(root)))
    return markers[:20]


def infer_model_path(markers: List[str], root: Path, entry_script: Optional[Path]) -> Optional[str]:
    if not markers:
        return None

    scores = {}
    entry_text = read_text(entry_script) if entry_script else ""

    for marker in markers:
        marker_path = Path(marker)
        parent = marker_path.parent
        if str(parent) in {"", "."}:
            continue

        score = 1
        if marker_path.name in {"config.json", "tokenizer.json", "tokenizer_config.json"}:
            score = 2
        if marker_path.name == "model.safetensors" or marker_path.suffix == ".ckpt":
            score = 3

        key = str(parent)
        scores[key] = scores.get(key, 0) + score

        normalized_parent = key.replace("\\", "/")
        if entry_text and (normalized_parent in entry_text or parent.name in entry_text):
            scores[key] += 3

    if not scores:
        return None

    ranked = sorted(
        scores.items(),
        key=lambda item: (-item[1], len(Path(item[0]).parts), item[0]),
    )
    return ranked[0][0]


def choose_config(configs: List[Path], entry_script: Optional[Path], root: Path) -> Optional[str]:
    if not configs:
        return None
    if entry_script:
        entry_text = read_text(entry_script)
        for config in configs:
            rel = str(config.relative_to(root))
            if rel in entry_text or config.name in entry_text:
                return rel
    ranked = sorted(
        configs,
        key=lambda path: (
            "train" not in path.name.lower() and "infer" not in path.name.lower(),
            len(path.parts),
            path.name,
        ),
    )
    return str(ranked[0].relative_to(root))


def build_execution_target(
    root: Path,
    target_hint: Optional[str],
    framework_hint: Optional[str],
    cann_path_hint: Optional[Path],
    entry_script_hint: Optional[Path],
    config_path_hint: Optional[Path],
    model_path_hint: Optional[Path],
    model_hub_id_hint: Optional[str],
    dataset_path_hint: Optional[Path],
    dataset_hub_id_hint: Optional[str],
    dataset_split_hint: Optional[str],
    checkpoint_path_hint: Optional[Path],
    task_smoke_cmd_hint: Optional[str],
    selected_python_hint: Optional[str],
    selected_env_root_hint: Optional[str],
) -> dict:
    evidence: List[str] = []
    candidate_scripts = find_candidate_scripts(root)
    configs = find_candidate_configs(root)
    markers = find_model_markers(root)
    requested_framework = normalize_framework_hint(framework_hint)
    model_hub_id = normalize_model_hub_id(model_hub_id_hint)
    dataset_hub_id = normalize_dataset_hub_id(dataset_hub_id_hint)
    python_selection = resolve_selected_python(
        root=root,
        selected_python=selected_python_hint,
        selected_env_root=selected_env_root_hint,
    )

    chosen_script: Optional[Path] = None
    local_framework = None
    discovered_target = target_hint

    if requested_framework:
        evidence.append("explicit framework_hint input provided")
    if cann_path_hint:
        evidence.append("explicit cann_path input provided")
    if model_hub_id:
        evidence.append("explicit model_hub_id input provided")
    if dataset_hub_id:
        evidence.append("explicit dataset_hub_id input provided")
    if dataset_split_hint:
        evidence.append("explicit dataset_split input provided")

    if entry_script_hint:
        chosen_script = entry_script_hint if entry_script_hint.is_absolute() else (root / entry_script_hint)
        evidence.append("explicit entry_script input provided")
        script_text = read_text(chosen_script)
        local_framework = infer_framework(script_text)
        if not model_hub_id:
            extracted_model_hub_id = extract_model_hub_id(script_text)
            if extracted_model_hub_id:
                model_hub_id = extracted_model_hub_id
                evidence.append("model_hub_id extracted from entry script")
        if not dataset_hub_id:
            extracted_dataset_hub_id = extract_dataset_hub_id(script_text)
            if extracted_dataset_hub_id:
                dataset_hub_id = extracted_dataset_hub_id
                evidence.append("dataset_hub_id extracted from entry script")
        if not dataset_split_hint:
            extracted_dataset_split = extract_dataset_split(script_text)
            if extracted_dataset_split:
                dataset_split_hint = extracted_dataset_split
                evidence.append("dataset_split extracted from entry script")
        discovered_target = discovered_target or score_script(chosen_script, root)[3]
    else:
        ranked: List[Tuple[int, Path, List[str], Optional[str], Optional[str]]] = []
        for candidate in candidate_scripts:
            score, reasons, framework, target_type = score_script(candidate, root)
            if score <= 0:
                continue
            ranked.append((score, candidate, reasons, framework, target_type))
        ranked.sort(key=lambda item: (-item[0], str(item[1])))
        if ranked:
            best = ranked[0]
            chosen_script = best[1]
            local_framework = best[3]
            discovered_target = discovered_target or best[4]
            evidence.extend(best[2])
            script_text = read_text(chosen_script)
            if not model_hub_id:
                extracted_model_hub_id = extract_model_hub_id(script_text)
                if extracted_model_hub_id:
                    model_hub_id = extracted_model_hub_id
                    evidence.append("model_hub_id extracted from discovered entry script")
            if not dataset_hub_id:
                extracted_dataset_hub_id = extract_dataset_hub_id(script_text)
                if extracted_dataset_hub_id:
                    dataset_hub_id = extracted_dataset_hub_id
                    evidence.append("dataset_hub_id extracted from discovered entry script")
            if not dataset_split_hint:
                extracted_dataset_split = extract_dataset_split(script_text)
                if extracted_dataset_split:
                    dataset_split_hint = extracted_dataset_split
                    evidence.append("dataset_split extracted from discovered entry script")
            if len(ranked) > 1 and ranked[1][0] == best[0]:
                evidence.append("multiple candidate scripts have equal evidence strength")

    recipe = resolve_example_recipe(discovered_target or target_hint, model_hub_id, dataset_hub_id)
    if recipe and not chosen_script:
        chosen_script = root / str(recipe["entry_script"])
        evidence.append(f"bundled training example selected: {recipe['recipe_id']}")
        discovered_target = discovered_target or "training"

    framework_path, framework_default_reason = default_framework_path(
        discovered_target or target_hint,
        requested_framework,
        local_framework,
        recipe.get("framework_path") if recipe else None,
    )
    if requested_framework and local_framework and requested_framework != local_framework:
        evidence.append(
            f"local workspace evidence suggests {local_framework}, but explicit framework hint requested {requested_framework}"
        )
    if framework_default_reason:
        evidence.append(framework_default_reason)

    config_path = None
    if config_path_hint:
        config_path = str(config_path_hint)
        evidence.append("explicit config_path input provided")
    else:
        config_path = choose_config(configs, chosen_script, root)
        if config_path:
            evidence.append("config path inferred from workspace evidence")

    model_path = None
    if model_path_hint:
        model_path = str(model_path_hint)
        evidence.append("explicit model_path input provided")
    else:
        model_path = infer_model_path(markers, root, chosen_script)
        if model_path:
            evidence.append("model path inferred from workspace model markers")

    dataset_path = None
    if dataset_path_hint:
        dataset_path = str(dataset_path_hint)
        evidence.append("explicit dataset_path input provided")

    dataset_split = dataset_split_hint or (str(recipe["dataset_split"]) if recipe and recipe.get("dataset_split") else None)
    model_hub_id = model_hub_id or (str(recipe["model_hub_id"]) if recipe and recipe.get("model_hub_id") else None)
    dataset_hub_id = dataset_hub_id or (str(recipe["dataset_hub_id"]) if recipe and recipe.get("dataset_hub_id") else None)

    if target_hint:
        evidence.append("explicit target input provided")
    if task_smoke_cmd_hint:
        evidence.append("explicit task_smoke_cmd input provided")

    confidence = "low"
    if target_hint and chosen_script:
        confidence = "high"
    elif chosen_script and framework_path and config_path:
        confidence = "high"
    elif chosen_script and (framework_path or config_path):
        confidence = "medium"

    launch_cmd = None
    if chosen_script:
        launch_cmd = f"python {chosen_script.relative_to(root)}"

    return {
        "working_dir": str(root),
        "target_type": discovered_target or "unknown",
        "entry_script": str(chosen_script.relative_to(root)) if chosen_script else None,
        "launch_cmd": launch_cmd,
        "framework_path": framework_path or "unknown",
        "framework_hint": requested_framework or "auto",
        "cann_path": str(cann_path_hint) if cann_path_hint else None,
        "config_path": config_path,
        "model_path": model_path,
        "model_hub_id": model_hub_id,
        "dataset_path": dataset_path,
        "dataset_hub_id": dataset_hub_id,
        "dataset_split": dataset_split,
        "checkpoint_path": str(checkpoint_path_hint) if checkpoint_path_hint else None,
        "selected_python": python_selection.get("selected_python"),
        "selected_env_root": python_selection.get("selected_env_root"),
        "selected_python_source": python_selection.get("selection_source"),
        "selected_python_status": python_selection.get("selection_status"),
        "selected_python_reason": python_selection.get("selection_reason"),
        "selected_python_version": python_selection.get("python_version"),
        "task_smoke_cmd": task_smoke_cmd_hint,
        "output_path": None,
        "asset_provider": "huggingface" if model_hub_id or dataset_hub_id else None,
        "example_recipe_id": recipe.get("recipe_id") if recipe else None,
        "example_template_path": recipe.get("template_path") if recipe else None,
        "reference_transformers_version": recipe.get("reference_transformers_version") if recipe else None,
        "expected_runtime_profile": recipe.get("runtime_profile") if recipe else [],
        "evidence": evidence,
        "confidence": confidence,
        "candidate_counts": {
            "scripts": len(candidate_scripts),
            "configs": len(configs),
            "model_markers": len(markers),
        },
        "model_markers": markers,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Discover an execution target for readiness-agent",
        epilog=(
            "Internal helper. Prefer the top-level readiness workflow entrypoint instead of "
            "calling this script directly. Do not substitute system python for a missing "
            "workspace-local environment."
        ),
    )
    parser.add_argument("--working-dir", required=True, help="workspace root")
    parser.add_argument("--target", default="auto", help="training, inference, or auto")
    parser.add_argument("--framework-hint", help="explicit framework preference such as mindspore or pta")
    parser.add_argument("--cann-path", help="explicit CANN root or set_env.sh path")
    parser.add_argument("--entry-script", help="explicit entry script path")
    parser.add_argument("--config-path", help="explicit config path")
    parser.add_argument("--model-path", help="explicit model path")
    parser.add_argument("--model-hub-id", help="explicit Hugging Face model repo ID")
    parser.add_argument("--dataset-path", help="explicit dataset path")
    parser.add_argument("--dataset-hub-id", help="explicit Hugging Face dataset repo ID")
    parser.add_argument("--dataset-split", help="explicit dataset split for remote dataset download")
    parser.add_argument("--checkpoint-path", help="explicit checkpoint path")
    parser.add_argument("--selected-python", help="explicit Python interpreter for the workspace")
    parser.add_argument("--selected-env-root", help="explicit environment root for the workspace")
    parser.add_argument("--task-smoke-cmd", help="explicit minimal task smoke command")
    parser.add_argument("--output-json", required=True, help="path to write execution target JSON")
    args = parser.parse_args()

    root = Path(args.working_dir).resolve()
    result = build_execution_target(
        root=root,
        target_hint=normalize_target_hint(args.target),
        framework_hint=args.framework_hint,
        cann_path_hint=Path(args.cann_path) if args.cann_path else None,
        entry_script_hint=Path(args.entry_script) if args.entry_script else None,
        config_path_hint=Path(args.config_path) if args.config_path else None,
        model_path_hint=Path(args.model_path) if args.model_path else None,
        model_hub_id_hint=args.model_hub_id,
        dataset_path_hint=Path(args.dataset_path) if args.dataset_path else None,
        dataset_hub_id_hint=args.dataset_hub_id,
        dataset_split_hint=args.dataset_split,
        checkpoint_path_hint=Path(args.checkpoint_path) if args.checkpoint_path else None,
        task_smoke_cmd_hint=args.task_smoke_cmd,
        selected_python_hint=args.selected_python,
        selected_env_root_hint=args.selected_env_root,
    )
    output = Path(args.output_json)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"target_type": result["target_type"], "confidence": result["confidence"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
