#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


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


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def normalize_target_hint(value: str | None) -> str | None:
    if not value:
        return None
    value = value.strip().lower()
    if value in {"training", "inference"}:
        return value
    return None


def infer_framework(text: str) -> str | None:
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


def score_script(path: Path, root: Path) -> tuple[int, list[str], str | None, str | None]:
    score = 0
    reasons: list[str] = []
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


def find_candidate_scripts(root: Path) -> list[Path]:
    candidates: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SCRIPT_SUFFIXES:
            continue
        if ".venv" in path.parts or "__pycache__" in path.parts:
            continue
        candidates.append(path)
    return candidates


def find_candidate_configs(root: Path) -> list[Path]:
    candidates: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in CONFIG_SUFFIXES:
            continue
        if ".venv" in path.parts or "__pycache__" in path.parts:
            continue
        candidates.append(path)
    return candidates


def find_model_markers(root: Path) -> list[str]:
    markers: list[str] = []
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


def choose_config(configs: list[Path], entry_script: Path | None, root: Path) -> str | None:
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
    target_hint: str | None,
    entry_script_hint: Path | None,
    config_path_hint: Path | None,
    model_path_hint: Path | None,
    dataset_path_hint: Path | None,
    checkpoint_path_hint: Path | None,
    task_smoke_cmd_hint: str | None,
) -> dict:
    evidence: list[str] = []
    candidate_scripts = find_candidate_scripts(root)
    configs = find_candidate_configs(root)
    markers = find_model_markers(root)

    chosen_script: Path | None = None
    framework_path = None
    discovered_target = target_hint

    if entry_script_hint:
        chosen_script = entry_script_hint if entry_script_hint.is_absolute() else (root / entry_script_hint)
        evidence.append("explicit entry_script input provided")
        script_text = read_text(chosen_script)
        framework_path = infer_framework(script_text)
        discovered_target = discovered_target or score_script(chosen_script, root)[3]
    else:
        ranked: list[tuple[int, Path, list[str], str | None, str | None]] = []
        for candidate in candidate_scripts:
            score, reasons, framework, target_type = score_script(candidate, root)
            if score <= 0:
                continue
            ranked.append((score, candidate, reasons, framework, target_type))
        ranked.sort(key=lambda item: (-item[0], str(item[1])))
        if ranked:
            best = ranked[0]
            chosen_script = best[1]
            framework_path = best[3]
            discovered_target = discovered_target or best[4]
            evidence.extend(best[2])
            if len(ranked) > 1 and ranked[1][0] == best[0]:
                evidence.append("multiple candidate scripts have equal evidence strength")

    config_path = None
    if config_path_hint:
        config_path = str(config_path_hint)
        evidence.append("explicit config_path input provided")
    else:
        config_path = choose_config(configs, chosen_script, root)
        if config_path:
            evidence.append("config path inferred from workspace evidence")

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
        "config_path": config_path,
        "model_path": str(model_path_hint) if model_path_hint else None,
        "dataset_path": str(dataset_path_hint) if dataset_path_hint else None,
        "checkpoint_path": str(checkpoint_path_hint) if checkpoint_path_hint else None,
        "task_smoke_cmd": task_smoke_cmd_hint,
        "output_path": None,
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
    parser = argparse.ArgumentParser(description="Discover an execution target for readiness-agent")
    parser.add_argument("--working-dir", required=True, help="workspace root")
    parser.add_argument("--target", default="auto", help="training, inference, or auto")
    parser.add_argument("--entry-script", help="explicit entry script path")
    parser.add_argument("--config-path", help="explicit config path")
    parser.add_argument("--model-path", help="explicit model path")
    parser.add_argument("--dataset-path", help="explicit dataset path")
    parser.add_argument("--checkpoint-path", help="explicit checkpoint path")
    parser.add_argument("--task-smoke-cmd", help="explicit minimal task smoke command")
    parser.add_argument("--output-json", required=True, help="path to write execution target JSON")
    args = parser.parse_args()

    root = Path(args.working_dir).resolve()
    result = build_execution_target(
        root=root,
        target_hint=normalize_target_hint(args.target),
        entry_script_hint=Path(args.entry_script) if args.entry_script else None,
        config_path_hint=Path(args.config_path) if args.config_path else None,
        model_path_hint=Path(args.model_path) if args.model_path else None,
        dataset_path_hint=Path(args.dataset_path) if args.dataset_path else None,
        checkpoint_path_hint=Path(args.checkpoint_path) if args.checkpoint_path else None,
        task_smoke_cmd_hint=args.task_smoke_cmd,
    )
    output = Path(args.output_json)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"target_type": result["target_type"], "confidence": result["confidence"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
