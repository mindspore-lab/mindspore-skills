#!/usr/bin/env python3
import ast
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Pattern, Tuple

from asset_registry import ENTRY_PATTERNS
from asset_schema import make_asset_candidate, make_asset_requirement, rank_asset_candidates
from candidate_utils import looks_like_hf_repo_id, looks_like_local_path
from environment_selection import resolve_optional_path


PATH_VALUE_KEYS = {
    "config": ("config", "config_file", "config_path"),
    "model": ("model_name_or_path", "model_path", "pretrained_model_name_or_path", "model_dir"),
    "tokenizer": ("tokenizer_name_or_path", "tokenizer_path", "tokenizer_dir"),
    "dataset": ("dataset_dir", "dataset_path", "data_dir", "data_path", "train_file", "validation_file", "dataset"),
    "checkpoint": ("checkpoint_path", "resume_from_checkpoint", "load_checkpoint", "ckpt_path"),
}
SCRIPT_SYMBOL_KEYS = {
    "config": {"config", "config_file", "config_path"},
    "model": {"model_repo_id", "model_name_or_path", "pretrained_model_name_or_path", "model_path"},
    "tokenizer": {"tokenizer_repo_id", "tokenizer_name_or_path", "tokenizer_path"},
    "dataset": {"dataset_repo_id", "dataset", "dataset_dir", "dataset_path", "data_dir", "data_path", "train_file", "validation_file"},
    "dataset_split": {"dataset_split", "split"},
    "checkpoint": {"checkpoint_path", "resume_from_checkpoint", "ckpt_path"},
}
LOCAL_WORKSPACE_DIRS = {
    "model": ("model", "models"),
    "tokenizer": ("tokenizer", "tokenizers"),
    "dataset": ("dataset", "data"),
}
HF_DATASET_CALL_PATTERN = re.compile(r"""load_dataset\(\s*["']([^"']+)["']""")
HF_MODEL_CALL_PATTERN = re.compile(r"""from_pretrained\(\s*["']([^"']+)["']""")
HF_SNAPSHOT_CALL_PATTERN = re.compile(r"""snapshot_download\([^)]*repo_id\s*=\s*["']([^"']+)["']""")
TRAINING_ARGUMENTS_PATTERN = re.compile(r"""\bTrainingArguments\s*\(""")


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


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


def resolve_entry_scripts(root: Path, files: List[Path], entry_candidates: List[Dict[str, object]]) -> List[Path]:
    results: List[Path] = []
    seen = set()
    for item in entry_candidates:
        path = resolve_optional_path(str(item.get("value") or ""), root)
        if not path or not path.exists():
            continue
        normalized = str(path.resolve())
        if normalized in seen:
            continue
        seen.add(normalized)
        results.append(path)
    if results:
        return results
    for name in ENTRY_PATTERNS:
        path = root / name
        if path.exists():
            results.append(path)
    return results


def resolve_hf_cache_layout(root: Path) -> Dict[str, object]:
    explicit_hub = os.environ.get("HUGGINGFACE_HUB_CACHE")
    explicit_datasets = os.environ.get("HF_DATASETS_CACHE")
    explicit_hf_home = os.environ.get("HF_HOME")

    if explicit_hub or explicit_datasets:
        hub_cache = Path(explicit_hub or (root / "huggingface-cache" / "hub")).resolve()
        datasets_cache = Path(explicit_datasets or (root / "huggingface-cache" / "datasets")).resolve()
        hf_home = Path(explicit_hf_home).resolve() if explicit_hf_home else None
        source = "explicit_cache_env"
    elif explicit_hf_home:
        hf_home = Path(explicit_hf_home).resolve()
        hub_cache = (hf_home / "hub").resolve()
        datasets_cache = (hf_home / "datasets").resolve()
        source = "explicit_hf_home"
    else:
        hf_home = (root / "huggingface-cache").resolve()
        hub_cache = (hf_home / "hub").resolve()
        datasets_cache = (hf_home / "datasets").resolve()
        source = "working_dir_default"

    return {
        "source": source,
        "hf_home": str(hf_home) if hf_home else None,
        "hub_cache": str(hub_cache),
        "datasets_cache": str(datasets_cache),
    }


def _repo_tokens(repo_id: str, kind: str) -> List[str]:
    owner, repo = repo_id.split("/", 1)
    tokens = [f"{owner}___{repo}", f"{owner}--{repo}", repo]
    if kind in {"model", "tokenizer"}:
        tokens.insert(0, f"models--{owner}--{repo}")
    return tokens


def _matching_cache_dirs(base_root: Path, repo_id: str, kind: str) -> List[Path]:
    if not base_root.exists() or not base_root.is_dir():
        return []
    tokens = [token.lower() for token in _repo_tokens(repo_id, kind)]
    matches: List[Path] = []
    for child in base_root.iterdir():
        if not child.is_dir():
            continue
        lowered = child.name.lower()
        if any(token in lowered for token in tokens):
            matches.append(child.resolve())
    return matches


def _dotted_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _dotted_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""


def _resolve_string_value(node: Optional[ast.AST], symbols: Dict[str, str]) -> Optional[str]:
    if node is None:
        return None
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        value = node.value.strip()
        return value or None
    if isinstance(node, ast.Name):
        value = symbols.get(node.id)
        if not value:
            return None
        normalized = str(value).strip()
        return normalized or None
    if isinstance(node, ast.JoinedStr):
        parts: List[str] = []
        for item in node.values:
            if isinstance(item, ast.Constant) and isinstance(item.value, str):
                parts.append(item.value)
                continue
            if isinstance(item, ast.FormattedValue):
                formatted = _resolve_string_value(item.value, symbols)
                if formatted is None:
                    return None
                parts.append(formatted)
                continue
            return None
        combined = "".join(parts).strip()
        return combined or None
    if isinstance(node, ast.Call):
        func_name = _dotted_name(node.func).split(".")[-1]
        if func_name in {"Path", "PurePath"} and node.args:
            return _resolve_string_value(node.args[0], symbols)
        if func_name == "str" and node.args:
            return _resolve_string_value(node.args[0], symbols)
    return None


def _line_text(text: str, lineno: int) -> str:
    lines = text.splitlines()
    if 1 <= lineno <= len(lines):
        return lines[lineno - 1].strip()
    return ""


def _call_argument(call: ast.Call, *, position: Optional[int] = 0, keyword_names: Tuple[str, ...] = ()) -> Optional[ast.AST]:
    if position is not None and len(call.args) > position:
        return call.args[position]
    keyword_set = set(keyword_names)
    for keyword in call.keywords:
        if keyword.arg in keyword_set:
            return keyword.value
    return None


def _script_hint(
    kind: str,
    entry_script: Path,
    lineno: int,
    line: str,
    *,
    source_type: str,
    locator: Dict[str, object],
    confidence: float,
) -> Dict[str, object]:
    return {
        "kind": kind,
        "source_type": source_type,
        "locator": locator,
        "entry_script": str(entry_script),
        "callsite": f"{entry_script.name}:{lineno}",
        "line": line,
        "confidence": confidence,
    }


def _hint_from_value(
    kind: str,
    entry_script: Path,
    lineno: int,
    line: str,
    value: str,
    *,
    split: Optional[str] = None,
    confidence: float,
) -> Optional[Dict[str, object]]:
    token = str(value or "").strip()
    if not token:
        return None
    resolved_local = resolve_optional_path(token, entry_script.parent)
    if kind in {"config", "checkpoint"}:
        local_path = resolved_local or Path(token).expanduser()
        return _script_hint(kind, entry_script, lineno, line, source_type="local_path", locator={"path": str(local_path)}, confidence=confidence)
    if resolved_local and resolved_local.exists():
        return _script_hint(kind, entry_script, lineno, line, source_type="local_path", locator={"path": str(resolved_local)}, confidence=confidence)
    if looks_like_local_path(token):
        return _script_hint(kind, entry_script, lineno, line, source_type="local_path", locator={"path": token}, confidence=confidence)
    if looks_like_hf_repo_id(token):
        locator: Dict[str, object] = {"repo_id": token}
        if split:
            locator["split"] = split
        return _script_hint(kind, entry_script, lineno, line, source_type="hf_hub", locator=locator, confidence=confidence)
    return None


def _build_symbol_table(tree: ast.AST) -> Tuple[Dict[str, str], Dict[str, int]]:
    symbols: Dict[str, str] = {}
    locations: Dict[str, int] = {}
    for statement in getattr(tree, "body", []):
        target_name: Optional[str] = None
        value_node: Optional[ast.AST] = None
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1 and isinstance(statement.targets[0], ast.Name):
            target_name = statement.targets[0].id
            value_node = statement.value
        elif isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
            target_name = statement.target.id
            value_node = statement.value
        if not target_name or value_node is None:
            continue
        resolved = _resolve_string_value(value_node, symbols)
        if resolved is None:
            continue
        symbols[target_name] = resolved
        locations[target_name] = int(getattr(statement, "lineno", 1))
    return symbols, locations


def _script_constant_hints(entry_script: Path, text: str, symbols: Dict[str, str], locations: Dict[str, int]) -> Dict[str, List[Dict[str, object]]]:
    hints = {kind: [] for kind in ("config", "model", "tokenizer", "dataset", "checkpoint")}
    dataset_split = next((value for name, value in symbols.items() if name.lower() in SCRIPT_SYMBOL_KEYS["dataset_split"]), None)
    for name, value in symbols.items():
        lowered = name.lower()
        lineno = locations.get(name, 1)
        line = _line_text(text, lineno)
        if lowered in SCRIPT_SYMBOL_KEYS["config"]:
            if looks_like_local_path(value):
                hints["config"].append(_script_hint("config", entry_script, lineno, line, source_type="local_path", locator={"path": value}, confidence=0.73))
            continue
        if lowered in SCRIPT_SYMBOL_KEYS["model"]:
            hint = _hint_from_value("model", entry_script, lineno, line, value, confidence=0.74)
            if hint:
                hints["model"].append(hint)
            continue
        if lowered in SCRIPT_SYMBOL_KEYS["tokenizer"]:
            hint = _hint_from_value("tokenizer", entry_script, lineno, line, value, confidence=0.74)
            if hint:
                hints["tokenizer"].append(hint)
            continue
        if lowered in SCRIPT_SYMBOL_KEYS["dataset"]:
            hint = _hint_from_value("dataset", entry_script, lineno, line, value, split=dataset_split, confidence=0.74)
            if hint:
                hints["dataset"].append(hint)
            continue
        if lowered in SCRIPT_SYMBOL_KEYS["checkpoint"]:
            hint = _hint_from_value("checkpoint", entry_script, lineno, line, value, confidence=0.72)
            if hint and hint.get("source_type") == "local_path":
                hints["checkpoint"].append(hint)
    return hints


def _from_pretrained_kind(func_name: str) -> str:
    receiver_chain = func_name.split(".")[:-1]
    receiver = receiver_chain[-1] if receiver_chain else ""
    if "Tokenizer" in receiver:
        return "tokenizer"
    return "model"


def _ast_call_hints(entry_script: Path, text: str, tree: ast.AST, symbols: Dict[str, str]) -> Dict[str, List[Dict[str, object]]]:
    hints = {kind: [] for kind in ("config", "model", "tokenizer", "dataset", "checkpoint")}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func_name = _dotted_name(node.func)
        short_name = func_name.split(".")[-1]
        line = _line_text(text, int(getattr(node, "lineno", 1)))
        lineno = int(getattr(node, "lineno", 1))

        if short_name == "TrainingArguments":
            hints["config"].append(
                _script_hint(
                    "config",
                    entry_script,
                    lineno,
                    line or "TrainingArguments(...) detected",
                    source_type="inline_config",
                    locator={"entry_script": str(entry_script), "callsite": f"{entry_script.name}:{lineno}"},
                    confidence=0.88,
                )
            )

        if short_name == "load_dataset":
            dataset_value = _resolve_string_value(_call_argument(node, position=0, keyword_names=("path", "name", "dataset")), symbols)
            split_value = _resolve_string_value(_call_argument(node, position=None, keyword_names=("split",)), symbols)
            if dataset_value:
                hint = _hint_from_value("dataset", entry_script, lineno, line, dataset_value, split=split_value, confidence=0.88)
                if hint:
                    hints["dataset"].append(hint)
            continue

        if short_name == "load_from_disk":
            dataset_path = _resolve_string_value(_call_argument(node, position=0, keyword_names=("dataset_path", "path")), symbols)
            if dataset_path:
                hint = _hint_from_value("dataset", entry_script, lineno, line, dataset_path, confidence=0.86)
                if hint and hint.get("source_type") == "local_path":
                    hints["dataset"].append(hint)
            continue

        if short_name == "snapshot_download":
            repo_id = _resolve_string_value(_call_argument(node, position=0, keyword_names=("repo_id",)), symbols)
            if repo_id:
                hint = _hint_from_value("model", entry_script, lineno, line, repo_id, confidence=0.84)
                if hint:
                    hints["model"].append(hint)

        if short_name == "from_pretrained":
            repo_or_path = _resolve_string_value(
                _call_argument(node, position=0, keyword_names=("pretrained_model_name_or_path", "model_name_or_path", "path")),
                symbols,
            )
            if repo_or_path:
                kind = _from_pretrained_kind(func_name)
                hint = _hint_from_value(kind, entry_script, lineno, line, repo_or_path, confidence=0.87 if kind == "model" else 0.85)
                if hint:
                    hints[kind].append(hint)

        for keyword in node.keywords:
            keyword_name = keyword.arg or ""
            if keyword_name in PATH_VALUE_KEYS["checkpoint"]:
                checkpoint_value = _resolve_string_value(keyword.value, symbols)
                if checkpoint_value:
                    hint = _hint_from_value("checkpoint", entry_script, lineno, line, checkpoint_value, confidence=0.82)
                    if hint and hint.get("source_type") == "local_path":
                        hints["checkpoint"].append(hint)
            if keyword_name in PATH_VALUE_KEYS["config"]:
                config_value = _resolve_string_value(keyword.value, symbols)
                if config_value and looks_like_local_path(config_value):
                    hints["config"].append(
                        _script_hint("config", entry_script, lineno, line, source_type="local_path", locator={"path": config_value}, confidence=0.76)
                    )
    return hints


def _callsite_matches(pattern: Pattern[str], text: str, entry_script: Path, *, kind: str, confidence: float, split: Optional[str] = None) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []
    lines = text.splitlines()
    for index, line in enumerate(lines, start=1):
        match = pattern.search(line)
        if not match:
            continue
        hint = _hint_from_value(kind, entry_script, index, line.strip(), match.group(1).strip(), split=split, confidence=confidence)
        if hint:
            results.append(hint)
    return results


def analyze_entry_scripts(entry_scripts: Iterable[Path]) -> Dict[str, object]:
    hints = {kind: [] for kind in ("config", "model", "tokenizer", "dataset", "checkpoint")}
    for entry_script in entry_scripts:
        text = read_text(entry_script)
        if not text:
            continue

        try:
            tree = ast.parse(text, filename=str(entry_script))
        except SyntaxError:
            tree = None

        if tree is not None:
            symbols, locations = _build_symbol_table(tree)
            constant_hints = _script_constant_hints(entry_script, text, symbols, locations)
            ast_hints = _ast_call_hints(entry_script, text, tree, symbols)
            for kind in hints:
                hints[kind].extend(constant_hints.get(kind) or [])
                hints[kind].extend(ast_hints.get(kind) or [])
        else:
            hints["model"].extend(_callsite_matches(HF_MODEL_CALL_PATTERN, text, entry_script, kind="model", confidence=0.79))
            hints["model"].extend(_callsite_matches(HF_SNAPSHOT_CALL_PATTERN, text, entry_script, kind="model", confidence=0.79))
            hints["dataset"].extend(_callsite_matches(HF_DATASET_CALL_PATTERN, text, entry_script, kind="dataset", confidence=0.80, split="train"))
            if TRAINING_ARGUMENTS_PATTERN.search(text):
                hints["config"].append(
                    _script_hint(
                        "config",
                        entry_script,
                        1,
                        "TrainingArguments(...) detected",
                        source_type="inline_config",
                        locator={"entry_script": str(entry_script), "callsite": f"{entry_script.name}:TrainingArguments"},
                        confidence=0.84,
                    )
                )
    return hints


def _local_candidate(kind: str, path: Path, label: str, source: str, confidence: float, evidence: Optional[List[str]] = None) -> Dict[str, object]:
    return make_asset_candidate(
        kind,
        "local_path",
        label=label,
        locator={"path": str(path)},
        confidence=confidence,
        selection_source=source,
        evidence=evidence,
        exists=path.exists(),
    )


def _hf_hub_candidate(kind: str, repo_id: str, label: str, source: str, confidence: float, *, split: Optional[str] = None, evidence: Optional[List[str]] = None) -> Dict[str, object]:
    locator: Dict[str, object] = {"repo_id": repo_id}
    if split:
        locator["split"] = split
    return make_asset_candidate(kind, "hf_hub", label=label, locator=locator, confidence=confidence, selection_source=source, evidence=evidence)


def _hf_cache_candidate(kind: str, repo_id: str, cache_path: Path, label: str, source: str, confidence: float, *, split: Optional[str] = None, evidence: Optional[List[str]] = None) -> Dict[str, object]:
    locator: Dict[str, object] = {"repo_id": repo_id, "cache_path": str(cache_path)}
    if split:
        locator["split"] = split
    return make_asset_candidate(
        kind,
        "hf_cache",
        label=label,
        locator=locator,
        confidence=confidence,
        selection_source=source,
        evidence=evidence,
        exists=cache_path.exists(),
    )
def _inline_config_candidate(hint: Dict[str, object]) -> Dict[str, object]:
    locator = hint.get("locator") if isinstance(hint.get("locator"), dict) else {}
    return make_asset_candidate(
        "config",
        "inline_config",
        label=f"inline config in {Path(str(locator.get('entry_script') or '')).name}",
        locator=locator,
        confidence=float(hint.get("confidence") or 0.88),
        selection_source="script_analysis",
        evidence=[str(hint.get("line") or ""), str(hint.get("callsite") or "")],
    )


def _config_candidates_from_workspace(root: Path, config_candidates: List[Dict[str, object]]) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []
    for item in config_candidates:
        path = resolve_optional_path(str(item.get("value") or ""), root)
        if not path:
            continue
        results.append(
            _local_candidate(
                "config",
                path,
                str(item.get("label") or f"config file {path.name}"),
                str(item.get("selection_source") or "workspace_scan"),
                float(item.get("confidence") or 0.74),
                evidence=[str(item.get("value") or "")],
            )
        )
    return results


def _config_hint_candidates(kind: str, root: Path, config_candidates: List[Dict[str, object]], repo_split: Optional[str] = None) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []
    for config_candidate in config_candidates:
        config_path = resolve_optional_path(str(config_candidate.get("value") or ""), root)
        if not config_path or not config_path.exists():
            continue
        text = read_text(config_path)
        for raw_value in parse_config_values(text, PATH_VALUE_KEYS[kind]):
            local = resolve_optional_path(raw_value, root) if looks_like_local_path(raw_value) else None
            evidence = [f"{config_path.name}: {raw_value}"]
            if local:
                results.append(_local_candidate(kind, local, f"{kind} path from {config_path.name}", "config_scan", 0.75, evidence))
            elif looks_like_hf_repo_id(raw_value):
                results.append(_hf_hub_candidate(kind, raw_value, f"HF Hub {kind} from {config_path.name}", "config_scan", 0.72, split=repo_split, evidence=evidence))
    return results


def _discover_cache_candidates(
    kind: str,
    repo_id: str,
    cache_layout: Dict[str, object],
    *,
    split: Optional[str] = None,
    source: str = "hf_cache_scan",
    confidence: float = 0.83,
    evidence: Optional[List[str]] = None,
) -> List[Dict[str, object]]:
    cache_root = Path(str(cache_layout.get("datasets_cache") or "")) if kind == "dataset" else Path(str(cache_layout.get("hub_cache") or ""))
    matches = _matching_cache_dirs(cache_root, repo_id, kind)
    return [
        _hf_cache_candidate(
            kind,
            repo_id,
            match,
            f"HF cache {kind}: {repo_id}",
            source,
            confidence,
            split=split,
            evidence=[*(evidence or []), str(match)],
        )
        for match in matches
    ]


def _script_hint_candidates(kind: str, script_hints: Dict[str, object], cache_layout: Dict[str, object]) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []
    for hint in script_hints.get(kind) or []:
        locator = hint.get("locator") if isinstance(hint.get("locator"), dict) else {}
        source_type = str(hint.get("source_type") or "")
        evidence = [str(hint.get("line") or ""), str(hint.get("callsite") or "")]
        confidence = float(hint.get("confidence") or 0.78)
        if source_type == "inline_config":
            results.append(_inline_config_candidate(hint))
            continue
        if source_type == "local_path":
            path = Path(str(locator.get("path") or "")).expanduser()
            results.append(_local_candidate(kind, path, f"script {kind} path {path.name or path}", "script_analysis", confidence, evidence=evidence))
            continue
        repo_id = str(locator.get("repo_id") or "")
        if source_type != "hf_hub" or not looks_like_hf_repo_id(repo_id):
            continue
        split = str(locator.get("split") or "").strip() or None
        results.append(_hf_hub_candidate(kind, repo_id, f"HF Hub {kind}: {repo_id}", "script_analysis", confidence, split=split, evidence=evidence))
        results.extend(
            _discover_cache_candidates(
                kind,
                repo_id,
                cache_layout,
                split=split,
                source="script_analysis",
                confidence=min(0.9, confidence + 0.02),
                evidence=evidence,
            )
        )
    return results


def _explicit_local_candidate(kind: str, root: Path, args: object, arg_name: str, label_prefix: str) -> Tuple[List[Dict[str, object]], Optional[str]]:
    candidates: List[Dict[str, object]] = []
    explicit_candidate_id = None
    explicit_path = resolve_optional_path(getattr(args, arg_name, None), root)
    if explicit_path:
        explicit_candidate = _local_candidate(kind, explicit_path, f"{label_prefix} {explicit_path.name}", "explicit_input", 0.99)
        explicit_candidate_id = explicit_candidate["id"]
        candidates.append(explicit_candidate)
    return candidates, explicit_candidate_id


def _build_config_assets(root: Path, args: object, config_candidates: List[Dict[str, object]], script_hints: Dict[str, object]) -> Dict[str, object]:
    candidates, explicit_candidate_id = _explicit_local_candidate("config", root, args, "config_path", "explicit config")
    candidates.extend(_config_candidates_from_workspace(root, config_candidates))
    candidates.extend(_script_hint_candidates("config", script_hints, {}))
    return {
        "requirement": make_asset_requirement("config", required=False, reason="config file is optional unless the launcher requires one"),
        "candidates": rank_asset_candidates(candidates),
        "explicit_candidate_id": explicit_candidate_id,
    }


def _build_remote_asset(kind: str, root: Path, args: object, config_candidates: List[Dict[str, object]], script_hints: Dict[str, object], cache_layout: Dict[str, object], target_hint: Optional[str]) -> Dict[str, object]:
    candidates: List[Dict[str, object]] = []
    explicit_candidate_id = None
    explicit_path_arg = f"{kind}_path"
    if hasattr(args, explicit_path_arg):
        explicit_candidates, explicit_candidate_id = _explicit_local_candidate(kind, root, args, explicit_path_arg, f"explicit {kind}")
        candidates.extend(explicit_candidates)

    explicit_hub_arg = f"{kind}_hub_id"
    explicit_hub = getattr(args, explicit_hub_arg, None) if hasattr(args, explicit_hub_arg) else None
    if explicit_hub:
        split = (getattr(args, "dataset_split", None) or "train") if kind == "dataset" else None
        explicit_hub_candidate = _hf_hub_candidate(kind, str(explicit_hub), f"explicit {kind} repo {explicit_hub}", "explicit_input", 0.99, split=split)
        explicit_candidate_id = explicit_hub_candidate["id"]
        candidates.append(explicit_hub_candidate)

    for name in LOCAL_WORKSPACE_DIRS.get(kind, ()):
        path = root / name
        if path.exists():
            candidates.append(_local_candidate(kind, path, f"workspace {kind} path {name}", "workspace_scan", 0.82))

    split = (getattr(args, "dataset_split", None) or "train") if kind == "dataset" else None
    candidates.extend(_config_hint_candidates(kind, root, config_candidates, repo_split=split))
    candidates.extend(_script_hint_candidates(kind, script_hints, cache_layout))

    if kind == "model":
        required = target_hint in {"training", "inference"} or bool(candidates)
        reason = "model weights or identifiers are needed before launch"
    elif kind == "dataset":
        required = target_hint == "training" or bool(candidates)
        reason = "training workloads need a dataset source before launch"
    else:
        required = bool(candidates)
        reason = "tokenizer assets are needed when the entry script loads a tokenizer explicitly"

    return {
        "requirement": make_asset_requirement(kind, required=required, reason=reason),
        "candidates": rank_asset_candidates(candidates),
        "explicit_candidate_id": explicit_candidate_id,
    }


def _build_checkpoint_assets(root: Path, args: object, config_candidates: List[Dict[str, object]], script_hints: Dict[str, object], files: List[Path]) -> Dict[str, object]:
    candidates, explicit_candidate_id = _explicit_local_candidate("checkpoint", root, args, "checkpoint_path", "explicit checkpoint")
    candidates.extend(_config_hint_candidates("checkpoint", root, config_candidates))
    candidates.extend(_script_hint_candidates("checkpoint", script_hints, {}))
    checkpoints_root = root / "checkpoints"
    if checkpoints_root.exists():
        candidates.append(_local_candidate("checkpoint", checkpoints_root, "workspace checkpoints directory", "workspace_scan", 0.78))
    for path in files:
        if path.suffix.lower() in {".ckpt", ".pt", ".bin"}:
            candidates.append(_local_candidate("checkpoint", path, f"workspace checkpoint file {path.name}", "workspace_scan", 0.7))
    return {
        "requirement": make_asset_requirement("checkpoint", required=False, reason="checkpoint is optional unless resuming from one"),
        "candidates": rank_asset_candidates(candidates),
        "explicit_candidate_id": explicit_candidate_id,
    }


def discover_asset_catalog(
    *,
    root: Path,
    files: List[Path],
    entry_candidates: List[Dict[str, object]],
    config_candidates: List[Dict[str, object]],
    args: object,
    target_hint: Optional[str],
) -> Dict[str, object]:
    entry_scripts = resolve_entry_scripts(root, files, entry_candidates)
    script_hints = analyze_entry_scripts(entry_scripts)
    cache_layout = resolve_hf_cache_layout(root)

    assets = {
        "config": _build_config_assets(root, args, config_candidates, script_hints),
        "model": _build_remote_asset("model", root, args, config_candidates, script_hints, cache_layout, target_hint),
        "tokenizer": _build_remote_asset("tokenizer", root, args, config_candidates, script_hints, cache_layout, target_hint),
        "dataset": _build_remote_asset("dataset", root, args, config_candidates, script_hints, cache_layout, target_hint),
        "checkpoint": _build_checkpoint_assets(root, args, config_candidates, script_hints, files),
    }

    return {
        "assets": assets,
        "script_hints": script_hints,
        "cache_layout": cache_layout,
        "entry_scripts": [str(path) for path in entry_scripts],
    }
