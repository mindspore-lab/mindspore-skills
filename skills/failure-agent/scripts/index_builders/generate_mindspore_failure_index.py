#!/usr/bin/env python3
"""Build the canonical MindSpore Mint API index artifacts for LLM use."""

from __future__ import annotations

import argparse
import ast
import copy
from datetime import datetime, timezone
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
FAILURE_AGENT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_OUTPUT_DIR = FAILURE_AGENT_ROOT / "reference" / "index"
DEFAULT_WORKSPACE_ROOT = SCRIPT_DIR / ".tmp" / "mindspore"
DEFAULT_REMOTE_URL = "https://atomgit.com/mindspore/mindspore.git"
GENERATOR_NAME = "generate_mindspore_failure_index.py"
GENERATOR_VERSION = "1.0.0"

PYTHON_ROOT = Path("mindspore/python")
MINT_ROOT = PYTHON_ROOT / "mindspore/mint"
API_DEF_ROOT = Path("mindspore/ops/api_def")
OP_DEF_ROOT = Path("mindspore/ops/op_def/yaml")
FUNC_OP_DEF_ROOT = Path("mindspore/ops/op_def/func_op")
ACLNN_CONFIG = Path("mindspore/python/mindspore/ops_generate/pyboost/aclnn_config.yaml")
GRAD_ROOT = Path("mindspore/ccsrc/frontend/expander/grad")
GEN_OPS = Path("mindspore/python/mindspore/ops_generate/gen_ops.py")
FALLBACK_ROOT = Path("mindspore/ops/fallback")
META_DSL_FUNC_OP_ROOT = Path("mindspore/ccsrc/frontend/operator/meta_dsl/func_op")
CPU_KERNEL_ROOT = Path("mindspore/ops/kernel/cpu")
GPU_KERNEL_ROOT = Path("mindspore/ops/kernel/gpu")
ASCEND_ACLNN_AUTO_GEN_ROOT = Path("mindspore/ops/kernel/ascend/aclnn/kernel_mod_impl/aclnn_auto_gen")
ASCEND_ACLNN_CUSTOMIZE_ROOT = Path("mindspore/ops/kernel/ascend/aclnn/kernel_mod_impl/customize")
ASCEND_ACLNN_REGISTER = Path("mindspore/ops/kernel/ascend/aclnn/kernel_mod_impl/auto_generate/aclnn_kernel_register_auto.cc")
ASCEND_PYBOOST_AUTO_GEN_ROOT = Path("mindspore/ops/kernel/ascend/aclnn/pyboost_impl/auto_generate")
ASCEND_PYBOOST_CUSTOMIZE_ROOT = Path("mindspore/ops/kernel/ascend/aclnn/pyboost_impl/customize")

PRIMITIVE_KERNEL_NAME_MAP = {
    "SumExt": ["ReduceSum"],
}

SYMBOL_OP_DEF_MAP = {
    "BCEWithLogitsLoss": ["binary_cross_entropy_with_logits_op.yaml"],
}


@dataclass
class EvidenceItem:
    path: str
    kind: str
    anchor: str
    summary: str

    def to_dict(self) -> dict[str, str]:
        return {
            "path": self.path,
            "kind": self.kind,
            "anchor": self.anchor,
            "summary": self.summary,
        }


@dataclass
class ImportBinding:
    local_name: str
    source_module: str
    source_name: str
    anchor: str

    @property
    def impl_symbol(self) -> str:
        return f"{self.source_module}.{self.source_name}"


@dataclass
class LocalBinding:
    local_name: str
    kind: str
    node: ast.AST
    anchor: str
    target_symbol: str = ""


@dataclass
class ClassExecution:
    entry: str
    calls: list[str]
    chain: list[str]
    branching_notes: list[str] = field(default_factory=list)


@dataclass
class ModuleInfo:
    module_name: str
    path: Path
    imports: dict[str, ImportBinding] = field(default_factory=dict)
    locals: dict[str, LocalBinding] = field(default_factory=dict)
    explicit_exports: list[str] = field(default_factory=list)
    extend_modules: list[str] = field(default_factory=list)
    star_imports: list[str] = field(default_factory=list)
    has_all: bool = False


@dataclass
class ResolvedExport:
    public_path: str
    module_name: str
    export_name: str
    impl_module: str
    impl_name: str
    impl_path: Optional[Path]
    api_kind: str
    source_kind: str
    evidence: list[EvidenceItem]
    resolved_symbol_chain: list[str] = field(default_factory=list)
    local_node: Optional[ast.AST] = None
    local_module: Optional[ModuleInfo] = None


@dataclass
class SymbolResolution:
    kind: Optional[str]
    impl_module: str
    impl_name: str
    impl_path: Optional[Path]
    local_node: Optional[ast.AST]
    local_module: Optional[ModuleInfo]
    chain: list[str]


def _resolution_score(target_name: str, resolution: SymbolResolution) -> tuple[int, int, int]:
    score = 0
    if resolution.kind is not None:
        score += 100
    if resolution.impl_name == target_name:
        score += 50
    if isinstance(resolution.local_node, (ast.ClassDef, ast.FunctionDef)) and resolution.local_node.name == target_name:
        score += 50
    if resolution.impl_path is not None:
        path_text = str(resolution.impl_path).replace("\\", "/").lower()
        target = target_name.lower()
        if target in path_text:
            score += 20
        module_tail = resolution.impl_module.split(".")[-1].lower()
        if module_tail in {"pooling", "loss", "activation", "normalization", "basic", "conv"}:
            score += 5
    return (score, -len(resolution.chain), 1 if resolution.kind == "class" else 0)


def _best_resolution(target_name: str, candidates: list[SymbolResolution]) -> Optional[SymbolResolution]:
    usable = [item for item in candidates if item.kind is not None or item.local_node is not None]
    if not usable:
        return None
    usable.sort(key=lambda item: _resolution_score(target_name, item), reverse=True)
    return usable[0]


class NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, data: Any) -> bool:
        return True


def yaml_dump(data: Any, path: Path) -> None:
    text = yaml.dump(data, Dumper=NoAliasDumper, allow_unicode=True, sort_keys=False)
    path.write_text(text, encoding="utf-8")


def uniq(seq: list[Any]) -> list[Any]:
    result = []
    seen = set()
    for item in seq:
        key = yaml.safe_dump(item, allow_unicode=True, sort_keys=True) if isinstance(item, (list, dict)) else repr(item)
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def _sort_scalar_list(values: list[Any]) -> list[Any]:
    if all(isinstance(item, str) for item in values):
        return sorted(values)
    return values


def _dict_sort_key(item: dict[str, Any]) -> tuple[str, ...]:
    preferred = ["api", "path", "kind", "anchor", "summary", "primitive", "op_yaml", "api_def", "dispatch"]
    parts = []
    for key in preferred:
        if key in item:
            parts.append(str(item.get(key, "")))
    if not parts:
        for key in sorted(item):
            parts.append(f"{key}={item[key]}")
    return tuple(parts)


def canonicalize_for_yaml(data: Any, parent_key: str = "") -> Any:
    sort_scalar_keys = {
        "primitive",
        "possible_primitives",
        "composed_of",
        "prelude_calls",
        "terminal_calls",
        "resolved_symbol_chain",
        "resolved_execution_chain",
        "func_op_expands_to",
        "flags",
        "interfaces",
        "effective_interfaces",
        "op_yamls",
        "meta_dsl_paths",
        "expanded_primitives",
    }
    sort_dict_list_keys = {
        "source",
        "primitive_sources",
        "dispatch_detail",
        "aclnn_evidence",
        "func_op_expansion_evidence",
        "grad",
        "notes",
    }
    if isinstance(data, dict):
        return {key: canonicalize_for_yaml(value, key) for key, value in data.items()}
    if isinstance(data, list):
        normalized = [canonicalize_for_yaml(item, parent_key) for item in data]
        if parent_key == "apis" and all(isinstance(item, dict) and "api" in item for item in normalized):
            return sorted(normalized, key=lambda item: str(item["api"]))
        if parent_key in sort_scalar_keys:
            return _sort_scalar_list(normalized)
        if parent_key in sort_dict_list_keys and all(isinstance(item, dict) for item in normalized):
            return sorted(normalized, key=_dict_sort_key)
        if all(isinstance(item, dict) and {"path", "kind", "anchor", "summary"}.issubset(item.keys()) for item in normalized):
            return sorted(normalized, key=_dict_sort_key)
        return normalized
    return data


def repo_commit_hint(repo_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return ""
    return result.stdout.strip()


def mindspore_version_hint(repo_root: Path) -> str:
    version_path = repo_root / "version.txt"
    if version_path.exists():
        return version_path.read_text(encoding="utf-8").strip()
    return ""


def camel_to_snake(name: str) -> str:
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    name = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    name = name.replace("__", "_").lower()
    name = re.sub(r"(\d)_([a-z])(?=_|$)", r"\1\2", name)
    return name


class SourceIndex:
    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.module_cache: dict[str, Optional[ModuleInfo]] = {}
        self.api_defs = self._load_api_defs()
        self.op_defs = self._load_op_defs()
        self.function_name_to_op_yaml = self._build_op_field_map("function", "name")
        self.class_name_to_op_yaml = self._build_op_field_map("class", "name")
        self.op_name_to_op_yaml = self._build_op_name_map()
        self.aclnn_map = self._load_yaml_map(repo_root / ACLNN_CONFIG)
        self.bprop_map = self._load_bprop_map()
        self.fallback_map = self._load_fallback_map()
        self.cpu_kernel_map = self._load_kernel_registry_map("cpu")
        self.gpu_kernel_map = self._load_kernel_registry_map("gpu")
        self.ascend_kbk_map = self._load_ascend_kbk_map()
        self.generated_primitive_names = self._load_generated_primitive_names()

    def relpath(self, path: Path) -> str:
        return path.resolve().relative_to(self.repo_root.resolve()).as_posix()

    def _module_path(self, module_name: str) -> Optional[Path]:
        rel = Path(*module_name.split("."))
        init_path = self.repo_root / PYTHON_ROOT / rel / "__init__.py"
        if init_path.exists():
            return init_path
        file_path = self.repo_root / PYTHON_ROOT / f"{rel}.py"
        if file_path.exists():
            return file_path
        return None

    def _load_yaml_map(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}

    def _load_api_defs(self) -> dict[str, dict[str, Any]]:
        root = self.repo_root / API_DEF_ROOT
        result = {}
        for path in sorted(root.glob("*.yaml")):
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or len(data) != 1:
                continue
            name = next(iter(data))
            result[name] = {"path": path, "payload": data[name]}
        return result

    def _load_op_defs(self) -> dict[str, dict[str, Any]]:
        result = {}
        for root in (self.repo_root / OP_DEF_ROOT, self.repo_root / FUNC_OP_DEF_ROOT):
            if not root.exists():
                continue
            for path in sorted(root.glob("*.yaml")):
                data = yaml.safe_load(path.read_text(encoding="utf-8"))
                if not isinstance(data, dict) or len(data) != 1:
                    continue
                name = next(iter(data))
                result[path.name] = {"op_name": name, "path": path, "payload": data[name]}
        return result

    def _build_op_field_map(self, section: str, field: str) -> dict[str, list[str]]:
        result: dict[str, list[str]] = defaultdict(list)
        for op_yaml, info in self.op_defs.items():
            payload = info["payload"]
            if not isinstance(payload, dict):
                continue
            block = payload.get(section)
            if not isinstance(block, dict):
                continue
            value = block.get(field)
            if isinstance(value, str) and value:
                result[value].append(op_yaml)
        return {key: uniq(values) for key, values in result.items()}

    def _build_op_name_map(self) -> dict[str, list[str]]:
        result: dict[str, list[str]] = defaultdict(list)
        for op_yaml, info in self.op_defs.items():
            result[str(info["op_name"])].append(op_yaml)
        return {key: uniq(values) for key, values in result.items()}

    def _load_generated_primitive_names(self) -> set[str]:
        module = self.load_module("mindspore.ops.auto_generate.gen_ops_prim")
        if module is None:
            return set()
        return {
            name
            for name, binding in module.locals.items()
            if binding.kind == "class" and isinstance(binding.node, ast.ClassDef)
        }

    def _load_bprop_map(self) -> dict[str, list[EvidenceItem]]:
        result: dict[str, list[EvidenceItem]] = defaultdict(list)
        for path in sorted((self.repo_root / GRAD_ROOT).glob("*.cc")):
            text = path.read_text(encoding="utf-8")
            for segment in text.split("REG_BPROP_BUILDER(")[1:]:
                quote = segment.split(")", 1)[0]
                if not quote.startswith('"'):
                    continue
                primitive = quote.split('"')[1]
                anchor = f'REG_BPROP_BUILDER("{primitive}")'
                result[primitive].append(
                    EvidenceItem(self.relpath(path), "direct", anchor, f"bprop builder for {primitive}")
                )
        return dict(result)

    def _load_fallback_map(self) -> dict[str, list[EvidenceItem]]:
        result: dict[str, list[EvidenceItem]] = defaultdict(list)
        root = self.repo_root / FALLBACK_ROOT
        if not root.exists():
            return {}
        for path in sorted(root.glob("*.cc")):
            text = path.read_text(encoding="utf-8")
            for match in re.finditer(r'REG_FALLBACK_BUILDER\("([^"]+)"\)', text):
                primitive = match.group(1)
                anchor = f'REG_FALLBACK_BUILDER("{primitive}")'
                result[primitive].append(
                    EvidenceItem(self.relpath(path), "direct", anchor, f"fallback builder for {primitive}")
                )
        return dict(result)

    def _load_kernel_registry_map(self, backend: str) -> dict[str, list[EvidenceItem]]:
        result: dict[str, list[EvidenceItem]] = defaultdict(list)
        if backend == "cpu":
            root = self.repo_root / CPU_KERNEL_ROOT
            patterns = [
                r"MS_KERNEL_FACTORY_REG_BY_CREATOR\(NativeCpuKernelMod,\s*([A-Za-z0-9_]+)",
                r"MS_KERNEL_FACTORY_REG\(NativeCpuKernelMod,\s*([A-Za-z0-9_]+)",
            ]
        else:
            root = self.repo_root / GPU_KERNEL_ROOT
            patterns = [
                r"MS_KERNEL_FACTORY_REG_BY_CREATOR\(NativeGpuKernelMod,\s*([A-Za-z0-9_]+)",
                r"MS_KERNEL_FACTORY_REG\(NativeGpuKernelMod,\s*([A-Za-z0-9_]+)",
            ]
        if not root.exists():
            return {}
        for path in sorted(root.rglob("*.cc")):
            text = path.read_text(encoding="utf-8", errors="ignore")
            for pattern in patterns:
                for match in re.finditer(pattern, text):
                    kernel_name = match.group(1)
                    anchor = match.group(0)
                    result[kernel_name].append(
                        EvidenceItem(self.relpath(path), "direct", anchor, f"{backend.upper()} kernel factory for {kernel_name}")
                    )
        return dict(result)

    def _load_ascend_kbk_map(self) -> dict[str, list[EvidenceItem]]:
        result: dict[str, list[EvidenceItem]] = defaultdict(list)
        for root, label in (
            (self.repo_root / ASCEND_ACLNN_AUTO_GEN_ROOT, "aclnn_auto_gen"),
            (self.repo_root / ASCEND_ACLNN_CUSTOMIZE_ROOT, "customize"),
        ):
            if not root.exists():
                continue
            for path in sorted(root.rglob("*")):
                if not path.is_file():
                    continue
                key = path.stem.lower()
                result[key].append(EvidenceItem(self.relpath(path), "direct", path.name, f"Ascend KBK {label} artifact"))
                if label == "customize" and path.suffix in {".cc", ".h"}:
                    text = path.read_text(encoding="utf-8", errors="ignore")
                    for match in re.finditer(r"MS_ACLNN_KERNEL_FACTORY_REG\(\s*([A-Za-z0-9_]+)\s*,\s*([A-Za-z0-9_]+)\s*\)", text):
                        primitive = match.group(1)
                        result[primitive.lower()].append(
                            EvidenceItem(self.relpath(path), "direct", match.group(0), f"Ascend KBK customize register entry for {primitive}")
                        )
        register_path = self.repo_root / ASCEND_ACLNN_REGISTER
        if register_path.exists():
            text = register_path.read_text(encoding="utf-8", errors="ignore")
            patterns = [
                r"MS_ACLNN_KERNEL_FACTORY_REG\(\s*([A-Za-z0-9_]+)\s*,\s*([A-Za-z0-9_]+)\s*\)",
                r"MS_ACLNN_COMMON_KERNEL_FACTORY_REG\(\s*([A-Za-z0-9_]+)\s*,\s*(aclnn[A-Za-z0-9_]+)\s*,\s*\d+\s*\)",
            ]
            for pattern in patterns:
                for match in re.finditer(pattern, text):
                    primitive = match.group(1)
                    anchor = match.group(0)
                    result[primitive.lower()].append(
                        EvidenceItem(self.relpath(register_path), "direct", anchor, f"Ascend KBK register entry for {primitive}")
                    )
        return dict(result)

    def load_module(self, module_name: str) -> Optional[ModuleInfo]:
        if module_name in self.module_cache:
            return self.module_cache[module_name]
        path = self._module_path(module_name)
        if path is None:
            self.module_cache[module_name] = None
            return None
        tree = ast.parse(path.read_text(encoding="utf-8"))
        info = ModuleInfo(module_name=module_name, path=path)
        imported_modules: dict[str, str] = {}
        for node in tree.body:
            if isinstance(node, ast.ImportFrom):
                source_module = self._resolve_import_from(module_name, node.module or "", node.level)
                for alias in node.names:
                    if alias.name == "*":
                        info.star_imports.append(source_module)
                        continue
                    local_name = alias.asname or alias.name
                    anchor = f"from {source_module} import {alias.name}"
                    info.imports[local_name] = ImportBinding(local_name, source_module, alias.name, anchor)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imported_modules[alias.asname or alias.name.split(".")[-1]] = alias.name
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                info.locals[node.name] = LocalBinding(node.name, "function", node, f"def {node.name}")
            elif isinstance(node, ast.ClassDef):
                info.locals[node.name] = LocalBinding(node.name, "class", node, f"class {node.name}")
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "__all__":
                        values = self._string_list(node.value)
                        if values is not None:
                            info.has_all = True
                            info.explicit_exports.extend(values)
                    elif isinstance(target, ast.Name) and isinstance(node.value, ast.Call):
                        target_symbol = extract_call_symbol(node.value, info) or ""
                        if target_symbol:
                            info.locals[target.id] = LocalBinding(
                                target.id,
                                "assigned_call",
                                node,
                                f"{target.id} = call",
                                target_symbol=target_symbol,
                            )
            elif isinstance(node, ast.AugAssign):
                if isinstance(node.target, ast.Name) and node.target.id == "__all__":
                    values = self._string_list(node.value)
                    if values is not None:
                        info.has_all = True
                        info.explicit_exports.extend(values)
            elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                call = node.value
                if isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name):
                    if call.func.value.id == "__all__" and call.func.attr == "extend" and len(call.args) == 1:
                        arg = call.args[0]
                        if isinstance(arg, ast.Attribute) and arg.attr == "__all__" and isinstance(arg.value, ast.Name):
                            import_name = arg.value.id
                            binding = info.imports.get(import_name)
                            if binding is not None:
                                info.has_all = True
                                info.extend_modules.append(binding.source_module)
                            elif import_name in imported_modules:
                                info.has_all = True
                                info.extend_modules.append(imported_modules[import_name])
        self.module_cache[module_name] = info
        return info

    def _resolve_import_from(self, current_module: str, module: str, level: int) -> str:
        if level == 0:
            return module
        current_path = self._module_path(current_module)
        parts = current_module.split(".")
        base = parts if current_path and current_path.name == "__init__.py" else parts[:-1]
        prefix = base[: len(base) - level + 1]
        if module:
            prefix.extend(module.split("."))
        return ".".join(prefix)

    def _string_list(self, node: ast.AST) -> Optional[list[str]]:
        if not isinstance(node, (ast.List, ast.Tuple)):
            return None
        values = []
        for elt in node.elts:
            if not isinstance(elt, ast.Constant) or not isinstance(elt.value, str):
                return None
            values.append(elt.value)
        return values


def infer_api_kind(module_name: str, export_name: str, hint: str) -> str:
    if hint == "class" or export_name[:1].isupper():
        return "class"
    return "function"


def classify_source_kind(source_module: str, api_kind: str) -> str:
    if api_kind == "class":
        return "class_api"
    if source_module.startswith("mindspore.ops.auto_generate"):
        return "generated_binding"
    if source_module.startswith("mindspore.ops.function") or source_module.startswith("mindspore.ops.functional"):
        return "ops_binding"
    return "python_wrapper"


def resolve_symbol(
    index: SourceIndex,
    module_name: str,
    symbol_name: str,
    visited: Optional[set[tuple[str, str]]] = None,
) -> SymbolResolution:
    visited = set() if visited is None else set(visited)
    key = (module_name, symbol_name)
    if key in visited:
        return SymbolResolution(None, module_name, symbol_name, index._module_path(module_name), None, None, [f"{module_name}.{symbol_name}"])
    visited.add(key)
    module = index.load_module(module_name)
    if module is None:
        return SymbolResolution(None, module_name, symbol_name, None, None, None, [f"{module_name}.{symbol_name}"])
    current = f"{module_name}.{symbol_name}"
    if symbol_name in module.locals:
        binding = module.locals[symbol_name]
        if binding.kind == "assigned_call" and binding.target_symbol:
            target_parts = binding.target_symbol.split(".")
            if len(target_parts) >= 2:
                resolved = resolve_symbol(index, ".".join(target_parts[:-1]), target_parts[-1], visited)
                return SymbolResolution(
                    resolved.kind,
                    resolved.impl_module,
                    resolved.impl_name,
                    resolved.impl_path,
                    resolved.local_node,
                    resolved.local_module,
                    [current] + resolved.chain,
                )
        return SymbolResolution(binding.kind, module_name, symbol_name, module.path, binding.node, module, [current])
    if symbol_name in module.imports:
        binding = module.imports[symbol_name]
        resolved = resolve_symbol(index, binding.source_module, binding.source_name, visited)
        return SymbolResolution(
            resolved.kind,
            resolved.impl_module,
            resolved.impl_name,
            resolved.impl_path,
            resolved.local_node,
            resolved.local_module,
            [current] + resolved.chain,
        )
    star_candidates = []
    for star_module in module.star_imports:
        resolved = resolve_symbol(index, star_module, symbol_name, visited)
        if resolved.kind is not None or resolved.local_node is not None:
            star_candidates.append(
                SymbolResolution(
                    resolved.kind,
                    resolved.impl_module,
                    resolved.impl_name,
                    resolved.impl_path,
                    resolved.local_node,
                    resolved.local_module,
                    [current] + resolved.chain,
                )
            )
    best = _best_resolution(symbol_name, star_candidates)
    if best is not None:
        return best
    return SymbolResolution(None, module_name, symbol_name, index._module_path(module_name), None, module, [current])


def resolve_call_alias(index: SourceIndex, symbol: str, depth: int = 0) -> str:
    if depth > 6:
        return symbol
    parts = symbol.split(".")
    if len(parts) < 2:
        return symbol
    for split in range(len(parts) - 1, 0, -1):
        module_name = ".".join(parts[:split])
        remainder = parts[split:]
        if index._module_path(module_name) is None:
            continue
        current_module = module_name
        current_symbol_parts = remainder
        changed = False
        while len(current_symbol_parts) > 1:
            head = current_symbol_parts[0]
            resolved = resolve_symbol(index, current_module, head)
            if resolved.impl_path is None:
                break
            next_module = f"{resolved.impl_module}.{resolved.impl_name}"
            if index._module_path(next_module) is None:
                break
            current_module = next_module
            current_symbol_parts = current_symbol_parts[1:]
            changed = True
        if changed:
            normalized = f"{current_module}.{'.'.join(current_symbol_parts)}"
            if normalized != symbol:
                return resolve_call_alias(index, normalized, depth + 1)
        break
    return symbol


def resolve_export(index: SourceIndex, module: ModuleInfo, export_name: str) -> Optional[ResolvedExport]:
    public_path = f"{module.module_name}.{export_name}"
    if export_name in module.locals:
        binding = module.locals[export_name]
        api_kind = infer_api_kind(module.module_name, export_name, binding.kind)
        return ResolvedExport(
            public_path,
            module.module_name,
            export_name,
            module.module_name,
            export_name,
            module.path,
            api_kind,
            "class_api" if binding.kind == "class" else "python_wrapper",
            [EvidenceItem(index.relpath(module.path), "direct", binding.anchor, f"local {binding.kind} export")],
            resolved_symbol_chain=[public_path],
            local_node=binding.node,
            local_module=module,
        )
    if export_name in module.imports:
        binding = module.imports[export_name]
        resolved = resolve_symbol(index, binding.source_module, binding.source_name)
        api_kind = infer_api_kind(module.module_name, export_name, resolved.kind or "function")
        return ResolvedExport(
            public_path,
            module.module_name,
            export_name,
            resolved.impl_module,
            resolved.impl_name,
            resolved.impl_path,
            api_kind,
            classify_source_kind(resolved.impl_module, api_kind),
            [EvidenceItem(index.relpath(module.path), "direct", binding.anchor, "public export binding")],
            resolved_symbol_chain=[public_path] + resolved.chain,
            local_node=resolved.local_node,
            local_module=resolved.local_module,
        )
    star_candidates: list[tuple[str, SymbolResolution]] = []
    for star_module in module.star_imports:
        resolved = resolve_symbol(index, star_module, export_name)
        if resolved.kind is not None or resolved.local_node is not None:
            star_candidates.append((star_module, resolved))
    if star_candidates:
        best_source, best = max(star_candidates, key=lambda item: _resolution_score(export_name, item[1]))
        api_kind = infer_api_kind(module.module_name, export_name, best.kind or "function")
        return ResolvedExport(
            public_path,
            module.module_name,
            export_name,
            best.impl_module,
            best.impl_name,
            best.impl_path,
            api_kind,
            classify_source_kind(best.impl_module, api_kind),
            [EvidenceItem(index.relpath(module.path), "derived", f"from {best_source} import *", "resolved through star import")],
            resolved_symbol_chain=[public_path] + best.chain,
            local_node=best.local_node,
            local_module=best.local_module,
        )
    return None


def expand_exports(index: SourceIndex, module_name: str, visiting: Optional[set[str]] = None) -> list[ResolvedExport]:
    visiting = set() if visiting is None else set(visiting)
    if module_name in visiting:
        return []
    visiting.add(module_name)
    module = index.load_module(module_name)
    if module is None:
        return []
    result = []
    for export_name in module.explicit_exports:
        resolved = resolve_export(index, module, export_name)
        if resolved is not None:
            result.append(resolved)
    for nested in module.extend_modules:
        for item in expand_exports(index, nested, visiting):
            result.append(
                ResolvedExport(
                    public_path=f"{module_name}.{item.export_name}",
                    module_name=module_name,
                    export_name=item.export_name,
                    impl_module=item.impl_module,
                    impl_name=item.impl_name,
                    impl_path=item.impl_path,
                    api_kind=item.api_kind,
                    source_kind=item.source_kind,
                    evidence=[EvidenceItem(index.relpath(module.path), "derived", f"__all__.extend({nested}.__all__)", "re-export nested __all__")] + item.evidence,
                    local_node=item.local_node,
                    local_module=item.local_module,
                )
            )
    return result


def gather_exports(index: SourceIndex) -> list[ResolvedExport]:
    seen = set()
    result = []
    for path in sorted((index.repo_root / MINT_ROOT).rglob("*.py")):
        if path.name.startswith("_") and path.name != "__init__.py":
            continue
        rel = path.relative_to(index.repo_root / PYTHON_ROOT)
        parts = list(rel.parts)
        if parts[-1] == "__init__.py":
            parts = parts[:-1]
        else:
            parts[-1] = parts[-1][:-3]
        module_name = ".".join(parts)
        module = index.load_module(module_name)
        if module is None or not module.has_all:
            continue
        for export in expand_exports(index, module_name):
            if export.public_path in seen:
                continue
            seen.add(export.public_path)
            result.append(export)
    return sorted(result, key=lambda item: item.public_path)


def collect_call_details(export: ResolvedExport) -> tuple[list[str], list[str]]:
    if export.local_node is None or export.local_module is None:
        return [], []
    all_calls: list[str] = []
    terminal_calls: list[str] = []

    class Visitor(ast.NodeVisitor):
        def visit_Return(self, node: ast.Return) -> None:
            if isinstance(node.value, ast.Call):
                symbol = extract_call_symbol(node.value, export.local_module)
                if symbol and not symbol.startswith(("Tensor", "self.", "ops.", "mindspore.Tensor")):
                    terminal_calls.append(symbol)
            self.generic_visit(node)

        def visit_Call(self, node: ast.Call) -> None:
            symbol = extract_call_symbol(node, export.local_module)
            if symbol and not symbol.startswith(("Tensor", "self.", "ops.", "mindspore.Tensor")):
                all_calls.append(symbol)
            self.generic_visit(node)

    Visitor().visit(export.local_node)
    if terminal_calls:
        prelude = []
        remaining = list(terminal_calls)
        for call in all_calls:
            if call in remaining:
                remaining.remove(call)
            else:
                prelude.append(call)
        return uniq(prelude), uniq(terminal_calls)
    return [], uniq(all_calls)


def infer_primitives_from_symbol_name(index: SourceIndex, symbol_name: str) -> list[str]:
    results: list[str] = []
    for name in uniq(op_yamls_from_index(index, symbol_name) + op_def_candidates_from_symbol(symbol_name)):
        primitive, _ = primitive_from_op(index, name)
        if primitive:
            results.append(primitive)
    generated = infer_generated_primitive(index, symbol_name)
    if generated:
        results.append(generated)
    return uniq(results)


def infer_possible_primitives_from_symbol(
    index: SourceIndex, symbol: str, depth: int = 0, visited: Optional[set[str]] = None
) -> list[str]:
    if depth > 2:
        return []
    visited = set() if visited is None else set(visited)
    if symbol in visited:
        return []
    visited.add(symbol)

    results = infer_primitives_from_symbol_name(index, symbol)
    parts = symbol.split(".")
    if len(parts) < 2:
        return uniq(results)
    resolution = None
    for split in range(len(parts) - 1, 0, -1):
        module_name = ".".join(parts[:split])
        symbol_name = ".".join(parts[split:])
        if index._module_path(module_name) is None:
            continue
        resolution = resolve_symbol(index, module_name, symbol_name.split(".")[0])
        break
    if resolution is None or resolution.local_node is None or resolution.local_module is None:
        return uniq(results)
    if isinstance(resolution.local_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        pseudo_export = ResolvedExport(
            public_path=symbol,
            module_name=resolution.local_module.module_name,
            export_name=resolution.impl_name,
            impl_module=resolution.impl_module,
            impl_name=resolution.impl_name,
            impl_path=resolution.impl_path,
            api_kind="function",
            source_kind="python_wrapper",
            evidence=[],
            local_node=resolution.local_node,
            local_module=resolution.local_module,
        )
        prelude, terminals = collect_call_details(pseudo_export)
        for call in uniq(prelude + terminals):
            normalized = resolve_call_alias(index, call)
            results.extend(infer_primitives_from_symbol_name(index, normalized))
            if isinstance(resolution.local_node, ast.FunctionDef):
                nested = find_nested_function(resolution.local_node, normalized.split(".")[-1])
                if nested is not None:
                    nested_export = ResolvedExport(
                        public_path=normalized,
                        module_name=resolution.local_module.module_name,
                        export_name=nested.name,
                        impl_module=resolution.local_module.module_name,
                        impl_name=nested.name,
                        impl_path=resolution.local_module.path,
                        api_kind="function",
                        source_kind="python_wrapper",
                        evidence=[],
                        local_node=nested,
                        local_module=resolution.local_module,
                    )
                    nested_prelude, nested_terminals = collect_call_details(nested_export)
                    for nested_call in uniq(nested_prelude + nested_terminals):
                        nested_normalized = resolve_call_alias(index, nested_call)
                        results.extend(infer_primitives_from_symbol_name(index, nested_normalized))
                        results.extend(
                            infer_possible_primitives_from_symbol(index, nested_normalized, depth + 1, visited)
                        )
            results.extend(infer_possible_primitives_from_symbol(index, normalized, depth + 1, visited))
    return uniq(results)


def attribute_chain(node: ast.AST) -> Optional[list[str]]:
    parts = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
        return list(reversed(parts))
    return None


def extract_call_symbol(node: ast.Call, module: ModuleInfo, locals_map: Optional[dict[str, LocalBinding]] = None) -> Optional[str]:
    func = node.func
    locals_map = module.locals if locals_map is None else locals_map
    if isinstance(func, ast.Name):
        binding = module.imports.get(func.id)
        if binding is not None:
            return binding.impl_symbol
        if func.id in locals_map:
            return f"{module.module_name}.{func.id}"
    chain = attribute_chain(func)
    if chain:
        base_name = chain[0]
        suffix = ".".join(chain[1:])
        binding = module.imports.get(base_name)
        if binding is not None:
            return binding.impl_symbol if not suffix else f"{binding.impl_symbol}.{suffix}"
        if base_name == "self":
            return base_name if not suffix else f"{base_name}.{suffix}"
        if base_name in locals_map:
            local_symbol = f"{module.module_name}.{base_name}"
            return local_symbol if not suffix else f"{local_symbol}.{suffix}"
    return None


def find_class_method(class_node: ast.ClassDef, method_name: str) -> Optional[ast.FunctionDef]:
    for item in class_node.body:
        if isinstance(item, ast.FunctionDef) and item.name == method_name:
            return item
    return None


def resolve_class_node(
    class_node: ast.ClassDef,
    module: ModuleInfo,
    class_name: str,
    visited: Optional[set[tuple[str, str]]] = None,
) -> Optional[tuple[ast.ClassDef, ModuleInfo]]:
    visited = set() if visited is None else set(visited)
    key = (module.module_name, class_name)
    if key in visited:
        return None
    visited.add(key)
    if class_node.name == class_name:
        return class_node, module
    local = module.locals.get(class_name)
    if local is not None and isinstance(local.node, ast.ClassDef):
        return local.node, module
    binding = module.imports.get(class_name)
    if binding is not None:
        source_module = binding.source_module
        source_name = binding.source_name
        source_info = module if source_module == module.module_name else None
        if source_info is None:
            return None
        if source_name in source_info.locals and isinstance(source_info.locals[source_name].node, ast.ClassDef):
            return source_info.locals[source_name].node, source_info
    return None


def iter_class_hierarchy(class_node: ast.ClassDef, module: ModuleInfo, visited: Optional[set[tuple[str, str]]] = None) -> list[tuple[ast.ClassDef, ModuleInfo]]:
    visited = set() if visited is None else set(visited)
    key = (module.module_name, class_node.name)
    if key in visited:
        return []
    visited.add(key)
    chain = [(class_node, module)]
    for base in class_node.bases:
        base_name = None
        if isinstance(base, ast.Name):
            base_name = base.id
        else:
            chain_parts = attribute_chain(base)
            if chain_parts:
                base_name = chain_parts[-1]
        if not base_name:
            continue
        resolved = resolve_class_node(class_node, module, base_name, visited)
        if resolved is not None:
            base_node, base_module = resolved
            chain.extend(iter_class_hierarchy(base_node, base_module, visited))
    return chain


def find_method_in_hierarchy(class_node: ast.ClassDef, module: ModuleInfo, method_name: str) -> Optional[tuple[ast.FunctionDef, ast.ClassDef, ModuleInfo]]:
    for node, owner_module in iter_class_hierarchy(class_node, module):
        method = find_class_method(node, method_name)
        if method is not None:
            return method, node, owner_module
    return None


def analyze_init_callables(class_node: ast.ClassDef, module: ModuleInfo) -> dict[str, list[tuple[str, str]]]:
    attrs: dict[str, list[tuple[str, str]]] = defaultdict(list)
    hierarchy = iter_class_hierarchy(class_node, module)

    def walk(stmts: list[ast.stmt], owner_module: ModuleInfo, condition: str = "") -> None:
        for stmt in stmts:
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
                target = stmt.targets[0]
                if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self":
                    symbol = None
                    value = stmt.value
                    if isinstance(value, ast.Call):
                        symbol = extract_call_symbol(value, owner_module)
                    elif isinstance(value, ast.Name):
                        binding = owner_module.imports.get(value.id)
                        if binding is not None:
                            symbol = binding.impl_symbol
                        elif value.id in owner_module.locals:
                            symbol = f"{owner_module.module_name}.{value.id}"
                    elif isinstance(value, ast.Attribute):
                        chain = attribute_chain(value)
                        if chain and chain[0] != "self":
                            base = owner_module.imports.get(chain[0])
                            if base is not None:
                                symbol = base.impl_symbol if len(chain) == 1 else f"{base.impl_symbol}.{'.'.join(chain[1:])}"
                    if symbol:
                        attrs[target.attr].append((symbol, condition))
            elif isinstance(stmt, ast.If):
                try:
                    cond = ast.unparse(stmt.test)
                except Exception:
                    cond = "branch_condition"
                walk(stmt.body, owner_module, cond)
                else_cond = f"not ({cond})"
                walk(stmt.orelse, owner_module, else_cond)

    for node, owner_module in reversed(hierarchy):
        init_method = find_class_method(node, "__init__")
        if init_method is not None:
            walk(init_method.body, owner_module)
    return {key: uniq(value) for key, value in attrs.items()}


def resolve_qualified_symbol(index: SourceIndex, symbol: str) -> tuple[str, Optional[SymbolResolution]]:
    normalized = resolve_call_alias(index, symbol)
    parts = normalized.split(".")
    for split in range(len(parts) - 1, 0, -1):
        module_name = ".".join(parts[:split])
        if index._module_path(module_name) is None:
            continue
        remainder = parts[split:]
        if len(remainder) != 1:
            continue
        return normalized, resolve_symbol(index, module_name, remainder[0])
    return normalized, None


def expand_bound_construct_target(
    index: SourceIndex,
    symbol: str,
    depth: int = 1,
) -> tuple[list[str], list[str], list[str]]:
    normalized, resolved = resolve_qualified_symbol(index, symbol)
    if depth <= 0 or resolved is None or not isinstance(resolved.local_node, ast.ClassDef) or resolved.local_module is None:
        return [normalized], [], []
    construct_info = find_method_in_hierarchy(resolved.local_node, resolved.local_module, "construct")
    if construct_info is None:
        return [normalized], [], []
    construct, owner_class, owner_module = construct_info
    direct_calls = extract_return_calls(construct, owner_module)
    if not direct_calls:
        return [normalized], [], []
    init_callables = analyze_init_callables(resolved.local_node, resolved.local_module)
    resolved_calls = []
    chain = [f"{resolved.impl_module}.{resolved.impl_name}.construct"]
    if owner_class.name != resolved.local_node.name or owner_module.module_name != resolved.local_module.module_name:
        chain.append(f"{owner_module.module_name}.{owner_class.name}.construct")
    branching_notes = []
    for call in direct_calls:
        if call.startswith("self."):
            member_name = call.split(".", 1)[1]
            helper_info = find_method_in_hierarchy(resolved.local_node, resolved.local_module, member_name)
            if helper_info is not None:
                helper, helper_class, helper_module = helper_info
                helper_calls = extract_return_calls(helper, helper_module)
                chain.append(call)
                if helper_class.name != resolved.local_node.name or helper_module.module_name != resolved.local_module.module_name:
                    chain.append(f"{helper_module.module_name}.{helper_class.name}.{member_name}")
                resolved_calls.extend(resolve_call_alias(index, item) for item in helper_calls)
                continue
            if member_name in init_callables:
                chain.append(call)
                bound = init_callables[member_name]
                for bound_symbol, condition in bound:
                    nested_calls, nested_chain, nested_notes = expand_bound_construct_target(index, bound_symbol, depth - 1)
                    resolved_calls.extend(nested_calls)
                    chain.extend(nested_chain)
                    branching_notes.extend(nested_notes)
                if len(bound) > 1:
                    note = f"{call} may resolve to " + ", ".join(
                        f"{bound_symbol} [{condition or 'default'}]" for bound_symbol, condition in bound
                    )
                    branching_notes.append(note)
                continue
        resolved_calls.append(resolve_call_alias(index, call))
    return uniq(resolved_calls) or [normalized], uniq(chain), uniq(branching_notes)


def extract_return_calls(method_node: ast.FunctionDef, module: ModuleInfo) -> list[str]:
    calls = []
    all_assignments: dict[str, list[str]] = defaultdict(list)
    nested_locals = {
        stmt.name: LocalBinding(stmt.name, "function", stmt, stmt.name)
        for stmt in method_node.body
        if isinstance(stmt, ast.FunctionDef)
    }

    def add_assignment(name: str, symbol: str) -> None:
        all_assignments[name].append(symbol)

    def walk(stmts: list[ast.stmt], scope: dict[str, str]) -> None:
        for stmt in stmts:
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                target_name = stmt.targets[0].id
                if isinstance(stmt.value, ast.Call):
                    symbol = extract_call_symbol(stmt.value, module, {**module.locals, **nested_locals})
                    if symbol:
                        scope[target_name] = symbol
                        add_assignment(target_name, symbol)
                elif isinstance(stmt.value, ast.Name) and stmt.value.id in scope:
                    scope[target_name] = scope[stmt.value.id]
                    add_assignment(target_name, scope[target_name])
            elif isinstance(stmt, ast.If):
                walk(stmt.body, dict(scope))
                walk(stmt.orelse, dict(scope))
            elif isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.Call):
                symbol = extract_call_symbol(stmt.value, module, {**module.locals, **nested_locals})
                if symbol:
                    calls.append(symbol)
            elif isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.Name):
                symbols = []
                if stmt.value.id in scope:
                    symbols.append(scope[stmt.value.id])
                symbols.extend(all_assignments.get(stmt.value.id, []))
                calls.extend(symbols)

    walk(method_node.body, {})
    return uniq(calls)


def find_nested_function(function_node: ast.FunctionDef, function_name: str) -> Optional[ast.FunctionDef]:
    for node in ast.walk(function_node):
        if isinstance(node, ast.FunctionDef) and node is not function_node and node.name == function_name:
            return node
    return None


def analyze_class_construct(index: SourceIndex, export: ResolvedExport, helper_depth: int = 1) -> Optional[ClassExecution]:
    if export.local_node is None or export.local_module is None or not isinstance(export.local_node, ast.ClassDef):
        return None
    construct_info = find_method_in_hierarchy(export.local_node, export.local_module, "construct")
    if construct_info is None:
        return None
    construct, owner_class, owner_module = construct_info
    direct_calls = extract_return_calls(construct, owner_module)
    if not direct_calls:
        return None
    init_callables = analyze_init_callables(export.local_node, export.local_module)
    chain = [f"{export.public_path}.construct"]
    if owner_class.name != export.local_node.name or owner_module.module_name != export.local_module.module_name:
        chain.append(f"{owner_module.module_name}.{owner_class.name}.construct")
    resolved = []
    branching_notes = []
    for call in direct_calls:
        if call.startswith("self.") and helper_depth > 0:
            member_name = call.split(".", 1)[1]
            helper_info = find_method_in_hierarchy(export.local_node, export.local_module, member_name)
            if helper_info is not None:
                helper, helper_class, helper_module = helper_info
                helper_calls = extract_return_calls(helper, helper_module)
                chain.append(call)
                if helper_class.name != export.local_node.name or helper_module.module_name != export.local_module.module_name:
                    chain.append(f"{helper_module.module_name}.{helper_class.name}.{member_name}")
                resolved.extend(helper_calls)
                continue
            if member_name in init_callables:
                chain.append(call)
                bound = init_callables[member_name]
                for symbol, _ in bound:
                    nested_calls, nested_chain, nested_notes = expand_bound_construct_target(index, symbol, depth=1)
                    resolved.extend(nested_calls)
                    chain.extend(nested_chain)
                    branching_notes.extend(nested_notes)
                if len(bound) > 1:
                    note = f"{call} may resolve to " + ", ".join(
                        f"{symbol} [{condition or 'default'}]" for symbol, condition in bound
                    )
                    branching_notes.append(note)
                continue
        resolved.append(call)
    return ClassExecution(entry="construct", calls=uniq(resolved), chain=chain + uniq(resolved), branching_notes=uniq(branching_notes))


def normalize_candidates(name: str) -> list[str]:
    candidates = [name.split(".")[-1]]
    stripped = candidates[0]
    for suffix in ("_ext", "_op", "_view", "_impl", "_v2"):
        if stripped.endswith(suffix):
            stripped = stripped[: -len(suffix)]
    candidates.append(stripped)
    if stripped.endswith("_"):
        candidates.append(stripped[:-1])
    if stripped == "round_op":
        candidates.append("round")
    return uniq([item for item in candidates if item])


def listify(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        return [payload]
    return []


def choose_api_def(index: SourceIndex, export: ResolvedExport, calls: list[str]) -> Optional[str]:
    candidates = normalize_candidates(export.export_name) + normalize_candidates(export.impl_name)
    for call in calls:
        candidates.extend(normalize_candidates(call.split(".")[-1]))
    for alias in uniq(candidates):
        if alias in index.api_defs:
            return alias
        op_yamls = index.function_name_to_op_yaml.get(alias, [])
        for api_name, api_info in index.api_defs.items():
            payloads = listify(api_info["payload"])
            if any(entry.get("op_yaml") in op_yamls for entry in payloads):
                return api_name
    return None


def primitive_from_op(index: SourceIndex, op_yaml: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    if not op_yaml:
        return None, None
    info = index.op_defs.get(op_yaml)
    if info is None:
        return None, None
    payload = info["payload"]
    class_info = payload.get("class") if isinstance(payload, dict) else None
    if isinstance(class_info, dict) and class_info.get("name"):
        return str(class_info["name"]), index.relpath(info["path"])
    op_name = info["op_name"]
    return "".join(part.capitalize() for part in str(op_name).split("_")), index.relpath(info["path"])


def infer_generated_primitive(index: SourceIndex, symbol_name: str) -> Optional[str]:
    candidate_names = []
    raw = symbol_name.split(".")[-1]
    candidate_names.extend(normalize_candidates(raw))
    candidate_names.extend(camel_to_snake(name) for name in list(candidate_names))
    candidate_names.extend(item.replace("_", "") for item in list(candidate_names) if "_" in item)
    for name in uniq(candidate_names):
        primitive = "".join(part.capitalize() for part in str(name).split("_"))
        if primitive in index.generated_primitive_names:
            return primitive
    return None


HELPER_PRIMITIVE_NOISE = {
    "Cast",
    "Shape",
    "Rank",
    "TupleToTensor",
    "FillScalar",
    "Floor",
    "Ones",
    "Select",
    "Equal",
    "EqualExt",
    "Gather",
    "GatherD",
    "MaskedFill",
    "MaskedSelect",
    "Flatten",
    "FlattenExt",
    "Narrow",
}


def filter_possible_primitives(primitives: list[str]) -> list[str]:
    if len(primitives) <= 2:
        return uniq(primitives)
    filtered = [item for item in uniq(primitives) if item not in HELPER_PRIMITIVE_NOISE]
    return filtered or uniq(primitives)


def op_yamls_from_index(index: SourceIndex, symbol_name: str) -> list[str]:
    candidates = []
    raw = symbol_name.split(".")[-1]
    raw_candidates = normalize_candidates(raw)
    snake_candidates = uniq(raw_candidates + [camel_to_snake(item) for item in raw_candidates])
    compact_candidates = uniq(snake_candidates + [item.replace("_", "") for item in snake_candidates if "_" in item])
    for name in uniq(raw_candidates + snake_candidates + compact_candidates):
        candidates.extend(index.function_name_to_op_yaml.get(name, []))
        candidates.extend(index.class_name_to_op_yaml.get(name, []))
        candidates.extend(index.op_name_to_op_yaml.get(name, []))
        primitive_name = "".join(part.capitalize() for part in str(name).split("_"))
        candidates.extend(index.class_name_to_op_yaml.get(primitive_name, []))
    return uniq(candidates)


def op_def_candidates_from_symbol(symbol_name: str) -> list[str]:
    base_names = normalize_candidates(symbol_name)
    base_names.extend(camel_to_snake(base) for base in list(base_names))
    base_names.extend(item.replace("_", "") for item in list(base_names) if "_" in item)
    candidates = list(SYMBOL_OP_DEF_MAP.get(symbol_name.split(".")[-1], []))
    for base in uniq(base_names):
        candidates.append(f"{base}_ext_op.yaml")
        candidates.append(f"{base}_op.yaml")
        candidates.append(f"{base}.yaml")
    return uniq(candidates)


def fallback_op_defs(index: SourceIndex, export: ResolvedExport, calls: list[str]) -> list[dict[str, str]]:
    candidate_names = []
    for symbol in [export.export_name, export.impl_name]:
        candidate_names.extend(op_yamls_from_index(index, symbol))
        candidate_names.extend(op_def_candidates_from_symbol(symbol))
    for call in calls:
        candidate_names.extend(op_yamls_from_index(index, call.split(".")[-1]))
        candidate_names.extend(op_def_candidates_from_symbol(call.split(".")[-1]))
    results = []
    for name in uniq(candidate_names):
        info = index.op_defs.get(name)
        if info is None:
            continue
        primitive, op_path = primitive_from_op(index, name)
        if primitive:
            results.append({"api_def": "", "op_yaml": name, "op_def_path": op_path or "", "primitive": primitive})
    if results:
        return results
    generated_candidates = [export.export_name, export.impl_name] + [call.split(".")[-1] for call in calls]
    seen = set()
    for symbol in generated_candidates:
        primitive = infer_generated_primitive(index, symbol)
        if not primitive or primitive in seen:
            continue
        seen.add(primitive)
        results.append({"api_def": "", "op_yaml": "", "op_def_path": "", "primitive": primitive})
    return results


def primitive_kernel_candidates(primitive: str) -> list[str]:
    candidates = [primitive]
    candidates.extend(PRIMITIVE_KERNEL_NAME_MAP.get(primitive, []))
    if primitive.endswith("Ext"):
        candidates.append(primitive[:-3])
    if primitive.endswith("View"):
        candidates.append(primitive[:-4])
    if primitive.endswith("ExtView"):
        candidates.append(primitive[:-4])
        candidates.append(primitive[:-7])
    return uniq([item for item in candidates if item])


def support_state() -> dict[str, str]:
    return {"ascend": "unknown", "cpu": "unknown", "gpu": "unknown"}


def support_state_with_no() -> dict[str, str]:
    return {"ascend": "unknown", "cpu": "unknown", "gpu": "unknown"}


def match_ascend_kbk(index: SourceIndex, primitive: str, primitive_sources: Optional[list[dict[str, str]]] = None) -> list[EvidenceItem]:
    candidates = primitive_kernel_candidates(primitive) + normalize_candidates(primitive)
    candidates = uniq(candidates + [camel_to_snake(item) for item in candidates])
    for source in primitive_sources or []:
        op_yaml = source.get("op_yaml", "")
        if op_yaml:
            stem = op_yaml.replace("_op.yaml", "").replace(".yaml", "")
            candidates.append(stem)
            candidates.append(stem.replace("_", ""))
    results = []
    for candidate in candidates:
        snake = candidate.lower()
        compact = snake.replace("_", "")
        exact_keys = {
            snake,
            compact,
            f"{snake}_aclnn_kernel",
            f"{snake}_kernel",
            f"{snake}_aclnn",
            f"{compact}_aclnn_kernel",
            f"{compact}_kernel",
            f"{compact}_aclnn",
        }
        for key in exact_keys:
            results.extend(index.ascend_kbk_map.get(key, []))
    return uniq(results)


def match_kernel_registry(registry: dict[str, list[EvidenceItem]], primitive: str) -> list[EvidenceItem]:
    results = []
    for candidate in primitive_kernel_candidates(primitive):
        results.extend(registry.get(candidate, []))
    return uniq(results)


def dispatch_map(index: SourceIndex, primitive_sources: list[dict[str, str]]) -> tuple[bool, dict[str, Any], list[dict[str, Any]]]:
    def normalize_dispatch_value(value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, str) and value.strip().lower() in {"none", "null"}:
            return None
        return value

    dispatch_enabled = False
    merged = {}
    detail: list[dict[str, Any]] = []
    for source in primitive_sources:
        op_yaml = source.get("op_yaml", "")
        op_info = index.op_defs.get(op_yaml)
        payload = op_info["payload"] if op_info is not None else {}
        dispatch = payload.get("dispatch") if isinstance(payload, dict) else None
        if not isinstance(dispatch, dict) or not dispatch.get("enable"):
            continue
        dispatch_enabled = True
        item = {"api_def": source.get("api_def", ""), "op_yaml": op_yaml, "dispatch": "enable"}
        for src, dst in (("Ascend", "ascend"), ("CPU", "cpu"), ("GPU", "gpu")):
            if src in dispatch:
                value = normalize_dispatch_value(dispatch[src])
                item[dst] = "None" if value is None else str(value)
                merged[dst] = value
        detail.append(item)
    return dispatch_enabled, merged, detail


def extract_aclnn_calls_from_text(text: str) -> list[str]:
    patterns = [
        r"LAUNCH_ACLNN\(\s*(aclnn[A-Za-z0-9_]+)",
        r'std::move\(\s*"(aclnn[A-Za-z0-9_]+)"\s*\)',
        r'op_type[^"\n]*"(aclnn[A-Za-z0-9_]+)"',
        r'"(aclnn[A-Za-z0-9_]+)"',
    ]
    results = []
    for pattern in patterns:
        results.extend(re.findall(pattern, text))
    return uniq(results)


def match_ascend_customize_files(index: SourceIndex, primitive: str) -> list[Path]:
    roots = [
        index.repo_root / ASCEND_ACLNN_CUSTOMIZE_ROOT,
        index.repo_root / ASCEND_PYBOOST_CUSTOMIZE_ROOT,
        index.repo_root / ASCEND_PYBOOST_AUTO_GEN_ROOT,
    ]
    names = []
    for candidate in primitive_kernel_candidates(primitive) + normalize_candidates(primitive):
        snake = camel_to_snake(candidate)
        names.extend(
            [
                f"{snake}_aclnn_kernel.cc",
                f"{snake}_aclnn_kernel.h",
                f"{snake}.cc",
                f"{snake}.h",
            ]
        )
    found = []
    for name in uniq(names):
        for root in roots:
            path = root / name
            if path.exists():
                found.append(path)
    return found


def match_ascend_customize_files_by_dispatch(
    index: SourceIndex, dispatch_platforms: dict[str, Any], primitives: list[str], primitive_sources: Optional[list[dict[str, str]]] = None
) -> list[Path]:
    found = []
    for primitive in primitives:
        found.extend(match_ascend_customize_files(index, primitive))
    for source in primitive_sources or []:
        op_yaml = source.get("op_yaml", "")
        if op_yaml:
            stem = op_yaml.replace("_op.yaml", "").replace(".yaml", "")
            found.extend(match_ascend_customize_files(index, stem))
            for root in [
                index.repo_root / ASCEND_ACLNN_CUSTOMIZE_ROOT,
                index.repo_root / ASCEND_PYBOOST_CUSTOMIZE_ROOT,
                index.repo_root / ASCEND_PYBOOST_AUTO_GEN_ROOT,
            ]:
                for suffix in (".cc", ".h", "_aclnn_kernel.cc", "_aclnn_kernel.h"):
                    path = root / f"{stem}{suffix}"
                    if path.exists():
                        found.append(path)
    ascend_name = dispatch_platforms.get("ascend")
    if isinstance(ascend_name, str) and ascend_name not in {"", "None"}:
        for candidate in normalize_candidates(ascend_name) + primitive_kernel_candidates(ascend_name):
            found.extend(match_ascend_customize_files(index, candidate))
            snake = camel_to_snake(candidate.replace("Ascend", "")).strip("_")
            for root in [index.repo_root / ASCEND_ACLNN_CUSTOMIZE_ROOT, index.repo_root / ASCEND_PYBOOST_CUSTOMIZE_ROOT]:
                for suffix in (".cc", ".h", "_aclnn_kernel.cc", "_aclnn_kernel.h"):
                    path = root / f"{snake}{suffix}"
                    if path.exists():
                        found.append(path)
    return uniq(found)


def analyze_aclnn(index: SourceIndex, primitives: list[str], primitive_sources: Optional[list[dict[str, str]]] = None) -> tuple[dict[str, Any], list[dict[str, str]]]:
    direct_interfaces = []
    for primitive in primitives:
        value = index.aclnn_map.get(primitive)
        if isinstance(value, str):
            direct_interfaces.append(value)
        elif isinstance(value, dict):
            for item in value.values():
                if isinstance(item, str) and item.startswith("aclnn"):
                    direct_interfaces.append(item)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, str) and item.startswith("aclnn"):
                    direct_interfaces.append(item)

    effective_interfaces = []
    aclnn_evidence: list[dict[str, str]] = []
    dispatch_platforms = dispatch_map(index, primitive_sources or [])[1] if primitive_sources else {}
    for path in match_ascend_customize_files_by_dispatch(index, dispatch_platforms, primitives, primitive_sources):
        text = path.read_text(encoding="utf-8", errors="ignore")
        for interface in extract_aclnn_calls_from_text(text):
            effective_interfaces.append(interface)
            aclnn_evidence.append(
                EvidenceItem(index.relpath(path), "direct", interface, f"customize path calls {interface}").to_dict()
            )

    direct_interfaces = uniq(direct_interfaces)
    effective_interfaces = uniq(effective_interfaces)
    if direct_interfaces and not effective_interfaces:
        effective_interfaces = list(direct_interfaces)
    if direct_interfaces:
        mode = "direct"
        path_kind = "direct_aclnn"
    elif effective_interfaces:
        mode = "indirect"
        path_kind = "customize_to_aclnn"
    elif primitives:
        mode = "none"
        path_kind = "none"
    else:
        mode = "unknown"
        path_kind = "unknown"
    return {
        "mode": mode,
        "interfaces": direct_interfaces,
        "effective_interfaces": effective_interfaces,
        "path_kind": path_kind,
    }, uniq(aclnn_evidence)


def extract_func_op_expansion(text: str) -> list[str]:
    results = []
    results.extend(re.findall(r"Prim\(([A-Za-z0-9_]+)\)", text))
    results.extend(re.findall(r'Emit\("([A-Za-z0-9_]+)"', text))
    return uniq(results)


def analyze_func_op(index: SourceIndex, primitive_sources: list[dict[str, str]], primitives: list[str]) -> dict[str, Any]:
    result = {
        "is_func_op": False,
        "op_yamls": [],
        "meta_dsl_paths": [],
        "expanded_primitives": [],
        "evidence": [],
    }
    candidate_stems = []
    for source in primitive_sources:
        op_yaml = source.get("op_yaml", "")
        if not op_yaml:
            continue
        op_info = index.op_defs.get(op_yaml)
        if op_info is None:
            continue
        payload = op_info.get("payload", {})
        if not isinstance(payload, dict) or payload.get("bprop_expander") is not False:
            continue
        result["is_func_op"] = True
        result["op_yamls"].append(op_yaml)
        result["evidence"].append(
            EvidenceItem(
                index.relpath(op_info["path"]),
                "direct",
                "bprop_expander: False",
                f"func_op definition for {source.get('primitive', '') or op_info['op_name']}",
            ).to_dict()
        )
        candidate_stems.append(str(op_info["op_name"]))
        function_block = payload.get("function")
        if isinstance(function_block, dict) and isinstance(function_block.get("name"), str):
            candidate_stems.append(function_block["name"])
        class_block = payload.get("class")
        if isinstance(class_block, dict) and isinstance(class_block.get("name"), str):
            candidate_stems.append(camel_to_snake(class_block["name"]))
    candidate_stems.extend(camel_to_snake(item) for item in primitives)
    root = index.repo_root / META_DSL_FUNC_OP_ROOT
    for stem in uniq(candidate_stems):
        if not stem:
            continue
        path = root / f"{stem}.cc"
        if not path.exists():
            continue
        result["meta_dsl_paths"].append(index.relpath(path))
        text = path.read_text(encoding="utf-8", errors="ignore")
        for primitive in extract_func_op_expansion(text):
            result["expanded_primitives"].append(primitive)
        result["evidence"].append(
            EvidenceItem(index.relpath(path), "direct", path.name, f"meta_dsl func_op expansion for {stem}").to_dict()
        )
    result["op_yamls"] = uniq(result["op_yamls"])
    result["meta_dsl_paths"] = uniq(result["meta_dsl_paths"])
    result["expanded_primitives"] = uniq(result["expanded_primitives"])
    result["evidence"] = uniq(result["evidence"])
    return result


def is_runtime_utility_api(public_path: str) -> bool:
    distributed_runtime_prefixes = (
        "mindspore.mint.distributed.get_",
        "mindspore.mint.distributed.destroy_",
    )
    distributed_runtime_names = {
        "mindspore.mint.distributed.all_gather_object",
        "mindspore.mint.distributed.broadcast_object_list",
        "mindspore.mint.distributed.gather_object",
        "mindspore.mint.distributed.init_process_group",
        "mindspore.mint.distributed.is_available",
        "mindspore.mint.distributed.is_initialized",
        "mindspore.mint.distributed.new_group",
        "mindspore.mint.distributed.P2POp",
        "mindspore.mint.distributed.recv_object_list",
        "mindspore.mint.distributed.scatter_object_list",
        "mindspore.mint.distributed.send_object_list",
    }
    if public_path in distributed_runtime_names:
        return True
    if public_path.startswith(distributed_runtime_prefixes):
        return True
    return False


def is_operator_mapping_not_applicable_api(public_path: str) -> bool:
    if is_runtime_utility_api(public_path):
        return True
    if public_path.startswith("mindspore.mint.optim."):
        return True
    return False


def is_distributed_comm_api(public_path: str) -> bool:
    return public_path.startswith("mindspore.mint.distributed.") and not is_runtime_utility_api(public_path)


def classify_terminal_kind(
    symbol: str, is_func_op: bool = False, module: Optional[ModuleInfo] = None
) -> str:
    if is_func_op:
        return "func_op"
    if module is not None:
        leaf = symbol.split(".")[-1]
        binding = module.locals.get(leaf)
        if binding is not None and binding.kind == "assigned_call":
            target = binding.target_symbol
            if "._c_expression." in target or target.startswith("mindspore._c_expression."):
                return "c_expression_instance"
            return "primitive_instance"
    if "._c_expression." in symbol or symbol.startswith("mindspore._c_expression."):
        return "c_expression_instance"
    if ".ops.auto_generate." in symbol:
        if ".gen_ops_prim." in symbol:
            return "auto_generate_primitive"
        return "auto_generate_function"
    if ".ops.functional_overload." in symbol:
        return "functional_overload"
    if ".ops." in symbol:
        return "ops_wrapper"
    return "ops_wrapper"


def merge_support_state(current: str, candidate: str) -> str:
    order = {"yes": 2, "unknown": 1, "no": 0}
    return candidate if order[candidate] > order[current] else current


def analyze_support(
    index: SourceIndex, primitives: list[str], primitive_sources: list[dict[str, str]]
) -> tuple[dict[str, str], dict[str, str], dict[str, dict[str, list[dict[str, str]]]], list[dict[str, Any]]]:
    if primitives:
        pynative = {"ascend": "no", "cpu": "no", "gpu": "no"}
        graph = {"ascend": "no", "cpu": "no", "gpu": "no"}
    else:
        pynative = support_state()
        graph = support_state()
    evidence = {
        "pynative": {"ascend": [], "cpu": [], "gpu": []},
        "graph_kbk_o0": {"ascend": [], "cpu": [], "gpu": []},
    }
    dispatch_enabled, dispatch_platforms, detail = dispatch_map(index, primitive_sources)

    def platform_disabled(name: str) -> bool:
        return dispatch_platforms.get(name, "__missing__") is None

    def platform_declared(name: str) -> bool:
        return name in dispatch_platforms

    for primitive in uniq(primitives):
        ascend_kbk = match_ascend_kbk(index, primitive, primitive_sources)
        ascend_customize = [
            EvidenceItem(index.relpath(path), "direct", path.name, "Ascend pyboost/customize artifact")
            for path in match_ascend_customize_files_by_dispatch(index, dispatch_platforms, [primitive], primitive_sources)
        ]
        cpu_kernel = match_kernel_registry(index.cpu_kernel_map, primitive)
        gpu_kernel = match_kernel_registry(index.gpu_kernel_map, primitive)
        fallback = index.fallback_map.get(primitive, [])
        primitive_pynative = support_state()
        primitive_graph = support_state()

        if dispatch_enabled:
            if platform_disabled("ascend"):
                primitive_pynative["ascend"] = "no"
            elif ascend_kbk or ascend_customize or primitive in index.aclnn_map:
                primitive_pynative["ascend"] = "yes"
                if primitive in index.aclnn_map:
                    evidence["pynative"]["ascend"].append(
                        EvidenceItem(index.relpath(index.repo_root / ACLNN_CONFIG), "direct", primitive, f"PYNATIVE Ascend aclnn entry for {primitive}").to_dict()
                    )
                evidence["pynative"]["ascend"].extend(item.to_dict() for item in ascend_kbk)
                evidence["pynative"]["ascend"].extend(item.to_dict() for item in ascend_customize)
            elif platform_declared("ascend"):
                primitive_pynative["ascend"] = "unknown"
            if platform_disabled("cpu"):
                primitive_pynative["cpu"] = "no"
            elif cpu_kernel:
                primitive_pynative["cpu"] = "yes"
                evidence["pynative"]["cpu"].extend(item.to_dict() for item in cpu_kernel)
            elif platform_declared("cpu") or dispatch_enabled:
                primitive_pynative["cpu"] = "no"
            if platform_disabled("gpu"):
                primitive_pynative["gpu"] = "no"
            elif gpu_kernel:
                primitive_pynative["gpu"] = "yes"
                evidence["pynative"]["gpu"].extend(item.to_dict() for item in gpu_kernel)
            elif platform_declared("gpu") or dispatch_enabled:
                primitive_pynative["gpu"] = "no"
        else:
            if cpu_kernel:
                primitive_pynative["cpu"] = "yes"
                evidence["pynative"]["cpu"].extend(item.to_dict() for item in cpu_kernel)
            if gpu_kernel:
                primitive_pynative["gpu"] = "yes"
                evidence["pynative"]["gpu"].extend(item.to_dict() for item in gpu_kernel)

        if platform_disabled("ascend"):
            primitive_graph["ascend"] = "no"
        elif ascend_kbk:
            primitive_graph["ascend"] = "yes"
            evidence["graph_kbk_o0"]["ascend"].extend(item.to_dict() for item in ascend_kbk)
        elif platform_declared("ascend"):
            primitive_graph["ascend"] = "unknown"
        if platform_disabled("cpu"):
            primitive_graph["cpu"] = "no"
        elif cpu_kernel:
            primitive_graph["cpu"] = "yes"
            evidence["graph_kbk_o0"]["cpu"].extend(item.to_dict() for item in cpu_kernel)
        elif fallback:
            primitive_graph["cpu"] = "yes"
            evidence["graph_kbk_o0"]["cpu"].extend(item.to_dict() for item in fallback)
        elif primitives:
            primitive_graph["cpu"] = "no"
        if platform_disabled("gpu"):
            primitive_graph["gpu"] = "no"
        elif gpu_kernel:
            primitive_graph["gpu"] = "yes"
            evidence["graph_kbk_o0"]["gpu"].extend(item.to_dict() for item in gpu_kernel)
        elif fallback:
            primitive_graph["gpu"] = "yes"
            evidence["graph_kbk_o0"]["gpu"].extend(item.to_dict() for item in fallback)
        elif primitives:
            primitive_graph["gpu"] = "no"

        for backend in ("ascend", "cpu", "gpu"):
            pynative[backend] = merge_support_state(pynative[backend], primitive_pynative[backend])
            graph[backend] = merge_support_state(graph[backend], primitive_graph[backend])
    for mode in evidence.values():
        for backend in mode:
            mode[backend] = uniq(mode[backend])
    return pynative, graph, evidence, detail


def inherit_construct_records(main_records: list[dict[str, Any]], evidence_records: list[dict[str, Any]]) -> None:
    record_map = {item["api"]: item for item in main_records}
    evidence_map = {item["api"]: item for item in evidence_records}

    for record in main_records:
        if record["api_level"] != "module_api" or not record["composed_of"] or "scenario_dependent" in record["flags"]:
            continue
        target_api = record["composed_of"][0]
        target_record = record_map.get(target_api)
        target_evidence = evidence_map.get(target_api)
        if not target_record or (
            not target_record.get("primitive_count")
            and not target_record.get("possible_primitives")
            and not target_record.get("func_op_expands_to")
        ):
            continue
        current_primitives = set(record.get("primitive", []))
        target_primitives = set(target_record.get("primitive", []))
        should_inherit = not current_primitives
        if not should_inherit and current_primitives != target_primitives:
            should_inherit = True
        if not should_inherit:
            for mode_name in ("pynative_support", "graph_kbk_o0_support"):
                current_mode = record.get(mode_name, {})
                target_mode = target_record.get(mode_name, {})
                if any(
                    current_mode.get(backend) == "unknown" and target_mode.get(backend) != "unknown"
                    for backend in ("ascend", "cpu", "gpu")
                ):
                    should_inherit = True
                    break
        if not should_inherit:
            current_aclnn = record.get("aclnn", {})
            target_aclnn = target_record.get("aclnn", {})
            if current_aclnn.get("mode") in {"unknown", "none"} and target_aclnn.get("mode") not in {"unknown", "none"}:
                should_inherit = True
        if not should_inherit:
            current_grad = record.get("grad", {})
            target_grad = target_record.get("grad", {})
            if current_grad.get("mode") == "unknown" and target_grad.get("mode") != "unknown":
                should_inherit = True
        if not should_inherit:
            continue

        record["primitive"] = list(target_record["primitive"])
        record["primitive_count"] = target_record["primitive_count"]
        record["possible_primitives"] = filter_possible_primitives(list(target_record.get("possible_primitives", [])))
        record["pynative_support"] = dict(target_record["pynative_support"])
        record["graph_kbk_o0_support"] = dict(target_record["graph_kbk_o0_support"])
        record["support_summary"] = target_record["support_summary"]
        record["func_op_expands_to"] = list(target_record.get("func_op_expands_to", []))
        record["graph_support_kind"] = target_record.get("graph_support_kind", record.get("graph_support_kind", "unknown"))
        record["support_reason_kind"] = target_record.get("support_reason_kind", record.get("support_reason_kind", "unknown"))
        record["semantic_kind"] = (
            "scenario_dependent_module"
            if not target_record.get("primitive_count") and target_record.get("possible_primitives")
            else record.get("semantic_kind", "high_level_module")
        )
        record["trust_level"] = "scenario_dependent" if record["semantic_kind"] == "scenario_dependent_module" else "inherited_fact"
        record["fact_origin"] = "inherited_from_construct"
        record["aclnn"] = {
            "mode": target_record["aclnn"]["mode"],
            "interfaces": list(target_record["aclnn"]["interfaces"]),
            "effective_interfaces": list(target_record["aclnn"]["effective_interfaces"]),
            "path_kind": target_record["aclnn"]["path_kind"],
        }
        record["grad"] = {
            "mode": target_record["grad"]["mode"],
            "impl": [dict(item) for item in target_record["grad"]["impl"]],
        }
        record["primary_paths"] = uniq(list(record["primary_paths"]) + list(target_record["primary_paths"]))[:3]
        record["confidence"] = "high" if target_record["confidence"] == "high" else "medium"
        flags = [flag for flag in record["flags"] if flag != "needs_manual_review"]
        if "construct_mapped" not in flags:
            flags.append("construct_mapped")
        if "func_op" in target_record.get("flags", []) and "func_op" not in flags:
            flags.append("func_op")
        if target_record.get("possible_primitives") and "scenario_dependent" not in flags:
            flags.append("scenario_dependent")
        record["flags"] = uniq(flags)

        evidence = evidence_map.get(record["api"])
        if evidence is not None:
            evidence["primitive_sources"] = [dict(item) for item in target_evidence.get("primitive_sources", [])] if target_evidence else []
            evidence["func_op"] = copy.deepcopy(target_evidence.get("func_op", {})) if target_evidence else evidence.get("func_op", {})
            evidence["support_evidence"] = copy.deepcopy(target_evidence.get("support_evidence", {})) if target_evidence else evidence["support_evidence"]
            evidence["dispatch_detail"] = [dict(item) for item in target_evidence.get("dispatch_detail", [])] if target_evidence else evidence["dispatch_detail"]
            evidence["aclnn_evidence"] = [dict(item) for item in target_evidence.get("aclnn_evidence", [])] if target_evidence else evidence["aclnn_evidence"]
            evidence["grad"] = [dict(item) for item in target_evidence.get("grad", [])] if target_evidence else evidence["grad"]
            inherited_note = f"inherited operator facts from construct target {target_api}"
            evidence["notes"] = uniq(list(evidence.get("notes", [])) + [inherited_note])

        record["status_summary"] = build_status_summary(record)
        record["unknown_reason"] = infer_unknown_reason(record)
        record["llm_summary"] = build_llm_summary(record)
        record["llm_warning"] = build_llm_warning(record)


def build_support_summary(
    pynative: dict[str, str],
    graph: dict[str, str],
    evidence: dict[str, dict[str, list[dict[str, str]]]],
    func_op_info: Optional[dict[str, Any]] = None,
    graph_support_kind: str = "unknown",
    runtime_utility: bool = False,
) -> str:
    if runtime_utility:
        return "runtime utility; operator mapping not applicable"

    def fmt(state: dict[str, str]) -> str:
        return ", ".join(f"{name.capitalize()}={state[name]}" for name in ("ascend", "cpu", "gpu"))

    suffix = ""
    cpu_sources = {item["summary"] for item in evidence["graph_kbk_o0"]["cpu"]}
    gpu_sources = {item["summary"] for item in evidence["graph_kbk_o0"]["gpu"]}
    if any("fallback builder" in item for item in cpu_sources | gpu_sources):
        suffix = " via fallback"
    if func_op_info and func_op_info.get("is_func_op"):
        expanded = ",".join(func_op_info.get("expanded_primitives", [])[:6])
        extra = "; graph_kbk_o0: via func_op expansion"
        if expanded:
            extra += f"; expanded_primitives={expanded}"
        return f"pynative: {fmt(pynative)}{extra}"
    if graph_support_kind == "scenario_dependent":
        return "scenario dependent; operator support depends on construct branch"
    return f"pynative: {fmt(pynative)}; graph_kbk_o0: {fmt(graph)}{suffix}"


def infer_category(public_path: str, impl_symbol: str) -> str:
    if ".mint.optim." in public_path or ".mint.optim." in impl_symbol:
        return "optim"
    if ".mint.nn." in public_path or ".nn." in impl_symbol:
        return "nn"
    if ".mint.linalg." in public_path or ".linalg." in impl_symbol:
        return "linalg"
    if ".mint.special." in public_path or ".special." in impl_symbol:
        return "special"
    if ".mint.distributed." in public_path or ".distributed." in impl_symbol:
        return "distributed"
    if public_path.startswith("mindspore.mint.optim."):
        return "optim"
    return "mint"


def infer_api_level(implementation_type: str, export: ResolvedExport, primitives: list[str]) -> str:
    if implementation_type == "high_level_module" or export.api_kind == "class":
        return "module_api"
    if implementation_type in {"single_op", "multi_overload_op"} and primitives:
        return "operator_api"
    return "wrapper_api"


def build_status_summary(item: dict[str, Any]) -> str:
    primitive = ",".join(item["primitive"]) if item["primitive"] else "none"
    possible = ",".join(item.get("possible_primitives", [])) or "none"
    if item.get("semantic_kind") == "runtime_utility":
        return "runtime utility; operator mapping not applicable"
    if item.get("unknown_reason") == "not_applicable" and item.get("category") == "distributed" and item.get("primitive_count"):
        return f"distributed communication operator; primitive={primitive}; standard kernel/aclnn support mapping not applicable"
    if item.get("unknown_reason") == "not_applicable" and item.get("category") == "optim":
        return "optimizer module; operator mapping not applicable"
    if "scenario_dependent" in item["flags"]:
        prefix = "high_level_module via construct" if item["api_level"] == "module_api" else "composite_op"
        return f"{prefix}; scenario dependent; possible_primitives={possible}; grad={item['grad']['mode']}"
    support = item["support_summary"]
    grad = item["grad"]["mode"]
    if "func_op" in item["flags"] and "construct_mapped" in item["flags"]:
        return f"high_level_module via construct; func_op; primitive={primitive}; {support}; grad={grad}"
    if "construct_mapped" in item["flags"]:
        return f"high_level_module via construct; primitive={primitive}; {support}; grad={grad}"
    if item["implementation_type"] == "multi_overload_op":
        return f"multi overload; primitives={primitive}; {support}; grad={grad}"
    if item["alias_of"]:
        return f"alias of {item['alias_of']}; primitive={primitive}; {support}; grad={grad}"
    return f"{item['implementation_type']}; primitive={primitive}; {support}; grad={grad}"


def aclnn_brief(aclnn: dict[str, Any]) -> str:
    effective = aclnn.get("effective_interfaces") or []
    direct = aclnn.get("interfaces") or []
    if effective:
        return ",".join(effective[:4])
    if direct:
        return ",".join(direct[:4])
    return aclnn.get("mode", "unknown")


def support_brief(support: dict[str, str]) -> str:
    return f"A={support['ascend']},C={support['cpu']},G={support['gpu']}"


def build_llm_summary(item: dict[str, Any]) -> str:
    primitive = ",".join(item.get("primitive") or []) or "none"
    possible = ",".join(item.get("possible_primitives") or []) or "none"
    aclnn = aclnn_brief(item.get("aclnn") or {})
    semantic_kind = item.get("semantic_kind")
    if semantic_kind == "runtime_utility":
        return "runtime utility; operator mapping not applicable"
    if item.get("unknown_reason") == "not_applicable" and item.get("category") == "distributed" and item.get("primitive_count"):
        return f"distributed communication operator; primitive={primitive}; standard kernel/aclnn support mapping not applicable"
    if item.get("unknown_reason") == "not_applicable" and item.get("category") == "optim":
        return "optimizer module; operator mapping not applicable"
    if semantic_kind == "scenario_dependent_module":
        return f"high-level module via construct; scenario dependent; possible_primitives={possible}"
    if not item.get("primitive") and item.get("possible_primitives"):
        return f"wrapper api; scenario dependent; possible_primitives={possible}"
    if item.get("graph_support_kind") == "func_op_expansion":
        return (
            f"func_op; primitive={primitive}; pynative({support_brief(item['pynative_support'])}); "
            f"graph works via expansion; aclnn={aclnn}"
        )
    if item.get("implementation_type") == "multi_overload_op":
        return (
            f"multi overload; primitives={primitive}; pynative({support_brief(item['pynative_support'])}); "
            f"graph({support_brief(item['graph_kbk_o0_support'])}); aclnn={aclnn}"
        )
    if semantic_kind == "high_level_module":
        return (
            f"high-level module via construct; primitive={primitive}; "
            f"pynative({support_brief(item['pynative_support'])}); "
            f"graph({support_brief(item['graph_kbk_o0_support'])}); aclnn={aclnn}"
        )
    return (
        f"direct operator; primitive={primitive}; "
        f"pynative({support_brief(item['pynative_support'])}); "
        f"graph({support_brief(item['graph_kbk_o0_support'])}); aclnn={aclnn}"
    )


def build_llm_warning(item: dict[str, Any]) -> str:
    semantic_kind = item.get("semantic_kind")
    if semantic_kind == "runtime_utility":
        return "Operator mapping is not applicable for this API."
    if item.get("unknown_reason") == "not_applicable" and item.get("category") == "distributed" and item.get("primitive_count"):
        return "Communication backend semantics are not represented by the standard kernel/aclnn support matrix."
    if item.get("unknown_reason") == "not_applicable" and item.get("category") == "optim":
        return "Optimizer behavior is defined by composite parameter and tensor updates, not a stable primitive mapping."
    if semantic_kind == "scenario_dependent_module":
        return "Do not infer a single primitive or support result without branch conditions."
    if not item.get("primitive") and item.get("possible_primitives"):
        return "Final operator path depends on wrapper branch or helper path."
    if item.get("graph_support_kind") == "func_op_expansion":
        return "GRAPH behavior is expansion-based; GRAPH unknown does not mean unsupported."
    if item.get("implementation_type") == "multi_overload_op":
        return "Support, aclnn, and grad are merged across overloads, not tied to one primitive."
    if semantic_kind == "high_level_module" and item.get("fact_origin") == "inherited_from_construct":
        return "Primitive and support facts are inherited through construct, not directly defined by the class."
    return ""


def infer_unknown_reason(item: dict[str, Any]) -> str:
    if item.get("semantic_kind") == "runtime_utility":
        return "not_applicable"
    if item.get("category") == "distributed" and item.get("primitive_count") and is_distributed_comm_api(item.get("api", "")):
        return "not_applicable"
    if item.get("category") == "optim" and not item.get("primitive_count"):
        return "not_applicable"
    if item.get("category") == "optim" and item.get("api_level") == "module_api":
        return "not_applicable"
    if item.get("semantic_kind") == "scenario_dependent_module":
        return "scenario_dependent"
    if not item.get("primitive_count") and item.get("possible_primitives"):
        return "scenario_dependent"
    if item.get("graph_support_kind") == "func_op_expansion" and any(
        item["graph_kbk_o0_support"][backend] == "unknown" for backend in ("ascend", "cpu", "gpu")
    ):
        return "func_op_expansion"
    if item.get("primitive_count") and item["graph_kbk_o0_support"]["ascend"] == "unknown":
        return "missing_kbk_evidence"
    if item.get("primitive_count") and any(
        item["pynative_support"][backend] == "unknown" or item["graph_kbk_o0_support"][backend] == "unknown"
        for backend in ("cpu", "gpu")
    ):
        return "missing_runtime_kernel_evidence"
    if not item.get("primitive_count") and (item.get("composed_of") or item.get("possible_primitives")):
        return "unresolved_static_chain"
    if any(
        item[field][backend] == "unknown"
        for field in ("pynative_support", "graph_kbk_o0_support")
        for backend in ("ascend", "cpu", "gpu")
    ):
        return "unresolved_static_chain"
    return ""


def infer_grad_not_applicable(public_path: str) -> bool:
    leaf = public_path.split(".")[-1].lower()
    non_diff_names = {
        "allclose",
        "isclose",
        "any",
        "all",
        "bincount",
        "histc",
        "argmax",
        "argmin",
        "nonzero",
    }
    if leaf in non_diff_names:
        return True
    distributed_markers = ("distributed.all_gather", "distributed.all_reduce", "distributed.broadcast", "distributed.scatter")
    if any(marker in public_path for marker in distributed_markers):
        return True
    return False


def build_record(index: SourceIndex, export: ResolvedExport) -> tuple[dict[str, Any], dict[str, Any]]:
    prelude_calls, terminal_calls = collect_call_details(export)
    class_execution = analyze_class_construct(index, export) if export.api_kind == "class" else None
    construct_calls = class_execution.calls if class_execution is not None else []
    effective_calls = uniq(resolve_call_alias(index, call) for call in (terminal_calls + construct_calls))
    api_def_name = choose_api_def(index, export, effective_calls)
    api_def_entry = index.api_defs.get(api_def_name) if api_def_name else None
    entries = []
    if api_def_entry is not None:
        for entry in listify(api_def_entry["payload"]):
            entry_copy = dict(entry)
            entry_copy["_api_def"] = api_def_name
            entries.append(entry_copy)

    aliases = [str(entry["alias"]) for entry in entries if "alias" in entry]
    alias_of = aliases[0] if aliases else ""
    alias_target = index.api_defs.get(alias_of) if alias_of else None
    alias_entries = []
    if alias_target is not None:
        for entry in listify(alias_target["payload"]):
            entry_copy = dict(entry)
            entry_copy["_api_def"] = alias_of
            alias_entries.append(entry_copy)

    effective_entries = entries
    if alias_of and all("op_yaml" not in entry for entry in entries):
        effective_entries = alias_entries

    primitives = []
    primitive_sources = []
    for entry in effective_entries:
        primitive, op_path = primitive_from_op(index, entry.get("op_yaml"))
        if primitive:
            primitives.append(primitive)
            primitive_sources.append(
                {
                    "api_def": entry.get("_api_def", ""),
                    "op_yaml": entry.get("op_yaml"),
                    "op_def_path": op_path or "",
                }
            )

    if not primitive_sources:
        for item in fallback_op_defs(index, export, effective_calls):
            primitives.append(item["primitive"])
            primitive_sources.append(
                {
                    "api_def": item["api_def"],
                    "op_yaml": item["op_yaml"],
                    "op_def_path": item["op_def_path"],
                }
            )

    pynative_support, graph_kbk_o0_support, support_evidence, dispatch_detail = analyze_support(index, primitives, primitive_sources)
    interface_forms = uniq(
        [part.strip() for entry in effective_entries for part in str(entry.get("interface", "")).split(",") if part.strip()]
    )

    aclnn_info, aclnn_evidence = analyze_aclnn(index, primitives, primitive_sources)
    func_op_info = analyze_func_op(index, primitive_sources, primitives)
    scenario_dependent = export.api_kind == "class" and class_execution is not None and bool(class_execution.branching_notes)
    runtime_utility = is_runtime_utility_api(export.public_path)
    operator_mapping_not_applicable = is_operator_mapping_not_applicable_api(export.public_path)
    possible_primitives = uniq(primitives) if scenario_dependent else []
    if not primitives and not runtime_utility:
        inferred_possible = []
        candidate_calls = uniq(effective_calls + terminal_calls)
        for call in candidate_calls:
            inferred_possible.extend(infer_possible_primitives_from_symbol(index, call))
        possible_primitives = filter_possible_primitives(uniq(possible_primitives + inferred_possible))

    grad_impl = []
    grad_mode = "unknown"
    for primitive in primitives:
        if primitive in index.bprop_map:
            grad_mode = "explicit_bprop"
            for ev in index.bprop_map[primitive]:
                grad_impl.append({"primitive": primitive, "kind": "bprop_builder", "path": ev.path, "anchor": ev.anchor})
    if grad_mode == "unknown" and export.api_kind == "class":
        grad_mode = "not_applicable" if export.public_path.startswith("mindspore.mint.optim.") else "autodiff"
    elif grad_mode == "unknown" and primitives:
        grad_mode = "unknown"
    if grad_mode == "unknown" and infer_grad_not_applicable(export.public_path):
        grad_mode = "not_applicable"
    if scenario_dependent:
        pynative_support = support_state()
        graph_kbk_o0_support = support_state()
        support_evidence = {"pynative": {"ascend": [], "cpu": [], "gpu": []}, "graph_kbk_o0": {"ascend": [], "cpu": [], "gpu": []}}
        aclnn_info = {"mode": "unknown", "interfaces": [], "effective_interfaces": [], "path_kind": "unknown"}
        aclnn_evidence = []
        grad_mode = "unknown"
        grad_impl = []
        primitives = []
        primitive_sources = []
        func_op_info = {"is_func_op": False, "op_yamls": [], "meta_dsl_paths": [], "expanded_primitives": [], "evidence": []}
    elif func_op_info["is_func_op"]:
        graph_kbk_o0_support = support_state()
        support_evidence["graph_kbk_o0"] = {"ascend": [], "cpu": [], "gpu": []}

    if runtime_utility:
        primitives = []
        primitive_sources = []
        possible_primitives = []
        pynative_support = support_state()
        graph_kbk_o0_support = support_state()
        aclnn_info = {"mode": "not_applicable", "interfaces": [], "effective_interfaces": [], "path_kind": "not_applicable"}
        aclnn_evidence = []
        grad_mode = "not_applicable"
        grad_impl = []
        func_op_info = {"is_func_op": False, "op_yamls": [], "meta_dsl_paths": [], "expanded_primitives": [], "evidence": []}

    implementation_type = "high_level_module" if export.api_kind == "class" else "composite_op"
    if alias_of:
        implementation_type = "alias"
    elif runtime_utility:
        implementation_type = "runtime_utility"
    elif export.api_kind != "class" and len(uniq(primitives)) > 1:
        implementation_type = "multi_overload_op"
    elif export.api_kind != "class" and primitives and len(effective_calls) <= 1 and len(uniq(primitives)) == 1:
        implementation_type = "single_op"

    api_level = infer_api_level(implementation_type, export, primitives)
    primary_paths = []
    if export.impl_path:
        primary_paths.append(index.relpath(export.impl_path))
    if api_def_entry is not None:
        primary_paths.append(index.relpath(api_def_entry["path"]))
    for source in primitive_sources:
        if source["op_def_path"]:
            primary_paths.append(source["op_def_path"])
    if aclnn_info["interfaces"] or aclnn_info["effective_interfaces"]:
        primary_paths.append(index.relpath(index.repo_root / ACLNN_CONFIG))
    if grad_impl:
        primary_paths.append(grad_impl[0]["path"])
    primary_paths = uniq(primary_paths)[:3]

    graph_support_kind = "unknown"
    if runtime_utility:
        graph_support_kind = "not_applicable"
    elif scenario_dependent or (not primitives and bool(possible_primitives)):
        graph_support_kind = "scenario_dependent"
    elif func_op_info["is_func_op"]:
        graph_support_kind = "func_op_expansion"
    elif any("fallback builder" in item["summary"] for item in support_evidence["graph_kbk_o0"]["cpu"] + support_evidence["graph_kbk_o0"]["gpu"]):
        graph_support_kind = "fallback"
    elif any(state == "yes" for state in graph_kbk_o0_support.values()):
        graph_support_kind = "direct_kernel"

    support_summary = build_support_summary(
        pynative_support,
        graph_kbk_o0_support,
        support_evidence,
        func_op_info,
        graph_support_kind=graph_support_kind,
        runtime_utility=runtime_utility,
    )
    if (scenario_dependent or (not primitives and possible_primitives)) and possible_primitives:
        support_summary = f"scenario dependent; possible_primitives={','.join(possible_primitives)}"
    confidence = "high" if primitives or alias_of else ("medium" if export.source_kind in {"generated_binding", "ops_binding"} else "low")
    flags = []
    if alias_of:
        flags.append("alias")
    if len(primitives) > 1:
        flags.append("multi_overload")
    if export.source_kind == "generated_binding":
        flags.append("generated_binding")
    if implementation_type == "high_level_module":
        flags.append("high_level_module")
    if func_op_info["is_func_op"]:
        flags.append("func_op")
    if class_execution is not None and (primitives or possible_primitives or construct_calls):
        flags.append("construct_mapped")
    if scenario_dependent or (not primitives and possible_primitives):
        flags.append("scenario_dependent")
    if (confidence == "low" or (implementation_type == "single_op" and not primitives)) and not runtime_utility:
        flags.append("needs_manual_review")

    semantic_kind = "direct_operator"
    trust_level = "direct_fact"
    fact_origin = "direct"
    if runtime_utility:
        semantic_kind = "runtime_utility"
        trust_level = "not_applicable"
        fact_origin = "not_applicable"
    elif scenario_dependent:
        semantic_kind = "scenario_dependent_module"
        trust_level = "scenario_dependent"
        fact_origin = "inherited_from_construct"
    elif not primitives and possible_primitives:
        semantic_kind = "scenario_dependent_module" if export.api_kind == "class" else "direct_operator"
        trust_level = "scenario_dependent"
        fact_origin = "inherited_from_construct" if export.api_kind == "class" else "direct"
    elif func_op_info["is_func_op"]:
        semantic_kind = "func_op_operator" if export.api_kind != "class" else "high_level_module"
        trust_level = "expansion_based"
        fact_origin = "expansion_derived" if export.api_kind != "class" else "inherited_from_construct"
    elif implementation_type == "high_level_module":
        semantic_kind = "high_level_module"
        trust_level = "inherited_fact" if "construct_mapped" in flags else "direct_fact"
        fact_origin = "inherited_from_construct" if "construct_mapped" in flags else "direct"
    elif implementation_type == "multi_overload_op":
        semantic_kind = "multi_overload_operator"
        trust_level = "direct_fact"
        fact_origin = "direct"
    elif alias_of:
        semantic_kind = "direct_operator"
        trust_level = "inherited_fact"
        fact_origin = "inherited_from_alias"

    support_reason_kind = "unknown"
    if runtime_utility:
        support_reason_kind = "runtime_utility"
    elif scenario_dependent or (not primitives and possible_primitives):
        support_reason_kind = "scenario_dependent"
    elif func_op_info["is_func_op"]:
        support_reason_kind = "func_op_expansion"
    elif aclnn_info["path_kind"] == "customize_to_aclnn":
        support_reason_kind = "customize_to_aclnn"
    elif export.api_kind != "class" and effective_calls and primitive_sources:
        support_reason_kind = "direct_primitive_runtime"
    elif export.api_kind != "class" and effective_calls:
        support_reason_kind = "composite_runtime"

    terminal_symbol = ""
    if terminal_calls:
        terminal_symbol = terminal_calls[0]
    elif effective_calls:
        terminal_symbol = effective_calls[0]
    elif export.impl_module and export.impl_name:
        terminal_symbol = f"{export.impl_module}.{export.impl_name}"
    terminal_kind = classify_terminal_kind(
        terminal_symbol,
        func_op_info["is_func_op"],
        export.local_module,
    ) if terminal_symbol else ("func_op" if func_op_info["is_func_op"] else "")

    evidence_ref = export.public_path
    main = {
        "api": export.public_path,
        "category": infer_category(export.public_path, f"{export.impl_module}.{export.impl_name}"),
        "api_level": api_level,
        "semantic_kind": semantic_kind,
        "trust_level": trust_level,
        "fact_origin": fact_origin,
        "status_summary": "",
        "implementation_type": implementation_type,
        "primitive": uniq(primitives),
        "primitive_count": len(uniq(primitives)),
        "possible_primitives": possible_primitives,
        "func_op_expands_to": func_op_info["expanded_primitives"],
        "graph_support_kind": graph_support_kind,
        "support_reason_kind": support_reason_kind,
        "pynative_support": pynative_support,
        "graph_kbk_o0_support": graph_kbk_o0_support,
        "support_summary": support_summary,
        "aclnn": aclnn_info,
        "grad": {"mode": grad_mode, "impl": uniq(grad_impl)},
        "composed_of": uniq(effective_calls)[:12],
        "branching_notes": class_execution.branching_notes if class_execution is not None else [],
        "alias_of": alias_of,
        "primary_paths": primary_paths,
        "confidence": confidence,
        "flags": uniq(flags),
        "evidence_ref": evidence_ref,
    }
    main["unknown_reason"] = infer_unknown_reason(main)
    main["status_summary"] = build_status_summary(main)
    main["llm_summary"] = build_llm_summary(main)
    main["llm_warning"] = build_llm_warning(main)

    evidence = {
        "api": export.public_path,
        "source": uniq([item.to_dict() for item in export.evidence]),
        "impl_symbol": f"{export.impl_module}.{export.impl_name}",
        "impl_path": index.relpath(export.impl_path) if export.impl_path else "",
        "execution_entry": class_execution.entry if class_execution is not None else "",
        "execution_chain": class_execution.chain if class_execution is not None else [],
        "prelude_calls": uniq(prelude_calls),
        "api_def": (
            {
                "name": api_def_name or "",
                "path": index.relpath(api_def_entry["path"]) if api_def_entry else "",
                "used_from_alias_target": bool(alias_of and effective_entries is alias_entries),
            }
        ),
        "terminal_symbol": terminal_symbol,
        "terminal_kind": terminal_kind,
        "primitive_sources": uniq(primitive_sources),
        "alias_chain": [alias_of] if alias_of else [],
        "func_op": {
            "is_func_op": func_op_info["is_func_op"],
            "op_yamls": func_op_info["op_yamls"],
            "meta_dsl_paths": func_op_info["meta_dsl_paths"],
            "expanded_primitives": func_op_info["expanded_primitives"],
        },
        "func_op_expansion_evidence": [dict(item) for item in func_op_info["evidence"]],
        "support_evidence": support_evidence,
        "dispatch_detail": dispatch_detail,
        "aclnn_evidence": uniq(aclnn_evidence + func_op_info["evidence"]),
        "grad": uniq(grad_impl),
        "branching_notes": class_execution.branching_notes if class_execution is not None else [],
        "notes": [],
    }
    if alias_of and effective_entries is alias_entries:
        evidence["notes"].append(f"inherited operator facts from alias target {alias_of}")
    if not api_def_entry:
        evidence["notes"].append("no matching api_def found from static resolution")
    evidence["resolved_symbol_chain"] = export.resolved_symbol_chain
    if primitive_sources and not api_def_entry:
        evidence["notes"].append("operator facts recovered from auto_generate/op_def fallback")
    return main, evidence


def build_review_queue(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    queue = {
        "support_should_be_no_but_still_unknown": [],
        "aclnn_effective_interface_missing": [],
        "primitive_missing_cpu_gpu_kernel_mapping": [],
        "graph_ascend_kbk_unclosed": [],
        "class_api_construct_chain_unclosed": [],
        "simple_wrapper_missing": [],
        "true_unresolved_mapping": [],
        "primitive_mapping_not_applicable": [],
    }
    for record in records:
        ref = {"api": record["api"], "status_summary": record["status_summary"], "evidence_ref": record["evidence_ref"]}
        if record.get("semantic_kind") == "runtime_utility" or record.get("unknown_reason") == "not_applicable":
            queue["primitive_mapping_not_applicable"].append(ref)
            continue
        if not record["primitive_count"] and not record.get("possible_primitives") and not record.get("func_op_expands_to"):
            if record["api_level"] == "wrapper_api" and record.get("composed_of"):
                queue["simple_wrapper_missing"].append(ref)
            else:
                queue["true_unresolved_mapping"].append(ref)
        if (
            record["primitive_count"]
            and record.get("unknown_reason") not in {"func_op_expansion", "missing_kbk_evidence", "not_applicable"}
            and any(record["pynative_support"][backend] == "unknown" for backend in ("ascend", "cpu", "gpu"))
        ):
            queue["support_should_be_no_but_still_unknown"].append(ref)
        if record["aclnn"]["mode"] in {"direct", "indirect"} and not record["aclnn"]["effective_interfaces"] and record["aclnn"]["path_kind"] != "direct_aclnn":
            queue["aclnn_effective_interface_missing"].append(ref)
        if record["primitive_count"] and record.get("graph_support_kind") != "func_op_expansion" and any(
            record["pynative_support"][backend] != "unknown" and record["graph_kbk_o0_support"][backend] == "unknown"
            for backend in ("cpu", "gpu")
        ):
            queue["primitive_missing_cpu_gpu_kernel_mapping"].append(ref)
        if (
            record["primitive_count"]
            and record["graph_kbk_o0_support"]["ascend"] == "unknown"
            and record.get("graph_support_kind") != "func_op_expansion"
            and record.get("unknown_reason") != "not_applicable"
        ):
            queue["graph_ascend_kbk_unclosed"].append(ref)
        if (
            record["api_level"] == "module_api"
            and "construct_mapped" not in record["flags"]
            and record["composed_of"]
            and record.get("unknown_reason") != "scenario_dependent"
        ):
            queue["class_api_construct_chain_unclosed"].append(ref)
    return queue


def build_review_markdown(records: list[dict[str, Any]], queue: dict[str, list[dict[str, Any]]]) -> str:
    lines = [
        "# mint_api_index review",
        "",
        f"- Total records: {len(records)}",
        f"- Operator APIs: {sum(1 for item in records if item['api_level'] == 'operator_api')}",
        f"- Wrapper APIs: {sum(1 for item in records if item['api_level'] == 'wrapper_api')}",
        f"- Module APIs: {sum(1 for item in records if item['api_level'] == 'module_api')}",
        f"- Needs manual review: {sum(1 for item in records if 'needs_manual_review' in item['flags'])}",
        "",
        "## Priority Queues",
        "",
    ]
    for key, items in queue.items():
        lines.append(f"### {key}")
        lines.append("")
        for item in items[:80]:
            lines.append(f"- `{item['api']}`: {item['status_summary']}")
        lines.append("")
    return "\n".join(lines)


def build_rulebook() -> str:
    lines = [
        "# MindSpore Mint API Index Rulebook",
        "",
        "- Main index: `mint_api_index.yaml`.",
        "- Evidence side table: `mint_api_evidence.yaml`.",
        "- Review queue: `mint_api_review_queue.yaml`.",
        "- All paths are relative to the user-provided MindSpore repo root. No absolute paths. No line numbers.",
        "- `api_level`: `operator_api`, `wrapper_api`, `module_api`.",
        "- `implementation_type`: `single_op`, `multi_overload_op`, `composite_op`, `alias`, `high_level_module`, `runtime_utility`.",
        "- `semantic_kind`: `direct_operator`, `multi_overload_operator`, `func_op_operator`, `high_level_module`, `scenario_dependent_module`, `runtime_utility`.",
        "- `trust_level`: `direct_fact`, `inherited_fact`, `scenario_dependent`, `expansion_based`, `not_applicable`.",
        "- `fact_origin`: `direct`, `inherited_from_construct`, `inherited_from_alias`, `expansion_derived`, `not_applicable`.",
        "- `graph_support_kind`: `direct_kernel`, `func_op_expansion`, `fallback`, `scenario_dependent`, `unknown`, `not_applicable`.",
        "- `support_reason_kind`: `direct_primitive_runtime`, `customize_to_aclnn`, `composite_runtime`, `func_op_expansion`, `scenario_dependent`, `runtime_utility`, `unknown`.",
        "- `unknown_reason`: `unresolved_static_chain`, `func_op_expansion`, `scenario_dependent`, `not_applicable`, `missing_kbk_evidence`, `missing_runtime_kernel_evidence`, or empty when no unknown explanation is needed.",
        "- `pynative_support` and `graph_kbk_o0_support` are matrices over `ascend/cpu/gpu` with values `yes`, `no`, `unknown`.",
        "- `llm_summary` is the high-density first-read view for LLM consumption. It compresses existing structured facts without introducing new ones.",
        "- `llm_warning` captures the main boundary that prevents a direct naive interpretation.",
        "- `support_summary` is a template summary for fast reading. It must not introduce new facts beyond structured fields.",
        "- `aclnn` contains `mode`, `interfaces`, `effective_interfaces`, `path_kind`.",
        "- `aclnn.mode`: `direct`, `indirect`, `none`, `unknown`, `not_applicable`.",
        "- `aclnn.path_kind`: `direct_aclnn`, `customize_to_aclnn`, `composite_to_aclnn`, `none`, `unknown`, `not_applicable`.",
        "- `grad.mode`: `explicit_bprop`, `grad_op`, `autodiff`, `not_applicable`, `unknown`.",
        "- Records flagged with `func_op` come from `ops/op_def/func_op` and usually have `bprop_expander: False`. Their GRAPH semantics are expansion-based, not direct-kernel based.",
        "- `func_op_expands_to` lists downstream primitives statically extracted from `ccsrc/frontend/operator/meta_dsl/func_op/*.cc`.",
        "- `multi_overload_op` means one public API maps to multiple primitive overloads. Do not interpret it as a single primitive API.",
        "- `high_level_module` with `construct_mapped` means primitive/support/aclnn/grad facts are inherited through `construct`, not directly defined by the class itself.",
        "- `runtime_utility` means primitive mapping is intentionally treated as not applicable.",
        "- `terminal_symbol` and `terminal_kind` explain the final static resolution endpoint in the evidence table, for example `auto_generate_function`, `ops_wrapper`, `c_expression_instance`, `primitive_instance`, `func_op`.",
        "- `prelude_calls` in the evidence table records helper/setup calls seen before the terminal operator call, such as seed generation or dtype preparation.",
        "- `dispatch.enable=True` only means an adapter layer is generated. It is not a final backend support conclusion.",
        "- PYNATIVE support combines `dispatch`, CPU/GPU kernel registration, and Ascend pyboost/aclnn evidence.",
        "- GRAPH `jit_level='O0'` support is judged separately: Ascend uses KBK aclnn evidence; CPU/GPU use kernel factory or fallback evidence.",
        "- If `dispatch.{platform}=None`, that platform adapter is not generated and the support result should prefer `no`.",
    ]
    return "\n".join(lines) + "\n"


def build_methodology() -> str:
    lines = [
        "# MindSpore Mint API Methodology",
        "",
        "## Error Priority Rules",
        "",
        "1. Prefer `unknown` over a guessed `yes` or `no`.",
        "2. For `func_op`, GRAPH `unknown` does not mean unsupported. It means the GRAPH path should be understood through expansion.",
        "3. `multi_overload_op` must not be interpreted with a single-primitive mental model.",
        "4. For `high_level_module`, inherited primitive/support facts from `construct` must be read as inherited facts, not direct class facts.",
        "5. `runtime_utility` APIs should not continue operator mapping; treat primitive mapping as `not_applicable`.",
        "6. `dispatch.enable=True` is only an adapter-layer clue, not a final support conclusion.",
        "",
        "## How To Read The Index",
        "",
        "1. Read `semantic_kind` first.",
        "2. Then read `trust_level` to know whether the fact is direct, inherited, scenario-dependent, or expansion-based.",
        "3. Then read `primitive` or `possible_primitives`.",
        "4. Then read `pynative_support` and `graph_kbk_o0_support`.",
        "5. Then read `aclnn`.",
        "6. If `llm_warning` is not empty, follow the warning before using the summary as a hard conclusion.",
        "",
        "## General Workflow",
        "",
        "1. Start from the public API and resolve re-exports, `from ... import *`, and aggregator modules. For class symbols in aggregator modules, do not stop at the first star import; collect candidates and choose the best same-name definition.",
        "2. For class APIs, analyze `construct` first. Allow one layer of `self.helper(...)`. Do not recurse through nested class APIs indefinitely.",
        "3. Besides direct `return mint.nn.functional.xxx(...)`, also support callable members bound in `__init__`, for example `self.xxx = ops.auto_generate.SomeOp(...)` and then `self.xxx(...)` inside `construct`.",
        "4. Primitive resolution priority: `api_def -> op_def -> class.name`, then `python symbol -> ops.auto_generate -> op_def`, then small manual mappings for generated names that do not match op_def names.",
        "5. `dispatch.enable=True` means an adapter layer is generated: pyboost for PYNATIVE and KBK for GRAPH `jit_level='O0'`.",
        "6. `dispatch.{platform}=None` means that platform adapter is not generated and should prefer `no` even if later name-based kernel matches exist.",
        "7. PYNATIVE support: Ascend uses pyboost customize, `LAUNCH_ACLNN(...)`, or downstream Ascend pyboost chains; CPU/GPU use kernel registration evidence.",
        "8. GRAPH KBK O0 support: Ascend uses KBK aclnn kernel-mod or register evidence; CPU/GPU use kernel factory or fallback evidence.",
        "9. Ascend KBK registration must include both `MS_ACLNN_KERNEL_FACTORY_REG(...)` and `MS_ACLNN_COMMON_KERNEL_FACTORY_REG(...)`.",
        "10. `aclnn.interfaces` means direct primitive aclnn integration; `aclnn.effective_interfaces` means the final aclnn interfaces actually reached by the execution chain, and it can be multi-valued.",
        "11. If CPU/GPU registry and fallback have both been checked and neither exists, emit `no` instead of leaving `unknown`.",
        "12. For `func_op`, PYNATIVE still follows dispatch/runtime evidence; GRAPH should be modeled through `meta_dsl/func_op` expansion rather than direct kernel registration.",
        "13. `yes/no/unknown` rule: closed evidence gives `yes`; in-scope negative evidence gives `no`; only unresolved static analysis gives `unknown`.",
        "14. For simple wrappers, helper/setup calls such as seed generation must be kept as `prelude_calls` and must not replace the returned operator call as the terminal symbol.",
        "15. Module-level primitive-instance bindings such as `randint_like_ = RandIntLike()` should be resolved as primitive-instance terminals when a wrapper returns `randint_like_(...)`.",
        "",
        "## Examples",
        "",
        "### mindspore.mint.AdaptiveAvgPool1d",
        "- API: `mindspore.mint.AdaptiveAvgPool1d`",
        "- Primitive: `AdaptiveAvgPool1D`",
        "- Execution chain: `construct -> mint.nn.functional.adaptive_avg_pool1d -> ops.auto_generate.adaptive_avg_pool1d`",
        "- PYNATIVE: Ascend=`yes`, CPU=`no`, GPU=`no`",
        "- GRAPH KBK O0: Ascend=`yes`, CPU=`no`, GPU=`no`",
        "- Effective aclnn interface: `aclnnAdaptiveAvgPool2d`",
        "- Conclusion: this class API reaches `aclnnAdaptiveAvgPool2d` through a customized Ascend path.",
        "",
        "### mindspore.mint.xlogy",
        "- API: `mindspore.mint.xlogy`",
        "- Primitive: `Xlogy` plus scalar overload primitive(s)",
        "- PYNATIVE: Ascend/CPU/GPU all close with runtime evidence",
        "- GRAPH KBK O0: Ascend/CPU/GPU all close with runtime evidence",
        "- aclnn: direct aclnn path exists",
        "- Conclusion: all three backends are supported in both modes.",
        "",
        "### mindspore.mint.sum",
        "- API: `mindspore.mint.sum`",
        "- Primitive: `SumExt`",
        "- CPU/GPU kernel mapping: `SumExt -> ReduceSum`",
        "- PYNATIVE: Ascend/CPU/GPU=`yes`",
        "- GRAPH KBK O0: Ascend/CPU/GPU=`yes`; CPU/GPU can also be supported by fallback",
        "- Conclusion: a small manual primitive-to-kernel-name map is necessary for some cases.",
        "",
        "### mindspore.mint.add",
        "- API: `mindspore.mint.add`",
        "- implementation_type: `multi_overload_op`",
        "- Primitive: `AddScalar`, `AddExt`",
        "- PYNATIVE: Ascend/CPU/GPU=`yes`",
        "- GRAPH KBK O0: Ascend/CPU/GPU=`yes`",
        "- fallback: `AddExt` can be supported in GRAPH CPU/GPU through `REG_FALLBACK_BUILDER(\"AddExt\")`",
        "- Conclusion: this is a multi-overload API; do not read it as a single-primitive operator.",
        "",
        "### mindspore.mint.AdaptiveAvgPool3d",
        "- API: `mindspore.mint.AdaptiveAvgPool3d`",
        "- Primitive: `AdaptiveAvgPool3DExt`",
        "- Execution chain: `construct -> mint.nn.functional.adaptive_avg_pool3d -> ops.auto_generate.adaptive_avg_pool3d_ext`",
        "- PYNATIVE: Ascend=`yes`, CPU=`yes`, GPU=`no`",
        "- GRAPH KBK O0: Ascend=`yes`, CPU=`yes`, GPU=`no`",
        "- Effective aclnn interfaces: `aclnnMean`, `aclnnAdaptiveAvgPool3d`",
        "- Conclusion: `dispatch.GPU=None` directly negates GPU; the Ascend customize path reaches two aclnn interfaces.",
        "",
        "### mindspore.mint.BCEWithLogitsLoss",
        "- API: `mindspore.mint.BCEWithLogitsLoss`",
        "- Primitive: `BCEWithLogitsLoss`",
        "- Execution chain: `construct -> self.bce_with_logits -> ops.auto_generate.BCEWithLogitsLoss`",
        "- PYNATIVE: Ascend/CPU/GPU=`yes`",
        "- GRAPH KBK O0: Ascend may still be `unknown`; CPU/GPU are already closed",
        "- Effective aclnn interface: `aclnnBinaryCrossEntropyWithLogits`",
        "- Conclusion: class analysis must also recognize callable members bound in `__init__`, not only direct calls inside `construct`.",
        "",
        "### mindspore.mint.minimum",
        "- API: `mindspore.mint.minimum`",
        "- Primitive: `Minimum`",
        "- PYNATIVE: Ascend/CPU/GPU=`yes`",
        "- GRAPH KBK O0: Ascend/CPU/GPU=`yes`",
        "- Ascend KBK evidence: `MS_ACLNN_COMMON_KERNEL_FACTORY_REG(Minimum, aclnnMinimum, 3)`",
        "- Conclusion: auto-generated `MS_ACLNN_COMMON_KERNEL_FACTORY_REG(...)` must be treated as valid Ascend KBK evidence.",
        "",
        "### mindspore.mint.CosineEmbeddingLoss",
        "- API: `mindspore.mint.CosineEmbeddingLoss`",
        "- Primitive: `CosineEmbeddingLoss`",
        "- Execution chain: `construct -> mint.nn.functional.cosine_embedding_loss -> ops.auto_generate.cosine_embedding_loss`",
        "- func_op: `cosine_embedding_loss_op.yaml` has `bprop_expander: False`; GRAPH expansion is implemented in `meta_dsl/func_op/cosine_embedding_loss.cc`",
        "- func_op_expands_to: `Mul`, `SumExt`, `Sqrt`, `Div`, `ClampMin`, `MeanExt`, ...",
        "- Conclusion: in GRAPH mode, this API should be understood through func_op expansion rather than direct-kernel registration.",
        "",
        "### mindspore.mint.randint_like",
        "- API: `mindspore.mint.randint_like`",
        "- Primitive: `RandIntLike`",
        "- Execution chain: `mint.randint_like -> ops.function.random_func.randint_like_ext -> randint_like_`",
        "- Prelude calls: `default_generator._step`",
        "- Terminal kind: `primitive_instance`",
        "- PYNATIVE: Ascend=`yes`, CPU=`no`, GPU=`no`",
        "- GRAPH KBK O0: Ascend=`yes`, CPU=`no`, GPU=`no`",
        "- Effective aclnn interface: `aclnnInplaceRandom`",
        "- Conclusion: for simple wrappers, helper calls must not replace the returned primitive-instance terminal.",
        "",
        "### mindspore.mint.distributed.get_rank",
        "- API: `mindspore.mint.distributed.get_rank`",
        "- semantic_kind: `runtime_utility`",
        "- Primitive mapping: not applicable",
        "- grad/aclnn: `not_applicable`",
        "- Conclusion: this is a runtime utility API and should not be forced into operator mapping.",
        "",
    ]
    return "\n".join(lines) + "\n"


def run_gen_ops(repo_root: Path) -> None:
    subprocess.run([sys.executable, str(repo_root / GEN_OPS)], check=True, cwd=repo_root)


def current_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_stdout(repo_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def repo_branch_hint(repo_root: Path) -> str:
    try:
        return git_stdout(repo_root, "branch", "--show-current")
    except Exception:
        return ""


def safe_rmtree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


def clone_remote_repo(remote_url: str, branch: str | None, workspace_root: Path) -> Path:
    safe_rmtree(workspace_root)
    workspace_root.mkdir(parents=True, exist_ok=True)
    repo_root = workspace_root / "repo"
    cmd = ["git", "-c", "core.longpaths=true", "clone", remote_url, str(repo_root)]
    if branch:
        cmd = [
            "git",
            "-c",
            "core.longpaths=true",
            "clone",
            "--branch",
            branch,
            "--single-branch",
            remote_url,
            str(repo_root),
        ]
    subprocess.run(cmd, check=True)
    return repo_root


def prepare_repo(args: argparse.Namespace) -> tuple[Path, str, str, str]:
    if args.repo and args.branch:
        raise SystemExit("--repo and --branch cannot be used together.")
    if args.repo:
        repo_root = Path(args.repo).resolve()
        if not repo_root.exists():
            raise FileNotFoundError(f"MindSpore repo not found: {repo_root}")
        return repo_root, "local", str(repo_root), repo_branch_hint(repo_root)
    workspace_root = Path(args.workspace_root).resolve()
    repo_root = clone_remote_repo(DEFAULT_REMOTE_URL, args.branch, workspace_root)
    return repo_root, "remote", DEFAULT_REMOTE_URL, args.branch or repo_branch_hint(repo_root)


def remove_legacy_outputs(out_dir: Path) -> None:
    legacy = out_dir / "mint_api_index.jsonl"
    if legacy.exists():
        legacy.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build compact MindSpore mint API index.")
    parser.add_argument("--repo", help="Path to a local MindSpore repo root.")
    parser.add_argument("--branch", help="Remote branch to clone from the default MindSpore repository.")
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT_DIR), help="Output directory.")
    parser.add_argument("--workspace-root", default=str(DEFAULT_WORKSPACE_ROOT), help="Temporary workspace root.")
    parser.add_argument("--keep-workspace", action="store_true", help="Keep the temporary cloned workspace.")
    parser.add_argument("--skip-gen-ops", action="store_true", help="Skip running gen_ops.py before indexing.")
    parser.add_argument("--with-evidence", action="store_true", help="Generate mint_api_evidence.yaml.")
    parser.add_argument("--with-review", action="store_true", help="Generate review queue and markdown outputs.")
    parser.add_argument("--with-rulebook", action="store_true", help="Generate mint_api_index_rulebook.md.")
    args = parser.parse_args()

    repo_root, source_mode, source_repo_url, source_branch = prepare_repo(args)
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        if not args.skip_gen_ops:
            run_gen_ops(repo_root)

        index = SourceIndex(repo_root)
        exports = gather_exports(index)
        main_records = []
        evidence_records = []
        for export in exports:
            main_record, evidence_record = build_record(index, export)
            main_records.append(main_record)
            evidence_records.append(evidence_record)

        inherit_construct_records(main_records, evidence_records)
        queue = build_review_queue(main_records)
        generated_at = current_timestamp()
        repo_commit = repo_commit_hint(repo_root)
        version_hint = mindspore_version_hint(repo_root)
        base_meta = {
            "generated_at": generated_at,
            "generator_name": GENERATOR_NAME,
            "generator_version": GENERATOR_VERSION,
            "source_mode": source_mode,
            "source_repo_url": source_repo_url,
            "source_branch": source_branch,
            "source_commit": repo_commit,
            "mindspore_version_hint": version_hint,
            "index_schema_version": "1.0",
            "generated_after_gen_ops": not args.skip_gen_ops,
        }
        main_payload = {
            "meta": {
                **base_meta,
                "repo_root_hint": "mindspore repo root",
                "api_count": len(main_records),
            },
            "stats": {
                "operator_api": sum(1 for item in main_records if item["api_level"] == "operator_api"),
                "wrapper_api": sum(1 for item in main_records if item["api_level"] == "wrapper_api"),
                "module_api": sum(1 for item in main_records if item["api_level"] == "module_api"),
                "needs_manual_review": sum(1 for item in main_records if "needs_manual_review" in item["flags"]),
            },
            "apis": main_records,
        }
        evidence_payload = {
            "meta": {
                **base_meta,
                "api_count": len(evidence_records),
            },
            "apis": evidence_records,
        }
        review_payload = {
            "meta": {
                **base_meta,
                "review_bucket_count": len(queue),
            },
            "queue": queue,
        }

        files = [str(out_dir / "mint_api_index.yaml"), str(out_dir / "mint_api_methodology.md")]
        yaml_dump(canonicalize_for_yaml(main_payload), out_dir / "mint_api_index.yaml")
        (out_dir / "mint_api_methodology.md").write_text(build_methodology(), encoding="utf-8")

        if args.with_evidence:
            yaml_dump(canonicalize_for_yaml(evidence_payload), out_dir / "mint_api_evidence.yaml")
            files.append(str(out_dir / "mint_api_evidence.yaml"))
        if args.with_review:
            yaml_dump(canonicalize_for_yaml(review_payload), out_dir / "mint_api_review_queue.yaml")
            (out_dir / "mint_api_index_review.md").write_text(
                build_review_markdown(main_records, queue),
                encoding="utf-8",
            )
            files.extend(
                [
                    str(out_dir / "mint_api_review_queue.yaml"),
                    str(out_dir / "mint_api_index_review.md"),
                ]
            )
        if args.with_rulebook:
            (out_dir / "mint_api_index_rulebook.md").write_text(build_rulebook(), encoding="utf-8")
            files.append(str(out_dir / "mint_api_index_rulebook.md"))

        remove_legacy_outputs(out_dir)
        summary = {"apis": len(main_records), "files": files}
        print(yaml.safe_dump(summary, allow_unicode=True, sort_keys=False))
    finally:
        if source_mode == "remote" and not args.keep_workspace:
            safe_rmtree(Path(args.workspace_root).resolve())


if __name__ == "__main__":
    main()
