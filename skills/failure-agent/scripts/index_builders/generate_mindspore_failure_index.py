#!/usr/bin/env python3
"""Build the canonical MindSpore Mint API index artifacts for LLM use.

Default manual command:
    Run from the failure-agent directory when a local MindSpore checkout is
    available. This avoids network clone and skips regenerating MindSpore
    generated ops when the checkout is already prepared.

    python scripts/index_builders/generate_mindspore_failure_index.py \
        --repo D:/path/to/mindspore \
        --skip-gen-ops \
        --deterministic \
        --with-yaml

Remote clone command:
    Omit --repo to clone the default MindSpore remote repository into the
    temporary workspace. Add --branch to pin a remote branch.

    python scripts/index_builders/generate_mindspore_failure_index.py \
        --branch master \
        --deterministic \
        --with-yaml

Parameters:
    --repo PATH
        Use an existing local MindSpore repository. When this is set, no remote
        clone is performed. Default: unset.
    --branch NAME
        Remote branch to clone from https://atomgit.com/mindspore/mindspore.git
        when --repo is not set. Default: remote default branch.
    --out DIR
        Output directory. Default: reference/index under failure-agent.
    --workspace-root DIR
        Temporary clone workspace used only for remote builds. Default:
        scripts/index_builders/.tmp/mindspore.
    --keep-workspace
        Keep the temporary remote clone after the build. Default: delete it.
    --skip-gen-ops
        Do not run mindspore/python/mindspore/ops_generate/gen_ops.py before
        indexing. Recommended for local prepared checkouts. Default: run it.
    --with-yaml
        Also write mint_api_index.yaml for manual review. Default: disabled.
    --with-evidence
        Also write mint_api_evidence.yaml. Default: disabled.
    --with-review
        Also write review queue and markdown review outputs. Default: disabled.
    --with-rulebook
        Also write mint_api_index_rulebook.md. Default: disabled.
    --deterministic
        Use deterministic metadata timestamps for reproducible outputs. Default:
        use current UTC timestamp.

Outputs:
    Always writes mint_api_index.db. Optional flags can additionally write YAML,
    evidence, review, and rulebook artifacts into --out.
"""

from __future__ import annotations

import argparse
import ast
import copy
from datetime import datetime, timezone
import json
import os
import re
import shutil
import sqlite3
import stat
import subprocess
import sys
import time
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
DEFAULT_DB_NAME = "mint_api_index.db"
GENERATOR_NAME = "generate_mindspore_failure_index.py"
GENERATOR_VERSION = "1.0.0"
INDEX_SCHEMA_VERSION = "2.0"
DETERMINISTIC_TIMESTAMP = "1970-01-01T00:00:00+00:00"

PYTHON_ROOT = Path("mindspore/python")
MINT_ROOT = PYTHON_ROOT / "mindspore/mint"
MINT_DOC_INDEX = Path("docs/api/api_python_en/mindspore.mint.rst")
API_DEF_ROOT = Path("mindspore/ops/api_def")
OP_DEF_ROOT = Path("mindspore/ops/op_def/yaml")
FUNC_OP_DEF_ROOT = Path("mindspore/ops/op_def/func_op")
ACLNN_CONFIG = Path("mindspore/python/mindspore/ops_generate/pyboost/aclnn_config.yaml")
GRAD_ROOT = Path("mindspore/ccsrc/frontend/expander/grad")
GEN_OPS = Path("mindspore/python/mindspore/ops_generate/gen_ops.py")
FALLBACK_ROOT = Path("mindspore/ops/fallback")
META_DSL_FUNC_OP_ROOT = Path("mindspore/ccsrc/frontend/operator/meta_dsl/func_op")
OPS_INFER_FUNC_IMPL_ROOT = Path("mindspore/ops/infer/ops_func_impl")
OPS_INFER_SYMBOL_IMPL_ROOT = Path("mindspore/ops/infer/symbol_ops_impl")
ASCEND_OP_ADAPTER_DECLARE_ROOT = Path("mindspore/ccsrc/plugin/ascend/res_manager/op_adapter/op_declare")
FUNCTIONAL_OVERLOAD_MODULE = "mindspore.ops.functional_overload"
FUNCTIONAL_MAP_PATH = Path("mindspore/ccsrc/frontend/operator/composite/auto_generate/functional_map.cc")
PYBOOST_OVERLOAD_FUNCTIONS_PATH = Path(
    "mindspore/ccsrc/pynative/forward/pyboost/auto_generate/pyboost_overload_functions.cc"
)
CPU_KERNEL_ROOT = Path("mindspore/ops/kernel/cpu")
GPU_KERNEL_ROOT = Path("mindspore/ops/kernel/gpu")
CPU_PYBOOST_ROOT = Path("mindspore/ops/kernel/cpu/pyboost")
GPU_PYBOOST_ROOT = Path("mindspore/ops/kernel/gpu/pyboost")
PYBOOST_COMPOSITE_ROOT = Path("mindspore/ccsrc/pynative/utils/pyboost/functions/composite")
PYBOOST_API_PATH = Path("mindspore/ccsrc/pynative/forward/pyboost/auto_generate/pyboost_api.cc")
PYBOOST_CORE_PATH = Path("mindspore/ccsrc/pynative/forward/pyboost/auto_generate/pyboost_core.cc")
PYBOOST_FUNCTIONS_PATH = Path("mindspore/ccsrc/pynative/utils/pyboost/functions/auto_generate/functions.cc")
ASCEND_ACLNN_AUTO_GEN_ROOT = Path("mindspore/ops/kernel/ascend/aclnn/kernel_mod_impl/aclnn_auto_gen")
ASCEND_ACLNN_CUSTOMIZE_ROOT = Path("mindspore/ops/kernel/ascend/aclnn/kernel_mod_impl/customize")
ASCEND_ACLNN_REGISTER = Path("mindspore/ops/kernel/ascend/aclnn/kernel_mod_impl/auto_generate/aclnn_kernel_register_auto.cc")
ASCEND_PYBOOST_AUTO_GEN_ROOT = Path("mindspore/ops/kernel/ascend/aclnn/pyboost_impl/auto_generate")
ASCEND_PYBOOST_CUSTOMIZE_ROOT = Path("mindspore/ops/kernel/ascend/aclnn/pyboost_impl/customize")
HOST_VIEW_KERNEL_ROOT = Path("mindspore/ops/kernel/host/view/kernel_mod_impl")
ASCEND_KERNEL_SELECT_PATH = Path("mindspore/ccsrc/plugin/ascend/kernel_executor/kernel_select_ascend.cc")
ANFALGO_PATH = Path("mindspore/ccsrc/utils/anfalgo.cc")
ASCEND_KERNEL_EXECUTOR_PATH = Path("mindspore/ccsrc/plugin/ascend/kernel_executor/ascend_kernel_executor.cc")
PYBOOST_CUSTOMIZE_FUNCTION_ROOT = Path("mindspore/ccsrc/pynative/utils/pyboost/functions/customize")

PRIMITIVE_KERNEL_NAME_MAP = {
    "SumExt": ["ReduceSum"],
}

SYMBOL_OP_DEF_MAP = {
    "BCEWithLogitsLoss": ["binary_cross_entropy_with_logits_op.yaml"],
    "LogSoftmaxExt": ["log_softmax_ext_op.yaml"],
}

GRAPH_FALLBACK_DISPATCH_NONE_OVERRIDE_PRIMITIVES = {"ArgMaxExt", "NonZeroExt"}
GRAPH_FALLBACK_COMPANION_PRIMITIVE_MAP = {"NonZero": "NonZeroExt"}
PURE_PYTHON_PRELUDE_PREFIXES = (
    "mindspore._checkparam.",
)
PURE_PYTHON_PRELUDE_SYMBOLS = {
    "isinstance",
    "tuple",
    "list",
    "int",
    "len",
    "range",
}
FRONTEND_GUARD_PREFIXES = (
    "mindspore._checkparam.",
    "mindspore.ops.functional.isconstant",
)
FRONTEND_GUARD_SYMBOL_PARTS = (
    ".check_",
    ".check",
)
CONSTRUCTOR_SETUP_PREFIXES = (
    "mindspore.common.initializer.",
    "mindspore.common.parameter.Parameter",
    "mindspore.common.tensor.Tensor",
    "mindspore.graph._utils.cell_attr_register.",
)
CONSTRUCTOR_SETUP_SYMBOLS = {
    "Parameter",
    "Tensor",
    "initializer",
    "math.sqrt",
    "mindspore._extends.cell_attr_register",
}


@dataclass
class PathEntry:
    path: str
    anchor: str = ""


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
        if not self.source_name:
            return self.source_module
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
class GradAnalysis:
    mode: str
    differentiable: str
    impl: list[dict[str, str]]
    backward_primitives: list[str]


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


def json_dump_text(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


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


def _path_sort_key(item: Any) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        return str(item.get("path", ""))
    return str(item)


def _sort_path_list(values: list[Any]) -> list[Any]:
    return sorted(values, key=_path_sort_key)


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
    sort_path_keys = {
        "api_def_paths",
        "dispatch_paths",
        "implementation_paths",
        "op_def_paths",
        "infer_paths",
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
        if parent_key in sort_path_keys:
            return _sort_path_list(normalized)
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
        self.aclop_adapter_map = self._load_aclop_adapter_map()
        self.cpu_pyboost_op_map = self._load_pyboost_op_map("cpu")
        self.gpu_pyboost_op_map = self._load_pyboost_op_map("gpu")
        self.cpu_pyboost_custom_kernel_map = self._load_pyboost_custom_kernel_map("cpu")
        self.gpu_pyboost_custom_kernel_map = self._load_pyboost_custom_kernel_map("gpu")
        self.cpu_pyboost_impl_map = self._load_pyboost_impl_map("cpu")
        self.gpu_pyboost_impl_map = self._load_pyboost_impl_map("gpu")
        self.pyboost_composite_inner_map = self._load_pyboost_composite_inner_map()
        self.ascend_kbk_map = self._load_ascend_kbk_map()
        self.host_view_kernel_map = self._load_host_view_kernel_map()
        self.generated_primitive_names = self._load_generated_primitive_names()
        self.functional_overload_graph_map = self._load_functional_overload_graph_map()
        self.functional_overload_pynative_map = self._load_functional_overload_pynative_map()
        self.python_function_modules = self._load_python_function_modules()
        self.infer_path_map = self._load_infer_path_map()
        self.pyboost_ascend_impl_map = self._load_pyboost_ascend_impl_map()
        self.ascend_kernel_impl_map = self._load_ascend_kernel_impl_map()
        self.view_op_names = self._build_view_op_names()
        self.view_pynative_impl_map = self._load_view_pynative_impl_map()
        self.rt_nop_view_map = self._load_rt_nop_view_map()
        self.fallback_reshape_map = self._load_fallback_reshape_map()

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

    def _build_view_op_names(self) -> set[str]:
        result: set[str] = set()
        for info in self.op_defs.values():
            payload = info.get("payload")
            if isinstance(payload, dict) and payload.get("view") is True:
                op_name = str(info["op_name"])
                result.add(op_name)
                class_info = payload.get("class") if isinstance(payload, dict) else None
                if isinstance(class_info, dict) and class_info.get("name"):
                    result.add(str(class_info["name"]))
                else:
                    result.add("".join(part.capitalize() for part in op_name.split("_")))
        return result

    def _load_view_pynative_impl_map(self) -> dict[str, list[PathEntry]]:
        result: dict[str, list[PathEntry]] = defaultdict(list)
        files = (
            (self.repo_root / PYBOOST_API_PATH, lambda primitive, snake: rf"\bPyboost{re.escape(primitive)}(?:Base|Op)\s*\("),
            (self.repo_root / PYBOOST_CORE_PATH, lambda primitive, snake: rf"\bPyboost{re.escape(primitive)}OpExec\s*\("),
            (
                self.repo_root / PYBOOST_FUNCTIONS_PATH,
                lambda primitive, snake: rf"\b(?:{re.escape(snake)}_view_impl|{re.escape(snake)})\s*\(",
            ),
        )
        for primitive in sorted(self.view_op_names):
            snake = camel_to_snake(primitive)
            for path, pattern_factory in files:
                if not path.exists():
                    continue
                text = path.read_text(encoding="utf-8", errors="ignore")
                match = re.search(pattern_factory(primitive, snake), text)
                if match:
                    result[primitive].append(PathEntry(self.relpath(path), match.group(0).rstrip()))
            if primitive == "FlattenExt":
                for rel_path, anchor in (
                    (PYBOOST_CUSTOMIZE_FUNCTION_ROOT / "flatten_ext_impl.cc", "flatten_ext_impl"),
                    (PYBOOST_CUSTOMIZE_FUNCTION_ROOT / "reshape_impl.cc", "reshape_impl"),
                ):
                    path = self.repo_root / rel_path
                    if path.exists() and anchor in path.read_text(encoding="utf-8", errors="ignore"):
                        result[primitive].append(PathEntry(self.relpath(path), anchor))
            elif primitive == "Reshape":
                path = self.repo_root / PYBOOST_CUSTOMIZE_FUNCTION_ROOT / "reshape_impl.cc"
                if path.exists() and "reshape_impl" in path.read_text(encoding="utf-8", errors="ignore"):
                    result[primitive].append(PathEntry(self.relpath(path), "reshape_impl"))
        return {key: uniq(values) for key, values in result.items()}

    def _load_rt_nop_view_map(self) -> dict[str, list[EvidenceItem]]:
        kernel_select = self.repo_root / ASCEND_KERNEL_SELECT_PATH
        anfalgo = self.repo_root / ANFALGO_PATH
        executor = self.repo_root / ASCEND_KERNEL_EXECUTOR_PATH
        if not kernel_select.exists() or not anfalgo.exists() or not executor.exists():
            return {}
        kernel_select_text = kernel_select.read_text(encoding="utf-8", errors="ignore")
        anfalgo_text = anfalgo.read_text(encoding="utf-8", errors="ignore")
        executor_text = executor.read_text(encoding="utf-8", errors="ignore")
        if "nop_op_to_memcpy_" not in executor_text or "MemoryCopyAsync" not in executor_text:
            return {}
        result: dict[str, list[EvidenceItem]] = {}
        for primitive in sorted(self.view_op_names):
            prim_token = f"prim::kPrim{primitive}"
            name_token = f'prim::kPrim{primitive}->name()'
            if prim_token not in kernel_select_text:
                continue
            if name_token not in anfalgo_text and f'"{primitive}"' not in anfalgo_text:
                continue
            result[primitive] = [
                EvidenceItem(
                    self.relpath(kernel_select),
                    "direct",
                    f"rt_kernel_ops:{primitive}",
                    f"Ascend graph RT_KERNEL selection for {primitive}",
                ),
                EvidenceItem(
                    self.relpath(anfalgo),
                    "direct",
                    f"IsNopNode:{primitive}",
                    f"Ascend graph NOP classification for {primitive}",
                ),
                EvidenceItem(
                    self.relpath(executor),
                    "direct",
                    "nop_op_to_memcpy_/MemoryCopyAsync",
                    f"Ascend graph NOP memcpy launch path for {primitive}",
                ),
            ]
        return result

    def _load_generated_primitive_names(self) -> set[str]:
        module = self.load_module("mindspore.ops.auto_generate.gen_ops_prim")
        if module is None:
            return set()
        return {
            name
            for name, binding in module.locals.items()
            if binding.kind == "class" and isinstance(binding.node, ast.ClassDef)
        }

    def _load_functional_overload_graph_map(self) -> dict[str, list[str]]:
        path = self.repo_root / FUNCTIONAL_MAP_PATH
        if not path.exists():
            return {}
        text = path.read_text(encoding="utf-8", errors="ignore")
        result: dict[str, list[str]] = defaultdict(list)
        for api_name, raw_values in re.findall(r'\{"([A-Za-z0-9_]+)"\s*,\s*\{([^}]*)\}\}', text):
            for token in raw_values.split(","):
                token = token.strip()
                if not token or token.startswith("Deprecated"):
                    continue
                if token.startswith("prim::kPrim"):
                    result[api_name].append(token.replace("prim::kPrim", ""))
        return {key: uniq(values) for key, values in result.items()}

    def _load_functional_overload_pynative_map(self) -> dict[str, list[str]]:
        path = self.repo_root / PYBOOST_OVERLOAD_FUNCTIONS_PATH
        if not path.exists():
            return {}
        text = path.read_text(encoding="utf-8", errors="ignore")
        pattern = re.compile(
            r'class\s+[A-Za-z0-9_]+Functional\s*:\s*public\s+Functional\s*\{.*?'
            r'PythonArgParser parser\(\{(.*?)\},\s*"([A-Za-z0-9_]+)"\);',
            re.S,
        )
        result: dict[str, list[str]] = defaultdict(list)
        for raw_signatures, api_name in pattern.findall(text):
            result[api_name].extend(re.findall(r'"([A-Za-z0-9_]+)\(', raw_signatures))
        return {key: uniq(values) for key, values in result.items()}

    def _load_python_function_modules(self) -> dict[str, list[str]]:
        root = self.repo_root / PYTHON_ROOT / "mindspore" / "ops"
        if not root.exists():
            return {}
        result: dict[str, list[str]] = defaultdict(list)
        for path in sorted(root.rglob("*.py")):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            rel = path.relative_to(self.repo_root / PYTHON_ROOT).with_suffix("")
            parts = list(rel.parts)
            if parts and parts[-1] == "__init__":
                parts = parts[:-1]
            if not parts:
                continue
            module_name = ".".join(parts)
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    result[node.name].append(module_name)
        return {key: uniq(values) for key, values in result.items()}

    def _load_infer_path_map(self) -> dict[str, list[PathEntry]]:
        result: dict[str, list[PathEntry]] = defaultdict(list)
        for root in (
            self.repo_root / OPS_INFER_FUNC_IMPL_ROOT,
            self.repo_root / OPS_INFER_SYMBOL_IMPL_ROOT,
        ):
            if not root.exists():
                continue
            is_symbol = str(root).replace("\\", "/").endswith("symbol_ops_impl")
            for path in sorted(root.glob("*.cc")):
                text = path.read_text(encoding="utf-8", errors="ignore")
                anchor = self._extract_infer_anchor(text, path.stem, is_symbol)
                result[path.stem].append(PathEntry(self.relpath(path), anchor))
        return {key: uniq(values) for key, values in result.items()}

    @staticmethod
    def _extract_infer_anchor(text: str, stem: str, is_symbol: bool) -> str:
        if is_symbol:
            m = re.search(r'REG_SYMBOL_OP_BUILDER\("([^"]+)"\)', text)
            if m:
                return f'REG_SYMBOL_OP_BUILDER("{m.group(1)}")'
            m = re.search(r"class\s+(?:OPS_API\s+)?(\w+)\s*:\s*public\s+InferShapeOp", text)
            if m:
                return f"class {m.group(1)}"
        else:
            m = re.search(r"REGISTER_SIMPLE_INFER\(\w+,\s*(\w+)\)", text)
            if m:
                return m.group(1)
            m = re.search(r"(\w+FuncImpl)::\w+", text)
            if m:
                return m.group(1)
        return stem

    def _load_pyboost_ascend_impl_map(self) -> dict[str, list[PathEntry]]:
        result: dict[str, list[PathEntry]] = defaultdict(list)
        roots = (
            self.repo_root / ASCEND_PYBOOST_AUTO_GEN_ROOT,
            self.repo_root / ASCEND_PYBOOST_CUSTOMIZE_ROOT,
        )
        for root in roots:
            if not root.exists():
                continue
            for path in sorted(root.rglob("*")):
                if not path.is_file() or path.suffix not in {".cc", ".h"}:
                    continue
                text = path.read_text(encoding="utf-8", errors="ignore")
                for primitive in set(re.findall(r"([A-Za-z0-9_]+)Ascend::Call\s*\(", text)):
                    result[primitive].append(PathEntry(self.relpath(path), f"{primitive}Ascend::Call("))
                for primitive in set(re.findall(r"([A-Za-z0-9_]+)AscendCustomize\s*\(", text)):
                    result[primitive].append(PathEntry(self.relpath(path), f"{primitive}AscendCustomize("))
        return {key: uniq(values) for key, values in result.items()}

    def _load_ascend_kernel_impl_map(self) -> dict[str, list[PathEntry]]:
        result: dict[str, list[PathEntry]] = defaultdict(list)
        for root in (
            self.repo_root / ASCEND_ACLNN_AUTO_GEN_ROOT,
            self.repo_root / ASCEND_ACLNN_CUSTOMIZE_ROOT,
        ):
            if not root.exists():
                continue
            for path in sorted(root.rglob("*")):
                if not path.is_file() or path.suffix not in {".cc", ".h"}:
                    continue
                text = path.read_text(encoding="utf-8", errors="ignore")
                class_matches = set(re.findall(r"class\s+([A-Za-z0-9_]+)Ascend\b", text))
                factory_matches = set(re.findall(r"MS_ACLNN_KERNEL_FACTORY_REG\(\s*([A-Za-z0-9_]+)\s*,", text))
                common_factory_matches = set(re.findall(r"MS_ACLNN_COMMON_KERNEL_FACTORY_REG\(\s*([A-Za-z0-9_]+)\s*,", text))
                all_primitives = class_matches | factory_matches | common_factory_matches
                rel = self.relpath(path)
                for primitive in all_primitives:
                    if primitive in class_matches:
                        anchor = f"class {primitive}Ascend"
                    elif primitive in factory_matches:
                        anchor = f"MS_ACLNN_KERNEL_FACTORY_REG({primitive},"
                    else:
                        anchor = f"MS_ACLNN_COMMON_KERNEL_FACTORY_REG({primitive},"
                    result[primitive].append(PathEntry(rel, anchor))
        return {key: uniq(values) for key, values in result.items()}

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

    def _load_fallback_reshape_map(self) -> dict[str, list[EvidenceItem]]:
        result: dict[str, list[EvidenceItem]] = defaultdict(list)
        root = self.repo_root / FALLBACK_ROOT
        if not root.exists():
            return {}
        pattern = re.compile(r'REG_FALLBACK_BUILDER\("([^"]+)"\)\.SetBody\(BODYFUNC\(ib\)\s*\{(.*?)\n\}\);', re.S)
        for path in sorted(root.glob("*.cc")):
            text = path.read_text(encoding="utf-8", errors="ignore")
            for primitive, body in pattern.findall(text):
                if "ib->Reshape" not in body:
                    continue
                result[primitive].append(
                    EvidenceItem(
                        self.relpath(path),
                        "derived",
                        f'REG_FALLBACK_BUILDER("{primitive}") -> ib->Reshape',
                        f"fallback builder for {primitive} reaches Reshape",
                    )
                )
        return dict(result)

    def _load_aclop_adapter_map(self) -> dict[str, list[EvidenceItem]]:
        result: dict[str, list[EvidenceItem]] = defaultdict(list)
        root = self.repo_root / ASCEND_OP_ADAPTER_DECLARE_ROOT
        if not root.exists():
            return {}
        pattern = re.compile(r"REG_ADPT_DESC\(\s*([A-Za-z0-9_]+)\s*,")
        for path in sorted(root.rglob("*.cc")):
            text = path.read_text(encoding="utf-8", errors="ignore")
            for match in pattern.finditer(text):
                primitive = match.group(1)
                anchor = match.group(0)
                result[primitive].append(
                    EvidenceItem(self.relpath(path), "direct", anchor, f"Ascend aclop adapter for {primitive}")
                )
        return dict(result)

    def _load_kernel_registry_map(self, backend: str) -> dict[str, list[EvidenceItem]]:
        result: dict[str, list[EvidenceItem]] = defaultdict(list)
        if backend == "cpu":
            root = self.repo_root / CPU_KERNEL_ROOT
            mod_class = "NativeCpuKernelMod"
        else:
            root = self.repo_root / GPU_KERNEL_ROOT
            mod_class = "NativeGpuKernelMod"
        direct_patterns = [
            rf"MS_KERNEL_FACTORY_REG_BY_CREATOR\({mod_class},\s*([A-Za-z0-9_]+)",
            rf"MS_KERNEL_FACTORY_REG\({mod_class},\s*([A-Za-z0-9_]+)",
        ]
        if not root.exists():
            return {}
        for path in sorted(root.rglob("*.cc")):
            text = path.read_text(encoding="utf-8", errors="ignore")
            define_spans = self._define_body_spans(text)
            for pattern in direct_patterns:
                for match in re.finditer(pattern, text):
                    if any(s <= match.start() < e for s, e in define_spans):
                        continue
                    kernel_name = match.group(1)
                    anchor = match.group(0) + ")"
                    result[kernel_name].append(
                        EvidenceItem(self.relpath(path), "direct", anchor, f"{backend.upper()} kernel factory for {kernel_name}")
                    )
            for macro_name, name_param_idx in self._find_wrapper_macros(text, mod_class):
                call_pat = re.compile(rf"(?<!\w){re.escape(macro_name)}\(([^)]+)\)")
                for m in call_pat.finditer(text):
                    if any(s <= m.start() < e for s, e in define_spans):
                        continue
                    args = [a.strip() for a in m.group(1).split(",")]
                    if name_param_idx < len(args):
                        kernel_name = args[name_param_idx]
                        if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", kernel_name):
                            anchor = m.group(0)
                            result[kernel_name].append(
                                EvidenceItem(self.relpath(path), "wrapper_macro", anchor, f"{backend.upper()} kernel via {macro_name}")
                            )
        return dict(result)

    def _load_pyboost_op_map(self, backend: str) -> dict[str, list[EvidenceItem]]:
        result: dict[str, list[EvidenceItem]] = defaultdict(list)
        root = self.repo_root / (CPU_PYBOOST_ROOT if backend == "cpu" else GPU_PYBOOST_ROOT)
        macro = "CPU" if backend == "cpu" else "GPU"
        if not root.exists():
            return {}
        pattern = re.compile(rf"MS_REG_PYBOOST_OP\(\s*{macro}\s*,\s*([A-Za-z0-9_]+)\s*\)")
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix not in {".cc", ".h"}:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for match in pattern.finditer(text):
                primitive = match.group(1)
                result[primitive].append(
                    EvidenceItem(
                        self.relpath(path),
                        "direct",
                        match.group(0),
                        f"{backend.upper()} pyboost op registration for {primitive}",
                    )
                )
        return dict(result)

    def _load_pyboost_custom_kernel_map(self, backend: str) -> dict[str, list[EvidenceItem]]:
        result: dict[str, list[EvidenceItem]] = defaultdict(list)
        root = self.repo_root / (CPU_PYBOOST_ROOT if backend == "cpu" else GPU_PYBOOST_ROOT)
        macro = "MS_REG_PYBOOST_CPU_CUSTOM_KERNEL" if backend == "cpu" else "MS_REG_PYBOOST_GPU_CUSTOM_KERNEL"
        if not root.exists():
            return {}
        pattern = re.compile(rf"{macro}\(\s*([A-Za-z0-9_]+)\s*\)")
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix not in {".cc", ".h"}:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for match in pattern.finditer(text):
                primitive = match.group(1)
                result[primitive].append(
                    EvidenceItem(
                        self.relpath(path),
                        "direct",
                        match.group(0),
                        f"{backend.upper()} pyboost custom kernel registration for {primitive}",
                    )
                )
        return dict(result)

    def _load_pyboost_impl_map(self, backend: str) -> dict[str, list[PathEntry]]:
        result: dict[str, list[PathEntry]] = defaultdict(list)
        root = self.repo_root / (CPU_PYBOOST_ROOT if backend == "cpu" else GPU_PYBOOST_ROOT)
        suffix = "CPU" if backend == "cpu" else "GPU"
        if not root.exists():
            return {}
        call_pattern = re.compile(rf"([A-Za-z0-9_]+){suffix}::Call\s*\(")
        customize_pattern = re.compile(rf"\b([A-Za-z0-9_]+){suffix}Customize\s*\(")
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix not in {".cc", ".h"}:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for primitive in set(call_pattern.findall(text)):
                result[primitive].append(PathEntry(self.relpath(path), f"{primitive}{suffix}::Call("))
            for primitive in set(customize_pattern.findall(text)):
                result[primitive].append(PathEntry(self.relpath(path), f"{primitive}{suffix}Customize("))
        return {key: uniq(values) for key, values in result.items()}

    def _load_pyboost_composite_inner_map(self) -> dict[str, list[dict[str, str]]]:
        result: dict[str, list[dict[str, str]]] = defaultdict(list)
        root = self.repo_root / PYBOOST_COMPOSITE_ROOT
        if not root.exists():
            return {}
        inner_function_to_primitive = {
            "inplace_uniform": "InplaceUniform",
            "inplace_random": "InplaceRandom",
            "inplace_normal": "InplaceNormal",
        }
        for path in sorted(root.glob("*.cc")):
            text = path.read_text(encoding="utf-8", errors="ignore")
            outer_key = path.stem.lower()
            rel = self.relpath(path)
            for inner_function, inner_primitive in inner_function_to_primitive.items():
                pattern = re.compile(rf"\b{re.escape(inner_function)}\s*\(")
                for match in pattern.finditer(text):
                    result[outer_key].append(
                        {
                            "inner_primitive": inner_primitive,
                            "path": rel,
                            "anchor": match.group(0),
                            "summary": f"pyboost composite {outer_key} calls {inner_function}",
                        }
                    )
        return {key: uniq(values) for key, values in result.items()}

    @staticmethod
    def _define_body_spans(text: str) -> list[tuple[int, int]]:
        """Return (start, end) spans of all #define bodies (including continuations)."""
        spans: list[tuple[int, int]] = []
        for m in re.finditer(
            r"#define\s+\w+(?:\([^)]*\))?(?:[^\\\n]|\\.)*(?:\\\n(?:[^\\\n]|\\.)*)*",
            text,
        ):
            spans.append((m.start(), m.end()))
        return spans

    @staticmethod
    def _find_wrapper_macros(text: str, mod_class: str) -> list[tuple[str, int]]:
        """Find #define macros that wrap MS_KERNEL_FACTORY_REG* for mod_class.

        Returns (macro_name, param_index_of_kernel_name) pairs.
        """
        results: list[tuple[str, int]] = []
        define_pat = re.compile(
            r"#define\s+([A-Za-z_]\w*)\(([^)]*)\)(?:[^\\\n]|\\.)*(?:\\\n(?:[^\\\n]|\\.)*)*",
        )
        for dm in define_pat.finditer(text):
            macro_name = dm.group(1)
            if macro_name.startswith("MS_KERNEL_FACTORY_REG"):
                continue
            param_str = dm.group(2).strip()
            if not param_str:
                continue
            params = [p.strip() for p in param_str.split(",")]
            body = dm.group(0)
            reg_match = re.search(
                rf"MS_KERNEL_FACTORY_REG(?:_BY_CREATOR)?\(\s*{mod_class},\s*([A-Za-z_]\w*)",
                body,
            )
            if not reg_match:
                continue
            name_ref = reg_match.group(1)
            if name_ref in params:
                results.append((macro_name, params.index(name_ref)))
        return results

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

    def _load_host_view_kernel_map(self) -> dict[str, list[EvidenceItem]]:
        result: dict[str, list[EvidenceItem]] = defaultdict(list)
        root = self.repo_root / HOST_VIEW_KERNEL_ROOT
        if not root.exists():
            return {}
        pattern = re.compile(r"MS_HOST_REG_KERNEL\(\s*([A-Za-z0-9_]+)\s*,\s*([A-Za-z0-9_]+)\s*\)")
        for path in sorted(root.glob("*.cc")):
            text = path.read_text(encoding="utf-8", errors="ignore")
            for match in pattern.finditer(text):
                primitive = match.group(1)
                result[primitive].append(
                    EvidenceItem(
                        self.relpath(path),
                        "direct",
                        match.group(0),
                        f"host view kernel register entry for {primitive}",
                    )
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
                    local_name = alias.asname or alias.name.split(".")[-1]
                    imported_modules[local_name] = alias.name
                    anchor = f"import {alias.name}" + (f" as {alias.asname}" if alias.asname else "")
                    info.imports[local_name] = ImportBinding(local_name, alias.name, "", anchor)
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
                    elif isinstance(target, ast.Name) and isinstance(node.value, (ast.Call, ast.Name, ast.Attribute)):
                        if isinstance(node.value, ast.Call):
                            target_symbol = extract_call_symbol(node.value, info) or ""
                        else:
                            target_symbol = extract_callable_symbol(node.value, info) or ""
                        if target_symbol:
                            info.locals[target.id] = LocalBinding(
                                target.id,
                                "assigned_call",
                                node,
                                f"{target.id} = alias",
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
    documented = load_documented_mint_api_names(index)
    if documented:
        return gather_documented_exports(index, documented)
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


def load_documented_mint_api_names(index: SourceIndex) -> list[str]:
    path = index.repo_root / MINT_DOC_INDEX
    if not path.exists():
        return []
    pattern = re.compile(r"^mindspore\.mint(?:\.[A-Za-z_][A-Za-z0-9_]*)+$")
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if not pattern.match(value):
            continue
        if value == "mindspore.mint":
            continue
        items.append(value)
    return sorted(dict.fromkeys(items))


def gather_documented_exports(index: SourceIndex, api_names: list[str]) -> list[ResolvedExport]:
    result = []
    seen = set()
    for public_path in api_names:
        module_name, _, export_name = public_path.rpartition(".")
        if not module_name or not export_name:
            continue
        module = index.load_module(module_name)
        if module is None:
            continue
        export = resolve_export(index, module, export_name)
        if export is None:
            continue
        if export.local_node is None:
            continue
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
    result = resolve_primitive_source_from_terminal(index, symbol_name, origin_kind="terminal_call")
    if not result:
        return []
    return [result["primitive"]]


def is_pure_python_prelude_symbol(symbol: str) -> bool:
    return symbol in PURE_PYTHON_PRELUDE_SYMBOLS or symbol.startswith(PURE_PYTHON_PRELUDE_PREFIXES)


def is_frontend_guard_symbol(symbol: str) -> bool:
    if symbol in PURE_PYTHON_PRELUDE_SYMBOLS:
        return True
    if symbol.startswith(FRONTEND_GUARD_PREFIXES):
        return True
    if symbol.startswith("mindspore.") and any(part in symbol for part in FRONTEND_GUARD_SYMBOL_PARTS):
        return True
    return False


def is_constructor_setup_symbol(symbol: str) -> bool:
    if symbol in CONSTRUCTOR_SETUP_SYMBOLS:
        return True
    if symbol.startswith(CONSTRUCTOR_SETUP_PREFIXES):
        return True
    leaf = symbol.split(".")[-1]
    return leaf in {"Normal", "Uniform", "HeUniform", "Constant"}


def is_non_backend_unresolved_symbol(item: dict[str, str], *, api_kind: str, has_support_targets: bool) -> bool:
    if not has_support_targets:
        return False
    symbol = str(item.get("symbol", "")).strip()
    origin_kind = str(item.get("origin_kind", "")).strip()
    if is_frontend_guard_symbol(symbol):
        return True
    if api_kind == "class" and is_constructor_setup_symbol(symbol):
        return True
    return False


def infer_possible_primitives_from_symbol(
    index: SourceIndex, symbol: str, depth: int = 0, visited: Optional[set[str]] = None
) -> list[str]:
    return []


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


def extract_callable_symbol(
    expr: ast.AST, module: ModuleInfo, locals_map: Optional[dict[str, LocalBinding]] = None
) -> Optional[str]:
    locals_map = module.locals if locals_map is None else locals_map
    if isinstance(expr, ast.Name):
        binding = module.imports.get(expr.id)
        if binding is not None:
            return binding.impl_symbol
        if expr.id in locals_map:
            local_binding = locals_map[expr.id]
            if local_binding.target_symbol:
                return local_binding.target_symbol
            return f"{module.module_name}.{expr.id}"
        if module.module_name == "mindspore.ops.auto_generate.gen_ops_def":
            return expr.id
        return None
    if isinstance(expr, ast.Attribute):
        chain = attribute_chain(expr)
        if not chain:
            return None
        base_name = chain[0]
        suffix = ".".join(chain[1:])
        binding = module.imports.get(base_name)
        if binding is not None:
            return binding.impl_symbol if not suffix else f"{binding.impl_symbol}.{suffix}"
        if base_name == "self":
            return base_name if not suffix else f"{base_name}.{suffix}"
        if base_name in locals_map:
            local_binding = locals_map[base_name]
            local_symbol = local_binding.target_symbol or f"{module.module_name}.{base_name}"
            return local_symbol if not suffix else f"{local_symbol}.{suffix}"
    return None


def extract_cached_primitive_symbol(
    expr: ast.AST, module: ModuleInfo, locals_map: Optional[dict[str, LocalBinding]] = None
) -> Optional[str]:
    if not isinstance(expr, ast.Call):
        return None
    callable_symbol = extract_callable_symbol(expr.func, module, locals_map)
    if callable_symbol and callable_symbol.endswith("._get_cache_prim") and expr.args:
        primitive_symbol = extract_callable_symbol(expr.args[0], module, locals_map)
        if primitive_symbol:
            return primitive_symbol
    return extract_cached_primitive_symbol(expr.func, module, locals_map)


def extract_call_symbol(node: ast.Call, module: ModuleInfo, locals_map: Optional[dict[str, LocalBinding]] = None) -> Optional[str]:
    cached_primitive = extract_cached_primitive_symbol(node, module, locals_map)
    if cached_primitive:
        return cached_primitive
    return extract_callable_symbol(node.func, module, locals_map)


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


def scan_calls_in_expr(expr: ast.AST, module: ModuleInfo) -> list[str]:
    calls: list[str] = []

    class Visitor(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call) -> None:
            symbol = extract_call_symbol(node, module)
            if symbol:
                calls.append(symbol)
            self.generic_visit(node)

    Visitor().visit(expr)
    return uniq(calls)


def extract_return_calls_with_metadata(method_node: ast.FunctionDef, module: ModuleInfo) -> tuple[list[str], list[str], list[str]]:
    calls = []
    inner_calls = []
    branching_notes = []
    all_assignments: dict[str, list[str]] = defaultdict(list)
    nested_locals = {
        stmt.name: LocalBinding(stmt.name, "function", stmt, stmt.name)
        for stmt in method_node.body
        if isinstance(stmt, ast.FunctionDef)
    }

    def add_assignment(name: str, symbols: list[str]) -> None:
        all_assignments[name].extend(symbols)

    def assign_name_targets(targets: list[ast.AST], symbols: list[str], scope: dict[str, list[str]]) -> None:
        for target in targets:
            if isinstance(target, ast.Name):
                scope[target.id] = list(symbols)
                add_assignment(target.id, list(symbols))

    def resolve_callable_map(
        node: ast.AST,
        scope: dict[str, list[str]],
        callable_maps: dict[str, dict[str, list[str]]],
    ) -> dict[str, list[str]]:
        if not isinstance(node, ast.Dict):
            return {}
        result: dict[str, list[str]] = {}
        for key_node, value_node in zip(node.keys, node.values):
            if not isinstance(key_node, ast.Constant) or not isinstance(key_node.value, str):
                continue
            symbols = resolve_callable_expr(value_node, scope, callable_maps)
            if symbols:
                result[str(key_node.value)] = symbols
        return result

    def resolve_callable_expr(
        expr: ast.AST,
        scope: dict[str, list[str]],
        callable_maps: dict[str, dict[str, list[str]]],
    ) -> list[str]:
        if isinstance(expr, ast.Call):
            cached_primitive = extract_cached_primitive_symbol(expr, module, {**module.locals, **nested_locals})
            if cached_primitive:
                return [cached_primitive]
        symbol = extract_callable_symbol(expr, module, {**module.locals, **nested_locals})
        if symbol:
            return [symbol]
        if isinstance(expr, ast.Name):
            return list(scope.get(expr.id, []))
        if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Attribute) and expr.func.attr == "get":
            owner = expr.func.value
            if isinstance(owner, ast.Name) and owner.id in callable_maps:
                table = callable_maps[owner.id]
                if expr.args and isinstance(expr.args[0], ast.Constant) and isinstance(expr.args[0].value, str):
                    return list(table.get(str(expr.args[0].value), []))
                values = uniq([item for symbols in table.values() for item in symbols])
                if values:
                    branching_notes.append(
                        f"{owner.id}.get(...) may resolve to " + ", ".join(values)
                    )
                return values
        if isinstance(expr, ast.Subscript) and isinstance(expr.value, ast.Name) and expr.value.id in callable_maps:
            table = callable_maps[expr.value.id]
            key_node = expr.slice
            if isinstance(key_node, ast.Constant) and isinstance(key_node.value, str):
                return list(table.get(str(key_node.value), []))
            values = uniq([item for symbols in table.values() for item in symbols])
            if values:
                branching_notes.append(
                    f"{expr.value.id}[...] may resolve to " + ", ".join(values)
                )
            return values
        return []

    def resolve_return_value(
        value: ast.AST,
        scope: dict[str, list[str]],
        callable_maps: dict[str, dict[str, list[str]]],
    ) -> list[str]:
        if isinstance(value, ast.Call):
            return resolve_callable_expr(value.func, scope, callable_maps)
        if isinstance(value, ast.Name):
            symbols = []
            symbols.extend(scope.get(value.id, []))
            symbols.extend(all_assignments.get(value.id, []))
            return uniq(symbols)
        if isinstance(value, ast.Tuple):
            symbols = []
            for element in value.elts:
                symbols.extend(resolve_return_value(element, scope, callable_maps))
            return uniq(symbols)
        return []

    def walk(
        stmts: list[ast.stmt],
        scope: dict[str, list[str]],
        callable_maps: dict[str, dict[str, list[str]]],
    ) -> None:
        for stmt in stmts:
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
                target = stmt.targets[0]
                if isinstance(stmt.value, ast.Call):
                    inner_calls.extend(scan_calls_in_expr(stmt.value, module))
                    symbols = resolve_callable_expr(stmt.value.func, scope, callable_maps)
                    if symbols:
                        if isinstance(target, ast.Name):
                            scope[target.id] = list(symbols)
                            add_assignment(target.id, list(symbols))
                        elif isinstance(target, ast.Tuple):
                            assign_name_targets(list(target.elts), list(symbols), scope)
                elif isinstance(target, ast.Name) and isinstance(stmt.value, ast.Name) and stmt.value.id in scope:
                    scope[target.id] = list(scope[stmt.value.id])
                    add_assignment(target.id, list(scope[target.id]))
                elif isinstance(target, ast.Tuple) and isinstance(stmt.value, ast.Name) and stmt.value.id in scope:
                    assign_name_targets(list(target.elts), list(scope[stmt.value.id]), scope)
                elif isinstance(target, ast.Name):
                    callable_map = resolve_callable_map(stmt.value, scope, callable_maps)
                    if callable_map:
                        callable_maps[target.id] = callable_map
            elif isinstance(stmt, ast.If):
                walk(stmt.body, copy.deepcopy(scope), copy.deepcopy(callable_maps))
                walk(stmt.orelse, copy.deepcopy(scope), copy.deepcopy(callable_maps))
            elif isinstance(stmt, ast.For):
                walk(stmt.body, copy.deepcopy(scope), copy.deepcopy(callable_maps))
                walk(stmt.orelse, copy.deepcopy(scope), copy.deepcopy(callable_maps))
            elif isinstance(stmt, ast.Return):
                inner_calls.extend(scan_calls_in_expr(stmt.value, module))
                calls.extend(resolve_return_value(stmt.value, scope, callable_maps))

    walk(method_node.body, {}, {})
    return uniq(calls), uniq(inner_calls), uniq(branching_notes)


def extract_return_calls(method_node: ast.FunctionDef, module: ModuleInfo) -> list[str]:
    calls, _, _ = extract_return_calls_with_metadata(method_node, module)
    return calls


def find_nested_function(function_node: ast.FunctionDef, function_name: str) -> Optional[ast.FunctionDef]:
    for node in ast.walk(function_node):
        if isinstance(node, ast.FunctionDef) and node is not function_node and node.name == function_name:
            return node
    return None


def analyze_class_construct(index: SourceIndex, export: ResolvedExport, helper_depth: int = 2) -> Optional[ClassExecution]:
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
                    nested_calls, nested_chain, nested_notes = expand_bound_construct_target(index, symbol, depth=helper_depth)
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
    raw = symbol_name.split(".")[-1]
    if raw in index.generated_primitive_names:
        return raw
    primitive = "".join(part.capitalize() for part in str(raw).split("_"))
    if primitive in index.generated_primitive_names:
        return primitive
    return None


def primitive_from_prim_base_name(base_name: str) -> Optional[str]:
    if not base_name.endswith("Prim_") or len(base_name) <= len("Prim_"):
        return None
    primitive = base_name[: -len("Prim_")]
    return primitive or None


def infer_pyboost_impl_primitive(index: SourceIndex, symbol_name: str) -> Optional[str]:
    raw = symbol_name.split(".")[-1]
    if not raw.endswith("_impl"):
        return None
    module = index.load_module("mindspore.ops.auto_generate.pyboost_inner_prim")
    if module is None:
        return None
    binding = module.locals.get(raw)
    if binding is None or binding.kind != "assigned_call" or not binding.target_symbol:
        return None
    target_parts = binding.target_symbol.split(".")
    if len(target_parts) < 2:
        return None
    resolution = resolve_symbol(index, ".".join(target_parts[:-1]), target_parts[-1])
    if not isinstance(resolution.local_node, ast.ClassDef) or resolution.local_module is None:
        return None
    primitive_candidates: list[str] = []
    for base in resolution.local_node.bases:
        base_symbol = extract_callable_symbol(base, resolution.local_module)
        if base_symbol:
            base_name = base_symbol.split(".")[-1]
        elif isinstance(base, ast.Name):
            base_name = base.id
        else:
            continue
        primitive = primitive_from_prim_base_name(base_name)
        if primitive:
            primitive_candidates.append(primitive)
    primitive_candidates = uniq(primitive_candidates)
    if len(primitive_candidates) == 1:
        return primitive_candidates[0]
    return None


def infer_pyboost_class_primitive(index: SourceIndex, symbol_name: str) -> Optional[str]:
    raw = symbol_name.split(".")[-1]
    if not raw.startswith("_Pyboost") or not raw.endswith("Prim"):
        return None
    parts = symbol_name.split(".")
    if len(parts) < 2:
        return None
    resolution = resolve_symbol(index, ".".join(parts[:-1]), parts[-1])
    if not isinstance(resolution.local_node, ast.ClassDef) or resolution.local_module is None:
        return None
    primitive_candidates: list[str] = []
    for base in resolution.local_node.bases:
        if isinstance(base, ast.Name):
            base_name = base.id
        else:
            base_symbol = extract_callable_symbol(base, resolution.local_module)
            base_name = base_symbol.split(".")[-1] if base_symbol else ""
        primitive = primitive_from_prim_base_name(base_name)
        if primitive:
            primitive_candidates.append(primitive)
    primitive_candidates = uniq(primitive_candidates)
    if len(primitive_candidates) == 1:
        return primitive_candidates[0]
    return None


def is_python_composite_function(index: SourceIndex, symbol: str) -> bool:
    normalized, resolution = resolve_qualified_symbol(index, symbol)
    if resolution is None or not isinstance(resolution.local_node, ast.FunctionDef) or resolution.local_module is None:
        return False
    if normalized.startswith("mindspore._c_expression.") or normalized.endswith("_instance"):
        return False
    if resolution.impl_module == FUNCTIONAL_OVERLOAD_MODULE:
        return False
    primitive, _ = resolve_primitive_source_from_terminal(index, normalized, origin_kind="effective_call")
    return primitive is None


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
}


def filter_possible_primitives(primitives: list[str]) -> list[str]:
    if len(primitives) <= 2:
        return uniq(primitives)
    filtered = [item for item in uniq(primitives) if item not in HELPER_PRIMITIVE_NOISE]
    return filtered or uniq(primitives)


def exact_op_yamls_from_symbol(index: SourceIndex, symbol_name: str) -> list[str]:
    raw = symbol_name.split(".")[-1]
    candidates = []
    candidates.extend(index.function_name_to_op_yaml.get(raw, []))
    candidates.extend(index.class_name_to_op_yaml.get(raw, []))
    candidates.extend(index.op_name_to_op_yaml.get(raw, []))
    candidates.extend(SYMBOL_OP_DEF_MAP.get(raw, []))
    if raw in index.op_defs:
        candidates.append(raw)
    if f"{raw}.yaml" in index.op_defs:
        candidates.append(f"{raw}.yaml")
    return uniq(candidates)


def resolve_primitive_source_from_terminal(
    index: SourceIndex,
    symbol_name: str,
    *,
    origin_kind: str,
) -> tuple[Optional[dict[str, str]], str]:
    op_yamls = exact_op_yamls_from_symbol(index, symbol_name)
    if len(op_yamls) == 1:
        primitive, op_path = primitive_from_op(index, op_yamls[0])
        if primitive:
            return {
                "api_def": "",
                "op_yaml": op_yamls[0],
                "op_def_path": op_path or "",
                "primitive": primitive,
                "origin_kind": origin_kind,
            }, ""
    if len(op_yamls) > 1:
        return None, "ambiguous_terminal_mapping"
    primitive = infer_generated_primitive(index, symbol_name)
    if not primitive:
        primitive = infer_pyboost_impl_primitive(index, symbol_name)
    if not primitive:
        primitive = infer_pyboost_class_primitive(index, symbol_name)
    if not primitive:
        raw = symbol_name.split(".")[-1]
        direct_source = primitive_source_for_name(index, raw, origin_kind=origin_kind)
        if direct_source.get("op_yaml"):
            primitive = raw
    if primitive:
        direct_source = primitive_source_for_name(index, primitive, origin_kind=origin_kind)
        if direct_source.get("op_yaml"):
            return direct_source, ""
        return {
            "api_def": "",
            "op_yaml": "",
            "op_def_path": "",
            "primitive": primitive,
            "origin_kind": origin_kind,
        }, ""
    return None, "terminal_symbol_unresolved"


def primitive_kernel_candidates(primitive: str, allow_suffix_stripping: bool = True) -> list[str]:
    candidates = [primitive]
    candidates.extend(PRIMITIVE_KERNEL_NAME_MAP.get(primitive, []))
    if allow_suffix_stripping and primitive.endswith("Ext"):
        candidates.append(primitive[:-3])
    if allow_suffix_stripping and primitive.endswith("View"):
        candidates.append(primitive[:-4])
    if allow_suffix_stripping and primitive.endswith("ExtView"):
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
    results = uniq(results)
    scoped_factory = [
        item
        for item in results
        if item.anchor.startswith(f"MS_ACLNN_KERNEL_FACTORY_REG({primitive},")
        or item.anchor.startswith(f"MS_ACLNN_COMMON_KERNEL_FACTORY_REG({primitive},")
    ]
    return uniq(scoped_factory) if scoped_factory else results


def match_kernel_registry(
    registry: dict[str, list[EvidenceItem]], primitive: str, allow_suffix_stripping: bool = False
) -> list[EvidenceItem]:
    results = []
    for candidate in primitive_kernel_candidates(primitive, allow_suffix_stripping=allow_suffix_stripping):
        results.extend(registry.get(candidate, []))
    return uniq(results)


def match_host_view_kernel(
    index: SourceIndex,
    primitive: str,
    primitive_sources: Optional[list[dict[str, str]]] = None,
) -> list[EvidenceItem]:
    if primitive not in index.view_op_names:
        return []
    sources = primitive_sources or []
    if sources and not any(op_yaml_has_graph_view(index, str(item.get("op_yaml", "")).strip()) for item in sources):
        return []
    return uniq(index.host_view_kernel_map.get(primitive, []))


def match_rt_nop_view_closure(
    index: SourceIndex,
    primitive: str,
    primitive_sources: Optional[list[dict[str, str]]] = None,
) -> list[EvidenceItem]:
    if primitive not in index.view_op_names:
        return []
    sources = primitive_sources or []
    if sources:
        op_yamls = [str(item.get("op_yaml", "")).strip() for item in sources if str(item.get("op_yaml", "")).strip()]
        if not op_yamls:
            return []
        for op_yaml in op_yamls:
            op_info = index.op_defs.get(op_yaml)
            payload = op_info.get("payload", {}) if isinstance(op_info, dict) else {}
            if not isinstance(payload, dict) or payload.get("view") is not True:
                return []
    return uniq(index.rt_nop_view_map.get(primitive, []))


def match_fallback_to_rt_nop_reshape(
    index: SourceIndex,
    primitive: str,
) -> list[EvidenceItem]:
    fallback = uniq(index.fallback_reshape_map.get(primitive, []))
    reshape_rt_nop = uniq(index.rt_nop_view_map.get("Reshape", []))
    if not fallback or not reshape_rt_nop:
        return []
    return uniq(fallback + reshape_rt_nop)


def analyze_pynative_exact_terminal_backend(
    index: SourceIndex,
    primitive: str,
    backend: str,
) -> tuple[Optional[str], list[dict[str, str]]]:
    if backend == "cpu":
        pyboost_op_map = index.cpu_pyboost_op_map
        pyboost_custom_kernel_map = index.cpu_pyboost_custom_kernel_map
        pyboost_impl_map = index.cpu_pyboost_impl_map
        kernel_registry_map = index.cpu_kernel_map
    else:
        pyboost_op_map = index.gpu_pyboost_op_map
        pyboost_custom_kernel_map = index.gpu_pyboost_custom_kernel_map
        pyboost_impl_map = index.gpu_pyboost_impl_map
        kernel_registry_map = index.gpu_kernel_map

    runner_registration = pyboost_op_map.get(primitive, [])
    runner_impl_paths = pyboost_impl_map.get(primitive, [])
    if not runner_registration and not runner_impl_paths:
        return None, []

    evidence: list[dict[str, str]] = []
    for item in runner_impl_paths:
        evidence.append(
            EvidenceItem(
                item.path,
                "direct",
                item.anchor,
                f"{backend.upper()} pyboost runner for {primitive}",
            ).to_dict()
        )
    evidence.extend(item.to_dict() for item in runner_registration)

    exact_custom_kernel = pyboost_custom_kernel_map.get(primitive, [])
    exact_kernel_registry = kernel_registry_map.get(primitive, [])
    if exact_custom_kernel or exact_kernel_registry:
        evidence.extend(item.to_dict() for item in exact_custom_kernel)
        evidence.extend(item.to_dict() for item in exact_kernel_registry)
        return "yes", uniq(evidence)

    evidence.append(
        EvidenceItem(
            runner_impl_paths[0].path if runner_impl_paths else runner_registration[0].path,
            "derived",
            primitive,
            f"{backend.upper()} exact terminal kernel closure missing for {primitive}",
        ).to_dict()
    )
    return "no", uniq(evidence)


def analyze_random_composite_inner_backend(
    index: SourceIndex,
    primitive: str,
    backend: str,
    primitive_sources: list[dict[str, str]],
) -> tuple[Optional[str], list[dict[str, str]]]:
    op_yaml_candidates = op_yaml_candidates_for_primitive(index, primitive, primitive_sources)
    composite_hits: list[dict[str, str]] = []
    for op_yaml in uniq(op_yaml_candidates):
        op_info = index.op_defs.get(op_yaml) or {}
        op_def = op_info.get("payload", {}) if isinstance(op_info, dict) else {}
        composite_value = op_def.get("composite", False)
        if composite_value is not True and str(composite_value).lower() != "true":
            continue
        op_key = Path(op_yaml).stem.removesuffix("_op").lower()
        composite_hits.extend(index.pyboost_composite_inner_map.get(op_key, []))
    composite_hits = uniq(composite_hits)
    if not composite_hits:
        return None, []

    inner_primitives = uniq([item.get("inner_primitive", "") for item in composite_hits if item.get("inner_primitive")])
    if not inner_primitives:
        return None, []

    evidence: list[dict[str, str]] = []
    states: list[str] = []
    for hit in composite_hits:
        evidence.append(
            EvidenceItem(
                hit["path"],
                "derived",
                hit["anchor"],
                f"random composite {primitive} reaches {hit['inner_primitive']}",
            ).to_dict()
        )
    for inner_primitive in inner_primitives:
        inner_state, inner_evidence = analyze_pynative_exact_terminal_backend(index, inner_primitive, backend)
        if inner_state is None:
            return None, []
        states.append(inner_state)
        evidence.extend(inner_evidence)
    if all(state == "yes" for state in states):
        return "yes", uniq(evidence)
    if any(state == "unknown" for state in states):
        return "unknown", uniq(evidence)
    return "no", uniq(evidence)


def op_yaml_candidates_for_primitive(
    index: SourceIndex,
    primitive: str,
    primitive_sources: Optional[list[dict[str, str]]] = None,
) -> list[str]:
    candidates: list[str] = []
    for source in primitive_sources or []:
        op_yaml = str(source.get("op_yaml", "")).strip()
        if op_yaml:
            candidates.append(op_yaml)
    candidates.extend(index.op_name_to_op_yaml.get(primitive, []))
    primitive_snake = camel_to_snake(primitive)
    primitive_compact = primitive_snake.replace("_", "")
    if not candidates:
        for op_yaml, op_info in index.op_defs.items():
            op_name = str(op_info.get("op_name", ""))
            op_compact = op_name.replace("_", "").lower()
            if op_name.lower() == primitive_snake or op_compact == primitive_compact:
                candidates.append(op_yaml)
    return uniq(candidates)


def enrich_support_targets_with_op_defs(index: SourceIndex, targets: list[dict[str, str]]) -> list[dict[str, str]]:
    enriched: list[dict[str, str]] = []
    for item in targets:
        updated = dict(item)
        primitive = str(updated.get("primitive", "")).strip()
        op_yaml = str(updated.get("op_yaml", "")).strip()
        if primitive and not op_yaml:
            candidates = op_yaml_candidates_for_primitive(index, primitive, [updated])
            if len(candidates) == 1:
                candidate = candidates[0]
                op_info = index.op_defs.get(candidate)
                updated["op_yaml"] = candidate
                if op_info is not None:
                    updated["op_def_path"] = index.relpath(op_info["path"])
        enriched.append(updated)
    return uniq(enriched)


def op_yaml_has_enabled_dispatch(index: SourceIndex, op_yaml: str) -> bool:
    op_info = index.op_defs.get(op_yaml)
    payload = op_info.get("payload", {}) if isinstance(op_info, dict) else {}
    dispatch = payload.get("dispatch") if isinstance(payload, dict) else None
    return isinstance(dispatch, dict) and bool(dispatch.get("enable"))


def op_yaml_has_graph_view(index: SourceIndex, op_yaml: str) -> bool:
    op_info = index.op_defs.get(op_yaml)
    payload = op_info.get("payload", {}) if isinstance(op_info, dict) else {}
    return isinstance(payload, dict) and payload.get("graph_view") is True


def primitive_sources_from_symbols(
    index: SourceIndex,
    symbols: list[str],
    *,
    origin_kind: str = "effective_call",
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    results = []
    seen = set()
    unresolved = []
    for symbol in uniq(symbols):
        item, reason = resolve_primitive_source_from_terminal(index, symbol, origin_kind=origin_kind)
        if item is None:
            unresolved.append({"symbol": symbol, "origin_kind": origin_kind, "reason": reason})
            continue
        key = (item.get("primitive", ""), item.get("op_yaml", ""))
        if key in seen:
            continue
        seen.add(key)
        results.append(item)
    return uniq(results), unresolved


def primitive_source_for_name(index: SourceIndex, primitive: str, *, origin_kind: str) -> dict[str, str]:
    op_yamls = op_yaml_candidates_for_primitive(index, primitive)
    if len(op_yamls) == 1:
        resolved_primitive, op_path = primitive_from_op(index, op_yamls[0])
        if resolved_primitive == primitive:
            return {
                "api_def": "",
                "op_yaml": op_yamls[0],
                "op_def_path": op_path or "",
                "primitive": primitive,
                "origin_kind": origin_kind,
            }
    return {"api_def": "", "op_yaml": "", "op_def_path": "", "primitive": primitive, "origin_kind": origin_kind}


def fixed_functional_overload_targets(
    index: SourceIndex,
    api_name: str,
    *,
    origin_kind: str,
) -> list[dict[str, str]]:
    primitives = uniq(index.functional_overload_graph_map.get(api_name, []))
    return [primitive_source_for_name(index, primitive, origin_kind=origin_kind) for primitive in primitives]


def is_python_identity_pass_through(export: ResolvedExport, class_execution: Optional[ClassExecution]) -> bool:
    if export.public_path != "mindspore.mint.nn.Identity":
        return False
    if export.api_kind != "class" or not isinstance(export.local_node, ast.ClassDef) or export.local_module is None:
        return False
    construct_info = find_method_in_hierarchy(export.local_node, export.local_module, "construct")
    if construct_info is None:
        return False
    construct, _, _ = construct_info
    if len(construct.args.args) < 2:
        return False
    input_name = construct.args.args[1].arg
    returns = [node for node in ast.walk(construct) if isinstance(node, ast.Return)]
    return len(returns) == 1 and isinstance(returns[0].value, ast.Name) and returns[0].value.id == input_name


def apply_special_terminal_resolution(
    index: SourceIndex,
    export: ResolvedExport,
    support_primitive_sources: list[dict[str, str]],
    unresolved_support_symbols: list[dict[str, str]],
    support_chain_facts: dict[str, Any],
) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, Any]]:
    if export.public_path == "mindspore.mint.float_power":
        primitives = ["Cast", "Pow", "PowTensorScalar", "PowScalarTensor"]
        support_primitive_sources = [
            primitive_source_for_name(index, primitive, origin_kind="scenario_branch")
            for primitive in primitives
        ]
        unresolved_support_symbols = []
        support_chain_facts["python_composite_used"] = True
        support_chain_facts["scenario_dependent"] = True
        support_chain_facts["possible_primitives"] = primitives
        support_chain_facts["execution_chain"].extend(
            [
                f"{export.impl_module}.{export.impl_name}",
                "branch: Tensor/Tensor -> Cast + Pow",
                "branch: Tensor/Scalar -> Cast + PowTensorScalar",
                "branch: Scalar/Tensor -> Cast + PowScalarTensor",
            ]
        )
        support_chain_facts["branching_notes"].append(
            "float_power selects Pow/PowTensorScalar/PowScalarTensor by Tensor/Number input kinds"
        )
    elif export.public_path == "mindspore.mint.nn.functional.conv2d":
        targets = fixed_functional_overload_targets(index, "conv2d", origin_kind="functional_overload_bridge")
        if targets:
            support_primitive_sources = targets
            unresolved_support_symbols = []
            support_chain_facts["python_composite_used"] = True
            support_chain_facts["scenario_dependent"] = True
            support_chain_facts["possible_primitives"] = [item["primitive"] for item in targets if item.get("primitive")]
            support_chain_facts["execution_chain"].extend(
                [
                    f"{export.impl_module}.{export.impl_name}",
                    f"{FUNCTIONAL_OVERLOAD_MODULE}.conv2d",
                    f"{FUNCTIONAL_MAP_PATH}:conv2d->{','.join(support_chain_facts['possible_primitives'])}",
                ]
            )
            support_chain_facts["branching_notes"].append(
                "conv2d functional_overload dispatches to Conv2DExt or Conv2DPadding"
            )
    elif export.public_path == "mindspore.mint.searchsorted" and support_primitive_sources:
        unresolved_support_symbols = [
            item
            for item in unresolved_support_symbols
            if not (
                item.get("origin_kind") == "prelude_call"
                and str(item.get("symbol", "")).startswith("mindspore._checkparam.")
            )
        ]
        if support_chain_facts.get("execution_chain") == ["mindspore._checkparam.check_value_type"]:
            support_chain_facts["execution_chain"] = []
        if not unresolved_support_symbols:
            support_chain_facts["python_composite_used"] = False
    support_chain_facts["possible_primitives"] = uniq(support_chain_facts.get("possible_primitives", []))
    support_chain_facts["execution_chain"] = uniq(support_chain_facts.get("execution_chain", []))
    support_chain_facts["branching_notes"] = uniq(support_chain_facts.get("branching_notes", []))
    return support_primitive_sources, unresolved_support_symbols, support_chain_facts


def drop_prelude_helper_primitives(
    *,
    export: ResolvedExport,
    support_primitive_sources: list[dict[str, str]],
    support_chain_facts: dict[str, Any],
) -> list[dict[str, str]]:
    support_unique_primitives = uniq(
        [str(item.get("primitive", "")) for item in support_primitive_sources if item.get("primitive")]
    )
    if (
        len(support_unique_primitives) == 1
        or support_chain_facts.get("scenario_dependent")
    ):
        return support_primitive_sources

    if "Cast" in support_unique_primitives:
        non_cast_primitives = [primitive for primitive in support_unique_primitives if primitive != "Cast"]
        if len(non_cast_primitives) == 1:
            return [
                item for item in support_primitive_sources if str(item.get("primitive", "")) == non_cast_primitives[0]
            ]

    if export.public_path in {
        "mindspore.mint.nn.functional.adaptive_avg_pool2d",
        "mindspore.mint.nn.AdaptiveAvgPool2d",
    } and set(support_unique_primitives) == {"AdaptiveAvgPool2DExt", "Shape"}:
        return [
            item
            for item in support_primitive_sources
            if str(item.get("primitive", "")) == "AdaptiveAvgPool2DExt"
        ]

    return support_primitive_sources


def _resolve_python_composite_from_function(
    index: SourceIndex,
    symbol: str,
    function_node: ast.FunctionDef,
    module: ModuleInfo,
    *,
    origin_kind: str,
    depth: int,
    visited: set[str],
) -> dict[str, Any]:
    chain = [symbol]
    if symbol in visited:
        return {
            "resolved_targets": [],
            "possible_primitives": [],
            "execution_chain": chain,
            "branching_notes": [],
            "unresolved_symbols": [{"symbol": symbol, "origin_kind": origin_kind, "reason": "unresolved_composite_chain"}],
            "scenario_dependent": False,
        }
    visited.add(symbol)
    if depth <= 0:
        return {
            "resolved_targets": [],
            "possible_primitives": [],
            "execution_chain": chain,
            "branching_notes": [],
            "unresolved_symbols": [{"symbol": symbol, "origin_kind": origin_kind, "reason": "terminal_symbol_unresolved"}],
            "scenario_dependent": False,
        }
    return_calls, inner_calls, local_notes = extract_return_calls_with_metadata(function_node, module)
    if not return_calls and not inner_calls:
        return {
            "resolved_targets": [],
            "possible_primitives": [],
            "execution_chain": chain,
            "branching_notes": local_notes,
            "unresolved_symbols": [{"symbol": symbol, "origin_kind": origin_kind, "reason": "terminal_symbol_unresolved"}],
            "scenario_dependent": False,
        }

    resolved_targets: list[dict[str, str]] = []
    possible_primitives: list[str] = []
    branching_notes = list(local_notes)
    unresolved_symbols: list[dict[str, str]] = []
    execution_chain = list(chain)
    scenario_dependent = len(uniq(return_calls)) > 1 or bool(local_notes)

    for call in uniq(return_calls + inner_calls):
        normalized_call = resolve_call_alias(index, call)
        execution_chain.append(normalized_call)
        item, reason = resolve_primitive_source_from_terminal(index, normalized_call, origin_kind=origin_kind)
        if item is not None:
            resolved_targets.append(item)
            possible_primitives.append(item["primitive"])
            continue
        if normalized_call.startswith("mindspore._c_expression.") or normalized_call.endswith("_instance"):
            unresolved_symbols.append({"symbol": normalized_call, "origin_kind": origin_kind, "reason": "c_expression_blackbox"})
            continue
        leaf = normalized_call.split(".")[-1]
        nested_function = find_nested_function(function_node, leaf)
        if nested_function is not None and normalized_call.startswith(f"{module.module_name}."):
            nested = _resolve_python_composite_from_function(
                index,
                normalized_call,
                nested_function,
                module,
                origin_kind="effective_call",
                depth=depth - 1,
                visited=visited,
            )
            resolved_targets.extend(nested["resolved_targets"])
            possible_primitives.extend(nested["possible_primitives"])
            execution_chain.extend(nested["execution_chain"])
            branching_notes.extend(nested["branching_notes"])
            unresolved_symbols.extend(nested["unresolved_symbols"])
            scenario_dependent = scenario_dependent or nested["scenario_dependent"]
            continue
        if is_python_composite_function(index, normalized_call):
            nested = resolve_python_composite_targets(
                index,
                normalized_call,
                origin_kind="effective_call",
                depth=depth - 1,
                visited=visited,
            )
            resolved_targets.extend(nested["resolved_targets"])
            possible_primitives.extend(nested["possible_primitives"])
            execution_chain.extend(nested["execution_chain"])
            branching_notes.extend(nested["branching_notes"])
            unresolved_symbols.extend(nested["unresolved_symbols"])
            scenario_dependent = scenario_dependent or nested["scenario_dependent"]
            continue
        unresolved_symbols.append({"symbol": normalized_call, "origin_kind": origin_kind, "reason": reason})

    return {
        "resolved_targets": uniq(resolved_targets),
        "possible_primitives": filter_possible_primitives(uniq(possible_primitives)),
        "execution_chain": uniq(execution_chain),
        "branching_notes": uniq(branching_notes),
        "unresolved_symbols": uniq(unresolved_symbols),
        "scenario_dependent": scenario_dependent,
    }


def resolve_python_composite_targets(
    index: SourceIndex,
    symbol: str,
    *,
    origin_kind: str = "effective_call",
    depth: int = 3,
    visited: Optional[set[str]] = None,
) -> dict[str, Any]:
    visited = set() if visited is None else set(visited)
    normalized, resolution = resolve_qualified_symbol(index, symbol)
    if resolution is None or not isinstance(resolution.local_node, ast.FunctionDef) or resolution.local_module is None:
        return {
            "resolved_targets": [],
            "possible_primitives": [],
            "execution_chain": [normalized],
            "branching_notes": [],
            "unresolved_symbols": [{"symbol": normalized, "origin_kind": origin_kind, "reason": "terminal_symbol_unresolved"}],
            "scenario_dependent": False,
        }
    return _resolve_python_composite_from_function(
        index,
        normalized,
        resolution.local_node,
        resolution.local_module,
        origin_kind=origin_kind,
        depth=depth,
        visited=visited,
    )


def resolve_support_primitive_sources(
    index: SourceIndex,
    terminal_calls: list[str],
    effective_calls: list[str],
    prelude_calls: list[str],
    api_kind: str = "",
) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, Any]]:
    results = []
    seen = set()
    unresolved = []
    facts = {
        "python_composite_used": False,
        "possible_primitives": [],
        "execution_chain": [],
        "branching_notes": [],
        "scenario_dependent": False,
    }
    symbol_groups = [
        ("terminal_call", terminal_calls),
        ("effective_call", effective_calls),
        ("prelude_call", prelude_calls),
    ]
    for origin_kind, symbols in symbol_groups:
        if api_kind == "class":
            symbols = [symbol for symbol in symbols if not is_constructor_setup_symbol(str(symbol))]
        items, unresolved_items = primitive_sources_from_symbols(index, symbols, origin_kind=origin_kind)
        for item in items:
            key = (item.get("primitive", ""), item.get("op_yaml", ""), item.get("op_def_path", ""))
            if key in seen:
                continue
            seen.add(key)
            results.append(item)
        for unresolved_item in unresolved_items:
            composite = None
            if unresolved_item["reason"] == "terminal_symbol_unresolved" and is_python_composite_function(index, unresolved_item["symbol"]):
                composite = resolve_python_composite_targets(
                    index,
                    unresolved_item["symbol"],
                    origin_kind=origin_kind,
                )
            if composite is None:
                unresolved.append(unresolved_item)
                continue
            facts["python_composite_used"] = True
            facts["possible_primitives"].extend(composite["possible_primitives"])
            facts["execution_chain"].extend(composite["execution_chain"])
            facts["branching_notes"].extend(composite["branching_notes"])
            facts["scenario_dependent"] = facts["scenario_dependent"] or composite["scenario_dependent"]
            for item in composite["resolved_targets"]:
                key = (item.get("primitive", ""), item.get("op_yaml", ""), item.get("op_def_path", ""))
                if key in seen:
                    continue
                seen.add(key)
                results.append(item)
            unresolved.extend(composite["unresolved_symbols"])
    facts["possible_primitives"] = filter_possible_primitives(uniq(facts["possible_primitives"]))
    facts["execution_chain"] = uniq(facts["execution_chain"])
    facts["branching_notes"] = uniq(facts["branching_notes"])
    return results, unresolved, facts


def resolve_functional_overload_func_op_bridge(
    index: SourceIndex,
    unresolved_symbols: list[dict[str, str]],
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Close narrow functional_overload C++ instance bridges to terminal func_ops.

    `einsum_ext` reaches `functional_overload.einsum`, whose Python body returns
    the C++ `_einsum_instance`. The real graph terminal is declared in
    `functional_map.cc` as `einsum -> EinsumExt`, and `EinsumExt` is a func_op.
    Keep this rule deliberately narrow to avoid treating ordinary overload
    wrappers as single-terminal func_ops.
    """
    targets: list[dict[str, str]] = []
    bridged_symbols: list[str] = []
    evidence: list[dict[str, str]] = []
    execution_chain: list[str] = []
    for unresolved in unresolved_symbols:
        symbol = str(unresolved.get("symbol", "")).strip()
        if not symbol:
            continue
        normalized = resolve_call_alias(index, symbol)
        if normalized != f"{FUNCTIONAL_OVERLOAD_MODULE}.einsum":
            continue
        graph_primitives = uniq(index.functional_overload_graph_map.get("einsum", []))
        if graph_primitives != ["EinsumExt"]:
            continue
        primitive = graph_primitives[0]
        op_yamls = exact_op_yamls_from_symbol(index, primitive)
        if not op_yamls:
            primitive_stem = camel_to_snake(primitive)
            op_yamls = uniq(
                list(index.function_name_to_op_yaml.get(primitive_stem, []))
                + list(index.op_name_to_op_yaml.get(primitive_stem, []))
            )
        if len(op_yamls) != 1:
            continue
        op_yaml = op_yamls[0]
        op_info = index.op_defs.get(op_yaml)
        if op_info is None:
            continue
        payload = op_info.get("payload", {})
        if not isinstance(payload, dict) or payload.get("bprop_expander") is not False:
            continue
        dispatch = payload.get("dispatch")
        if not isinstance(dispatch, dict) or not dispatch.get("enable") or "Ascend" not in dispatch:
            continue
        primitive_value, op_path = primitive_from_op(index, op_yaml)
        if primitive_value != primitive:
            continue
        targets.append(
            {
                "api_def": "",
                "op_yaml": op_yaml,
                "op_def_path": op_path or "",
                "primitive": primitive,
                "origin_kind": "functional_overload_func_op_bridge",
            }
        )
        bridged_symbols.append(symbol)
        execution_chain.extend([normalized, f"{FUNCTIONAL_MAP_PATH}:einsum->{primitive}"])
        evidence.append(
            EvidenceItem(
                str(FUNCTIONAL_MAP_PATH).replace("\\", "/"),
                "direct",
                '"einsum" -> prim::kPrimEinsumExt',
                "functional_overload bridge resolves einsum to EinsumExt",
            ).to_dict()
        )
        evidence.append(
            EvidenceItem(
                index.relpath(op_info["path"]),
                "direct",
                "dispatch.Ascend",
                "graph func_op dispatch declares Ascend support for EinsumExt",
            ).to_dict()
        )
    return uniq(targets), {
        "symbols": uniq(bridged_symbols),
        "evidence": uniq(evidence),
        "execution_chain": uniq(execution_chain),
    }


def support_targets_from_func_op(index: SourceIndex, func_op_info: dict[str, Any]) -> list[dict[str, str]]:
    expanded_primitives = uniq(func_op_info.get("expanded_primitives", []))
    if not expanded_primitives:
        return []
    exact_targets, _ = primitive_sources_from_symbols(index, expanded_primitives, origin_kind="func_expansion")
    return exact_targets or [
        {"api_def": "", "op_yaml": "", "op_def_path": "", "primitive": primitive, "origin_kind": "func_expansion"}
        for primitive in expanded_primitives
    ]


def is_functional_overload_export(export: ResolvedExport) -> bool:
    return export.impl_module == FUNCTIONAL_OVERLOAD_MODULE or FUNCTIONAL_OVERLOAD_MODULE in export.resolved_symbol_chain


def overload_api_def_entries(
    export: ResolvedExport,
    effective_entries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if export.api_kind == "class":
        return []
    results = []
    for entry in effective_entries:
        interface_forms = {
            part.strip().lower()
            for part in str(entry.get("interface", "")).split(",")
            if part.strip()
        }
        if interface_forms and "function" not in interface_forms:
            continue
        results.append(entry)
    return results


def resolve_functional_overload_targets(
    index: SourceIndex,
    export: ResolvedExport,
    api_def_name: Optional[str],
    effective_entries: list[dict[str, Any]],
) -> tuple[list[dict[str, str]], list[str], dict[str, list[str]]]:
    branch_entries = overload_api_def_entries(export, effective_entries)
    if not branch_entries:
        return [], [], {"graph": [], "pynative": []}
    api_name = api_def_name or export.export_name
    graph_primitives = uniq(index.functional_overload_graph_map.get(api_name, []))
    pynative_primitives = uniq(index.functional_overload_pynative_map.get(api_name, []))
    targets: list[dict[str, str]] = []
    branch_primitives: list[str] = []
    for entry in branch_entries:
        primitive, op_path = primitive_from_op(index, entry.get("op_yaml"))
        if not primitive:
            continue
        targets.append(
            {
                "api_def": entry.get("_api_def", api_def_name or ""),
                "op_yaml": str(entry.get("op_yaml", "")),
                "op_def_path": op_path or "",
                "primitive": primitive,
                "origin_kind": "overload_branch",
                "py_method": str(entry.get("py_method", "")),
                "ascend_value": str(entry.get("Ascend", "")),
                "cpu_value": str(entry.get("CPU", "")),
                "gpu_value": str(entry.get("GPU", "")),
            }
        )
        branch_primitives.append(primitive)
    return uniq(targets), uniq(branch_primitives), {"graph": graph_primitives, "pynative": pynative_primitives}


def aggregate_overload_support_states(states: list[dict[str, str]]) -> dict[str, str]:
    if not states:
        return support_state()
    result = {}
    for backend in ("ascend", "cpu", "gpu"):
        values = [state[backend] for state in states]
        if values and all(value == "yes" for value in values):
            result[backend] = "yes"
        elif values and all(value == "no" for value in values):
            result[backend] = "no"
        else:
            result[backend] = "unknown"
    return result


def resolve_py_method_symbol(index: SourceIndex, py_method_name: str) -> Optional[SymbolResolution]:
    if not py_method_name:
        return None
    modules = index.python_function_modules.get(py_method_name, [])
    if len(modules) != 1:
        return None
    return resolve_symbol(index, modules[0], py_method_name)


def _raise_stub_message(node: ast.Raise) -> str:
    exc = node.exc
    if isinstance(exc, ast.Call):
        if exc.args:
            first = exc.args[0]
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                return first.value.lower()
            try:
                return ast.unparse(first).lower()
            except Exception:
                return ""
        func = exc.func
        if isinstance(func, ast.Name):
            return func.id.lower()
        if isinstance(func, ast.Attribute):
            chain = attribute_chain(func)
            return ".".join(chain).lower()
    elif isinstance(exc, ast.Constant) and isinstance(exc.value, str):
        return exc.value.lower()
    return ""


def _is_raise_stub_message(message: str) -> bool:
    checks = (
        "not supported",
        "only supports",
        "only support",
        "only supported on",
        "should not come here",
        "has not been implemented",
        "not been implemented",
        "this func has not been implemented",
        "no branch to go",
    )
    return any(token in message for token in checks)


def _function_is_raise_stub(function_node: ast.FunctionDef) -> bool:
    for node in ast.walk(function_node):
        if isinstance(node, ast.Raise):
            exc = node.exc
            if isinstance(exc, ast.Call):
                func_name = ""
                if isinstance(exc.func, ast.Name):
                    func_name = exc.func.id
                elif isinstance(exc.func, ast.Attribute):
                    func_name = ".".join(attribute_chain(exc.func))
                if func_name.endswith("NotImplementedError"):
                    return True
            if _is_raise_stub_message(_raise_stub_message(node)):
                return True
    return False


def resolve_py_method_backend_support(
    index: SourceIndex,
    py_method_name: str,
    exec_mode: str,
    backend: str,
    depth: int = 0,
    visited: Optional[set[str]] = None,
) -> tuple[str, list[dict[str, str]]]:
    if not py_method_name:
        return "unknown", []
    if depth > 4:
        return "unknown", []
    visited = set() if visited is None else set(visited)
    if py_method_name in visited:
        return "unknown", []
    visited.add(py_method_name)
    resolved = resolve_py_method_symbol(index, py_method_name)
    if resolved is None or not isinstance(resolved.local_node, ast.FunctionDef) or resolved.local_module is None or resolved.impl_path is None:
        return "unknown", []
    function_node = resolved.local_node
    evidence = [
        EvidenceItem(
            index.relpath(resolved.impl_path),
            "direct",
            f"def {resolved.impl_name}",
            f"py_method fallback for {exec_mode} {backend}",
        ).to_dict()
    ]
    return_calls, _, _ = extract_return_calls_with_metadata(function_node, resolved.local_module)
    if return_calls:
        nested_states: list[str] = []
        nested_evidence: list[dict[str, str]] = []
        for call in return_calls:
            normalized, target = resolve_qualified_symbol(index, call)
            if target is None or target.impl_path is None:
                nested_states.append("unknown")
                continue
            if isinstance(target.local_node, ast.FunctionDef) and target.impl_module == resolved.impl_module:
                state, sub_evidence = resolve_py_method_backend_support(
                    index,
                    target.impl_name,
                    exec_mode,
                    backend,
                    depth + 1,
                    visited,
                )
                nested_states.append(state)
                nested_evidence.extend(sub_evidence)
            elif target.impl_module.startswith("mindspore.ops.auto_generate") or target.impl_module.startswith("mindspore.ops.function"):
                nested_states.append("yes")
                nested_evidence.append(
                    EvidenceItem(
                        index.relpath(target.impl_path),
                        "direct",
                        normalized,
                        f"py_method reaches executable callable for {exec_mode} {backend}",
                    ).to_dict()
                )
            elif isinstance(target.local_node, ast.FunctionDef) and target.impl_module.startswith("mindspore.ops.tensor_method"):
                if target.impl_name == resolved.impl_name:
                    nested_states.append("unknown")
                    continue
                state, sub_evidence = resolve_py_method_backend_support(
                    index,
                    target.impl_name,
                    exec_mode,
                    backend,
                    depth + 1,
                    visited,
                )
                nested_states.append(state)
                nested_evidence.extend(sub_evidence)
            else:
                nested_states.append("yes")
                nested_evidence.append(
                    EvidenceItem(
                        index.relpath(target.impl_path),
                        "direct",
                        normalized,
                        f"py_method reaches executable callable for {exec_mode} {backend}",
                    ).to_dict()
                )
        evidence.extend(nested_evidence)
        evidence = uniq(evidence)
        if nested_states and all(state == "yes" for state in nested_states):
            return "yes", evidence
        if nested_states and all(state == "no" for state in nested_states):
            return "no", evidence
        return "unknown", evidence
    if _function_is_raise_stub(function_node):
        evidence.append(
            EvidenceItem(
                index.relpath(resolved.impl_path),
                "direct",
                f"def {resolved.impl_name}",
                f"py_method raise stub for {exec_mode} {backend}",
            ).to_dict()
        )
        return "no", uniq(evidence)
    return "unknown", evidence


def analyze_functional_overload_support(
    index: SourceIndex,
    support_targets: list[dict[str, str]],
    overload_facts: dict[str, list[str]],
) -> tuple[dict[str, str], dict[str, str], dict[str, dict[str, list[dict[str, str]]]], str]:
    graph_known = set(overload_facts.get("graph", []))
    pynative_known = set(overload_facts.get("pynative", []))
    branch_pynative: list[dict[str, str]] = []
    branch_graph: list[dict[str, str]] = []
    evidence = {
        "pynative": {"ascend": [], "cpu": [], "gpu": []},
        "graph_kbk_o0": {"ascend": [], "cpu": [], "gpu": []},
    }
    backend_value_keys = {"ascend": "ascend_value", "cpu": "cpu_value", "gpu": "gpu_value"}
    for item in support_targets:
        primitive = str(item.get("primitive", ""))
        if not primitive:
            continue
        primitive_pynative, primitive_graph, primitive_evidence, _ = analyze_support(index, [primitive], [item])
        branch_pynative_state = support_state()
        branch_graph_state = support_state()
        api_def_name = str(item.get("api_def", ""))
        op_yaml = str(item.get("op_yaml", ""))
        py_method_name = str(item.get("py_method", ""))
        api_def_info = index.api_defs.get(api_def_name) if api_def_name else None
        api_def_path = index.relpath(api_def_info["path"]) if api_def_info else ""
        for backend, value_key in backend_value_keys.items():
            backend_value = str(item.get(value_key, "")).strip().lower()
            if backend_value == "pyboost":
                branch_pynative_state[backend] = "yes"
                branch_graph_state[backend] = "yes"
                if api_def_path:
                    anchor = f"{api_def_name}:{op_yaml}:{backend}:pyboost"
                    evidence["pynative"][backend].append(
                        EvidenceItem(api_def_path, "direct", anchor, f"functional_overload pyboost branch for {backend}").to_dict()
                    )
                    evidence["graph_kbk_o0"][backend].append(
                        EvidenceItem(api_def_path, "direct", anchor, f"functional_overload KBK pyboost branch for {backend}").to_dict()
                    )
                evidence["pynative"][backend].extend(primitive_evidence["pynative"][backend])
                evidence["graph_kbk_o0"][backend].extend(primitive_evidence["graph_kbk_o0"][backend])
                continue
            if backend_value == "py_method":
                pynative_value, pynative_evidence = resolve_py_method_backend_support(index, py_method_name, "pynative", backend)
                graph_value, graph_evidence = resolve_py_method_backend_support(index, py_method_name, "graph_kbk_o0", backend)
                branch_pynative_state[backend] = pynative_value
                branch_graph_state[backend] = graph_value
                evidence["pynative"][backend].extend(pynative_evidence)
                evidence["graph_kbk_o0"][backend].extend(graph_evidence)
                if api_def_path:
                    anchor = f"{api_def_name}:{op_yaml}:{backend}:py_method:{py_method_name}"
                    evidence["pynative"][backend].append(
                        EvidenceItem(api_def_path, "direct", anchor, f"functional_overload py_method branch for {backend}").to_dict()
                    )
                    evidence["graph_kbk_o0"][backend].append(
                        EvidenceItem(api_def_path, "direct", anchor, f"functional_overload KBK py_method branch for {backend}").to_dict()
                    )
                continue
            branch_pynative_state[backend] = primitive_pynative[backend]
            branch_graph_state[backend] = primitive_graph[backend]
        if pynative_known and primitive not in pynative_known:
            for backend, value_key in backend_value_keys.items():
                if str(item.get(value_key, "")).strip().lower() not in {"pyboost", "py_method"}:
                    branch_pynative_state[backend] = "unknown"
        if graph_known and primitive not in graph_known:
            for backend, value_key in backend_value_keys.items():
                if str(item.get(value_key, "")).strip().lower() not in {"pyboost", "py_method"}:
                    branch_graph_state[backend] = "unknown"
        branch_pynative.append(branch_pynative_state)
        branch_graph.append(branch_graph_state)
        if primitive in pynative_known:
            source = EvidenceItem(index.relpath(index.repo_root / PYBOOST_OVERLOAD_FUNCTIONS_PATH), "direct", primitive, "functional_overload pyboost branch")
            for backend in ("ascend", "cpu", "gpu"):
                evidence["pynative"][backend].append(source.to_dict())
                evidence["pynative"][backend].extend(primitive_evidence["pynative"][backend])
        if primitive in graph_known:
            source = EvidenceItem(index.relpath(index.repo_root / FUNCTIONAL_MAP_PATH), "direct", primitive, "functional_overload graph branch")
            for backend in ("ascend", "cpu", "gpu"):
                evidence["graph_kbk_o0"][backend].append(source.to_dict())
                evidence["graph_kbk_o0"][backend].extend(primitive_evidence["graph_kbk_o0"][backend])
    for mode in evidence.values():
        for backend in mode:
            mode[backend] = uniq(mode[backend])
    unknown_reason = ""
    pynative_state = aggregate_overload_support_states(branch_pynative)
    graph_state = aggregate_overload_support_states(branch_graph)
    if not support_targets:
        unknown_reason = "overload_dispatch_unresolved"
    elif any(value == "unknown" for value in pynative_state.values()) or any(value == "unknown" for value in graph_state.values()):
        unknown_reason = "scenario_dependent_overload"
    return (pynative_state, graph_state, evidence, unknown_reason)


def analyze_python_scenario_support(
    index: SourceIndex,
    support_targets: list[dict[str, str]],
) -> tuple[dict[str, str], dict[str, str], dict[str, dict[str, list[dict[str, str]]]], str]:
    target_primitives = {str(item.get("primitive", "")) for item in support_targets if item.get("primitive")}
    filtered_targets = []
    for item in support_targets:
        primitive = str(item.get("primitive", ""))
        companion = GRAPH_FALLBACK_COMPANION_PRIMITIVE_MAP.get(primitive, "")
        if companion and companion in target_primitives:
            continue
        filtered_targets.append(item)
    support_targets = filtered_targets
    branch_pynative: list[dict[str, str]] = []
    branch_graph: list[dict[str, str]] = []
    evidence = {
        "pynative": {"ascend": [], "cpu": [], "gpu": []},
        "graph_kbk_o0": {"ascend": [], "cpu": [], "gpu": []},
    }
    for item in support_targets:
        primitive = str(item.get("primitive", ""))
        if not primitive:
            continue
        primitive_pynative, primitive_graph, primitive_evidence, _ = analyze_support(index, [primitive], [item])
        branch_pynative.append(primitive_pynative)
        branch_graph.append(primitive_graph)
        for backend in ("ascend", "cpu", "gpu"):
            evidence["pynative"][backend].extend(primitive_evidence["pynative"][backend])
            evidence["graph_kbk_o0"][backend].extend(primitive_evidence["graph_kbk_o0"][backend])
    for mode in evidence.values():
        for backend in mode:
            mode[backend] = uniq(mode[backend])
    pynative_state = aggregate_overload_support_states(branch_pynative)
    graph_state = aggregate_overload_support_states(branch_graph)
    unknown_reason = ""
    if not support_targets or any(value == "unknown" for value in pynative_state.values()) or any(value == "unknown" for value in graph_state.values()):
        unknown_reason = "scenario_dependent_call_chain"
    return pynative_state, graph_state, evidence, unknown_reason


def classify_call_chain(
    export: ResolvedExport,
    *,
    class_execution: Optional[ClassExecution],
    terminal_symbol: str,
    support_targets: list[dict[str, str]],
    possible_primitives: list[str],
    func_op_info: dict[str, Any],
    is_functional_overload: bool,
    python_composite_used: bool = False,
) -> tuple[str, str]:
    if is_functional_overload:
        return "functional_overload", ("overload_dispatch" if support_targets else "unknown")
    if class_execution is not None:
        if class_execution.branching_notes:
            return "scenario_dependent", ("scenario_candidates" if possible_primitives else "unknown")
        if func_op_info.get("is_func_op"):
            return "construct_mapped", ("func_expansion" if func_op_info.get("expanded_primitives") else "unknown")
        if python_composite_used and possible_primitives:
            return "construct_mapped", "scenario_candidates"
        return "construct_mapped", ("real_terminal" if support_targets else "unknown")
    if func_op_info.get("is_func_op"):
        if func_op_info.get("terminal_func_op_bridge"):
            return "func_op", "real_terminal"
        return "func_op", ("func_expansion" if func_op_info.get("expanded_primitives") else "unknown")
    if python_composite_used:
        return "python_composite_wrapper", ("scenario_candidates" if possible_primitives else ("real_terminal" if support_targets else "unknown"))
    if export.source_kind == "generated_binding" or export.impl_module.startswith("mindspore.ops.auto_generate"):
        return "generated_binding", ("real_terminal" if support_targets else "unknown")
    if len(uniq([item.get("primitive", "") for item in support_targets if item.get("primitive")])) > 1:
        return "composite_python", ("real_terminal" if support_targets else "unknown")
    if terminal_symbol.startswith("mindspore._c_expression") or terminal_symbol.endswith("_instance"):
        return "c_expression_blackbox", "unknown"
    if possible_primitives and not support_targets:
        return "scenario_dependent", "scenario_candidates"
    if support_targets:
        return "direct_python_call", "real_terminal"
    return "unresolved", "unknown"


def aggregate_support_states(states: list[dict[str, str]]) -> dict[str, str]:
    if not states:
        return support_state()
    result = {}
    for backend in ("ascend", "cpu", "gpu"):
        values = [state[backend] for state in states]
        if values and all(value == "yes" for value in values):
            result[backend] = "yes"
        elif any(value == "no" for value in values):
            result[backend] = "no"
        else:
            result[backend] = "unknown"
    return result


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


def _extract_braced_block(text: str, open_brace_pos: int) -> str:
    if open_brace_pos < 0 or open_brace_pos >= len(text) or text[open_brace_pos] != "{":
        return ""
    depth = 0
    for pos in range(open_brace_pos, len(text)):
        char = text[pos]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[open_brace_pos : pos + 1]
    return ""


def _extract_function_region(text: str, function_name: str) -> str:
    match = re.search(rf"\b{re.escape(function_name)}\s*\([^;{{]*\)\s*{{", text, re.S)
    if not match:
        return ""
    open_brace = text.find("{", match.start())
    return _extract_braced_block(text, open_brace)


def _extract_class_region(text: str, class_name: str) -> str:
    match = re.search(rf"\bclass\s+{re.escape(class_name)}\b[^{{;]*{{", text, re.S)
    if not match:
        return ""
    open_brace = text.find("{", match.start())
    return _extract_braced_block(text, open_brace)


def _extract_class_method_regions(text: str, class_name: str) -> list[str]:
    regions = []
    pattern = re.compile(rf"\b{re.escape(class_name)}::[A-Za-z_]\w*\s*\([^;{{]*\)\s*{{", re.S)
    for match in pattern.finditer(text):
        open_brace = text.find("{", match.start())
        region = _extract_braced_block(text, open_brace)
        if region:
            regions.append(region)
    return regions


def _factory_classes_for_primitive(text: str, primitive: str) -> list[str]:
    patterns = [
        rf"MS_ACLNN_KERNEL_FACTORY_REG\(\s*{re.escape(primitive)}\s*,\s*([A-Za-z0-9_]+)\s*\)",
        rf"MS_ACLNN_COMMON_KERNEL_FACTORY_REG\(\s*{re.escape(primitive)}\s*,\s*([A-Za-z0-9_]+)\s*,",
    ]
    classes = []
    for pattern in patterns:
        classes.extend(re.findall(pattern, text))
    return uniq(classes)


def _scoped_aclnn_evidence_from_region(index: SourceIndex, path: Path, region: str, summary_prefix: str) -> list[dict[str, str]]:
    evidence = []
    for interface in extract_aclnn_calls_from_text(region):
        evidence.append(
            EvidenceItem(index.relpath(path), "direct", interface, f"{summary_prefix} calls {interface}").to_dict()
        )
    return evidence


def _candidate_header_for_source(path: Path) -> Optional[Path]:
    if path.suffix != ".cc":
        return None
    header = path.with_suffix(".h")
    return header if header.exists() else None


def collect_pyboost_scoped_aclnn_evidence(index: SourceIndex, primitive: str) -> list[dict[str, str]]:
    evidence: list[dict[str, str]] = []
    function_names = {
        f"{primitive}Ascend::Call",
        f"{primitive}AscendCustomize",
    }
    for entry in index.pyboost_ascend_impl_map.get(primitive, []):
        path = index.repo_root / entry.path
        if not path.exists() or path.suffix not in {".cc", ".h"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        regions = []
        for function_name in function_names:
            region = _extract_function_region(text, function_name)
            if region:
                regions.append(region)
        for region in regions:
            evidence.extend(_scoped_aclnn_evidence_from_region(index, path, region, f"Ascend pynative {primitive} path"))
    return uniq(evidence)


def collect_graph_scoped_aclnn_evidence(index: SourceIndex, primitive: str) -> list[dict[str, str]]:
    evidence: list[dict[str, str]] = []
    candidate_paths = []
    for entry in index.ascend_kernel_impl_map.get(primitive, []):
        path = index.repo_root / entry.path
        if path.exists() and path.suffix in {".cc", ".h"}:
            candidate_paths.append(path)
            header = _candidate_header_for_source(path)
            if header:
                candidate_paths.append(header)
    for path in uniq(candidate_paths):
        text = path.read_text(encoding="utf-8", errors="ignore")
        class_names = _factory_classes_for_primitive(text, primitive)
        if not class_names and path.suffix == ".h":
            # The factory registration usually lives in the .cc file. Re-read it
            # to identify the exact kernel mod class before scanning the header.
            source = path.with_suffix(".cc")
            if source.exists():
                class_names = _factory_classes_for_primitive(source.read_text(encoding="utf-8", errors="ignore"), primitive)
        for class_name in class_names:
            regions = []
            class_region = _extract_class_region(text, class_name)
            if class_region:
                regions.append(class_region)
            regions.extend(_extract_class_method_regions(text, class_name))
            for region in regions:
                evidence.extend(_scoped_aclnn_evidence_from_region(index, path, region, f"Ascend graph {primitive} {class_name} path"))
    return uniq(evidence)


def collect_scoped_aclnn_evidence(index: SourceIndex, primitives: list[str]) -> list[dict[str, str]]:
    evidence: list[dict[str, str]] = []
    for primitive in primitives:
        evidence.extend(collect_pyboost_scoped_aclnn_evidence(index, primitive))
        evidence.extend(collect_graph_scoped_aclnn_evidence(index, primitive))
    return uniq(evidence)


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


def _path_is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def analyze_pynative_ascend(
    index: SourceIndex,
    primitive: str,
    primitive_sources: Optional[list[dict[str, str]]] = None,
) -> tuple[bool, list[dict[str, str]]]:
    evidence: list[dict[str, str]] = []
    primitive_sources = primitive_sources or []

    direct_interfaces = []
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
    if direct_interfaces:
        aclnn_config_path = index.repo_root / ACLNN_CONFIG
        for interface in uniq(direct_interfaces):
            evidence.append(
                EvidenceItem(
                    index.relpath(aclnn_config_path),
                    "direct",
                    interface,
                    f"PYNATIVE Ascend aclnn entry for {primitive}",
                ).to_dict()
            )

    pyboost_hit = False
    scoped_aclnn_evidence = collect_pyboost_scoped_aclnn_evidence(index, primitive)
    evidence.extend(scoped_aclnn_evidence)
    for entry in index.pyboost_ascend_impl_map.get(primitive, []):
        path = index.repo_root / entry.path
        if not path.exists() or path.suffix not in {".cc", ".h"}:
            continue
        pyboost_hit = True
        if not any(item.get("path") == entry.path for item in scoped_aclnn_evidence):
            evidence.append(
                EvidenceItem(
                    entry.path,
                    "direct",
                    entry.anchor,
                    f"Ascend pyboost implementation artifact for {primitive}",
                ).to_dict()
            )

    return bool(direct_interfaces) or pyboost_hit, uniq(evidence)


def analyze_pynative_view_op_backend(
    index: SourceIndex,
    primitive: str,
    backend: str,
) -> tuple[str, list[dict[str, str]]]:
    if primitive not in index.view_op_names:
        return "unknown", []
    impl_paths = uniq(index.view_pynative_impl_map.get(primitive, []))
    if len(impl_paths) < 3:
        return "unknown", []
    evidence = [
        EvidenceItem(
            entry.path,
            "direct",
            entry.anchor,
            f"view-op pynative {backend} path for {primitive}",
        ).to_dict()
        for entry in impl_paths
    ]
    return "yes", uniq(evidence)


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
    for item in collect_scoped_aclnn_evidence(index, primitives):
        interface = str(item.get("anchor", ""))
        if interface.startswith("aclnn"):
            effective_interfaces.append(interface)
            aclnn_evidence.append(item)

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


def analyze_functional_overload_aclnn(
    index: SourceIndex,
    support_targets: list[dict[str, str]],
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    if not support_targets:
        return {"mode": "unknown", "interfaces": [], "effective_interfaces": [], "path_kind": "unknown"}, []
    branch_infos: list[dict[str, Any]] = []
    evidence: list[dict[str, str]] = []
    for item in support_targets:
        primitive = str(item.get("primitive", ""))
        if not primitive:
            continue
        branch_info, branch_evidence = analyze_aclnn(index, [primitive], [item])
        branch_infos.append(branch_info)
        evidence.extend(branch_evidence)
    if not branch_infos:
        return {"mode": "unknown", "interfaces": [], "effective_interfaces": [], "path_kind": "unknown"}, uniq(evidence)

    interfaces = uniq([entry for info in branch_infos for entry in info.get("interfaces", [])])
    effective_interfaces = uniq([entry for info in branch_infos for entry in info.get("effective_interfaces", [])])
    branch_modes = [str(info.get("mode", "unknown")) for info in branch_infos]
    path_kinds = [str(info.get("path_kind", "unknown")) for info in branch_infos]

    has_proven_interfaces = bool(interfaces or effective_interfaces)
    if any(mode == "unknown" for mode in branch_modes):
        mode = "unknown"
    elif has_proven_interfaces and any(mode == "none" for mode in branch_modes):
        mode = "unknown"
    elif branch_modes and all(mode == "none" for mode in branch_modes):
        mode = "none"
    elif branch_modes and all(mode == "direct" for mode in branch_modes):
        mode = "direct"
    elif any(mode in {"direct", "indirect"} for mode in branch_modes):
        mode = "indirect"
    else:
        mode = "unknown"

    if any(kind == "unknown" for kind in path_kinds):
        path_kind = "unknown"
    elif has_proven_interfaces and any(kind == "none" for kind in path_kinds):
        path_kind = "unknown"
    elif path_kinds and all(kind == "none" for kind in path_kinds):
        path_kind = "none"
    elif path_kinds and all(kind == "direct_aclnn" for kind in path_kinds):
        path_kind = "direct_aclnn"
    elif any(kind == "customize_to_aclnn" for kind in path_kinds):
        direct_kinds = {kind for kind in path_kinds if kind == "direct_aclnn"}
        customize_kinds = {kind for kind in path_kinds if kind == "customize_to_aclnn"}
        if direct_kinds and customize_kinds:
            path_kind = "composite_to_aclnn"
        else:
            path_kind = "customize_to_aclnn"
    elif any(kind == "composite_to_aclnn" for kind in path_kinds):
        path_kind = "composite_to_aclnn"
    else:
        path_kind = "unknown"

    return {
        "mode": mode,
        "interfaces": interfaces,
        "effective_interfaces": effective_interfaces,
        "path_kind": path_kind,
    }, uniq(evidence)


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


def analyze_support(
    index: SourceIndex,
    primitives: list[str],
    primitive_sources: list[dict[str, str]],
    *,
    allow_no_dispatch_single_primitive_closure: bool = False,
) -> tuple[dict[str, str], dict[str, str], dict[str, dict[str, list[dict[str, str]]]], list[dict[str, Any]]]:
    per_primitive_pynative: list[dict[str, str]] = []
    per_primitive_graph: list[dict[str, str]] = []
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
        primitive_specific_sources = [item for item in primitive_sources if item.get("primitive") == primitive] or primitive_sources
        primitive_has_no_dispatch = bool(primitive_specific_sources) and all(
            str(item.get("op_yaml", "")).strip() and not op_yaml_has_enabled_dispatch(index, str(item.get("op_yaml", "")).strip())
            for item in primitive_specific_sources
        )
        aclop_adapter = uniq(index.aclop_adapter_map.get(primitive, []))
        ascend_kbk = match_ascend_kbk(index, primitive, primitive_specific_sources)
        host_view_graph = match_host_view_kernel(index, primitive, primitive_specific_sources)
        rt_nop_view_graph = match_rt_nop_view_closure(index, primitive, primitive_specific_sources)
        fallback_rt_nop_reshape_graph = match_fallback_to_rt_nop_reshape(index, primitive)
        pynative_ascend_hit, pynative_ascend_evidence = analyze_pynative_ascend(
            index,
            primitive,
            primitive_specific_sources,
        )
        view_pynative_ascend_state, view_pynative_ascend_evidence = analyze_pynative_view_op_backend(
            index,
            primitive,
            "ascend",
        )
        view_pynative_cpu_state, view_pynative_cpu_evidence = analyze_pynative_view_op_backend(
            index,
            primitive,
            "cpu",
        )
        view_pynative_gpu_state, view_pynative_gpu_evidence = analyze_pynative_view_op_backend(
            index,
            primitive,
            "gpu",
        )
        func_dispatch_graph_support, func_dispatch_graph_evidence, func_dispatch_used = analyze_func_dispatch_graph_support(
            index,
            primitive,
            primitive_specific_sources,
        )
        cpu_kernel = match_kernel_registry(index.cpu_kernel_map, primitive, allow_suffix_stripping=False)
        gpu_kernel = match_kernel_registry(index.gpu_kernel_map, primitive, allow_suffix_stripping=False)
        exact_cpu_state, exact_cpu_evidence = analyze_pynative_exact_terminal_backend(index, primitive, "cpu")
        exact_gpu_state, exact_gpu_evidence = analyze_pynative_exact_terminal_backend(index, primitive, "gpu")
        random_cpu_state, random_cpu_evidence = analyze_random_composite_inner_backend(
            index, primitive, "cpu", primitive_specific_sources
        )
        random_gpu_state, random_gpu_evidence = analyze_random_composite_inner_backend(
            index, primitive, "gpu", primitive_specific_sources
        )
        fallback = index.fallback_map.get(primitive, [])
        if not fallback and primitive in GRAPH_FALLBACK_COMPANION_PRIMITIVE_MAP:
            fallback = index.fallback_map.get(GRAPH_FALLBACK_COMPANION_PRIMITIVE_MAP[primitive], [])
        graph_fallback_overrides_dispatch_none = bool(
            fallback and primitive in GRAPH_FALLBACK_DISPATCH_NONE_OVERRIDE_PRIMITIVES
        )
        if primitive in GRAPH_FALLBACK_COMPANION_PRIMITIVE_MAP and fallback:
            graph_fallback_overrides_dispatch_none = True
        primitive_pynative = support_state()
        primitive_graph = support_state()

        if platform_disabled("ascend"):
            primitive_pynative["ascend"] = "no"
        elif pynative_ascend_hit:
            primitive_pynative["ascend"] = "yes"
            evidence["pynative"]["ascend"].extend(pynative_ascend_evidence)
        elif view_pynative_ascend_state == "yes":
            primitive_pynative["ascend"] = "yes"
            evidence["pynative"]["ascend"].extend(view_pynative_ascend_evidence)
        elif allow_no_dispatch_single_primitive_closure and primitive_has_no_dispatch and aclop_adapter:
            primitive_pynative["ascend"] = "yes"
            evidence["pynative"]["ascend"].extend(item.to_dict() for item in aclop_adapter)
        elif platform_declared("ascend") or dispatch_enabled:
            primitive_pynative["ascend"] = "unknown"

        if dispatch_enabled:
            if platform_disabled("cpu"):
                primitive_pynative["cpu"] = "no"
            elif view_pynative_cpu_state == "yes":
                primitive_pynative["cpu"] = "yes"
                evidence["pynative"]["cpu"].extend(view_pynative_cpu_evidence)
            elif cpu_kernel:
                primitive_pynative["cpu"] = "yes"
                evidence["pynative"]["cpu"].extend(item.to_dict() for item in cpu_kernel)
            elif exact_cpu_state in {"yes", "no"}:
                primitive_pynative["cpu"] = exact_cpu_state
                evidence["pynative"]["cpu"].extend(exact_cpu_evidence)
            elif random_cpu_state in {"yes", "no"}:
                primitive_pynative["cpu"] = random_cpu_state
                evidence["pynative"]["cpu"].extend(random_cpu_evidence)
            elif platform_declared("cpu") or dispatch_enabled:
                primitive_pynative["cpu"] = "no"
            if platform_disabled("gpu"):
                primitive_pynative["gpu"] = "no"
            elif view_pynative_gpu_state == "yes":
                primitive_pynative["gpu"] = "yes"
                evidence["pynative"]["gpu"].extend(view_pynative_gpu_evidence)
            elif gpu_kernel:
                primitive_pynative["gpu"] = "yes"
                evidence["pynative"]["gpu"].extend(item.to_dict() for item in gpu_kernel)
            elif exact_gpu_state in {"yes", "no"}:
                primitive_pynative["gpu"] = exact_gpu_state
                evidence["pynative"]["gpu"].extend(exact_gpu_evidence)
            elif random_gpu_state in {"yes", "no"}:
                primitive_pynative["gpu"] = random_gpu_state
                evidence["pynative"]["gpu"].extend(random_gpu_evidence)
            elif platform_declared("gpu") or dispatch_enabled:
                primitive_pynative["gpu"] = "no"
        else:
            if view_pynative_cpu_state == "yes":
                primitive_pynative["cpu"] = "yes"
                evidence["pynative"]["cpu"].extend(view_pynative_cpu_evidence)
            elif cpu_kernel:
                primitive_pynative["cpu"] = "yes"
                evidence["pynative"]["cpu"].extend(item.to_dict() for item in cpu_kernel)
            elif exact_cpu_state in {"yes", "no"}:
                primitive_pynative["cpu"] = exact_cpu_state
                evidence["pynative"]["cpu"].extend(exact_cpu_evidence)
            elif random_cpu_state in {"yes", "no"}:
                primitive_pynative["cpu"] = random_cpu_state
                evidence["pynative"]["cpu"].extend(random_cpu_evidence)
            if view_pynative_gpu_state == "yes":
                primitive_pynative["gpu"] = "yes"
                evidence["pynative"]["gpu"].extend(view_pynative_gpu_evidence)
            elif gpu_kernel:
                primitive_pynative["gpu"] = "yes"
                evidence["pynative"]["gpu"].extend(item.to_dict() for item in gpu_kernel)
            elif exact_gpu_state in {"yes", "no"}:
                primitive_pynative["gpu"] = exact_gpu_state
                evidence["pynative"]["gpu"].extend(exact_gpu_evidence)
            elif random_gpu_state in {"yes", "no"}:
                primitive_pynative["gpu"] = random_gpu_state
                evidence["pynative"]["gpu"].extend(random_gpu_evidence)

        if platform_disabled("ascend"):
            primitive_graph["ascend"] = "no"
        elif ascend_kbk:
            primitive_graph["ascend"] = "yes"
            evidence["graph_kbk_o0"]["ascend"].extend(item.to_dict() for item in ascend_kbk)
        elif host_view_graph:
            primitive_graph["ascend"] = "yes"
            evidence["graph_kbk_o0"]["ascend"].extend(item.to_dict() for item in host_view_graph)
        elif rt_nop_view_graph:
            primitive_graph["ascend"] = "yes"
            evidence["graph_kbk_o0"]["ascend"].extend(item.to_dict() for item in rt_nop_view_graph)
        elif allow_no_dispatch_single_primitive_closure and primitive_has_no_dispatch and aclop_adapter:
            primitive_graph["ascend"] = "yes"
            evidence["graph_kbk_o0"]["ascend"].extend(item.to_dict() for item in aclop_adapter)
        elif fallback_rt_nop_reshape_graph:
            primitive_graph["ascend"] = "yes"
            evidence["graph_kbk_o0"]["ascend"].extend(item.to_dict() for item in fallback_rt_nop_reshape_graph)
        elif func_dispatch_used and func_dispatch_graph_support["ascend"] == "yes":
            primitive_graph["ascend"] = "yes"
            evidence["graph_kbk_o0"]["ascend"].extend(func_dispatch_graph_evidence["ascend"])
        elif platform_declared("ascend"):
            primitive_graph["ascend"] = "unknown"
        if platform_disabled("cpu") and not graph_fallback_overrides_dispatch_none:
            primitive_graph["cpu"] = "no"
        elif cpu_kernel:
            primitive_graph["cpu"] = "yes"
            evidence["graph_kbk_o0"]["cpu"].extend(item.to_dict() for item in cpu_kernel)
        elif func_dispatch_used and func_dispatch_graph_support["cpu"] == "yes":
            primitive_graph["cpu"] = "yes"
            evidence["graph_kbk_o0"]["cpu"].extend(func_dispatch_graph_evidence["cpu"])
        elif fallback:
            primitive_graph["cpu"] = "yes"
            evidence["graph_kbk_o0"]["cpu"].extend(item.to_dict() for item in fallback)
        elif primitives:
            primitive_graph["cpu"] = "no"
        if platform_disabled("gpu") and not graph_fallback_overrides_dispatch_none:
            primitive_graph["gpu"] = "no"
        elif gpu_kernel:
            primitive_graph["gpu"] = "yes"
            evidence["graph_kbk_o0"]["gpu"].extend(item.to_dict() for item in gpu_kernel)
        elif func_dispatch_used and func_dispatch_graph_support["gpu"] == "yes":
            primitive_graph["gpu"] = "yes"
            evidence["graph_kbk_o0"]["gpu"].extend(func_dispatch_graph_evidence["gpu"])
        elif fallback:
            primitive_graph["gpu"] = "yes"
            evidence["graph_kbk_o0"]["gpu"].extend(item.to_dict() for item in fallback)
        elif primitives:
            primitive_graph["gpu"] = "no"

        per_primitive_pynative.append(primitive_pynative)
        per_primitive_graph.append(primitive_graph)
    for mode in evidence.values():
        for backend in mode:
            mode[backend] = uniq(mode[backend])
    pynative = aggregate_support_states(per_primitive_pynative)
    graph = aggregate_support_states(per_primitive_graph)
    return pynative, graph, evidence, detail


def analyze_func_dispatch_graph_support(
    index: SourceIndex,
    primitive: str,
    primitive_sources: list[dict[str, str]],
) -> tuple[dict[str, str], dict[str, list[dict[str, str]]], bool]:
    graph = support_state()
    evidence = {"ascend": [], "cpu": [], "gpu": []}
    func_op_info = analyze_func_op(index, primitive_sources, [primitive])
    if not func_op_info.get("is_func_op"):
        return graph, evidence, False
    expanded_primitives = list(func_op_info.get("expanded_primitives", []))
    if expanded_primitives:
        _, unresolved_expansion = primitive_sources_from_symbols(index, expanded_primitives, origin_kind="func_expansion")
        if not unresolved_expansion:
            return graph, evidence, False
    meta_dsl_paths = [str(path) for path in func_op_info.get("meta_dsl_paths", []) if path]
    if not meta_dsl_paths:
        return graph, evidence, False
    dispatch_enabled, dispatch_platforms, detail = dispatch_map(index, primitive_sources)
    if not dispatch_enabled:
        return graph, evidence, False

    used = False
    dispatch_keys = {"ascend": "ascend", "cpu": "cpu", "gpu": "gpu"}
    source_labels = {"ascend": "graph func_op dispatch", "cpu": "graph func_op dispatch", "gpu": "graph func_op dispatch"}
    for backend, dispatch_key in dispatch_keys.items():
        dispatch_value = dispatch_platforms.get(dispatch_key, "__missing__")
        if dispatch_value is None:
            graph[backend] = "no"
            continue
        if dispatch_value == "__missing__":
            continue
        graph[backend] = "yes"
        used = True
        for path in meta_dsl_paths:
            evidence[backend].append(
                EvidenceItem(path, "direct", Path(path).name, f"{source_labels[backend]} via meta_dsl func_op").to_dict()
            )
        for item in detail:
            api_def_name = str(item.get("api_def", ""))
            op_yaml = str(item.get("op_yaml", ""))
            anchor = f"{api_def_name}:{op_yaml}:{dispatch_key}" if api_def_name or op_yaml else dispatch_key
            evidence[backend].append(
                EvidenceItem(
                    index.relpath(index.repo_root / FUNC_OP_DEF_ROOT),
                    "direct",
                    anchor,
                    f"dispatch declares {dispatch_key} func_op support",
                ).to_dict()
            )
    for backend in evidence:
        evidence[backend] = uniq(evidence[backend])
    return graph, evidence, used


def inherit_construct_records(main_records: list[dict[str, Any]], evidence_records: list[dict[str, Any]]) -> None:
    record_map = {item["api"]: item for item in main_records}
    evidence_map = {item["api"]: item for item in evidence_records}
    impl_symbol_to_api: dict[str, str] = {}
    for evidence in evidence_records:
        impl_sym = str(evidence.get("impl_symbol", "")).strip()
        if impl_sym:
            impl_symbol_to_api[impl_sym] = evidence["api"]

    for record in main_records:
        if record["api_level"] != "module_api" or not record["composed_of"]:
            continue
        target_api = record["composed_of"][0]
        target_record = record_map.get(target_api)
        target_evidence = evidence_map.get(target_api)
        if target_record is None:
            delegate_api = impl_symbol_to_api.get(target_api)
            if delegate_api:
                target_record = record_map.get(delegate_api)
                target_evidence = evidence_map.get(delegate_api)
                target_api = delegate_api
        if not target_record or (
            not target_record.get("primitive")
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
        record["possible_primitives"] = filter_possible_primitives(list(target_record.get("possible_primitives", [])))
        record["pynative_support"] = dict(target_record["pynative_support"])
        record["graph_kbk_o0_support"] = dict(target_record["graph_kbk_o0_support"])
        record["func_op_expands_to"] = list(target_record.get("func_op_expands_to", []))
        record["support_reason_kind"] = target_record.get("support_reason_kind", record.get("support_reason_kind", "unknown"))
        record["fact_origin"] = "inherited_from_construct"
        record["aclnn"] = {
            "mode": target_record["aclnn"]["mode"],
            "interfaces": list(target_record["aclnn"]["interfaces"]),
            "effective_interfaces": list(target_record["aclnn"]["effective_interfaces"]),
            "path_kind": target_record["aclnn"]["path_kind"],
        }
        record["grad"] = {
            "mode": target_record["grad"]["mode"],
            "differentiable": target_record["grad"].get("differentiable", "unknown"),
            "backward_primitives": list(target_record["grad"].get("backward_primitives", [])),
            "impl": [dict(item) for item in target_record["grad"]["impl"]],
        }
        record["path_hints"] = merge_path_hints(record.get("path_hints", _empty_path_hints()), target_record.get("path_hints", _empty_path_hints()))

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

        record["unknown_reason"] = infer_unknown_reason(record)
        record["trust_level"] = infer_trust_level(
            api=record["api"],
            call_chain_kind=record.get("call_chain_kind", ""),
            resolution_kind=record.get("resolution_kind", ""),
            implementation_type=record.get("implementation_type", ""),
            unknown_reason=record["unknown_reason"],
            primitive=list(record.get("primitive", [])),
            possible_primitives=list(record.get("possible_primitives", [])),
            func_op_expands_to=list(record.get("func_op_expands_to", [])),
            runtime_utility=record.get("implementation_type") == "runtime_utility",
        )
        if "needs_manual_review" in record["flags"] and record["trust_level"] in ("certain", "strong"):
            record["flags"] = [f for f in record["flags"] if f != "needs_manual_review"]
        record["summary"] = build_summary(record)


def infer_category(public_path: str, impl_symbol: str) -> str:
    return "mint"


def is_optim_api(public_path: str) -> bool:
    return public_path.startswith("mindspore.mint.optim.")


def infer_api_level(implementation_type: str, export: ResolvedExport, primitives: list[str]) -> str:
    if implementation_type == "high_level_module" or export.api_kind == "class":
        return "module_api"
    if implementation_type in {"single_op", "multi_overload_op"} and primitives:
        return "operator_api"
    return "wrapper_api"


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


def build_summary(item: dict[str, Any]) -> str:
    primitive = ",".join(item.get("primitive") or []) or "none"
    possible = ",".join(item.get("possible_primitives") or []) or "none"
    aclnn = aclnn_brief(item.get("aclnn") or {})
    if item.get("implementation_type") == "runtime_utility":
        return "runtime utility; operator mapping not applicable"
    if item.get("implementation_type") == "python_pass_through":
        return (
            "python pass-through Cell; primitive=none; "
            f"pynative({support_brief(item['pynative_support'])}); "
            f"graph({support_brief(item['graph_kbk_o0_support'])}); aclnn={aclnn}"
        )
    if item.get("call_chain_kind") == "functional_overload":
        suffix = ""
        if item.get("unknown_reason"):
            suffix = f"; unknown_reason={item['unknown_reason']}"
        return (
            f"functional_overload/{item.get('resolution_kind')}; possible_primitives={possible}; "
            f"pynative({support_brief(item['pynative_support'])}); "
            f"graph({support_brief(item['graph_kbk_o0_support'])}); aclnn={aclnn}{suffix}"
        )
    if item.get("unknown_reason") == "not_applicable" and is_distributed_comm_api(item.get("api", "")) and item.get("primitive"):
        return f"distributed communication operator; primitive={primitive}; standard kernel/aclnn support mapping not applicable"
    if item.get("unknown_reason") == "not_applicable" and is_optim_api(item.get("api", "")):
        return "optimizer module; operator mapping not applicable"
    if item.get("unknown_reason") == "terminal_symbol_unresolved":
        return f"real execution chain unresolved; primitive={primitive}; pynative({support_brief(item['pynative_support'])}); graph({support_brief(item['graph_kbk_o0_support'])}); aclnn={aclnn}"
    if item.get("unknown_reason") == "scenario_dependent_call_chain":
        return (
            f"python composite wrapper/{item.get('resolution_kind')}; possible_primitives={possible}; "
            f"pynative({support_brief(item['pynative_support'])}); "
            f"graph({support_brief(item['graph_kbk_o0_support'])}); aclnn={aclnn}"
        )
    if item.get("api_level") == "module_api" and not item.get("primitive") and item.get("possible_primitives"):
        if item.get("support_reason_kind") == "func_dispatch":
            return (
                f"high-level module via construct; scenario dependent; possible_primitives={possible}; "
                f"graph({support_brief(item['graph_kbk_o0_support'])}) via func dispatch"
            )
        return f"high-level module via construct; scenario dependent; possible_primitives={possible}"
    if not item.get("primitive") and item.get("possible_primitives"):
        return f"wrapper api; scenario dependent; possible_primitives={possible}"
    if item.get("resolution_kind") == "func_expansion":
        graph_phrase = "via expansion"
        if item.get("support_reason_kind") == "func_dispatch":
            graph_phrase = "via func dispatch"
        return (
            f"func_op; primitive={primitive}; pynative({support_brief(item['pynative_support'])}); "
            f"graph({support_brief(item['graph_kbk_o0_support'])}) {graph_phrase}; aclnn={aclnn}"
        )
    if item.get("implementation_type") == "multi_overload_op":
        return (
            f"multi overload; primitives={primitive}; pynative({support_brief(item['pynative_support'])}); "
            f"graph({support_brief(item['graph_kbk_o0_support'])}); aclnn={aclnn}"
        )
    if item.get("implementation_type") == "high_level_module":
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


def infer_unknown_reason(item: dict[str, Any]) -> str:
    if item.get("implementation_type") == "runtime_utility":
        return "not_applicable"
    if item.get("primitive") and is_distributed_comm_api(item.get("api", "")):
        return "not_applicable"
    if is_optim_api(item.get("api", "")) and not item.get("primitive"):
        return "not_applicable"
    if is_optim_api(item.get("api", "")) and item.get("api_level") == "module_api":
        return "not_applicable"
    if item.get("call_chain_kind") == "functional_overload":
        if not item.get("possible_primitives"):
            return "overload_dispatch_unresolved"
        if any(
            item[field][backend] == "unknown"
            for field in ("pynative_support", "graph_kbk_o0_support")
            for backend in ("ascend", "cpu", "gpu")
        ):
            return "scenario_dependent_overload"
        return ""
    if item.get("call_chain_kind") == "python_composite_wrapper" and item.get("possible_primitives"):
        return "scenario_dependent_call_chain"
    if not item.get("primitive") and item.get("possible_primitives"):
        return "scenario_dependent"
    if not item.get("primitive") and not item.get("possible_primitives") and not item.get("func_op_expands_to"):
        if item.get("composed_of"):
            return "unresolved_composite_chain"
        return "terminal_symbol_unresolved"
    if item.get("resolution_kind") == "func_expansion" and any(
        item["graph_kbk_o0_support"][backend] == "unknown" for backend in ("ascend", "cpu", "gpu")
    ):
        return "func_op_expansion"
    if item.get("primitive") and item["graph_kbk_o0_support"]["ascend"] == "unknown":
        return "missing_kbk_evidence"
    if item.get("primitive") and any(
        item["pynative_support"][backend] == "unknown" or item["graph_kbk_o0_support"][backend] == "unknown"
        for backend in ("cpu", "gpu")
    ):
        return "missing_runtime_kernel_evidence"
    if not item.get("primitive") and (item.get("composed_of") or item.get("possible_primitives")):
        return "unresolved_static_chain"
    if any(
        item[field][backend] == "unknown"
        for field in ("pynative_support", "graph_kbk_o0_support")
        for backend in ("ascend", "cpu", "gpu")
    ):
        return "unresolved_static_chain"
    return ""


def infer_trust_level(
    *,
    api: str,
    call_chain_kind: str,
    resolution_kind: str,
    implementation_type: str,
    unknown_reason: str,
    primitive: list[str],
    possible_primitives: list[str],
    func_op_expands_to: list[str],
    runtime_utility: bool,
) -> str:
    if runtime_utility or unknown_reason == "not_applicable":
        return "not_applicable"
    if call_chain_kind == "functional_overload" or resolution_kind == "overload_dispatch":
        return "conditional"
    if unknown_reason in {
        "terminal_symbol_unresolved",
        "ambiguous_terminal_mapping",
        "unresolved_composite_chain",
        "unresolved_static_chain",
        "missing_kbk_evidence",
        "missing_runtime_kernel_evidence",
    }:
        return "weak"
    if resolution_kind == "func_expansion" or (call_chain_kind == "func_op" and func_op_expands_to):
        return "strong"
    if resolution_kind == "scenario_candidates" or unknown_reason in {"scenario_dependent", "scenario_dependent_call_chain", "scenario_dependent_overload", "overload_dispatch_unresolved"}:
        return "conditional"
    if implementation_type == "high_level_module" or call_chain_kind == "construct_mapped":
        return "strong"
    if not primitive and possible_primitives:
        return "conditional"
    if call_chain_kind in {"direct_python_call", "generated_binding"} or implementation_type in {"single_op", "alias"}:
        return "certain"
    return "certain" if primitive else "weak"


def _in_scope_primitives_for_flags(
    primitive: list[str],
    possible_primitives: list[str],
    func_op_expands_to: list[str],
    support_targets: list[dict[str, Any]],
) -> list[str]:
    values: list[str] = []
    values.extend(str(item) for item in primitive if item)
    values.extend(str(item) for item in possible_primitives if item)
    values.extend(str(item) for item in func_op_expands_to if item)
    values.extend(str(item.get("primitive", "")) for item in support_targets if item.get("primitive"))
    return uniq([item for item in values if item])


def should_mark_view_op(
    *,
    primitive: list[str],
    possible_primitives: list[str],
    func_op_expands_to: list[str],
    call_chain_kind: str,
    view_op_names: set[str],
) -> bool:
    if len(primitive) != 1:
        return False
    if possible_primitives or func_op_expands_to:
        return False
    if call_chain_kind in {"functional_overload", "python_composite_wrapper", "scenario_dependent"}:
        return False
    return primitive[0] in view_op_names


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


def infer_grad_scope_kind(
    *,
    origin_kind: str,
    call_chain_kind: str,
    resolution_kind: str,
) -> str:
    if origin_kind == "func_expansion" or resolution_kind == "func_expansion":
        return "func_expansion"
    if origin_kind == "overload_branch" or call_chain_kind == "functional_overload":
        return "overload_branch"
    if resolution_kind == "scenario_candidates":
        return "python_branch"
    if call_chain_kind in {"construct_mapped", "python_composite_wrapper"}:
        return "composite_chain"
    return "real_terminal"


def collect_grad_scope_targets(
    *,
    call_chain_kind: str,
    resolution_kind: str,
    support_targets: list[dict[str, str]],
    func_op_expands_to: list[str],
) -> list[dict[str, str]]:
    if resolution_kind == "func_expansion":
        return [
            {
                "primitive": primitive,
                "origin_kind": "func_expansion",
                "scope_kind": "func_expansion",
            }
            for primitive in uniq(func_op_expands_to)
            if primitive
        ]
    targets = []
    for item in support_targets:
        primitive = str(item.get("primitive", ""))
        if not primitive:
            continue
        targets.append(
            {
                "primitive": primitive,
                "origin_kind": str(item.get("origin_kind", "")),
                "scope_kind": infer_grad_scope_kind(
                    origin_kind=str(item.get("origin_kind", "")),
                    call_chain_kind=call_chain_kind,
                    resolution_kind=resolution_kind,
                ),
            }
        )
    return uniq(targets)


def scan_function_calls(function_node: ast.FunctionDef, module: ModuleInfo) -> list[str]:
    calls: list[str] = []

    class Visitor(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call) -> None:
            symbol = extract_call_symbol(node, module)
            if symbol:
                calls.append(symbol)
            self.generic_visit(node)

    Visitor().visit(function_node)
    return uniq(calls)


def resolve_backward_primitives_from_symbols(index: SourceIndex, symbols: list[str]) -> list[str]:
    primitives = []
    for symbol in symbols:
        resolved, _ = resolve_primitive_source_from_terminal(index, symbol, origin_kind="effective_call")
        if resolved and resolved.get("primitive"):
            primitives.append(str(resolved["primitive"]))
    return uniq(primitives)


def collect_python_bprop_candidates(
    index: SourceIndex,
    export: ResolvedExport,
    class_execution: Optional[ClassExecution],
) -> list[tuple[str, ast.FunctionDef, ModuleInfo]]:
    candidates: list[tuple[str, ast.FunctionDef, ModuleInfo]] = []
    seen: set[tuple[str, str]] = set()
    if export.api_kind == "class" and isinstance(export.local_node, ast.ClassDef) and export.local_module is not None:
        method_info = find_method_in_hierarchy(export.local_node, export.local_module, "bprop")
        if method_info is not None:
            _, owner_class, owner_module = method_info
            key = (owner_module.module_name, owner_class.name)
            if key not in seen:
                seen.add(key)
                candidates.append((f"{owner_module.module_name}.{owner_class.name}", method_info[0], owner_module))
    chain = class_execution.chain if class_execution is not None else []
    for item in chain:
        if not item.endswith(".construct"):
            continue
        class_symbol = item[:-len(".construct")]
        normalized, resolution = resolve_qualified_symbol(index, class_symbol)
        if resolution is None or not isinstance(resolution.local_node, ast.ClassDef) or resolution.local_module is None:
            continue
        method_info = find_method_in_hierarchy(resolution.local_node, resolution.local_module, "bprop")
        if method_info is None:
            continue
        _, owner_class, owner_module = method_info
        key = (owner_module.module_name, owner_class.name)
        if key in seen:
            continue
        seen.add(key)
        candidates.append((normalized, method_info[0], owner_module))
    return candidates


def can_collapse_functional_overload_grad(index: SourceIndex, export: ResolvedExport) -> bool:
    api_def = index.api_defs.get(export.impl_name)
    if api_def is None:
        return False
    payload = api_def.get("payload", {})
    if not isinstance(payload, list):
        return False
    function_entries = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        interface_value = str(item.get("interface", ""))
        interfaces = {part.strip() for part in interface_value.split(",") if part.strip()}
        if "function" not in interfaces:
            continue
        op_yaml = str(item.get("op_yaml", ""))
        if op_yaml.startswith("deprecated/"):
            continue
        function_entries.append(item)
    if not function_entries:
        return False
    py_methods = {str(item.get("py_method", "")).strip() for item in function_entries}
    if "" in py_methods or "place_holder" in py_methods:
        return False
    return len(py_methods) == 1


def analyze_grad(
    index: SourceIndex,
    export: ResolvedExport,
    *,
    call_chain_kind: str,
    resolution_kind: str,
    implementation_type: str,
    unknown_reason: str,
    grad_scope_targets: list[dict[str, str]],
    class_execution: Optional[ClassExecution],
) -> GradAnalysis:
    if infer_grad_not_applicable(export.public_path):
        return GradAnalysis(mode="not_applicable", differentiable="no", impl=[], backward_primitives=[])

    grad_scope_primitives = uniq([str(item.get("primitive", "")) for item in grad_scope_targets if item.get("primitive")])
    overload_grad_collapsible = call_chain_kind == "functional_overload" and can_collapse_functional_overload_grad(index, export)
    scenario_like = resolution_kind == "scenario_candidates" or (
        call_chain_kind == "functional_overload" and not overload_grad_collapsible
    )
    unresolved_scope = unknown_reason in {
        "terminal_symbol_unresolved",
        "ambiguous_terminal_mapping",
        "unresolved_composite_chain",
        "scenario_dependent_call_chain",
        "scenario_dependent_overload",
        "overload_dispatch_unresolved",
    }

    grad_impl: list[dict[str, str]] = []
    backward_primitives: list[str] = []

    for target in grad_scope_targets:
        primitive = str(target.get("primitive", ""))
        if not primitive or primitive not in index.bprop_map:
            continue
        backward_primitives.append(primitive)
        for ev in index.bprop_map[primitive]:
            grad_impl.append(
                {
                    "primitive": primitive,
                    "kind": "cpp_bprop_builder",
                    "path": ev.path,
                    "anchor": ev.anchor,
                    "scope_kind": str(target.get("scope_kind", "real_terminal")),
                }
            )

    for class_symbol, bprop_method, owner_module in collect_python_bprop_candidates(index, export, class_execution):
        call_symbols = scan_function_calls(bprop_method, owner_module)
        resolved_backward = resolve_backward_primitives_from_symbols(index, call_symbols)
        grad_impl.append(
            {
                "primitive": class_symbol.split(".")[-1],
                "kind": "python_cell_bprop",
                "path": index.relpath(owner_module.path),
                "anchor": f"{class_symbol}.bprop",
                "scope_kind": "composite_chain",
            }
        )
        backward_primitives.extend(resolved_backward)

    grad_impl = uniq(grad_impl)
    backward_primitives = uniq(backward_primitives)
    has_python_bprop = any(str(item.get("kind")) == "python_cell_bprop" for item in grad_impl)

    if has_python_bprop and not scenario_like:
        return GradAnalysis(mode="explicit_bprop", differentiable="yes", impl=grad_impl, backward_primitives=backward_primitives)
    if grad_impl and not unresolved_scope and not scenario_like:
        return GradAnalysis(mode="explicit_bprop", differentiable="yes", impl=grad_impl, backward_primitives=backward_primitives)
    if grad_impl and scenario_like:
        return GradAnalysis(mode="unknown", differentiable="unknown", impl=grad_impl, backward_primitives=backward_primitives)
    if export.api_kind == "class":
        if implementation_type == "runtime_utility":
            return GradAnalysis(mode="not_applicable", differentiable="no", impl=[], backward_primitives=[])
        if grad_scope_primitives and not unresolved_scope and not scenario_like:
            return GradAnalysis(mode="autodiff", differentiable="yes", impl=[], backward_primitives=[])
        return GradAnalysis(mode="unknown", differentiable="unknown", impl=grad_impl, backward_primitives=backward_primitives)
    if grad_scope_primitives and not unresolved_scope and not scenario_like and call_chain_kind in {"direct_python_call", "generated_binding"}:
        return GradAnalysis(mode="autodiff", differentiable="yes", impl=[], backward_primitives=[])
    return GradAnalysis(mode="unknown", differentiable="unknown", impl=grad_impl, backward_primitives=backward_primitives)


def _empty_kernel_path_hints() -> dict[str, dict[str, list[Any]]]:
    return {
        "pynative": {"ascend": [], "cpu": [], "gpu": []},
        "graph_kbk_o0": {"ascend": [], "cpu": [], "gpu": []},
    }


def _empty_path_hints() -> dict[str, Any]:
    return {
        "api_def_paths": [],
        "dispatch_paths": [],
        "implementation_paths": [],
        "op_def_paths": [],
        "kernel_paths": _empty_kernel_path_hints(),
        "infer_paths": [],
    }


def merge_path_hints(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    result = _empty_path_hints()
    for key in ("api_def_paths", "dispatch_paths", "implementation_paths", "op_def_paths", "infer_paths"):
        result[key] = uniq(list(left.get(key, [])) + list(right.get(key, [])))[:12]
    for exec_mode in ("pynative", "graph_kbk_o0"):
        for backend in ("ascend", "cpu", "gpu"):
            result["kernel_paths"][exec_mode][backend] = uniq(
                list(left.get("kernel_paths", {}).get(exec_mode, {}).get(backend, []))
                + list(right.get("kernel_paths", {}).get(exec_mode, {}).get(backend, []))
            )[:12]
    result["api_def_paths"] = result["api_def_paths"][:6]
    return result


def collect_infer_paths(
    index: SourceIndex,
    primitive_sources: list[dict[str, Any]],
    primitives: list[str],
    possible_primitives: list[str],
    func_op_op_yamls: list[str],
) -> list[PathEntry]:
    candidates: list[PathEntry] = []
    stems: list[str] = []
    for item in primitive_sources:
        op_yaml = str(item.get("op_yaml", "")).strip()
        if op_yaml:
            stems.append(op_yaml.removesuffix(".yaml").removesuffix("_op").removesuffix("_ext"))
    for primitive in list(primitives) + list(possible_primitives):
        if primitive:
            stems.append(camel_to_snake(str(primitive)))
    for op_yaml in func_op_op_yamls:
        if op_yaml:
            stems.append(str(op_yaml).removesuffix(".yaml"))
    for stem in uniq([item for item in stems if item]):
        candidates.extend(index.infer_path_map.get(stem, []))
    return uniq(candidates)[:12]


def collect_kernel_path_hints(
    index: SourceIndex,
    support_targets: list[dict[str, Any]],
    support_evidence: dict[str, dict[str, list[dict[str, Any]]]],
    dispatch_paths: set[str],
    aclnn_evidence: Optional[list[dict[str, Any]]] = None,
) -> dict[str, dict[str, list[PathEntry]]]:
    result: dict[str, dict[str, list[PathEntry]]] = {
        "pynative": {"ascend": [], "cpu": [], "gpu": []},
        "graph_kbk_o0": {"ascend": [], "cpu": [], "gpu": []},
    }
    for exec_mode in ("pynative", "graph_kbk_o0"):
        for backend in ("ascend", "cpu", "gpu"):
            bucket: list[PathEntry] = []
            for item in support_evidence.get(exec_mode, {}).get(backend, []):
                path = str(item.get("path", "")).strip()
                if path and path not in dispatch_paths and not path.endswith(".yaml") and not path.endswith(".py"):
                    bucket.append(PathEntry(path, str(item.get("anchor", ""))))
            result[exec_mode][backend] = uniq(bucket)[:12]
    for item in aclnn_evidence or []:
        path = str(item.get("path", "")).strip()
        if path and path not in dispatch_paths and not path.endswith(".yaml") and not path.endswith(".py"):
            entry = PathEntry(path, str(item.get("anchor", "")))
            result["pynative"]["ascend"].append(entry)
            result["graph_kbk_o0"]["ascend"].append(entry)
    # Tighten Ascend to branch/primitive-level direct implementation hits
    # using content-based maps that key by actual class names found in source
    # (e.g. SubExtAscend::Call), avoiding suffix-stripping false matches.
    ascend_primitives = uniq([str(item.get("primitive", "")).strip() for item in support_targets if str(item.get("primitive", "")).strip()])
    if ascend_primitives:
        pynative_ascend: list[PathEntry] = []
        graph_ascend: list[PathEntry] = []
        for primitive in ascend_primitives:
            pynative_ascend.extend(index.pyboost_ascend_impl_map.get(primitive, []))
            graph_ascend.extend(index.ascend_kernel_impl_map.get(primitive, []))
        if pynative_ascend:
            result["pynative"]["ascend"] = uniq(pynative_ascend)[:12]
        if graph_ascend:
            result["graph_kbk_o0"]["ascend"] = uniq(graph_ascend)[:12]
    return result


def collect_dispatch_paths(
    api_def_paths: list[PathEntry],
    dispatch_detail: list[dict[str, Any]],
    func_meta_dsl_paths: list[str],
    implementation_type: str,
    primitives: list[str],
) -> list[PathEntry]:
    result: list[PathEntry] = []
    if implementation_type == "overload_wrapper":
        result.append(PathEntry(index_rel_functional_overload_path(), "functional_overload"))
        prim_anchor = ", ".join(primitives[:3]) if primitives else ""
        result.append(PathEntry(str(FUNCTIONAL_MAP_PATH).replace("\\", "/"), prim_anchor))
        result.append(PathEntry(str(PYBOOST_OVERLOAD_FUNCTIONS_PATH).replace("\\", "/"), prim_anchor))
    for item in dispatch_detail:
        api_def = str(item.get("api_def", "")).strip()
        if api_def:
            result.extend(api_def_paths)
            break
    for p in func_meta_dsl_paths:
        result.append(PathEntry(p))
    return uniq(result)[:12]


def index_rel_functional_overload_path() -> str:
    return "mindspore/python/mindspore/ops/functional_overload.py"


def collect_implementation_paths(
    index: SourceIndex,
    export: ResolvedExport,
    support_evidence: dict[str, dict[str, list[dict[str, Any]]]],
    support_targets: list[dict[str, Any]],
    dispatch_paths: set[str],
) -> list[PathEntry]:
    paths: list[PathEntry] = []
    if export.impl_path:
        rel = index.relpath(export.impl_path)
        if rel not in dispatch_paths:
            anchor = export.impl_name or ""
            paths.append(PathEntry(rel, anchor))
    for exec_mode in ("pynative", "graph_kbk_o0"):
        for backend in ("ascend", "cpu", "gpu"):
            for item in support_evidence.get(exec_mode, {}).get(backend, []):
                path = str(item.get("path", "")).strip()
                if not path or path in dispatch_paths:
                    continue
                lower = path.lower()
                if lower.endswith(".py") and "kernel/" not in lower and "fallback/" not in lower and "aclnn/" not in lower:
                    paths.append(PathEntry(path, str(item.get("anchor", ""))))
    for item in support_targets:
        op_def_path = str(item.get("op_def_path", "")).strip()
        if op_def_path:
            continue
    return uniq(paths)[:12]


def build_path_hints(
    index: SourceIndex,
    export: ResolvedExport,
    record: dict[str, Any],
    *,
    api_def_entry: Optional[dict[str, Any]],
    support_primitive_sources: list[dict[str, Any]],
    metadata_primitive_sources: list[dict[str, Any]],
    support_targets: list[dict[str, Any]],
    support_evidence: dict[str, dict[str, list[dict[str, Any]]]],
    aclnn_evidence: list[dict[str, Any]],
    dispatch_detail: list[dict[str, Any]],
    grad_analysis: GradAnalysis,
    func_op_info: dict[str, Any],
) -> dict[str, Any]:
    api_name_short = export.impl_name or export.public_path.rsplit(".", 1)[-1]
    api_def_paths: list[PathEntry] = []
    api_def_used = any(str(item.get("api_def", "")).strip() for item in support_targets)
    if str(record.get("call_chain_kind", "")) == "functional_overload" or str(record.get("resolution_kind", "")) == "overload_dispatch":
        api_def_used = True
    if api_def_entry is not None and api_def_used:
        api_def_paths.append(PathEntry(index.relpath(api_def_entry["path"]), api_name_short))
    op_def_paths: list[PathEntry] = []
    for item in uniq(list(metadata_primitive_sources) + list(support_primitive_sources) + list(support_targets)):
        op_def_path = str(item.get("op_def_path", "")).strip()
        if op_def_path:
            op_info = index.op_defs.get(str(item.get("op_yaml", "")))
            op_yaml_name = op_info["op_name"] if op_info else str(item.get("op_yaml", "")).replace(".yaml", "")
            op_def_paths.append(PathEntry(op_def_path, op_yaml_name))
    for op_yaml in func_op_info.get("op_yamls", []):
        info = index.op_defs.get(str(op_yaml))
        if info is not None:
            op_def_paths.append(PathEntry(index.relpath(info["path"]), info["op_name"]))
    possible_primitives = list(record.get("possible_primitives", []))
    dispatch_primitives = list(record.get("primitive", [])) or possible_primitives
    dispatch_paths = collect_dispatch_paths(
        api_def_paths,
        dispatch_detail,
        list(func_op_info.get("meta_dsl_paths", [])),
        str(record.get("implementation_type", "")),
        dispatch_primitives,
    )
    if record.get("api") == "mindspore.mint.einsum" and "EinsumExt" in dispatch_primitives:
        dispatch_paths.extend(
            [
                PathEntry(index_rel_functional_overload_path(), "functional_overload.einsum"),
                PathEntry(str(FUNCTIONAL_MAP_PATH).replace("\\", "/"), '"einsum" -> prim::kPrimEinsumExt'),
            ]
        )
        dispatch_paths = uniq(dispatch_paths)[:12]
    if record.get("api") == "mindspore.mint.nn.functional.conv2d" and {
        "Conv2DExt",
        "Conv2DPadding",
    }.issubset(set(dispatch_primitives)):
        dispatch_paths.extend(
            [
                PathEntry(index_rel_functional_overload_path(), "functional_overload.conv2d"),
                PathEntry(
                    str(FUNCTIONAL_MAP_PATH).replace("\\", "/"),
                    '"conv2d" -> prim::kPrimConv2DExt / prim::kPrimConv2DPadding',
                ),
                PathEntry(str(PYBOOST_OVERLOAD_FUNCTIONS_PATH).replace("\\", "/"), "conv2d overload dispatch"),
            ]
        )
        dispatch_paths = uniq(dispatch_paths)[:12]
    dispatch_set = {entry.path for entry in dispatch_paths}
    kernel_paths = collect_kernel_path_hints(
        index,
        support_targets,
        support_evidence,
        dispatch_set,
        aclnn_evidence=aclnn_evidence,
    )
    implementation_paths = collect_implementation_paths(
        index,
        export,
        support_evidence,
        support_primitive_sources,
        dispatch_set,
    )
    infer_paths = collect_infer_paths(
        index,
        support_primitive_sources,
        list(record.get("primitive", [])),
        list(record.get("possible_primitives", [])),
        list(func_op_info.get("op_yamls", [])),
    )
    return {
        "api_def_paths": uniq(api_def_paths)[:6],
        "dispatch_paths": uniq(dispatch_paths)[:12],
        "implementation_paths": uniq(implementation_paths)[:12],
        "op_def_paths": uniq(op_def_paths)[:12],
        "kernel_paths": kernel_paths,
        "infer_paths": infer_paths,
    }


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

    metadata_primitives = []
    metadata_primitive_sources = []
    for entry in effective_entries:
        primitive, op_path = primitive_from_op(index, entry.get("op_yaml"))
        if primitive:
            metadata_primitives.append(primitive)
            metadata_primitive_sources.append(
                {
                    "api_def": entry.get("_api_def", ""),
                    "op_yaml": entry.get("op_yaml"),
                    "op_def_path": op_path or "",
                    "primitive": primitive,
                }
            )
    support_primitive_sources, unresolved_support_symbols, support_chain_facts = resolve_support_primitive_sources(
        index, terminal_calls, effective_calls, prelude_calls, api_kind=export.api_kind
    )
    support_primitive_sources, unresolved_support_symbols, support_chain_facts = apply_special_terminal_resolution(
        index,
        export,
        support_primitive_sources,
        unresolved_support_symbols,
        support_chain_facts,
    )
    metadata_unique_primitives = uniq(metadata_primitives)
    support_unique_primitives = uniq([str(item.get("primitive", "")) for item in support_primitive_sources if item.get("primitive")])
    if (
        len(metadata_unique_primitives) == 1
        and metadata_unique_primitives[0] in support_unique_primitives
        and not support_chain_facts.get("scenario_dependent")
        and set(support_unique_primitives) - {metadata_unique_primitives[0]} <= {"Cast"}
    ):
        # Cast used only for argument preparation must not turn a single-op wrapper
        # such as cumprod into a multi-primitive support target.
        support_primitive_sources = [
            item for item in support_primitive_sources if str(item.get("primitive", "")) == metadata_unique_primitives[0]
        ]
    elif not support_chain_facts.get("scenario_dependent") and "Cast" in support_unique_primitives:
        non_cast_primitives = [primitive for primitive in support_unique_primitives if primitive != "Cast"]
        if len(non_cast_primitives) == 1:
            support_primitive_sources = [
                item for item in support_primitive_sources if str(item.get("primitive", "")) == non_cast_primitives[0]
            ]
    support_primitive_sources = drop_prelude_helper_primitives(
        export=export,
        support_primitive_sources=support_primitive_sources,
        support_chain_facts=support_chain_facts,
    )
    bridge_func_op_targets, bridge_func_op_facts = resolve_functional_overload_func_op_bridge(
        index, unresolved_support_symbols
    )
    if bridge_func_op_targets:
        bridged_symbols = set(bridge_func_op_facts.get("symbols", []))
        support_primitive_sources = uniq(list(support_primitive_sources) + list(bridge_func_op_targets))
        unresolved_support_symbols = [
            item
            for item in unresolved_support_symbols
            if str(item.get("symbol", "")).strip() not in bridged_symbols and item.get("origin_kind") != "prelude_call"
        ]
        support_chain_facts["execution_chain"].extend(bridge_func_op_facts.get("execution_chain", []))
        support_chain_facts["functional_overload_func_op_bridge"] = True
    else:
        bridge_func_op_facts = {"symbols": [], "evidence": [], "execution_chain": []}
        support_chain_facts["functional_overload_func_op_bridge"] = False
    support_primitive_sources = enrich_support_targets_with_op_defs(index, support_primitive_sources)
    support_primitives = uniq([item["primitive"] for item in support_primitive_sources if item.get("primitive")])
    is_functional_overload = is_functional_overload_export(export)
    overload_targets, overload_possible_primitives, overload_facts = resolve_functional_overload_targets(
        index,
        export,
        api_def_name,
        effective_entries,
    ) if is_functional_overload else ([], [], {"graph": [], "pynative": []})
    fixed_conv2d_functional_bridge = export.public_path == "mindspore.mint.nn.functional.conv2d" and bool(
        support_chain_facts.get("possible_primitives")
    )
    func_op_info = (
        analyze_func_op(index, support_primitive_sources, support_primitives)
        if not overload_targets and not fixed_conv2d_functional_bridge
        else {
        "is_func_op": False,
        "op_yamls": [],
        "meta_dsl_paths": [],
        "expanded_primitives": [],
        "evidence": [],
        }
    )
    bridge_func_op_terminal = bool(bridge_func_op_targets) and support_primitives == ["EinsumExt"]
    if bridge_func_op_terminal:
        func_op_info["is_func_op"] = True
        func_op_info["op_yamls"] = uniq([item.get("op_yaml", "") for item in bridge_func_op_targets if item.get("op_yaml")])
        func_op_info["expanded_primitives"] = ["EinsumExt"]
        func_op_info["terminal_func_op_bridge"] = True
        func_op_info["evidence"] = uniq(list(func_op_info.get("evidence", [])) + list(bridge_func_op_facts.get("evidence", [])))
    support_targets = support_primitive_sources
    overload_unknown_reason = ""
    resolution_unknown_reason = ""
    support_blocking_unknown_reason = ""
    single_no_dispatch_direct_terminal_closure = False
    support_target_primitives = uniq([item["primitive"] for item in support_targets if item.get("primitive")])
    no_dispatch_single_primitive_candidate = (
        export.api_kind != "class"
        and not overload_targets
        and not support_chain_facts["scenario_dependent"]
        and not support_chain_facts["python_composite_used"]
        and len(support_target_primitives) == 1
        and bool(support_targets)
        and all(str(item.get("op_yaml", "")).strip() for item in support_targets)
        and all(not op_yaml_has_enabled_dispatch(index, str(item.get("op_yaml", "")).strip()) for item in support_targets)
    )
    if unresolved_support_symbols:
        unresolved_reasons = {item["reason"] for item in unresolved_support_symbols}
        if "ambiguous_terminal_mapping" in unresolved_reasons:
            resolution_unknown_reason = "ambiguous_terminal_mapping"
        elif support_primitive_sources:
            resolution_unknown_reason = "unresolved_composite_chain"
        else:
            resolution_unknown_reason = next(iter(unresolved_reasons))
        unique_support_primitive_count = len(
            uniq([item.get("primitive", "") for item in support_primitive_sources if item.get("primitive")])
        )
        allow_prelude_unresolved_support = (
            bool(support_primitive_sources)
            and not func_op_info["is_func_op"]
            and not overload_targets
            and not support_chain_facts["python_composite_used"]
            and not support_chain_facts["scenario_dependent"]
            and unique_support_primitive_count == 1
        )
        blocking_unresolved = [
            item
            for item in unresolved_support_symbols
            if not (
                is_non_backend_unresolved_symbol(
                    item,
                    api_kind=export.api_kind,
                    has_support_targets=bool(support_primitive_sources),
                )
                or (
                item.get("origin_kind") == "prelude_call"
                and unique_support_primitive_count == 1
                and (
                    allow_prelude_unresolved_support
                    or is_pure_python_prelude_symbol(str(item.get("symbol", "")))
                )
                )
            )
        ]
        if not blocking_unresolved:
            resolution_unknown_reason = ""
        if no_dispatch_single_primitive_candidate and blocking_unresolved and all(
            item.get("reason") == "terminal_symbol_unresolved" for item in blocking_unresolved
        ):
            blocking_unresolved = []
            resolution_unknown_reason = ""
        blocking_reasons = {item["reason"] for item in blocking_unresolved}
        if not support_chain_facts["scenario_dependent"]:
            if "ambiguous_terminal_mapping" in blocking_reasons:
                support_blocking_unknown_reason = "ambiguous_terminal_mapping"
            elif blocking_reasons and support_primitive_sources:
                support_blocking_unknown_reason = "unresolved_composite_chain"
            elif blocking_reasons:
                support_blocking_unknown_reason = next(iter(blocking_reasons))
    if overload_targets:
        support_targets = overload_targets
        support_primitives = []
        support_target_primitives = uniq([item["primitive"] for item in support_targets if item.get("primitive")])
        resolution_unknown_reason = ""
        pynative_support, graph_kbk_o0_support, support_evidence, overload_unknown_reason = analyze_functional_overload_support(
            index,
            support_targets,
            overload_facts,
        )
        dispatch_detail = []
    else:
        if func_op_info["is_func_op"] and func_op_info["expanded_primitives"]:
            support_targets = support_targets_from_func_op(index, func_op_info)
        support_targets = enrich_support_targets_with_op_defs(index, support_targets)
        support_target_primitives = uniq([item["primitive"] for item in support_targets if item.get("primitive")])
        if support_chain_facts["scenario_dependent"] and support_chain_facts["possible_primitives"]:
            pynative_support, graph_kbk_o0_support, support_evidence, overload_unknown_reason = analyze_python_scenario_support(
                index,
                support_targets,
            )
            dispatch_detail = []
        else:
            pynative_support, graph_kbk_o0_support, support_evidence, dispatch_detail = analyze_support(
                index,
                support_target_primitives,
                support_targets,
                allow_no_dispatch_single_primitive_closure=no_dispatch_single_primitive_candidate,
            )
            single_no_dispatch_direct_terminal_closure = no_dispatch_single_primitive_candidate and len(
                support_target_primitives
            ) == 1
            if bridge_func_op_terminal:
                graph_kbk_o0_support["ascend"] = "yes"
                for mode_name in ("pynative", "graph_kbk_o0"):
                    for backend in ("cpu", "gpu"):
                        support_evidence[mode_name][backend] = []
                pynative_support["cpu"] = "unknown"
                pynative_support["gpu"] = "unknown"
                graph_kbk_o0_support["cpu"] = "unknown"
                graph_kbk_o0_support["gpu"] = "unknown"
                support_evidence["graph_kbk_o0"]["ascend"].extend(bridge_func_op_facts.get("evidence", []))
                support_evidence["pynative"]["ascend"].extend(
                    EvidenceItem(
                        item.path,
                        "direct",
                        item.anchor,
                        "PYNATIVE Ascend pyboost implementation for EinsumExt",
                    ).to_dict()
                    for item in index.pyboost_ascend_impl_map.get("EinsumExt", [])
                )
        if support_blocking_unknown_reason:
            pynative_support = support_state()
            graph_kbk_o0_support = support_state()
            support_evidence = {"pynative": {"ascend": [], "cpu": [], "gpu": []}, "graph_kbk_o0": {"ascend": [], "cpu": [], "gpu": []}}
            dispatch_detail = []
        elif single_no_dispatch_direct_terminal_closure and all(
            state == "yes" for state in pynative_support.values()
        ) and all(state == "yes" for state in graph_kbk_o0_support.values()):
            resolution_unknown_reason = ""
    interface_forms = uniq(
        [part.strip() for entry in effective_entries for part in str(entry.get("interface", "")).split(",") if part.strip()]
    )

    if not overload_targets and func_op_info.get("is_func_op") and func_op_info.get("expanded_primitives"):
        aclnn_primitives = support_target_primitives
        aclnn_sources = support_targets
    else:
        aclnn_primitives = support_primitives
        aclnn_sources = support_primitive_sources

    aclnn_info, aclnn_evidence = (
        analyze_functional_overload_aclnn(index, support_targets)
        if overload_targets
        else analyze_aclnn(index, aclnn_primitives, aclnn_sources)
    )
    if func_op_info.get("is_func_op") and func_op_info.get("expanded_primitives"):
        if aclnn_info.get("path_kind") in ("customize_to_aclnn", "direct_aclnn"):
            aclnn_info["path_kind"] = "composite_to_aclnn"
    composite_branching_notes = list(support_chain_facts["branching_notes"])
    class_branching_scenario = export.api_kind == "class" and class_execution is not None and bool(class_execution.branching_notes)
    scenario_dependent = class_branching_scenario or bool(support_chain_facts["scenario_dependent"])
    runtime_utility = is_runtime_utility_api(export.public_path)
    python_pass_through = is_python_identity_pass_through(export, class_execution)
    operator_mapping_not_applicable = is_operator_mapping_not_applicable_api(export.public_path)
    possible_primitives = overload_possible_primitives if overload_targets else (
        support_chain_facts["possible_primitives"] if support_chain_facts["scenario_dependent"] else (uniq(support_primitives) if scenario_dependent else [])
    )
    terminal_resolved = bool(support_primitive_sources)
    if not scenario_dependent and not terminal_resolved and not overload_targets:
        possible_primitives = []
    record_primitives = [] if support_chain_facts["scenario_dependent"] and support_chain_facts["possible_primitives"] else support_primitives

    if class_branching_scenario and not overload_targets:
        pynative_support = support_state()
        graph_kbk_o0_support = support_state()
        support_evidence = {"pynative": {"ascend": [], "cpu": [], "gpu": []}, "graph_kbk_o0": {"ascend": [], "cpu": [], "gpu": []}}
        aclnn_info = {"mode": "unknown", "interfaces": [], "effective_interfaces": [], "path_kind": "unknown"}
        aclnn_evidence = []
        metadata_primitives = []
        metadata_primitive_sources = []
        support_primitives = []
        support_primitive_sources = []
        func_op_info = {"is_func_op": False, "op_yamls": [], "meta_dsl_paths": [], "expanded_primitives": [], "evidence": []}

    if runtime_utility:
        metadata_primitives = []
        metadata_primitive_sources = []
        support_primitives = []
        support_primitive_sources = []
        support_targets = []
        support_target_primitives = []
        possible_primitives = []
        pynative_support = support_state()
        graph_kbk_o0_support = support_state()
        aclnn_info = {"mode": "not_applicable", "interfaces": [], "effective_interfaces": [], "path_kind": "not_applicable"}
        aclnn_evidence = []
        func_op_info = {"is_func_op": False, "op_yamls": [], "meta_dsl_paths": [], "expanded_primitives": [], "evidence": []}

    if python_pass_through:
        metadata_primitives = []
        metadata_primitive_sources = []
        support_primitives = []
        support_primitive_sources = []
        support_targets = []
        support_target_primitives = []
        possible_primitives = []
        record_primitives = []
        pynative_support = {"ascend": "yes", "cpu": "yes", "gpu": "yes"}
        graph_kbk_o0_support = {"ascend": "yes", "cpu": "yes", "gpu": "yes"}
        support_evidence = {"pynative": {"ascend": [], "cpu": [], "gpu": []}, "graph_kbk_o0": {"ascend": [], "cpu": [], "gpu": []}}
        for mode_name in ("pynative", "graph_kbk_o0"):
            for backend in ("ascend", "cpu", "gpu"):
                support_evidence[mode_name][backend].append(
                    EvidenceItem(
                        index.relpath(export.impl_path) if export.impl_path else "",
                        "direct",
                        "Identity.construct returns input",
                        f"python pass-through support for {mode_name} {backend}",
                    ).to_dict()
                )
        aclnn_info = {"mode": "not_applicable", "interfaces": [], "effective_interfaces": [], "path_kind": "not_applicable"}
        aclnn_evidence = []
        func_op_info = {"is_func_op": False, "op_yamls": [], "meta_dsl_paths": [], "expanded_primitives": [], "evidence": []}
        scenario_dependent = False
        class_branching_scenario = False

    implementation_type = "high_level_module" if export.api_kind == "class" else "composite_op"
    if alias_of:
        implementation_type = "alias"
    elif runtime_utility:
        implementation_type = "runtime_utility"
    elif python_pass_through:
        implementation_type = "python_pass_through"
    elif overload_targets:
        implementation_type = "overload_wrapper"
    elif export.api_kind != "class" and len(uniq(metadata_primitives)) > 1:
        implementation_type = "multi_overload_op"
    elif export.api_kind != "class" and metadata_primitives and len(effective_calls) <= 1 and len(uniq(metadata_primitives)) == 1:
        implementation_type = "single_op"

    api_level = infer_api_level(implementation_type, export, metadata_primitives)

    review_risk = "high" if metadata_primitives or alias_of else ("medium" if export.source_kind in {"generated_binding", "ops_binding"} else "low")
    flags = []
    if alias_of:
        flags.append("alias")
    if export.source_kind == "generated_binding":
        flags.append("generated_binding")
    if (review_risk == "low" or (implementation_type == "single_op" and not metadata_primitives)) and not runtime_utility:
        flags.append("needs_manual_review")

    fact_origin = "direct"
    if runtime_utility:
        fact_origin = "not_applicable"
    elif python_pass_through:
        fact_origin = "direct"
    elif overload_targets:
        fact_origin = "overload_dispatch"
    elif class_branching_scenario:
        fact_origin = "inherited_from_construct"
    elif func_op_info["is_func_op"]:
        fact_origin = "expansion_derived" if export.api_kind != "class" else "inherited_from_construct"
    elif implementation_type == "high_level_module":
        inherited_construct = class_execution is not None and (support_primitives or possible_primitives or construct_calls)
        fact_origin = "inherited_from_construct" if inherited_construct else "direct"
    elif alias_of:
        fact_origin = "inherited_from_alias"

    func_dispatch_evidence_present = any(
        "graph func_op dispatch" in str(item.get("summary", ""))
        for backend in ("ascend", "cpu", "gpu")
        for item in support_evidence["graph_kbk_o0"][backend]
    )
    support_reason_kind = "unknown"
    if runtime_utility:
        support_reason_kind = "not_applicable"
    elif python_pass_through:
        support_reason_kind = "direct_support"
    elif func_dispatch_evidence_present:
        support_reason_kind = "func_dispatch"
    elif overload_targets:
        support_reason_kind = "overload_dispatch"
    elif support_blocking_unknown_reason:
        support_reason_kind = "unknown"
    elif scenario_dependent or (not support_primitives and possible_primitives):
        support_reason_kind = "scenario_dependent"
    elif func_op_info["is_func_op"]:
        support_reason_kind = "func_expansion" if func_op_info["expanded_primitives"] else "func_dispatch"
    elif any("fallback builder" in item["summary"] for item in support_evidence["graph_kbk_o0"]["cpu"] + support_evidence["graph_kbk_o0"]["gpu"]):
        support_reason_kind = "fallback_closure"
    elif single_no_dispatch_direct_terminal_closure and len(record_primitives) == 1 and all(
        state == "yes" for state in pynative_support.values()
    ) and all(state == "yes" for state in graph_kbk_o0_support.values()):
        support_reason_kind = "direct_support"
    elif aclnn_info["path_kind"] == "customize_to_aclnn":
        support_reason_kind = "direct_support"
    elif export.api_kind != "class" and effective_calls and metadata_primitive_sources:
        support_reason_kind = "direct_support"
    elif export.api_kind != "class" and effective_calls:
        support_reason_kind = "composite_closure"
    elif any(state == "yes" for state in graph_kbk_o0_support.values()):
        support_reason_kind = "direct_support"

    terminal_symbol = ""
    if overload_targets:
        terminal_symbol = f"{FUNCTIONAL_OVERLOAD_MODULE}.{export.impl_name}"
    elif terminal_resolved:
        if terminal_calls:
            terminal_symbol = terminal_calls[0]
        elif effective_calls:
            terminal_symbol = effective_calls[0]
    terminal_kind = "functional_overload" if overload_targets else (
        classify_terminal_kind(
            terminal_symbol,
            func_op_info["is_func_op"],
            export.local_module,
        ) if terminal_symbol else ("func_op" if func_op_info["is_func_op"] else "")
    )
    call_chain_kind, resolution_kind = classify_call_chain(
        export,
        class_execution=class_execution,
        terminal_symbol=terminal_symbol,
        support_targets=support_targets,
        possible_primitives=possible_primitives,
        func_op_info=func_op_info,
        is_functional_overload=bool(overload_targets),
        python_composite_used=bool(support_chain_facts["python_composite_used"]),
    )
    if python_pass_through:
        terminal_symbol = f"{export.impl_module}.{export.impl_name}.construct"
        terminal_kind = "python_pass_through"
        call_chain_kind = "construct_mapped"
        resolution_kind = "real_terminal"

    main = {
        "api": export.public_path,
        "category": infer_category(export.public_path, f"{export.impl_module}.{export.impl_name}"),
        "api_level": api_level,
        "trust_level": "",
        "fact_origin": fact_origin,
        "call_chain_kind": call_chain_kind,
        "resolution_kind": resolution_kind,
        "implementation_type": implementation_type,
        "primitive": record_primitives,
        "possible_primitives": possible_primitives,
        "func_op_expands_to": func_op_info["expanded_primitives"],
        "support_reason_kind": support_reason_kind,
        "pynative_support": pynative_support,
        "graph_kbk_o0_support": graph_kbk_o0_support,
        "aclnn": aclnn_info,
        "grad": {"mode": "unknown", "differentiable": "unknown", "backward_primitives": [], "impl": []},
        "composed_of": uniq(effective_calls)[:12],
        "branching_notes": uniq((class_execution.branching_notes if class_execution is not None else []) + composite_branching_notes),
        "alias_of": alias_of,
        "path_hints": {},
        "flags": uniq(flags),
        "summary": "",
    }
    main["unknown_reason"] = infer_unknown_reason(main)
    if python_pass_through:
        main["unknown_reason"] = ""
    if export.public_path in {
        "mindspore.mint.float_power",
        "mindspore.mint.nn.functional.conv2d",
    } and not any(value == "unknown" for value in pynative_support.values()) and not any(
        value == "unknown" for value in graph_kbk_o0_support.values()
    ):
        main["unknown_reason"] = ""
    if bridge_func_op_terminal:
        main["unknown_reason"] = ""
    if resolution_unknown_reason:
        main["unknown_reason"] = resolution_unknown_reason
    if overload_unknown_reason:
        main["unknown_reason"] = overload_unknown_reason
    grad_scope_targets = collect_grad_scope_targets(
        call_chain_kind=call_chain_kind,
        resolution_kind=resolution_kind,
        support_targets=support_targets,
        func_op_expands_to=main["func_op_expands_to"],
    )
    grad_analysis = analyze_grad(
        index,
        export,
        call_chain_kind=call_chain_kind,
        resolution_kind=resolution_kind,
        implementation_type=implementation_type,
        unknown_reason=main["unknown_reason"],
        grad_scope_targets=grad_scope_targets,
        class_execution=class_execution,
    )
    main["grad"] = {
        "mode": grad_analysis.mode,
        "differentiable": grad_analysis.differentiable,
        "backward_primitives": grad_analysis.backward_primitives,
        "impl": grad_analysis.impl,
    }
    main["trust_level"] = infer_trust_level(
        api=main["api"],
        call_chain_kind=call_chain_kind,
        resolution_kind=resolution_kind,
        implementation_type=implementation_type,
        unknown_reason=main["unknown_reason"],
        primitive=main["primitive"],
        possible_primitives=main["possible_primitives"],
        func_op_expands_to=main["func_op_expands_to"],
        runtime_utility=runtime_utility,
    )
    main["path_hints"] = build_path_hints(
        index,
        export,
        main,
        api_def_entry=api_def_entry,
        support_primitive_sources=support_primitive_sources,
        metadata_primitive_sources=metadata_primitive_sources,
        support_targets=support_targets,
        support_evidence=support_evidence,
        aclnn_evidence=aclnn_evidence + func_op_info["evidence"],
        dispatch_detail=dispatch_detail,
        grad_analysis=grad_analysis,
        func_op_info=func_op_info,
    )
    all_op_names = _in_scope_primitives_for_flags(
        main["primitive"],
        main["possible_primitives"],
        main["func_op_expands_to"],
        support_targets,
    )
    is_view_op = should_mark_view_op(
        primitive=main["primitive"],
        possible_primitives=main["possible_primitives"],
        func_op_expands_to=main["func_op_expands_to"],
        call_chain_kind=call_chain_kind,
        view_op_names=index.view_op_names,
    )
    if any(name in index.view_op_names for name in all_op_names) and not is_view_op:
        main["flags"].append("has_view_op")
    if is_view_op:
        main["flags"].append("view_op")
    main["summary"] = build_summary(main)
    if "needs_manual_review" in main["flags"] and main["trust_level"] in ("certain", "strong"):
        main["flags"] = [f for f in main["flags"] if f != "needs_manual_review"]

    evidence = {
        "api": export.public_path,
        "source": uniq([item.to_dict() for item in export.evidence]),
        "impl_symbol": f"{export.impl_module}.{export.impl_name}",
        "impl_path": index.relpath(export.impl_path) if export.impl_path else "",
        "execution_entry": class_execution.entry if class_execution is not None else "",
        "execution_chain": uniq((class_execution.chain if class_execution is not None else []) + list(support_chain_facts["execution_chain"])),
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
        "call_chain_kind": call_chain_kind,
        "resolution_kind": resolution_kind,
        "primitive_sources": uniq(support_targets),
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
        "grad": [dict(item) for item in grad_analysis.impl],
        "grad_scope_targets": grad_scope_targets,
        "grad_backward_primitives": list(grad_analysis.backward_primitives),
        "branching_notes": uniq((class_execution.branching_notes if class_execution is not None else []) + composite_branching_notes),
        "notes": [],
    }
    if alias_of and effective_entries is alias_entries:
        evidence["notes"].append(f"inherited operator facts from alias target {alias_of}")
    if not api_def_entry:
        evidence["notes"].append("no matching api_def found from static resolution")
    if not support_targets and not runtime_utility and not scenario_dependent and not overload_targets:
        evidence["notes"].append("real terminal symbol could not be statically resolved")
    evidence["resolved_symbol_chain"] = export.resolved_symbol_chain
    if metadata_primitive_sources and not api_def_entry:
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
        ref = {"api": record["api"], "summary": record["summary"]}
        if record.get("implementation_type") == "runtime_utility" or record.get("unknown_reason") == "not_applicable":
            queue["primitive_mapping_not_applicable"].append(ref)
            continue
        if not record["primitive"] and not record.get("possible_primitives") and not record.get("func_op_expands_to"):
            if record["api_level"] == "wrapper_api" and record.get("composed_of"):
                queue["simple_wrapper_missing"].append(ref)
            else:
                queue["true_unresolved_mapping"].append(ref)
        if (
            record["primitive"]
            and record.get("unknown_reason") not in {"func_op_expansion", "missing_kbk_evidence", "not_applicable"}
            and any(record["pynative_support"][backend] == "unknown" for backend in ("ascend", "cpu", "gpu"))
        ):
            queue["support_should_be_no_but_still_unknown"].append(ref)
        if record["aclnn"]["mode"] in {"direct", "indirect"} and not record["aclnn"]["effective_interfaces"] and record["aclnn"]["path_kind"] != "direct_aclnn":
            queue["aclnn_effective_interface_missing"].append(ref)
        if record["primitive"] and record.get("resolution_kind") != "func_expansion" and any(
            record["pynative_support"][backend] != "unknown" and record["graph_kbk_o0_support"][backend] == "unknown"
            for backend in ("cpu", "gpu")
        ):
            queue["primitive_missing_cpu_gpu_kernel_mapping"].append(ref)
        if (
            record["primitive"]
            and record["graph_kbk_o0_support"]["ascend"] == "unknown"
            and record.get("resolution_kind") != "func_expansion"
            and record.get("unknown_reason") != "not_applicable"
        ):
            queue["graph_ascend_kbk_unclosed"].append(ref)
        if (
            record["api_level"] == "module_api"
            and record.get("call_chain_kind") != "construct_mapped"
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
            lines.append(f"- `{item['api']}`: {item['summary']}")
        lines.append("")
    return "\n".join(lines)


def build_rulebook() -> str:
    lines = [
        "# MindSpore Mint API Index Rulebook",
        "",
        "- Main index: `mint_api_index.db`.",
        "- Optional human review export: `mint_api_index.yaml`.",
        "- Evidence side table: `mint_api_evidence.yaml`.",
        "- Review queue: `mint_api_review_queue.yaml`.",
        "- All paths are relative to the user-provided MindSpore repo root. No absolute paths. No line numbers.",
        "- `api_level`: `operator_api`, `wrapper_api`, `module_api`.",
        "- `implementation_type`: `single_op`, `multi_overload_op`, `composite_op`, `alias`, `high_level_module`, `runtime_utility`, `overload_wrapper`.",
        "- `category`: current mint index domain marker. For now all records use `mint`.",
        "- `trust_level`: reliability tier over the whole record: `certain`, `strong`, `conditional`, `weak`, `not_applicable`.",
        "- `fact_origin`: `direct`, `inherited_from_construct`, `inherited_from_alias`, `expansion_derived`, `not_applicable`.",
        "- `support_reason_kind`: `direct_support`, `func_expansion`, `func_dispatch`, `overload_dispatch`, `fallback_closure`, `composite_closure`, `scenario_dependent`, `not_applicable`, `unknown`.",
        "- `unknown_reason`: `terminal_symbol_unresolved`, `unresolved_static_chain`, `func_op_expansion`, `scenario_dependent`, `not_applicable`, `missing_kbk_evidence`, `missing_runtime_kernel_evidence`, or empty when no unknown explanation is needed.",
        "- `pynative_support` and `graph_kbk_o0_support` are matrices over `ascend/cpu/gpu` with values `yes`, `no`, `unknown`.",
        "- For ordinary built-in PYNATIVE CPU/GPU paths, exact terminal kernel closure uses the real terminal primitive name: pyboost runners may only close to `yes` through exact-name custom-kernel or kernel-factory registrations, not by borrowing old primitive names or plugin/fallback paths.",
        "- Single-primitive and view-composite PYNATIVE view paths are exceptions to exact-name kernel closure: static `pyboost_api.cc -> pyboost_core.cc -> kernel::pyboost::<view_op>() -> *_view_impl` or `FlattenExt -> flatten_ext_impl -> reshape_impl` closure is positive CPU/GPU/Ascend evidence.",
        "- Random composite primitives such as `RandExt`/`Randn`/`RandInt` may close CPU/GPU support through their real inner `InplaceUniform`/`InplaceNormal`/`InplaceRandom` pyboost path; the outer composite primitive itself is only the entry point.",
        "- In `graph_kbk_o0`, a proven fallback builder is positive backend evidence. For the currently enabled direct/generated terminal fallback cases (`ArgMaxExt`, `NonZeroExt`), `dispatch.<backend> = None` does not override a matching fallback closure.",
        "- In `graph_kbk_o0` on Ascend, base view primitives such as `Reshape` and `Squeeze` may close through `RT_KERNEL + IsNopNode + nop_op_to_memcpy_/MemoryCopyAsync`; view composites such as `FlattenExt` may close only through an explicit fallback builder to such an inner primitive.",
        "- For single-primitive `real_terminal` APIs whose resolved `op_yaml` has no `dispatch`, support may still close directly: Ascend can use `REG_ADPT_DESC(...)` adapter registrations under `op_adapter/op_declare`, and CPU/GPU can use exact kernel factory registrations.",
        "- Module-level aliases to generated functions, such as `tensor_gt = greater`, should resolve through the imported generated function and then to the terminal primitive.",
        "- `_get_cache_prim(P.Xxx)` and `_get_cache_prim(Xxx)` with a statically known primitive class are equivalent to constructing and calling that primitive. Dynamic `_get_cache_prim(expr)` must remain unresolved.",
        "- Pyboost inner primitive classes such as `_PyboostSearchSortedPrim` can be resolved through their generated base class (`SearchSortedPrim_ -> SearchSorted`) when the inheritance chain is static.",
        "- Fixed functional-overload bridges must use exact mapped primitives only. Example: `nn.functional.conv2d -> functional_overload.conv2d -> functional_map.cc` resolves to `Conv2DExt` / `Conv2DPadding`; do not borrow legacy `Conv2D` kernel evidence for CPU/GPU.",
        "- A class whose `construct` is a pure `return input` pass-through should be modeled as Python pass-through support rather than as an `Identity` primitive. Do not add `identity_op.yaml` or borrow `Identity` kernel evidence for this case.",
        "- `summary` is the only human/LLM helper string. It compresses existing structured facts without introducing new ones.",
        "- `aclnn` contains `mode`, `interfaces`, `effective_interfaces`, `path_kind`.",
        "- `aclnn.mode`: `direct`, `indirect`, `none`, `unknown`, `not_applicable`.",
        "- `aclnn.path_kind`: `direct_aclnn`, `customize_to_aclnn`, `composite_to_aclnn`, `none`, `unknown`, `not_applicable`.",
        "- `grad.mode`: `explicit_bprop`, `grad_op`, `autodiff`, `not_applicable`, `unknown`.",
        "- Records flagged with `func_op` come from `ops/op_def/func_op` and usually have `bprop_expander: False`. Backend support must be judged from the real func_op expansion chain, not sibling overload hits.",
        "- `func_op_expands_to` lists downstream primitives statically extracted from `ccsrc/frontend/operator/meta_dsl/func_op/*.cc`.",
        "- `multi_overload_op` means one public API maps to multiple primitive overloads. Structural overload facts may still be listed, but backend support must follow the real execution chain.",
        "- `high_level_module` with `construct_mapped` means primitive/support/aclnn/grad facts are inherited through `construct`, not directly defined by the class itself.",
        "- `runtime_utility` means primitive mapping is intentionally treated as not applicable.",
        "- `terminal_symbol` and `terminal_kind` explain the final static resolution endpoint in the evidence table, for example `auto_generate_function`, `ops_wrapper`, `c_expression_instance`, `primitive_instance`, `func_op`.",
        "- `prelude_calls` in the evidence table records helper/setup calls seen before the terminal operator call, such as seed generation or dtype preparation.",
        "- Prelude helper primitives must not block backend support when a wrapper has a proven single terminal primitive. Example: `adaptive_avg_pool2d_ext` may call `shape_(input)` only to normalize `output_size=None`; the support primitive remains `AdaptiveAvgPool2DExt`, not `AdaptiveAvgPool2DExt + Shape`.",
        "- Pure Python prelude such as `validator.check_value_type`, `isinstance`, `tuple(...)`, or `None` normalization is not an operator chain and must not turn a closed terminal primitive into `unresolved_composite_chain`.",
        "- Frontend guards and constructor setup are not backend support terminals: `F.isconstant`, `Validator/validator.check_*`, `Parameter`, `Tensor`, and `initializer` calls may explain wrapper/setup behavior, but must not enter `support_targets` or block a proven operator-producing chain. Primitive members used by `construct` remain real support targets.",
        "- `dispatch.enable=True` only means an adapter layer is generated. It is not a final backend support conclusion.",
        "- PYNATIVE support combines `dispatch`, CPU/GPU kernel registration, and Ascend pyboost/aclnn evidence, but a backend is `yes` only if the real execution chain closes on that backend.",
        "- GRAPH `jit_level='O0'` support is judged separately: Ascend uses KBK aclnn evidence; CPU/GPU use kernel factory or fallback evidence. Fallback only closes the specific terminal primitive that uses it.",
        "- If `dispatch.{platform}=None`, that platform adapter is not generated and the support result should prefer `no`.",
        "- The current no-`dispatch` direct-support rule is intentionally narrow: only single-primitive `real_terminal` APIs may use Ascend adapter + CPU/GPU kernel factory closure. Do not generalize it to multi-primitive, overload, or scenario-dependent chains.",
    ]
    return "\n".join(lines) + "\n"


def initialize_sqlite_pragmas(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        PRAGMA journal_mode = OFF;
        PRAGMA synchronous = OFF;
        PRAGMA temp_store = MEMORY;
        PRAGMA foreign_keys = OFF;
        """
    )


def drop_sqlite_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        DROP TABLE IF EXISTS review_bucket_item;
        DROP TABLE IF EXISTS api_evidence;
        DROP TABLE IF EXISTS api_grad_impl;
        DROP TABLE IF EXISTS api_grad_primitive;
        DROP TABLE IF EXISTS api_dispatch_detail;
        DROP TABLE IF EXISTS api_aclnn_interface;
        DROP TABLE IF EXISTS api_flag;
        DROP TABLE IF EXISTS api_path;
        DROP TABLE IF EXISTS api_relation;
        DROP TABLE IF EXISTS api_support_target;
        DROP TABLE IF EXISTS api_primitive;
        DROP TABLE IF EXISTS api_support;
        DROP TABLE IF EXISTS api;
        DROP TABLE IF EXISTS build_stats;
        DROP TABLE IF EXISTS source_repository;
        DROP TABLE IF EXISTS schema_meta;
        """
    )


def create_sqlite_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        PRAGMA foreign_keys = ON;

        CREATE TABLE schema_meta (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            schema_version TEXT NOT NULL,
            generator_name TEXT NOT NULL,
            generator_version TEXT NOT NULL,
            generated_at TEXT NOT NULL,
            source_mode TEXT NOT NULL,
            source_repo_url TEXT NOT NULL,
            source_branch TEXT NOT NULL,
            source_commit TEXT NOT NULL,
            mindspore_version_hint TEXT NOT NULL,
            generated_after_gen_ops INTEGER NOT NULL,
            repo_root_hint TEXT NOT NULL,
            api_count INTEGER NOT NULL
        );

        CREATE TABLE source_repository (
            repo_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            repo_url TEXT NOT NULL,
            branch TEXT NOT NULL,
            commit_hash TEXT NOT NULL,
            source_type TEXT NOT NULL
        );

        CREATE TABLE build_stats (
            stat_key TEXT PRIMARY KEY,
            stat_value INTEGER NOT NULL
        );

        CREATE TABLE api (
            api_id INTEGER PRIMARY KEY,
            api_name TEXT NOT NULL UNIQUE,
            category TEXT NOT NULL,
            api_level TEXT NOT NULL,
            trust_level TEXT NOT NULL,
            fact_origin TEXT NOT NULL,
            call_chain_kind TEXT NOT NULL,
            resolution_kind TEXT NOT NULL,
            implementation_type TEXT NOT NULL,
            support_reason_kind TEXT NOT NULL,
            alias_of TEXT NOT NULL DEFAULT '',
            unknown_reason TEXT NOT NULL DEFAULT '',
            grad_mode TEXT NOT NULL,
            grad_differentiable TEXT NOT NULL,
            aclnn_mode TEXT NOT NULL,
            aclnn_path_kind TEXT NOT NULL,
            terminal_symbol TEXT NOT NULL DEFAULT '',
            terminal_kind TEXT NOT NULL DEFAULT '',
            execution_entry TEXT NOT NULL DEFAULT '',
            func_op_is_func_op INTEGER NOT NULL DEFAULT 0,
            summary TEXT NOT NULL
        );

        CREATE TABLE api_support (
            api_id INTEGER NOT NULL,
            exec_mode TEXT NOT NULL,
            backend TEXT NOT NULL,
            support_state TEXT NOT NULL,
            PRIMARY KEY (api_id, exec_mode, backend),
            FOREIGN KEY (api_id) REFERENCES api(api_id) ON DELETE CASCADE,
            CHECK (exec_mode IN ('pynative', 'graph_kbk_o0')),
            CHECK (backend IN ('ascend', 'cpu', 'gpu')),
            CHECK (support_state IN ('yes', 'no', 'unknown'))
        );

        CREATE TABLE api_primitive (
            api_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            ordinal INTEGER NOT NULL,
            primitive_name TEXT NOT NULL,
            PRIMARY KEY (api_id, role, ordinal),
            FOREIGN KEY (api_id) REFERENCES api(api_id) ON DELETE CASCADE,
            CHECK (role IN ('support_terminal', 'possible', 'func_expanded'))
        );

        CREATE TABLE api_support_target (
            api_id INTEGER NOT NULL,
            ordinal INTEGER NOT NULL,
            primitive_name TEXT NOT NULL,
            api_def_name TEXT NOT NULL DEFAULT '',
            op_yaml TEXT NOT NULL DEFAULT '',
            op_def_path TEXT NOT NULL DEFAULT '',
            origin_kind TEXT NOT NULL,
            PRIMARY KEY (api_id, ordinal),
            FOREIGN KEY (api_id) REFERENCES api(api_id) ON DELETE CASCADE,
            CHECK (origin_kind IN (
                'terminal_call',
                'effective_call',
                'prelude_call',
                'func_expansion',
                'overload_branch',
                'scenario_branch',
                'functional_overload_bridge'
            ))
        );

        CREATE TABLE api_relation (
            api_id INTEGER NOT NULL,
            relation_kind TEXT NOT NULL,
            ordinal INTEGER NOT NULL,
            value_text TEXT NOT NULL,
            PRIMARY KEY (api_id, relation_kind, ordinal),
            FOREIGN KEY (api_id) REFERENCES api(api_id) ON DELETE CASCADE,
            CHECK (relation_kind IN ('composed_of', 'prelude_call', 'execution_chain', 'resolved_symbol_chain', 'alias_chain', 'branching_note'))
        );

        CREATE TABLE api_path (
            api_id INTEGER NOT NULL,
            path_kind TEXT NOT NULL,
            ordinal INTEGER NOT NULL,
            path TEXT NOT NULL,
            anchor TEXT NOT NULL DEFAULT '',
            PRIMARY KEY (api_id, path_kind, ordinal),
            FOREIGN KEY (api_id) REFERENCES api(api_id) ON DELETE CASCADE,
            CHECK (path_kind IN (
                'api_def_path',
                'dispatch_path',
                'implementation_path',
                'op_def_path',
                'kernel_path_pynative_ascend',
                'kernel_path_pynative_cpu',
                'kernel_path_pynative_gpu',
                'kernel_path_graph_kbk_o0_ascend',
                'kernel_path_graph_kbk_o0_cpu',
                'kernel_path_graph_kbk_o0_gpu',
                'infer_path'
            ))
        );

        CREATE TABLE api_flag (
            api_id INTEGER NOT NULL,
            flag TEXT NOT NULL,
            ordinal INTEGER NOT NULL,
            PRIMARY KEY (api_id, flag),
            FOREIGN KEY (api_id) REFERENCES api(api_id) ON DELETE CASCADE
        );

        CREATE TABLE api_aclnn_interface (
            api_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            ordinal INTEGER NOT NULL,
            interface_name TEXT NOT NULL,
            PRIMARY KEY (api_id, role, ordinal),
            FOREIGN KEY (api_id) REFERENCES api(api_id) ON DELETE CASCADE,
            CHECK (role IN ('direct', 'effective'))
        );

        CREATE TABLE api_dispatch_detail (
            api_id INTEGER NOT NULL,
            ordinal INTEGER NOT NULL,
            api_def_name TEXT NOT NULL DEFAULT '',
            op_yaml TEXT NOT NULL DEFAULT '',
            dispatch_state TEXT NOT NULL,
            ascend_value TEXT,
            cpu_value TEXT,
            gpu_value TEXT,
            PRIMARY KEY (api_id, ordinal),
            FOREIGN KEY (api_id) REFERENCES api(api_id) ON DELETE CASCADE
        );

        CREATE TABLE api_grad_impl (
            api_id INTEGER NOT NULL,
            ordinal INTEGER NOT NULL,
            primitive_name TEXT NOT NULL,
            kind TEXT NOT NULL,
            path TEXT NOT NULL,
            anchor TEXT NOT NULL,
            scope_kind TEXT NOT NULL,
            PRIMARY KEY (api_id, ordinal),
            FOREIGN KEY (api_id) REFERENCES api(api_id) ON DELETE CASCADE
        );

        CREATE TABLE api_grad_primitive (
            api_id INTEGER NOT NULL,
            ordinal INTEGER NOT NULL,
            primitive_name TEXT NOT NULL,
            origin_kind TEXT NOT NULL,
            PRIMARY KEY (api_id, ordinal),
            FOREIGN KEY (api_id) REFERENCES api(api_id) ON DELETE CASCADE
        );

        CREATE TABLE api_evidence (
            evidence_id INTEGER PRIMARY KEY,
            api_id INTEGER NOT NULL,
            domain TEXT NOT NULL,
            exec_mode TEXT,
            backend TEXT,
            primitive_name TEXT NOT NULL DEFAULT '',
            path TEXT NOT NULL,
            kind TEXT NOT NULL,
            anchor TEXT NOT NULL,
            summary TEXT NOT NULL,
            ordinal INTEGER NOT NULL,
            extra_json TEXT NOT NULL DEFAULT '{}',
            FOREIGN KEY (api_id) REFERENCES api(api_id) ON DELETE CASCADE,
            CHECK (domain IN ('export_source', 'support', 'aclnn', 'func_op_expansion', 'grad'))
        );

        CREATE TABLE review_bucket_item (
            bucket TEXT NOT NULL,
            ordinal INTEGER NOT NULL,
            api_id INTEGER NOT NULL,
            summary TEXT NOT NULL,
            PRIMARY KEY (bucket, ordinal),
            FOREIGN KEY (api_id) REFERENCES api(api_id) ON DELETE CASCADE
        );

        CREATE INDEX idx_api_level ON api(api_level);
        CREATE INDEX idx_api_impl_type ON api(implementation_type);
        CREATE INDEX idx_api_unknown_reason ON api(unknown_reason);
        CREATE INDEX idx_support_lookup ON api_support(exec_mode, backend, support_state);
        CREATE INDEX idx_primitive_name ON api_primitive(primitive_name, role);
        CREATE INDEX idx_support_target_primitive ON api_support_target(primitive_name, origin_kind);
        CREATE INDEX idx_relation_value ON api_relation(relation_kind, value_text);
        CREATE INDEX idx_path_value ON api_path(path_kind, path);
        CREATE INDEX idx_evidence_lookup ON api_evidence(domain, exec_mode, backend, primitive_name);
        """
    )


def _insert_relation_rows(conn: sqlite3.Connection, api_id: int, relation_kind: str, values: list[str]) -> None:
    conn.executemany(
        "INSERT INTO api_relation(api_id, relation_kind, ordinal, value_text) VALUES (?, ?, ?, ?)",
        [(api_id, relation_kind, ordinal, value) for ordinal, value in enumerate(values)],
    )


def _insert_path_rows(conn: sqlite3.Connection, api_id: int, path_kind: str, values: list[Any]) -> None:
    rows = []
    for ordinal, value in enumerate(values):
        if isinstance(value, PathEntry):
            rows.append((api_id, path_kind, ordinal, value.path, value.anchor))
        else:
            rows.append((api_id, path_kind, ordinal, str(value), ""))
    conn.executemany(
        "INSERT INTO api_path(api_id, path_kind, ordinal, path, anchor) VALUES (?, ?, ?, ?, ?)",
        rows,
    )


def _insert_path_hint_rows(conn: sqlite3.Connection, api_id: int, path_hints: dict[str, Any]) -> None:
    _insert_path_rows(conn, api_id, "api_def_path", list(path_hints.get("api_def_paths", [])))
    _insert_path_rows(conn, api_id, "dispatch_path", list(path_hints.get("dispatch_paths", [])))
    _insert_path_rows(conn, api_id, "implementation_path", list(path_hints.get("implementation_paths", [])))
    _insert_path_rows(conn, api_id, "op_def_path", list(path_hints.get("op_def_paths", [])))
    _insert_path_rows(conn, api_id, "infer_path", list(path_hints.get("infer_paths", [])))
    kernel_paths = path_hints.get("kernel_paths", {})
    for exec_mode in ("pynative", "graph_kbk_o0"):
        for backend in ("ascend", "cpu", "gpu"):
            _insert_path_rows(
                conn,
                api_id,
                f"kernel_path_{exec_mode}_{backend}",
                list(kernel_paths.get(exec_mode, {}).get(backend, [])),
            )


def _path_list_from_db(conn: sqlite3.Connection, query: str, params: tuple[Any, ...]) -> list[Any]:
    """Read path+anchor rows.  Return ``str`` when anchor is empty, ``dict`` otherwise."""
    result: list[Any] = []
    for row in conn.execute(query, params).fetchall():
        path = str(row[0])
        anchor = str(row[1]) if len(row) > 1 else ""
        if anchor:
            result.append({"path": path, "anchor": anchor})
        else:
            result.append(path)
    return result


def _path_hints_from_db(conn: sqlite3.Connection, api_id: int) -> dict[str, Any]:
    result = _empty_path_hints()
    result["api_def_paths"] = _path_list_from_db(
        conn,
        "SELECT path, anchor FROM api_path WHERE api_id = ? AND path_kind = 'api_def_path' ORDER BY ordinal",
        (api_id,),
    )
    result["dispatch_paths"] = _path_list_from_db(
        conn,
        "SELECT path, anchor FROM api_path WHERE api_id = ? AND path_kind = 'dispatch_path' ORDER BY ordinal",
        (api_id,),
    )
    result["implementation_paths"] = _path_list_from_db(
        conn,
        "SELECT path, anchor FROM api_path WHERE api_id = ? AND path_kind = 'implementation_path' ORDER BY ordinal",
        (api_id,),
    )
    result["op_def_paths"] = _path_list_from_db(
        conn,
        "SELECT path, anchor FROM api_path WHERE api_id = ? AND path_kind = 'op_def_path' ORDER BY ordinal",
        (api_id,),
    )
    result["infer_paths"] = _path_list_from_db(
        conn,
        "SELECT path, anchor FROM api_path WHERE api_id = ? AND path_kind = 'infer_path' ORDER BY ordinal",
        (api_id,),
    )
    for exec_mode in ("pynative", "graph_kbk_o0"):
        for backend in ("ascend", "cpu", "gpu"):
            result["kernel_paths"][exec_mode][backend] = _path_list_from_db(
                conn,
                f"SELECT path, anchor FROM api_path WHERE api_id = ? AND path_kind = 'kernel_path_{exec_mode}_{backend}' ORDER BY ordinal",
                (api_id,),
            )
    return result


def _insert_primitive_rows(conn: sqlite3.Connection, api_id: int, role: str, values: list[str]) -> None:
    conn.executemany(
        "INSERT INTO api_primitive(api_id, role, ordinal, primitive_name) VALUES (?, ?, ?, ?)",
        [(api_id, role, ordinal, value) for ordinal, value in enumerate(values)],
    )


def validate_record_evidence_consistency(record: dict[str, Any], evidence: dict[str, Any]) -> None:
    if record.get("call_chain_kind") == "functional_overload":
        assert not record.get("primitive"), record["api"]
        assert record.get("possible_primitives"), record["api"]
        support_targets = list(evidence.get("primitive_sources", []))
        assert support_targets, record["api"]
        assert all(item.get("origin_kind") == "overload_branch" for item in support_targets), record["api"]
    grad = record.get("grad", {})
    if grad.get("mode") in {"unknown", "not_applicable"} and not grad.get("backward_primitives"):
        pass
    for item in grad.get("impl", []):
        scope_kind = str(item.get("scope_kind", ""))
        assert scope_kind in {"real_terminal", "func_expansion", "overload_branch", "python_branch", "composite_chain"}, record["api"]
    if record.get("resolution_kind") == "func_expansion":
        allowed = set(record.get("func_op_expands_to", []))
        for item in grad.get("impl", []):
            if str(item.get("kind")) == "cpp_bprop_builder":
                assert str(item.get("primitive", "")) in allowed, record["api"]
        for primitive in grad.get("backward_primitives", []):
            if primitive in allowed:
                continue
    if record.get("call_chain_kind") == "functional_overload":
        allowed = set(record.get("possible_primitives", []))
        for item in grad.get("impl", []):
            if str(item.get("kind")) == "cpp_bprop_builder":
                assert str(item.get("primitive", "")) in allowed, record["api"]
        return
    if record.get("resolution_kind") == "func_expansion":
        support_targets = [item for item in evidence.get("primitive_sources", []) if item.get("primitive")]
        assert support_targets, record["api"]
        return
    if record.get("unknown_reason") in {"terminal_symbol_unresolved", "ambiguous_terminal_mapping"} or (
        record.get("unknown_reason") == "unresolved_composite_chain" and record.get("support_reason_kind") == "unknown"
    ):
        if record.get("unknown_reason") == "terminal_symbol_unresolved":
            assert not record.get("primitive"), record["api"]
            assert not evidence.get("primitive_sources"), record["api"]
        for mode in ("pynative_support", "graph_kbk_o0_support"):
            assert all(record[mode][backend] == "unknown" for backend in ("ascend", "cpu", "gpu")), record["api"]
        return
    if record.get("support_reason_kind") == "not_applicable":
        return
    support_primitives = list(record.get("primitive", []))
    if not support_primitives:
        return
    support_targets = [item for item in evidence.get("primitive_sources", []) if item.get("primitive")]
    assert support_targets, record["api"]
    target_primitives = uniq([str(item["primitive"]) for item in support_targets])
    assert target_primitives == support_primitives, record["api"]


def _insert_evidence_rows(
    conn: sqlite3.Connection,
    api_id: int,
    domain: str,
    items: list[dict[str, Any]],
    *,
    exec_mode: str | None = None,
    backend: str | None = None,
    primitive_name: str = "",
    extra_factory: Optional[callable] = None,
) -> None:
    rows = []
    for ordinal, item in enumerate(items):
        extra_json = json_dump_text(extra_factory(item) if extra_factory is not None else {})
        rows.append(
            (
                api_id,
                domain,
                exec_mode,
                backend,
                primitive_name or str(item.get("primitive", "")),
                str(item.get("path", "")),
                str(item.get("kind", "")),
                str(item.get("anchor", "")),
                str(item.get("summary", "")),
                ordinal,
                extra_json,
            )
        )
    conn.executemany(
        """
        INSERT INTO api_evidence(
            api_id, domain, exec_mode, backend, primitive_name, path, kind, anchor, summary, ordinal, extra_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )


def write_sqlite_snapshot(
    db_path: Path,
    main_payload: dict[str, Any],
    evidence_payload: dict[str, Any],
    review_payload: dict[str, Any],
    deterministic: bool = False,
) -> None:
    conn = sqlite3.connect(db_path)
    try:
        initialize_sqlite_pragmas(conn)
        drop_sqlite_schema(conn)
        conn.execute("VACUUM")
        create_sqlite_schema(conn)
        meta = main_payload["meta"]
        conn.execute(
            """
            INSERT INTO schema_meta(
                id, schema_version, generator_name, generator_version, generated_at,
                source_mode, source_repo_url, source_branch, source_commit,
                mindspore_version_hint, generated_after_gen_ops, repo_root_hint, api_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                1,
                str(meta["index_schema_version"]),
                str(meta["generator_name"]),
                str(meta["generator_version"]),
                str(meta["generated_at"]),
                str(meta["source_mode"]),
                str(meta["source_repo_url"]),
                str(meta["source_branch"]),
                str(meta["source_commit"]),
                str(meta["mindspore_version_hint"]),
                1 if meta["generated_after_gen_ops"] else 0,
                str(meta["repo_root_hint"]),
                int(meta["api_count"]),
            ),
        )
        conn.executemany(
            """
            INSERT INTO source_repository(repo_id, name, repo_url, branch, commit_hash, source_type)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    idx + 1,
                    str(item.get("name", "")),
                    str(item.get("repo_url", "")),
                    str(item.get("branch", "")),
                    str(item.get("commit", "")),
                    str(item.get("source_type", "")),
                )
                for idx, item in enumerate(meta.get("source_repositories", []))
            ],
        )
        conn.executemany(
            "INSERT INTO build_stats(stat_key, stat_value) VALUES (?, ?)",
            [(key, int(value)) for key, value in main_payload["stats"].items()],
        )

        evidence_by_api = {item["api"]: item for item in evidence_payload["apis"]}
        api_id_by_name: dict[str, int] = {}
        for api_id, record in enumerate(main_payload["apis"], start=1):
            api_id_by_name[str(record["api"])] = api_id
            evidence = evidence_by_api.get(record["api"], {})
            validate_record_evidence_consistency(record, evidence)
            conn.execute(
                """
                INSERT INTO api(
                    api_id, api_name, category, api_level, trust_level, fact_origin,
                    call_chain_kind, resolution_kind,
                    implementation_type, support_reason_kind,
                    alias_of, unknown_reason, grad_mode, grad_differentiable, aclnn_mode, aclnn_path_kind,
                    terminal_symbol, terminal_kind, execution_entry, func_op_is_func_op, summary
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    api_id,
                    str(record["api"]),
                    str(record["category"]),
                    str(record["api_level"]),
                    str(record["trust_level"]),
                    str(record["fact_origin"]),
                    str(record.get("call_chain_kind", "")),
                    str(record.get("resolution_kind", "")),
                    str(record["implementation_type"]),
                    str(record["support_reason_kind"]),
                    str(record.get("alias_of", "")),
                    str(record.get("unknown_reason", "")),
                    str(record["grad"]["mode"]),
                    str(record["grad"].get("differentiable", "unknown")),
                    str(record["aclnn"]["mode"]),
                    str(record["aclnn"]["path_kind"]),
                    str(evidence.get("terminal_symbol", "")),
                    str(evidence.get("terminal_kind", "")),
                    str(evidence.get("execution_entry", "")),
                    1 if evidence.get("func_op", {}).get("is_func_op") else 0,
                    str(record["summary"]),
                ),
            )
            for exec_mode, field_name in (("pynative", "pynative_support"), ("graph_kbk_o0", "graph_kbk_o0_support")):
                conn.executemany(
                    "INSERT INTO api_support(api_id, exec_mode, backend, support_state) VALUES (?, ?, ?, ?)",
                    [(api_id, exec_mode, backend, str(record[field_name][backend])) for backend in ("ascend", "cpu", "gpu")],
                )
            _insert_primitive_rows(conn, api_id, "support_terminal", list(record.get("primitive", [])))
            _insert_primitive_rows(conn, api_id, "possible", list(record.get("possible_primitives", [])))
            _insert_primitive_rows(conn, api_id, "func_expanded", list(record.get("func_op_expands_to", [])))
            _insert_relation_rows(conn, api_id, "composed_of", list(record.get("composed_of", [])))
            _insert_relation_rows(conn, api_id, "branching_note", list(record.get("branching_notes", [])))
            _insert_path_hint_rows(conn, api_id, dict(record.get("path_hints", {})))
            for ordinal, flag in enumerate(record.get("flags", [])):
                conn.execute("INSERT INTO api_flag(api_id, flag, ordinal) VALUES (?, ?, ?)", (api_id, str(flag), ordinal))
            for role, items in (("direct", record["aclnn"].get("interfaces", [])), ("effective", record["aclnn"].get("effective_interfaces", []))):
                conn.executemany(
                    "INSERT INTO api_aclnn_interface(api_id, role, ordinal, interface_name) VALUES (?, ?, ?, ?)",
                    [(api_id, role, ordinal, str(item)) for ordinal, item in enumerate(items)],
                )
            for ordinal, item in enumerate(record["grad"].get("impl", [])):
                conn.execute(
                    """
                    INSERT INTO api_grad_impl(api_id, ordinal, primitive_name, kind, path, anchor, scope_kind)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        api_id,
                        ordinal,
                        str(item.get("primitive", "")),
                        str(item.get("kind", "")),
                        str(item.get("path", "")),
                        str(item.get("anchor", "")),
                        str(item.get("scope_kind", "real_terminal")),
                    ),
                )
            for ordinal, primitive in enumerate(record["grad"].get("backward_primitives", [])):
                conn.execute(
                    """
                    INSERT INTO api_grad_primitive(api_id, ordinal, primitive_name, origin_kind)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        api_id,
                        ordinal,
                        str(primitive),
                        "backward_impl",
                    ),
                )

            support_targets = list(evidence.get("primitive_sources", []))
            if record.get("func_op_expands_to"):
                support_targets.extend(
                    [
                        {
                            "primitive": primitive,
                            "api_def": "",
                            "op_yaml": "",
                            "op_def_path": "",
                            "origin_kind": "func_expansion",
                        }
                        for primitive in record["func_op_expands_to"]
                    ]
                )
            for ordinal, item in enumerate(support_targets):
                conn.execute(
                    """
                    INSERT INTO api_support_target(
                        api_id, ordinal, primitive_name, api_def_name, op_yaml, op_def_path, origin_kind
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        api_id,
                        ordinal,
                        str(item.get("primitive", "")),
                        str(item.get("api_def", "")),
                        str(item.get("op_yaml", "")),
                        str(item.get("op_def_path", "")),
                        str(item.get("origin_kind", "func_expansion" if item.get("origin_kind") == "func_expansion" else "effective_call")),
                    ),
                )
            _insert_relation_rows(conn, api_id, "prelude_call", list(evidence.get("prelude_calls", [])))
            _insert_relation_rows(conn, api_id, "execution_chain", list(evidence.get("execution_chain", [])))
            _insert_relation_rows(conn, api_id, "resolved_symbol_chain", list(evidence.get("resolved_symbol_chain", [])))
            _insert_relation_rows(conn, api_id, "alias_chain", list(evidence.get("alias_chain", [])))
            for ordinal, item in enumerate(evidence.get("dispatch_detail", [])):
                conn.execute(
                    """
                    INSERT INTO api_dispatch_detail(
                        api_id, ordinal, api_def_name, op_yaml, dispatch_state, ascend_value, cpu_value, gpu_value
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        api_id,
                        ordinal,
                        str(item.get("api_def", "")),
                        str(item.get("op_yaml", "")),
                        str(item.get("dispatch", "")),
                        None if "ascend" not in item else str(item.get("ascend")),
                        None if "cpu" not in item else str(item.get("cpu")),
                        None if "gpu" not in item else str(item.get("gpu")),
                    ),
                )
            _insert_evidence_rows(conn, api_id, "export_source", list(evidence.get("source", [])))
            support_evidence = evidence.get("support_evidence", {})
            for exec_mode in ("pynative", "graph_kbk_o0"):
                for backend in ("ascend", "cpu", "gpu"):
                    _insert_evidence_rows(
                        conn,
                        api_id,
                        "support",
                        list(support_evidence.get(exec_mode, {}).get(backend, [])),
                        exec_mode=exec_mode,
                        backend=backend,
                    )
            _insert_evidence_rows(conn, api_id, "aclnn", list(evidence.get("aclnn_evidence", [])))
            _insert_evidence_rows(conn, api_id, "func_op_expansion", list(evidence.get("func_op_expansion_evidence", [])))
            _insert_evidence_rows(
                conn,
                api_id,
                "grad",
                list(record["grad"].get("impl", [])),
                extra_factory=lambda item: {"primitive": str(item.get("primitive", ""))},
            )

        for bucket, items in review_payload["queue"].items():
            for ordinal, item in enumerate(items):
                conn.execute(
                    """
                    INSERT INTO review_bucket_item(bucket, ordinal, api_id, summary)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        str(bucket),
                        ordinal,
                        api_id_by_name[str(item["api"])],
                        str(item.get("summary", "")),
                    ),
                )
        conn.commit()
    finally:
        conn.close()
    if deterministic:
        normalize_sqlite_header_for_determinism(db_path)


def normalize_sqlite_header_for_determinism(db_path: Path) -> None:
    # SQLite updates a few header counters on each rebuild even when table content is identical.
    # Normalizing them after closing the connection keeps deterministic snapshots byte-stable.
    fixed_value = (1).to_bytes(4, "big")
    with db_path.open("r+b") as handle:
        for offset in (24, 40, 92):
            handle.seek(offset)
            handle.write(fixed_value)


def _support_state_from_db(conn: sqlite3.Connection, api_id: int, exec_mode: str) -> dict[str, str]:
    rows = conn.execute(
        "SELECT backend, support_state FROM api_support WHERE api_id = ? AND exec_mode = ? ORDER BY backend",
        (api_id, exec_mode),
    ).fetchall()
    result = support_state()
    for row in rows:
        result[str(row["backend"])] = str(row["support_state"])
    return result


def _scalar_list_from_db(conn: sqlite3.Connection, query: str, params: tuple[Any, ...]) -> list[str]:
    return [str(row[0]) for row in conn.execute(query, params).fetchall()]


def load_main_payload_from_db(db_path: Path) -> dict[str, Any]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        meta_row = conn.execute("SELECT * FROM schema_meta WHERE id = 1").fetchone()
        if meta_row is None:
            raise RuntimeError(f"schema_meta is empty: {db_path}")
        source_repositories = [
            {
                "name": str(row["name"]),
                "repo_url": str(row["repo_url"]),
                "branch": str(row["branch"]),
                "commit": str(row["commit_hash"]),
                "source_type": str(row["source_type"]),
            }
            for row in conn.execute("SELECT * FROM source_repository ORDER BY repo_id").fetchall()
        ]
        stats = {str(row["stat_key"]): int(row["stat_value"]) for row in conn.execute("SELECT stat_key, stat_value FROM build_stats").fetchall()}
        ordered_stats = {}
        for key in ("operator_api", "wrapper_api", "module_api", "needs_manual_review"):
            if key in stats:
                ordered_stats[key] = stats[key]
        for key in sorted(stats):
            if key not in ordered_stats:
                ordered_stats[key] = stats[key]

        records = []
        for row in conn.execute("SELECT * FROM api ORDER BY api_name").fetchall():
            api_id = int(row["api_id"])
            aclnn = {
                "mode": str(row["aclnn_mode"]),
                "interfaces": _scalar_list_from_db(
                    conn,
                    "SELECT interface_name FROM api_aclnn_interface WHERE api_id = ? AND role = 'direct' ORDER BY ordinal",
                    (api_id,),
                ),
                "effective_interfaces": _scalar_list_from_db(
                    conn,
                    "SELECT interface_name FROM api_aclnn_interface WHERE api_id = ? AND role = 'effective' ORDER BY ordinal",
                    (api_id,),
                ),
                "path_kind": str(row["aclnn_path_kind"]),
            }
            grad_impl = [
                {
                    "primitive": str(item["primitive_name"]),
                    "kind": str(item["kind"]),
                    "path": str(item["path"]),
                    "anchor": str(item["anchor"]),
                    "scope_kind": str(item["scope_kind"]),
                }
                for item in conn.execute(
                    "SELECT primitive_name, kind, path, anchor, scope_kind FROM api_grad_impl WHERE api_id = ? ORDER BY ordinal",
                    (api_id,),
                ).fetchall()
            ]
            records.append(
                {
                    "api": str(row["api_name"]),
                    "category": str(row["category"]),
                    "api_level": str(row["api_level"]),
                    "trust_level": str(row["trust_level"]),
                    "fact_origin": str(row["fact_origin"]),
                    "call_chain_kind": str(row["call_chain_kind"]),
                    "resolution_kind": str(row["resolution_kind"]),
                    "implementation_type": str(row["implementation_type"]),
                    "primitive": _scalar_list_from_db(
                        conn,
                        "SELECT primitive_name FROM api_primitive WHERE api_id = ? AND role = 'support_terminal' ORDER BY ordinal",
                        (api_id,),
                    ),
                    "possible_primitives": _scalar_list_from_db(
                        conn,
                        "SELECT primitive_name FROM api_primitive WHERE api_id = ? AND role = 'possible' ORDER BY ordinal",
                        (api_id,),
                    ),
                    "func_op_expands_to": _scalar_list_from_db(
                        conn,
                        "SELECT primitive_name FROM api_primitive WHERE api_id = ? AND role = 'func_expanded' ORDER BY ordinal",
                        (api_id,),
                    ),
                    "support_reason_kind": str(row["support_reason_kind"]),
                    "pynative_support": _support_state_from_db(conn, api_id, "pynative"),
                    "graph_kbk_o0_support": _support_state_from_db(conn, api_id, "graph_kbk_o0"),
                    "aclnn": aclnn,
                    "grad": {
                        "mode": str(row["grad_mode"]),
                        "differentiable": str(row["grad_differentiable"]),
                        "backward_primitives": _scalar_list_from_db(
                            conn,
                            "SELECT primitive_name FROM api_grad_primitive WHERE api_id = ? ORDER BY ordinal",
                            (api_id,),
                        ),
                        "impl": grad_impl,
                    },
                    "composed_of": _scalar_list_from_db(
                        conn,
                        "SELECT value_text FROM api_relation WHERE api_id = ? AND relation_kind = 'composed_of' ORDER BY ordinal",
                        (api_id,),
                    ),
                    "branching_notes": _scalar_list_from_db(
                        conn,
                        "SELECT value_text FROM api_relation WHERE api_id = ? AND relation_kind = 'branching_note' ORDER BY ordinal",
                        (api_id,),
                    ),
                    "alias_of": str(row["alias_of"]),
                    "path_hints": _path_hints_from_db(conn, api_id),
                    "flags": _scalar_list_from_db(conn, "SELECT flag FROM api_flag WHERE api_id = ? ORDER BY ordinal", (api_id,)),
                    "unknown_reason": str(row["unknown_reason"]),
                    "summary": str(row["summary"]),
                }
            )

        return {
            "meta": {
                "generated_at": str(meta_row["generated_at"]),
                "generator_name": str(meta_row["generator_name"]),
                "generator_version": str(meta_row["generator_version"]),
                "index_schema_version": str(meta_row["schema_version"]),
                "source_mode": str(meta_row["source_mode"]),
                "source_repo_url": str(meta_row["source_repo_url"]),
                "source_branch": str(meta_row["source_branch"]),
                "source_commit": str(meta_row["source_commit"]),
                "source_repository_count": len(source_repositories),
                "source_repositories": source_repositories,
                "mindspore_version_hint": str(meta_row["mindspore_version_hint"]),
                "generated_after_gen_ops": bool(meta_row["generated_after_gen_ops"]),
                "repo_root_hint": str(meta_row["repo_root_hint"]),
                "api_count": int(meta_row["api_count"]),
            },
            "stats": ordered_stats,
            "apis": records,
        }
    finally:
        conn.close()


def run_gen_ops(repo_root: Path) -> None:
    subprocess.run([sys.executable, str(repo_root / GEN_OPS)], check=True, cwd=repo_root)


def current_timestamp(*, deterministic: bool = False) -> str:
    if deterministic:
        return DETERMINISTIC_TIMESTAMP
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


def _force_remove_readonly(func, path, _exc_info):
    os.chmod(path, stat.S_IWRITE)
    func(path)


def safe_rmtree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path, onerror=_force_remove_readonly)


def prune_empty_parents(path: Path, *, stop_at: Path) -> None:
    current = path
    stop_at = stop_at.resolve()
    while current.exists() and current.resolve() != stop_at:
        try:
            current.rmdir()
        except OSError:
            break
        current = current.parent


def unlink_with_retries(path: Path, *, attempts: int = 50, delay_sec: float = 0.2) -> None:
    last_error: OSError | None = None
    for _ in range(attempts):
        try:
            if path.exists():
                path.unlink()
            return
        except OSError as exc:
            last_error = exc
            time.sleep(delay_sec)
    if last_error is not None:
        raise last_error


def run_workspace_root(base_root: Path, *, keep_workspace: bool) -> Path:
    if keep_workspace:
        return base_root
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
    return base_root / f"run-{stamp}"


def build_source_repositories(
    repo_root: Path,
    *,
    source_mode: str,
    source_repo_url: str,
    source_branch: str,
    source_commit: str,
) -> list[dict[str, str]]:
    return [
        {
            "name": "mindspore",
            "repo_url": source_repo_url or str(repo_root),
            "branch": source_branch,
            "commit": source_commit,
            "source_type": source_mode,
        }
    ]


def clone_remote_repo(remote_url: str, branch: str | None, workspace_root: Path) -> Path:
    safe_rmtree(workspace_root)
    workspace_root.mkdir(parents=True, exist_ok=True)
    repo_root = workspace_root / "repo"
    cmd = ["git", "-c", "core.longpaths=true", "clone", "--depth", "1", "--no-tags", remote_url, str(repo_root)]
    if branch:
        cmd = [
            "git",
            "-c",
            "core.longpaths=true",
            "clone",
            "--depth",
            "1",
            "--no-tags",
            "--branch",
            branch,
            "--single-branch",
            remote_url,
            str(repo_root),
        ]
    try:
        subprocess.run(cmd, check=True)
    except Exception:
        safe_rmtree(workspace_root)
        prune_empty_parents(workspace_root.parent, stop_at=SCRIPT_DIR)
        raise
    return repo_root


def prepare_repo(args: argparse.Namespace) -> tuple[Path, str, str, str, Path | None]:
    if args.repo and args.branch:
        raise SystemExit("--repo and --branch cannot be used together.")
    if args.repo:
        repo_root = Path(args.repo).resolve()
        if not repo_root.exists():
            raise FileNotFoundError(f"MindSpore repo not found: {repo_root}")
        return repo_root, "local", str(repo_root), repo_branch_hint(repo_root), None
    workspace_root = run_workspace_root(Path(args.workspace_root).resolve(), keep_workspace=args.keep_workspace)
    repo_root = clone_remote_repo(DEFAULT_REMOTE_URL, args.branch, workspace_root)
    return repo_root, "remote", DEFAULT_REMOTE_URL, args.branch or repo_branch_hint(repo_root), workspace_root


def remove_legacy_outputs(out_dir: Path) -> None:
    for legacy_name in ("mint_api_index.jsonl", "mint_api_index.db.tmp", "mint_api_index.db.tmp-journal"):
        legacy = out_dir / legacy_name
        if legacy.exists():
            try:
                legacy.unlink()
            except OSError:
                pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Build compact MindSpore mint API index.")
    parser.add_argument("--repo", help="Path to a local MindSpore repo root.")
    parser.add_argument("--branch", help="Remote branch to clone from the default MindSpore repository.")
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT_DIR), help="Output directory.")
    parser.add_argument("--workspace-root", default=str(DEFAULT_WORKSPACE_ROOT), help="Temporary workspace root.")
    parser.add_argument("--keep-workspace", action="store_true", help="Keep the temporary cloned workspace.")
    parser.add_argument("--skip-gen-ops", action="store_true", help="Skip running gen_ops.py before indexing.")
    parser.add_argument("--with-yaml", action="store_true", help="Generate optional mint_api_index.yaml for manual review.")
    parser.add_argument("--with-evidence", action="store_true", help="Generate mint_api_evidence.yaml.")
    parser.add_argument("--with-review", action="store_true", help="Generate review queue and markdown outputs.")
    parser.add_argument("--with-rulebook", action="store_true", help="Generate mint_api_index_rulebook.md.")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic metadata for reproducible outputs.")
    args = parser.parse_args()

    repo_root, source_mode, source_repo_url, source_branch, workspace_root = prepare_repo(args)
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
        generated_at = current_timestamp(deterministic=args.deterministic)
        repo_commit = repo_commit_hint(repo_root)
        version_hint = mindspore_version_hint(repo_root)
        source_repositories = build_source_repositories(
            repo_root,
            source_mode=source_mode,
            source_repo_url=source_repo_url,
            source_branch=source_branch,
            source_commit=repo_commit,
        )
        base_meta = {
            "generated_at": generated_at,
            "generator_name": GENERATOR_NAME,
            "generator_version": GENERATOR_VERSION,
            "index_schema_version": INDEX_SCHEMA_VERSION,
            "source_mode": source_mode,
            "source_repo_url": source_repo_url,
            "source_branch": source_branch,
            "source_commit": repo_commit,
            "source_repository_count": len(source_repositories),
            "source_repositories": source_repositories,
            "mindspore_version_hint": version_hint,
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
        main_payload = canonicalize_for_yaml(main_payload)
        evidence_payload = canonicalize_for_yaml(evidence_payload)
        review_payload = canonicalize_for_yaml(review_payload)

        db_path = out_dir / DEFAULT_DB_NAME
        write_sqlite_snapshot(
            db_path,
            main_payload,
            evidence_payload,
            review_payload,
            deterministic=args.deterministic,
        )
        files = [str(db_path)]
        if args.with_yaml:
            yaml_dump(canonicalize_for_yaml(load_main_payload_from_db(db_path)), out_dir / "mint_api_index.yaml")
            files.append(str(out_dir / "mint_api_index.yaml"))

        if args.with_evidence:
            yaml_dump(evidence_payload, out_dir / "mint_api_evidence.yaml")
            files.append(str(out_dir / "mint_api_evidence.yaml"))
        if args.with_review:
            yaml_dump(review_payload, out_dir / "mint_api_review_queue.yaml")
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
        if source_mode == "remote" and workspace_root is not None and not args.keep_workspace:
            safe_rmtree(workspace_root)
            prune_empty_parents(workspace_root.parent, stop_at=SCRIPT_DIR)


if __name__ == "__main__":
    main()
