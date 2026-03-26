#!/usr/bin/env python3
import argparse
import ast
import json
from pathlib import Path
from typing import Optional

from perf_common import read_json, write_json


PTA_ENV_FEATURES = ["env_pta_allocator_tune", "env_aclnn_cache"]
INFERENCE_FEATURES = [
    "model_eval",
    "use_cache",
    "pad_token_id",
    "sync_device_timing",
    "warmup_inference",
]


def parse_expr(snippet: str) -> ast.expr:
    return ast.parse(snippet, mode="eval").body


def parse_stmts(snippet: str) -> list[ast.stmt]:
    return ast.parse(snippet).body


def insert_import(module: ast.Module, import_snippet: str) -> None:
    import_node = ast.parse(import_snippet).body[0]
    for node in module.body:
        if ast.dump(node) == ast.dump(import_node):
            return
    insert_at = 0
    if module.body and isinstance(module.body[0], ast.Expr) and isinstance(module.body[0].value, ast.Constant):
        if isinstance(module.body[0].value.value, str):
            insert_at = 1
    while insert_at < len(module.body) and isinstance(module.body[insert_at], (ast.Import, ast.ImportFrom)):
        insert_at += 1
    module.body.insert(insert_at, import_node)


def unique(items: list[str]) -> list[str]:
    seen = set()
    result = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def contains_generate_call(node: ast.AST) -> bool:
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            func = child.func
            if isinstance(func, ast.Attribute) and func.attr == "generate":
                return True
            if isinstance(func, ast.Name) and func.id == "generate":
                return True
    return False


def is_name_assign(stmt: ast.stmt, name: str) -> bool:
    if not isinstance(stmt, ast.Assign):
        return False
    for target in stmt.targets:
        if isinstance(target, ast.Name) and target.id == name:
            return True
    return False


def is_time_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "time"
        and node.func.attr == "time"
    )


def is_start_timer_assign(stmt: ast.stmt) -> bool:
    return is_name_assign(stmt, "start") and isinstance(stmt, ast.Assign) and is_time_call(stmt.value)


def is_gen_time_assign(stmt: ast.stmt) -> bool:
    if not is_name_assign(stmt, "gen_time") or not isinstance(stmt, ast.Assign):
        return False
    value = stmt.value
    return (
        isinstance(value, ast.BinOp)
        and isinstance(value.left, ast.Call)
        and is_time_call(value.left)
        and isinstance(value.op, ast.Sub)
        and isinstance(value.right, ast.Name)
        and value.right.id == "start"
    )


def loop_iter_name(stmt: ast.stmt) -> Optional[str]:
    if isinstance(stmt, ast.For) and isinstance(stmt.iter, ast.Name):
        return stmt.iter.id
    return None


def env_stmt(feature: str) -> Optional[list[ast.stmt]]:
    if feature == "env_pta_allocator_tune":
        return parse_stmts("os.environ.setdefault('PYTORCH_NPU_ALLOC_CONF', 'max_split_size_mb:512')")
    if feature == "env_aclnn_cache":
        return parse_stmts("os.environ.setdefault('ACLNN_CACHE_LIMIT', '100000')")
    return None


def sync_stmt() -> list[ast.stmt]:
    return parse_stmts(
        """
if "torch_npu" in globals() and hasattr(torch_npu, "synchronize"):
    torch_npu.synchronize()
""".strip()
    )


def model_eval_stmt() -> list[ast.stmt]:
    return parse_stmts(
        """
if hasattr(model, "eval"):
    model.eval()
""".strip()
    )


def contiguous_inputs_stmt() -> list[ast.stmt]:
    return parse_stmts(
        """
if isinstance(inputs, dict):
    inputs = {k: (v.contiguous() if hasattr(v, "contiguous") else v) for k, v in inputs.items()}
elif hasattr(inputs, "contiguous"):
    inputs = inputs.contiguous()
""".strip()
    )


def warmup_stmt(iter_name: str) -> list[ast.stmt]:
    return parse_stmts(
        f"""
if {iter_name}:
    _perf_warmup_prompt = {iter_name}[0]
    _perf_warmup_inputs = tokenizer(_perf_warmup_prompt, return_tensors="pt")
    if isinstance(_perf_warmup_inputs, dict) and hasattr(model, "device"):
        _perf_warmup_inputs = {{k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in _perf_warmup_inputs.items()}}
    elif hasattr(_perf_warmup_inputs, "to") and hasattr(model, "device"):
        _perf_warmup_inputs = _perf_warmup_inputs.to(model.device)
    if isinstance(_perf_warmup_inputs, dict):
        for _perf_warmup_idx in range(2):
            _ = model.generate(
                **_perf_warmup_inputs,
                max_new_tokens=8,
                do_sample=False,
                use_cache=True,
                pad_token_id=getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", None),
            )
    if "torch_npu" in globals() and hasattr(torch_npu, "synchronize"):
        torch_npu.synchronize()
""".strip()
    )


def select_features(profile: dict, bottlenecks: dict, hotspot: Optional[dict]) -> tuple[list[str], list[str], list[str]]:
    features: list[str] = []
    rationale: list[str] = []
    metrics_to_watch = ["throughput", "first_prompt_throughput", "steady_state_throughput", "mean_prompt_latency"]

    stack = profile.get("stack")
    workload = profile.get("workload_type")
    primary = ((bottlenecks or {}).get("primary_candidate") or {}).get("name")
    lead_operator = None
    if hotspot and hotspot.get("top_operators"):
        lead_operator = hotspot["top_operators"][0].get("operator")

    if stack == "pta":
        features.extend(PTA_ENV_FEATURES)
        rationale.append("Apply safe PTA runtime environment defaults on the copied script before trying code-local optimizations.")

    if workload == "inference":
        features.extend(INFERENCE_FEATURES)
        rationale.append("Use an inference feature pack that warms the copied script, preserves cache-aware generation, and validates steady-state throughput.")

    if primary in {"host_framework_overhead", "graph_compile"}:
        rationale.append("The primary bottleneck points to launch/compile overhead, so warmup and accurate timing should be tried before lower-signal tweaks.")

    if lead_operator and "transdata" in lead_operator.lower():
        features.append("contiguous_inputs")
        rationale.append("The lead hotspot is a TransData operator, so making token inputs contiguous is a low-risk first trial.")

    if not features:
        features.extend(["model_eval"])
        rationale.append("No targeted pack was available, so apply a minimal copied-script inference-safety feature first.")

    return unique(features), rationale, metrics_to_watch


class FeatureTransformer(ast.NodeTransformer):
    def __init__(self, selected_features: list[str]) -> None:
        self.selected_features = set(selected_features)
        self.applied: set[str] = set()
        self.warmup_inserted = False

    def visit_Module(self, node: ast.Module):  # type: ignore[override]
        if any(feature.startswith("env_") for feature in self.selected_features):
            insert_import(node, "import os")
        node.body = self._transform_body(node.body)
        env_snippets = []
        for feature in ("env_pta_allocator_tune", "env_aclnn_cache"):
            if feature in self.selected_features:
                env_snippets.extend(env_stmt(feature) or [])
                self.applied.add(feature)
        if env_snippets:
            insert_at = 0
            if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant):
                if isinstance(node.body[0].value.value, str):
                    insert_at = 1
            while insert_at < len(node.body) and isinstance(node.body[insert_at], (ast.Import, ast.ImportFrom)):
                insert_at += 1
            node.body[insert_at:insert_at] = env_snippets
        return ast.fix_missing_locations(node)

    def visit_FunctionDef(self, node: ast.FunctionDef):  # type: ignore[override]
        node.body = self._transform_body(node.body)
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):  # type: ignore[override]
        node.body = self._transform_body(node.body)
        return node

    def visit_If(self, node: ast.If):  # type: ignore[override]
        node.body = self._transform_body(node.body)
        node.orelse = self._transform_body(node.orelse)
        return node

    def visit_For(self, node: ast.For):  # type: ignore[override]
        node.body = self._transform_body(node.body)
        node.orelse = self._transform_body(node.orelse)
        return self.generic_visit(node)

    def visit_While(self, node: ast.While):  # type: ignore[override]
        node.body = self._transform_body(node.body)
        node.orelse = self._transform_body(node.orelse)
        return self.generic_visit(node)

    def visit_With(self, node: ast.With):  # type: ignore[override]
        node.body = self._transform_body(node.body)
        return self.generic_visit(node)

    def visit_Try(self, node: ast.Try):  # type: ignore[override]
        node.body = self._transform_body(node.body)
        node.orelse = self._transform_body(node.orelse)
        node.finalbody = self._transform_body(node.finalbody)
        for handler in node.handlers:
            handler.body = self._transform_body(handler.body)
        return self.generic_visit(node)

    def visit_Call(self, node: ast.Call):  # type: ignore[override]
        node = self.generic_visit(node)
        func = node.func
        is_generate = isinstance(func, ast.Attribute) and func.attr == "generate"
        if not is_generate:
            return node

        keyword_names = {item.arg for item in node.keywords if item.arg}
        if "use_cache" in self.selected_features and "use_cache" not in keyword_names:
            node.keywords.append(ast.keyword(arg="use_cache", value=ast.Constant(value=True)))
            self.applied.add("use_cache")
        if "pad_token_id" in self.selected_features and "pad_token_id" not in keyword_names:
            node.keywords.append(
                ast.keyword(
                    arg="pad_token_id",
                    value=parse_expr("getattr(tokenizer, 'pad_token_id', None) or getattr(tokenizer, 'eos_token_id', None)"),
                )
            )
            self.applied.add("pad_token_id")
        return node

    def _transform_body(self, body: list[ast.stmt]) -> list[ast.stmt]:
        new_body: list[ast.stmt] = []
        for stmt in body:
            if (
                "warmup_inference" in self.selected_features
                and not self.warmup_inserted
                and loop_iter_name(stmt)
                and contains_generate_call(stmt)
            ):
                new_body.extend(warmup_stmt(loop_iter_name(stmt) or "prompts"))
                self.applied.add("warmup_inference")
                self.warmup_inserted = True

            if "sync_device_timing" in self.selected_features and is_start_timer_assign(stmt):
                new_body.extend(sync_stmt())
                self.applied.add("sync_device_timing")

            transformed = self.visit(stmt) if not isinstance(stmt, (ast.Module,)) else stmt

            if "sync_device_timing" in self.selected_features and is_gen_time_assign(stmt):
                new_body.extend(sync_stmt())
                self.applied.add("sync_device_timing")

            new_body.append(transformed)

            if "model_eval" in self.selected_features and is_name_assign(stmt, "model"):
                new_body.extend(model_eval_stmt())
                self.applied.add("model_eval")

            if "contiguous_inputs" in self.selected_features and is_name_assign(stmt, "inputs"):
                new_body.extend(contiguous_inputs_stmt())
                self.applied.add("contiguous_inputs")

        return new_body


def main() -> int:
    parser = argparse.ArgumentParser(description="Select and apply one copied-script performance feature pack.")
    parser.add_argument("--profile-json", required=True, help="performance profile JSON path")
    parser.add_argument("--bottlenecks-json", required=True, help="bottleneck classification JSON path")
    parser.add_argument("--source-script", required=True, help="source workload script path")
    parser.add_argument("--output-script", required=True, help="optimized copied script path")
    parser.add_argument("--output-json", required=True, help="path to write applied feature metadata")
    parser.add_argument("--hotspot-json", help="optional hotspot summary JSON path")
    args = parser.parse_args()

    profile = read_json(Path(args.profile_json))
    bottlenecks = read_json(Path(args.bottlenecks_json))
    hotspot = read_json(Path(args.hotspot_json)) if args.hotspot_json else None

    selected_features, rationale, metrics_to_watch = select_features(profile, bottlenecks, hotspot)
    source_path = Path(args.source_script).resolve()
    output_path = Path(args.output_script).resolve()
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    transformer = FeatureTransformer(selected_features)
    transformed = transformer.visit(tree)
    instrumented = ast.unparse(ast.fix_missing_locations(transformed)) + "\n"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(instrumented, encoding="utf-8")

    applied_features = [feature for feature in selected_features if feature in transformer.applied]
    skipped_features = [feature for feature in selected_features if feature not in transformer.applied]
    payload = {
        "schema_version": "performance-agent/0.1",
        "skill": "performance-agent",
        "source_script": str(source_path),
        "output_script": str(output_path),
        "workload_type": profile.get("workload_type"),
        "stack": profile.get("stack"),
        "primary_bottleneck": (bottlenecks.get("primary_candidate") or {}).get("name"),
        "selected_features": selected_features,
        "applied_features": applied_features,
        "skipped_features": skipped_features,
        "rationale": rationale,
        "metrics_to_watch": metrics_to_watch,
    }
    write_json(Path(args.output_json), payload)
    print(json.dumps({"applied_features": applied_features, "skipped_features": skipped_features}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
