from pathlib import Path
import ast
import importlib.util
import json
import os
import pytest
import sqlite3
import subprocess
import sys


SKILL_ROOT = Path(__file__).resolve().parents[1]
CANN_BUILDER = SKILL_ROOT / "scripts" / "index_builders" / "generate_cann_failure_index.py"
MINDSPORE_BUILDER = SKILL_ROOT / "scripts" / "index_builders" / "generate_mindspore_failure_index.py"
CANN_ERROR_DB = SKILL_ROOT / "reference" / "index" / "cann_error_index.db"
CANN_ACLNN_DB = SKILL_ROOT / "reference" / "index" / "cann_aclnn_api_index.db"
MINT_DB = SKILL_ROOT / "reference" / "index" / "mint_api_index.db"
MINT_METHODOLOGY = SKILL_ROOT / "scripts" / "index_builders" / "mint_api_methodology.md"
CANN_QUERY_SCRIPT = SKILL_ROOT / "scripts" / "query_cann_index.py"
MINT_QUERY_SCRIPT = SKILL_ROOT / "scripts" / "query_mint_api_index.py"
QUERY_SCRIPT = MINT_QUERY_SCRIPT
BUILDER_SCRIPT = MINDSPORE_BUILDER
DETERMINISTIC_TIMESTAMP = "1970-01-01T00:00:00+00:00"


def local_mindspore_repo() -> Path:
    repo = os.environ.get("MINDSPORE_TEST_REPO")
    if not repo:
        pytest.skip("MINDSPORE_TEST_REPO is not set")
    path = Path(repo)
    if not path.exists():
        pytest.skip(f"MINDSPORE_TEST_REPO does not exist: {path}")
    return path


# CANN query index semantics

def run_cann_query(*args: str) -> dict:
    result = subprocess.run(
        [sys.executable, str(CANN_QUERY_SCRIPT), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def run_cann_query_raw(*args: str) -> str:
    result = subprocess.run(
        [sys.executable, str(CANN_QUERY_SCRIPT), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


@pytest.mark.parametrize(
    ("db_path", "db_kind", "count_field"),
    (
        (CANN_ERROR_DB, "error_index", "entry_count"),
        (CANN_ACLNN_DB, "aclnn_index", "api_count"),
    ),
)
def test_cann_query_meta_returns_db_kind(db_path: Path, db_kind: str, count_field: str):
    payload = run_cann_query("--db", str(db_path), "meta")
    assert payload["ok"] is True
    assert payload["db_kind"] == db_kind
    assert payload["meta"][count_field] >= 1


def test_error_code_lookup_returns_stable_record():
    payload = run_cann_query("--db", str(CANN_ERROR_DB), "code", "--value", "561003")
    assert payload["ok"] is True
    assert payload["entry"]["code"] == 561003


@pytest.mark.parametrize(
    ("args", "db_kind", "payload_path", "expected"),
    (
        (("561003",), "error_index", ("entry", "code"), 561003),
        (("code", "--value", "561003"), "error_index", ("entry", "code"), 561003),
        (("aclnnAbs",), "aclnn_index", ("api", "api_name"), "aclnnAbs"),
        (("api", "--name", "aclnnAbs"), "aclnn_index", ("api", "api_name"), "aclnnAbs"),
    ),
)
def test_cann_query_uses_default_db_for_common_short_forms(args, db_kind, payload_path, expected):
    payload = run_cann_query(*args)
    assert payload["ok"] is True
    assert payload["db_kind"] == db_kind
    current = payload
    for key in payload_path:
        current = current[key]
    assert current == expected


def test_query_cli_accepts_llm_friendly_argument_orders_and_aliases():
    expected = run_cann_query("--db", str(CANN_ERROR_DB), "code", "--value", "561003")
    variants = [
        run_cann_query("code", "--db", str(CANN_ERROR_DB), "--value", "561003"),
        run_cann_query("code", "--value", "561003", "--db", str(CANN_ERROR_DB)),
        run_cann_query("code", "561003", "--db", str(CANN_ERROR_DB)),
        run_cann_query("error", "--value", "561003", "--db", str(CANN_ERROR_DB)),
    ]
    for payload in variants:
        assert payload["ok"] is True
        assert payload["entry"] == expected["entry"]


def test_query_cli_normalizes_prefixed_error_code_without_argparse_failure():
    payload = run_cann_query("code", "--value", "EZ1001", "--db", str(CANN_ERROR_DB))
    assert payload["ok"] is False
    assert payload["error"] in {"code not found", "invalid code"}
    if payload["error"] == "code not found":
        assert payload["value"] == "EZ1001"
        assert payload["normalized_value"] == "1001"


def test_cann_query_keeps_core_lookup_contracts():
    search = run_cann_query("--db", str(CANN_ERROR_DB), "search", "--text", "aclnn")
    assert search["ok"] is True
    assert search["results"]

    api = run_cann_query("--db", str(CANN_ACLNN_DB), "api", "--name", "aclnnAbs")
    assert api["ok"] is True
    assert api["api"]["workspace_api"] == "aclnnAbsGetWorkspaceSize"
    assert api["api"]["execute_api"] == "aclnnAbs"

    resolved = run_cann_query("--db", str(CANN_ACLNN_DB), "resolve", "--symbol", "aclnnAbsGetWorkspaceSize")
    assert resolved["ok"] is True
    assert resolved["api"]["api_name"] == "aclnnAbs"
    assert resolved["resolved_symbol"] == "aclnnAbsGetWorkspaceSize"


def test_command_db_mismatch_returns_explicit_error():
    payload = run_cann_query("--db", str(CANN_ERROR_DB), "api", "--name", "aclnnAbs")
    assert payload["ok"] is False
    assert payload["error"] == "command/db mismatch"
    payload = run_cann_query("--db", str(CANN_ACLNN_DB), "code", "--value", "561003")
    assert payload["ok"] is False
    assert payload["error"] == "command/db mismatch"


def test_query_script_reports_missing_index():
    missing_db = SKILL_ROOT / "reference" / "index" / "missing_cann.db"
    payload = run_cann_query("--db", str(missing_db), "meta")
    assert payload["ok"] is False
    assert payload["error"] == "index unavailable"


def test_query_cli_returns_json_hints_for_common_argument_errors():
    payload = run_cann_query("--db", str(CANN_ERROR_DB), "code")
    assert payload["ok"] is False
    assert payload["error"] == "missing value"
    assert "code --value" in payload["hint"]

    payload = run_cann_query("--db", str(CANN_ERROR_DB), "unknown")
    assert payload["ok"] is False
    assert payload["error"] == "unknown command"
    assert payload["examples"]


def test_cann_query_pretty_json_is_human_readable_and_equivalent():
    compact = run_cann_query("api", "--name", "aclnnAbs")

    pretty_prefix = run_cann_query_raw("-p", "api", "--name", "aclnnAbs")
    assert "\n  " in pretty_prefix
    assert json.loads(pretty_prefix) == compact

    pretty_suffix = run_cann_query_raw("api", "--name", "aclnnAbs", "--pretty")
    assert "\n  " in pretty_suffix
    assert json.loads(pretty_suffix) == compact

# Mint query CLI

def run_mint_query(*args: str) -> dict:
    result = subprocess.run(
        [sys.executable, str(MINT_QUERY_SCRIPT), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def run_mint_query_raw(*args: str) -> str:
    result = subprocess.run(
        [sys.executable, str(MINT_QUERY_SCRIPT), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


@pytest.mark.parametrize(
    "args",
    (
        ("api", "--name", "sum", "--module", "mint", "--db", str(MINT_DB)),
        ("api", "--name", "sum", "--db", str(MINT_DB)),
        ("--name", "mindspore.mint.sum"),
        ("--name", "sum", "--module", "mint"),
        ("sum",),
        ("mint.sum",),
        ("api", "--name", "mint.sum"),
    ),
)
def test_mint_query_resolves_common_sum_short_forms(args):
    payload = run_mint_query(*args)
    assert payload["ok"] is True
    assert payload["api"]["api"] == "mindspore.mint.sum"
    assert payload["api"]["primitive"] == ["SumExt"]


@pytest.mark.parametrize(
    ("args", "expected_api"),
    (
        (("api", "--name", "relu", "--module", "nn.functional"), "mindspore.mint.nn.functional.relu"),
        (("api", "--name", "relu", "--module", "mint.nn.functional"), "mindspore.mint.nn.functional.relu"),
        (("api", "--name", "log_softmax", "--module", "special"), "mindspore.mint.special.log_softmax"),
        (("api", "--name", "log_softmax", "--module", "mindspore.mint.special"), "mindspore.mint.special.log_softmax"),
    ),
)
def test_mint_query_module_is_relative_to_mint_namespace(args, expected_api):
    payload = run_mint_query(*args)
    assert payload["ok"] is True
    assert payload["api"]["api"] == expected_api


def test_mint_query_module_does_not_probe_non_mint_namespace():
    payload = run_mint_query("api", "--name", "relu", "--module", "nn.functional.missing")
    assert payload["ok"] is False
    assert payload["error"] == "api not found"
    assert "mindspore.nn.functional.missing.relu" not in payload["tried"]
    assert "mindspore.mint.nn.functional.missing.relu" in payload["tried"]


def test_mint_query_explain_and_search_are_llm_friendly():
    payload = run_mint_query("explain", "mint.sum")
    assert payload["ok"] is True
    assert payload["api_core"]["api_name"] == "mindspore.mint.sum"

    payload = run_mint_query("search", "--text", "reduce sum", "--db", str(MINT_DB))
    assert payload["ok"] is True
    assert any(item["api"] == "mindspore.mint.sum" for item in payload["results"])

    payload = run_mint_query("search", "reduce sum")
    assert payload["ok"] is True
    assert any(item["api"] == "mindspore.mint.sum" for item in payload["results"])

    payload = run_mint_query("explain", "--text", "reduce sum", "--db", str(MINT_DB))
    assert payload["ok"] is True
    assert any(item["api"] == "mindspore.mint.sum" for item in payload["results"])


def test_mint_query_returns_json_hints_for_common_argument_errors():
    payload = run_mint_query("api", "--module", "mint", "--db", str(MINT_DB))
    assert payload["ok"] is False
    assert payload["error"] == "missing name"
    assert payload["examples"]

    payload = run_mint_query("--db", str(MINT_DB), "unknown", "extra")
    assert payload["ok"] is False
    assert payload["error"] == "unknown command"
    assert payload["examples"]

    payload = run_mint_query("api", "--name", "mint.unknown")
    assert payload["ok"] is False
    assert payload["error"] == "api not found"
    assert "mint.unknown" in payload["tried"]
    assert "mindspore.mint.unknown" in payload["tried"]


def test_mint_query_pretty_json_is_human_readable_and_equivalent():
    compact = run_mint_query("explain", "mint.sum")

    pretty_prefix = run_mint_query_raw("-p", "explain", "mint.sum")
    assert "\n  " in pretty_prefix
    assert json.loads(pretty_prefix) == compact

    pretty_suffix = run_mint_query_raw("explain", "mint.sum", "--pretty")
    assert "\n  " in pretty_suffix
    assert json.loads(pretty_suffix) == compact

    error_pretty = run_mint_query_raw("-p", "api", "--name", "mint.unknown")
    assert "\n  " in error_pretty
    error_payload = json.loads(error_pretty)
    assert error_payload["ok"] is False
    assert error_payload["error"] == "api not found"

# Index builder/schema contract

def test_builder_public_cli_contracts_are_declared():
    text = CANN_BUILDER.read_text(encoding="utf-8")
    assert "cann_error_index.db" in text
    assert "cann_aclnn_api_index.db" in text
    assert "--no-tags" in text
    assert "--deterministic" in text
    assert "--keep-workspace" in text

    text = MINDSPORE_BUILDER.read_text(encoding="utf-8")
    assert "https://atomgit.com/mindspore/mindspore.git" in text
    assert "mint_api_index.db" in text
    assert "--depth" in text
    assert "--no-tags" in text
    assert "--repo" in text
    assert "--branch" in text
    assert "--deterministic" in text
    assert "--keep-workspace" in text
    assert "--with-yaml" in text
    assert "--with-evidence" in text
    assert "--with-review" in text
    assert "--with-rulebook" in text


def test_builders_emit_provenance_fields():
    cann_text = CANN_BUILDER.read_text(encoding="utf-8")
    mint_text = MINDSPORE_BUILDER.read_text(encoding="utf-8")
    for field in (
        "generated_at",
        "generator_name",
        "generator_version",
        "index_schema_version",
        "source_mode",
        "source_repo_url",
        "source_branch",
        "source_commit",
        "source_repository_count",
        "source_repositories",
    ):
        assert field in cann_text
        assert field in mint_text


def test_generated_indexes_include_provenance_meta():
    for path, count_field in ((CANN_ERROR_DB, "entry_count"), (CANN_ACLNN_DB, "api_count")):
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        try:
            meta = conn.execute("SELECT * FROM schema_meta WHERE id = 1").fetchone()
            assert meta is not None
            for field in (
                "generated_at",
                "generator_name",
                "generator_version",
                "schema_version",
                "source_mode",
                "source_repo_url",
                "source_branch",
                "source_commit",
                "source_repository_count",
                count_field,
            ):
                assert field in meta.keys()
        finally:
            conn.close()
    conn = sqlite3.connect(MINT_DB)
    try:
        conn.row_factory = sqlite3.Row
        meta = conn.execute("SELECT * FROM schema_meta WHERE id = 1").fetchone()
        assert meta is not None
        for field in (
            "generated_at",
            "generator_name",
            "generator_version",
            "schema_version",
            "source_mode",
            "source_repo_url",
            "source_branch",
            "source_commit",
            "mindspore_version_hint",
        ):
            assert field in meta.keys()
    finally:
        conn.close()


def test_generated_mint_db_schema_is_slimmed():
    conn = sqlite3.connect(MINT_DB)
    try:
        columns = {
            row[1]
            for row in conn.execute("PRAGMA table_info(api)").fetchall()
        }
        assert "summary" in columns
        assert "category" in columns
        assert "trust_level" in columns
        assert "grad_differentiable" in columns
        for removed in (
            "semantic_kind",
            "graph_support_kind",
            "primitive_count",
            "evidence_ref",
            "status_summary",
            "support_summary",
            "llm_summary",
            "llm_warning",
            "confidence",
        ):
            assert removed not in columns
        grad_primitive_columns = {
            row[1]
            for row in conn.execute("PRAGMA table_info(api_grad_primitive)").fetchall()
        }
        assert grad_primitive_columns == {"api_id", "ordinal", "primitive_name", "origin_kind"}
        api_path_columns = {
            row[1]
            for row in conn.execute("PRAGMA table_info(api_path)").fetchall()
        }
        assert "anchor" in api_path_columns
        path_kinds = {
            row[0]
            for row in conn.execute("SELECT DISTINCT path_kind FROM api_path").fetchall()
        }
        assert "primary_path" not in path_kinds
        assert {
            "api_def_path",
            "dispatch_path",
            "implementation_path",
            "op_def_path",
            "kernel_path_pynative_ascend",
            "kernel_path_pynative_cpu",
            "kernel_path_pynative_gpu",
            "kernel_path_graph_kbk_o0_ascend",
            "kernel_path_graph_kbk_o0_cpu",
            "kernel_path_graph_kbk_o0_gpu",
            "infer_path",
        }.issubset(path_kinds)
    finally:
        conn.close()


def test_generated_indexes_use_deterministic_timestamp():
    for path in (CANN_ERROR_DB, CANN_ACLNN_DB):
        conn = sqlite3.connect(path)
        try:
            value = conn.execute("SELECT generated_at FROM schema_meta WHERE id = 1").fetchone()
            assert value is not None
            assert value[0] == DETERMINISTIC_TIMESTAMP
        finally:
            conn.close()
    conn = sqlite3.connect(MINT_DB)
    try:
        value = conn.execute("SELECT generated_at FROM schema_meta WHERE id = 1").fetchone()
        assert value is not None
        assert value[0] == DETERMINISTIC_TIMESTAMP
    finally:
        conn.close()


# Mint DB semantic regression

def load_builder_module():
    spec = importlib.util.spec_from_file_location("mint_builder", BUILDER_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def query_api(api: str) -> dict:
    result = subprocess.run(
        [sys.executable, str(QUERY_SCRIPT), "--db", str(MINT_DB), "api", "--name", api],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    return payload["api"]


def query_meta() -> dict:
    result = subprocess.run(
        [sys.executable, str(QUERY_SCRIPT), "--db", str(MINT_DB), "meta"],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    return payload["meta"]


def query_explain(api: str) -> dict:
    result = subprocess.run(
        [sys.executable, str(QUERY_SCRIPT), "--db", str(MINT_DB), "explain", "--name", api],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    return payload


def query_aclnn(api: str) -> tuple[str, str, list[str], list[str]]:
    conn = sqlite3.connect(MINT_DB)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            """
            SELECT a.api_id, a.aclnn_mode, a.aclnn_path_kind
            FROM api a
            WHERE a.api_name = ?
            """,
            (api,),
        ).fetchone()
        assert row is not None
        direct = [
            str(item[0])
            for item in conn.execute(
                "SELECT interface_name FROM api_aclnn_interface WHERE api_id = ? AND role = 'direct' ORDER BY ordinal",
                (row["api_id"],),
            ).fetchall()
        ]
        effective = [
            str(item[0])
            for item in conn.execute(
                "SELECT interface_name FROM api_aclnn_interface WHERE api_id = ? AND role = 'effective' ORDER BY ordinal",
                (row["api_id"],),
            ).fetchall()
        ]
        return str(row["aclnn_mode"]), str(row["aclnn_path_kind"]), direct, effective
    finally:
        conn.close()


def test_query_meta_returns_index_build_information():
    meta = query_meta()
    assert meta["schema_version"] == "2.0"
    assert meta["generator_name"] == "generate_mindspore_failure_index.py"
    assert meta["mindspore_version_hint"] == "2.9.0"
    assert meta["source_branch"] == "master"
    assert len(meta["source_repositories"]) >= 1
    assert meta["source_repositories"][0]["name"] == "mindspore"


def test_dropout_support_requires_full_func_op_closure():
    entry = query_api("mindspore.mint.nn.functional.dropout")
    assert entry["category"] == "mint"
    assert entry["trust_level"] == "weak"
    assert entry["primitive"] == ["FuncDropoutExt"]
    assert entry["call_chain_kind"] == "func_op"
    assert entry["resolution_kind"] == "func_expansion"
    assert entry["support_reason_kind"] == "unknown"
    assert entry["support_matrix"]["pynative"]["cpu"] == "unknown"
    assert entry["support_matrix"]["graph_kbk_o0"]["cpu"] == "unknown"
    assert "func_op" in entry["summary"]
    assert "graph(A=unknown,C=unknown,G=unknown) via expansion" in entry["summary"]


def test_sum_keeps_explicit_kernel_name_mapping():
    entry = query_api("mindspore.mint.sum")
    payload = query_explain("mindspore.mint.sum")
    assert entry["category"] == "mint"
    assert entry["trust_level"] == "certain"
    assert "confidence" not in entry
    assert entry["primitive"] == ["SumExt"]
    assert entry["aclnn"] == {
        "mode": "indirect",
        "interfaces": [],
        "effective_interfaces": ["aclnnReduceSum"],
        "path_kind": "customize_to_aclnn",
    }
    assert payload["aclnn"] == entry["aclnn"]
    assert entry["support_matrix"]["pynative"]["cpu"] == "yes"
    assert entry["support_matrix"]["pynative"]["gpu"] == "yes"
    assert entry["support_matrix"]["graph_kbk_o0"]["cpu"] == "yes"
    assert entry["support_matrix"]["graph_kbk_o0"]["gpu"] == "yes"


def test_mint_query_returns_stable_empty_aclnn_shape():
    entry = query_api("mindspore.mint.any")
    payload = query_explain("mindspore.mint.any")
    assert entry["aclnn"] == {
        "mode": "none",
        "interfaces": [],
        "effective_interfaces": [],
        "path_kind": "none",
    }
    assert payload["aclnn"] == entry["aclnn"]


def test_randint_like_closes_ascend_support_from_real_pynative_and_kbk_evidence():
    entry = query_api("mindspore.mint.randint_like")
    assert entry["primitive"] == ["RandIntLike"]
    assert entry["support_reason_kind"] == "direct_support"
    assert entry["support_matrix"]["pynative"]["ascend"] == "yes"
    assert entry["support_matrix"]["graph_kbk_o0"]["ascend"] == "yes"


def test_trace_multi_overload_support_is_not_merged_across_siblings():
    entry = query_api("mindspore.mint.trace")
    assert entry["call_chain_kind"] == "generated_binding"
    assert entry["implementation_type"] == "composite_op"
    assert entry["support_matrix"]["pynative"]["cpu"] == "no"
    assert entry["support_matrix"]["graph_kbk_o0"]["cpu"] == "no"
    assert "direct operator" in entry["summary"]


def test_where_is_resolved_as_functional_overload_dispatch():
    entry = query_api("mindspore.mint.where")
    assert entry["trust_level"] == "conditional"
    assert entry["call_chain_kind"] == "functional_overload"
    assert entry["resolution_kind"] == "overload_dispatch"
    assert entry["support_reason_kind"] == "overload_dispatch"
    assert entry["primitive"] == []
    assert entry["possible_primitives"] == ["NonZeroExt", "Select"]
    assert entry["unknown_reason"] == ""
    assert entry["support_matrix"]["pynative"]["cpu"] == "yes"
    assert entry["support_matrix"]["graph_kbk_o0"]["cpu"] == "yes"


def test_graph_fallback_closure_can_override_dispatch_gpu_none_for_argmax_and_nonzero():
    for api, primitive in (
        ("mindspore.mint.argmax", "ArgMaxExt"),
        ("mindspore.mint.nonzero", "NonZeroExt"),
    ):
        payload = query_explain(api)
        assert payload["api_core"]["support_reason_kind"] == "fallback_closure"
        assert payload["support_matrix"]["graph_kbk_o0"]["cpu"] == "yes"
        assert payload["support_matrix"]["graph_kbk_o0"]["gpu"] == "yes"
        gpu_kernel_paths = " ".join(_mixed_paths(payload["path_hints"]["kernel_paths"]["graph_kbk_o0"]["gpu"]))
        assert "mindspore/ops/fallback/nn_ops.cc" in gpu_kernel_paths
        support_summaries = " ".join(
            item["summary"]
            for item in payload["evidence_digest"]
            if item["domain"] == "support" and item["exec_mode"] == "graph_kbk_o0" and item["backend"] == "gpu"
        )
        assert f'fallback builder for {primitive}' in support_summaries


def test_query_explain_returns_support_targets_and_current_grad_shape():
    payload = query_explain("mindspore.mint.nn.functional.dropout")
    assert payload["primitive"] == ["FuncDropoutExt"]
    assert payload["api_core"]["call_chain_kind"] == "func_op"
    assert payload["api_core"]["resolution_kind"] == "func_expansion"
    assert "support_targets" in payload
    assert payload["grad"]["mode"] == "unknown"
    assert payload["grad"]["differentiable"] == "unknown"
    assert "summary" in payload["warnings"]
    assert payload["grad"]["impl"]
    assert all(item["scope_kind"] == "func_expansion" for item in payload["grad"]["impl"])
    assert set(payload["grad"]["backward_primitives"]).issubset(
        {"DropoutExt", "InplaceCopy", "Zeros", "Mul", "InplaceMul", "Size"}
    )


def test_index_select_uses_real_terminal_support_target_instead_of_api_def_guess():
    payload = query_explain("mindspore.mint.index_select")
    assert payload["api_core"]["trust_level"] == "certain"
    assert payload["primitive"] == ["IndexSelect"]
    assert payload["support_targets"] == [
        {
            "primitive": "IndexSelect",
            "api_def": "",
            "op_yaml": "index_select_op.yaml",
            "op_def_path": "mindspore/ops/op_def/yaml/index_select_op.yaml",
            "origin_kind": "terminal_call",
        }
    ]
    assert payload["support_matrix"]["pynative"]["ascend"] == "yes"


def test_logsumexp_uses_real_terminal_support_target_instead_of_api_def_guess():
    payload = query_explain("mindspore.mint.logsumexp")
    assert payload["primitive"] == ["LogSumExp"]
    assert payload["support_targets"] == [
        {
            "primitive": "LogSumExp",
            "api_def": "",
            "op_yaml": "logsumexp_op.yaml",
            "op_def_path": "mindspore/ops/op_def/yaml/logsumexp_op.yaml",
            "origin_kind": "terminal_call",
        }
    ]


def test_generated_binding_impl_instance_resolves_stack_ext_primitive():
    payload = query_explain("mindspore.mint.stack")
    assert payload["api_core"]["call_chain_kind"] == "generated_binding"
    assert payload["api_core"]["resolution_kind"] == "real_terminal"
    assert payload["primitive"] == ["StackExt"]
    assert payload["support_targets"] == [
        {
            "primitive": "StackExt",
            "api_def": "",
            "op_yaml": "stack_ext_op.yaml",
            "op_def_path": "mindspore/ops/op_def/yaml/stack_ext_op.yaml",
            "origin_kind": "terminal_call",
        }
    ]


def test_generated_binding_impl_instance_resolution_is_not_stack_specific():
    roll = query_explain("mindspore.mint.roll")
    assert roll["primitive"] == ["Roll"]
    assert roll["api_core"]["resolution_kind"] == "real_terminal"
    assert roll["api_core"].get("unknown_reason", "") in {"", "missing_runtime_kernel_evidence"}
    assert roll["support_targets"][0]["primitive"] == "Roll"

    nan_to_num = query_explain("mindspore.mint.nan_to_num")
    assert nan_to_num["primitive"] == ["NanToNum"]
    assert nan_to_num["api_core"]["resolution_kind"] == "real_terminal"
    assert nan_to_num["api_core"].get("unknown_reason", "") in {"", "missing_runtime_kernel_evidence"}
    assert nan_to_num["support_targets"][0]["primitive"] == "NanToNum"


def test_bitwise_wrappers_close_pynative_cpu_gpu_to_no_without_api_def_dispatch():
    for api in ("mindspore.mint.bitwise_and", "mindspore.mint.bitwise_or", "mindspore.mint.bitwise_xor"):
        payload = query_explain(api)
        assert payload["api_core"]["call_chain_kind"] == "composite_python"
        assert payload["support_matrix"]["pynative"]["cpu"] == "no"
        assert payload["support_matrix"]["pynative"]["gpu"] == "no"
        dispatch_paths = " ".join(_mixed_paths(payload["path_hints"]["dispatch_paths"]))
        assert "mindspore/ops/api_def/bitwise_" not in dispatch_paths


def test_diff_resolves_private_helpers_into_scenario_dependent_inner_primitives():
    payload = query_explain("mindspore.mint.diff")
    entry_full = query_api("mindspore.mint.diff")
    entry = payload["api_core"]
    assert entry["call_chain_kind"] == "python_composite_wrapper"
    assert entry["support_reason_kind"] == "scenario_dependent"
    assert entry_full["unknown_reason"] == "scenario_dependent_call_chain"
    assert payload["primitive"] == []
    assert {"Concat", "Narrow", "SubExt", "LogicalXor"} <= set(payload["possible_primitives"])
    support_targets = payload["support_targets"]
    assert len(support_targets) >= 3
    assert {item["primitive"] for item in support_targets} >= {"Concat", "Narrow", "SubExt", "LogicalXor"}
    assert not (len(support_targets) == 1 and support_targets[0]["primitive"] == "Concat")
    op_def_paths = set(_mixed_paths(payload["path_hints"]["op_def_paths"]))
    assert "mindspore/ops/op_def/yaml/concat_op.yaml" in op_def_paths
    assert "mindspore/ops/op_def/yaml/narrow_op.yaml" in op_def_paths
    assert "mindspore/ops/op_def/yaml/sub_ext_op.yaml" in op_def_paths
    assert "mindspore/ops/op_def/yaml/logical_xor_op.yaml" in op_def_paths
    assert "view_op" not in entry_full["flags"]
    assert "has_view_op" in entry_full["flags"]


def test_view_flags_split_between_pure_view_api_and_composite_view_usage():
    narrow = query_api("mindspore.mint.narrow")
    assert "view_op" in narrow["flags"]
    assert "has_view_op" not in narrow["flags"]

    where = query_api("mindspore.mint.where")
    assert "view_op" not in where["flags"]
    assert "has_view_op" not in where["flags"]


def test_single_primitive_view_op_closes_pynative_ascend_from_view_pyboost_path():
    payload = query_explain("mindspore.mint.narrow")
    assert payload["primitive"] == ["NarrowView"]
    assert payload["api_core"]["support_reason_kind"] == "direct_support"
    assert payload["support_matrix"]["pynative"]["ascend"] == "yes"
    assert payload["support_matrix"]["graph_kbk_o0"]["ascend"] == "yes"
    ascend_kernel_paths = set(_mixed_paths(payload["path_hints"]["kernel_paths"]["pynative"]["ascend"]))
    assert "mindspore/ccsrc/pynative/forward/pyboost/auto_generate/pyboost_api.cc" in ascend_kernel_paths
    assert "mindspore/ccsrc/pynative/forward/pyboost/auto_generate/pyboost_core.cc" in ascend_kernel_paths
    assert (
        "mindspore/ccsrc/pynative/utils/pyboost/functions/auto_generate/functions.cc" in ascend_kernel_paths
    )
    ascend_support_summaries = " ".join(
        item["summary"]
        for item in payload["evidence_digest"]
        if item["domain"] == "support" and item["exec_mode"] == "pynative" and item["backend"] == "ascend"
    )
    assert "view-op pynative ascend path for NarrowView" in ascend_support_summaries


def test_single_primitive_graph_view_ops_close_graph_ascend_from_host_view_kernels():
    expected = {
        "mindspore.mint.unsqueeze": (
            "ExpandDimsView",
            "mindspore/ops/kernel/host/view/kernel_mod_impl/expand_dims_view.cc",
            "MS_HOST_REG_KERNEL(ExpandDimsView, ExpandDimsView)",
            "direct_support",
        ),
        "mindspore.mint.unbind": (
            "UnstackExtView",
            "mindspore/ops/kernel/host/view/kernel_mod_impl/unstack_ext_view.cc",
            "MS_HOST_REG_KERNEL(UnstackExtView, UnstackExtView)",
            "fallback_closure",
        ),
    }
    for api, (primitive, host_view_path, host_view_anchor, expected_reason) in expected.items():
        payload = query_explain(api)
        assert payload["primitive"] == [primitive]
        assert payload["api_core"]["support_reason_kind"] == expected_reason
        assert payload["support_matrix"]["graph_kbk_o0"]["ascend"] == "yes"
        graph_ascend_paths = set(_mixed_paths(payload["path_hints"]["kernel_paths"]["graph_kbk_o0"]["ascend"]))
        assert host_view_path in graph_ascend_paths
        support_anchors = " ".join(
            item["anchor"]
            for item in payload["evidence_digest"]
            if item["domain"] == "support" and item["exec_mode"] == "graph_kbk_o0" and item["backend"] == "ascend"
        )
        assert host_view_anchor in support_anchors


def test_exact_terminal_kernel_closure_updates_direct_and_generated_examples():
    logical_xor = query_api("mindspore.mint.logical_xor")
    assert logical_xor["support_matrix"]["pynative"]["cpu"] == "yes"
    assert logical_xor["support_matrix"]["pynative"]["gpu"] == "no"

    roll = query_api("mindspore.mint.roll")
    assert roll["support_matrix"]["pynative"]["cpu"] == "no"
    assert roll["support_matrix"]["pynative"]["gpu"] == "yes"

    nan_to_num = query_api("mindspore.mint.nan_to_num")
    assert nan_to_num["support_matrix"]["pynative"]["cpu"] == "yes"
    assert nan_to_num["support_matrix"]["pynative"]["gpu"] == "no"


def test_graph_fallback_gpu_none_override_does_not_expand_to_argmin_or_complex_fallback_cases():
    argmin = query_api("mindspore.mint.argmin")
    assert argmin["support_matrix"]["graph_kbk_o0"]["gpu"] == "no"

    for api in (
        "mindspore.mint.nn.BatchNorm1d",
        "mindspore.mint.nn.functional.silu",
        "mindspore.mint.nn.functional.kl_div",
    ):
        entry = query_api(api)
        assert entry["support_matrix"]["graph_kbk_o0"]["gpu"] == "no"


def test_random_composite_wrappers_close_cpu_gpu_from_inner_inplace_primitives():
    expected = {
        "mindspore.mint.rand": "InplaceUniform",
        "mindspore.mint.rand_like": "InplaceUniform",
        "mindspore.mint.randn": "InplaceNormal",
        "mindspore.mint.randn_like": "InplaceNormal",
        "mindspore.mint.randint": "InplaceRandom",
        "mindspore.mint.randint_like": "InplaceRandom",
    }
    for api, inner_primitive in expected.items():
        entry = query_api(api)
        payload = query_explain(api)
        assert entry["support_matrix"]["pynative"]["cpu"] == "no"
        assert entry["support_matrix"]["pynative"]["gpu"] == "no"
        assert entry["unknown_reason"] in {"", "unresolved_composite_chain"}
        cpu_kernel_paths = " ".join(_mixed_paths(payload["path_hints"]["kernel_paths"]["pynative"]["cpu"]))
        assert "pyboost_cpu" in cpu_kernel_paths
        summaries = " ".join(
            item["summary"]
            for item in payload["evidence_digest"]
            if item["domain"] == "support" and item["exec_mode"] == "pynative" and item["backend"] in {"cpu", "gpu"}
        )
        assert inner_primitive in summaries
        dispatch_paths = " ".join(_mixed_paths(payload["path_hints"]["dispatch_paths"]))
        assert "mindspore/ops/api_def/" not in dispatch_paths


def test_random_composite_rule_resolves_inner_inplace_primitive_from_composite_file():
    builder = load_builder_module()
    index = builder.SourceIndex(local_mindspore_repo())
    for primitive, op_yaml, inner in (
        ("RandExt", "rand_ext_op.yaml", "InplaceUniform"),
        ("Randn", "randn_op.yaml", "InplaceNormal"),
        ("RandInt", "randint_op.yaml", "InplaceRandom"),
    ):
        state, evidence = builder.analyze_random_composite_inner_backend(
            index,
            primitive,
            "cpu",
            [{"primitive": primitive, "op_yaml": op_yaml, "op_def_path": "", "origin_kind": "terminal_call"}],
        )
        assert state == "no"
        assert any(inner in item["summary"] for item in evidence)


def test_no_dispatch_single_primitive_support_closes_cumprod_and_cdist():
    for api, primitive, ascend_adapter_file, cpu_file, gpu_file, expected_reason in (
        (
            "mindspore.mint.cumprod",
            "CumProd",
            "mindspore/ccsrc/plugin/ascend/res_manager/op_adapter/op_declare/selection_ops_declare.cc",
            "mindspore/ops/kernel/cpu/native/cumprod_cpu_kernel.cc",
            "mindspore/ops/kernel/gpu/cuda/math/cumprod_gpu_kernel.cc",
            "direct_support",
        ),
        (
            "mindspore.mint.cdist",
            "Cdist",
            "mindspore/ccsrc/plugin/ascend/res_manager/op_adapter/op_declare/math_ops_declare.cc",
            "mindspore/ops/kernel/cpu/native/cdist_cpu_kernel.cc",
            "mindspore/ops/kernel/gpu/cuda/math/cdist_gpu_kernel.cc",
            "direct_support",
        ),
    ):
        entry = query_api(api)
        payload = query_explain(api)
        assert payload["primitive"] == [primitive]
        assert payload["api_core"]["support_reason_kind"] == expected_reason
        assert payload["warnings"]["summary"].count("unknown") == 0
        assert entry["unknown_reason"] == ""
        for exec_mode in ("pynative", "graph_kbk_o0"):
            assert payload["support_matrix"][exec_mode]["ascend"] == "yes"
            assert payload["support_matrix"][exec_mode]["cpu"] == "yes"
            assert payload["support_matrix"][exec_mode]["gpu"] == "yes"
        assert payload["support_targets"] == [
            {
                "primitive": primitive,
                "api_def": "",
                "op_yaml": "cum_prod_op.yaml" if primitive == "CumProd" else "cdist_op.yaml",
                "op_def_path": (
                    "mindspore/ops/op_def/yaml/cum_prod_op.yaml"
                    if primitive == "CumProd"
                    else "mindspore/ops/op_def/yaml/cdist_op.yaml"
                ),
                "origin_kind": "terminal_call",
            }
        ]
        evidence_paths = {item["path"] for item in payload["evidence_digest"] if item["domain"] == "support"}
        assert ascend_adapter_file in evidence_paths
        assert cpu_file in evidence_paths
        assert gpu_file in evidence_paths
        assert ascend_adapter_file in _mixed_paths(payload["path_hints"]["kernel_paths"]["pynative"]["ascend"])
        assert cpu_file in _mixed_paths(payload["path_hints"]["kernel_paths"]["pynative"]["cpu"])
        assert gpu_file in _mixed_paths(payload["path_hints"]["kernel_paths"]["pynative"]["gpu"])


def test_no_dispatch_single_primitive_rule_does_not_expand_to_norm_family():
    for api in (
        "mindspore.mint.linalg.vector_norm",
        "mindspore.mint.linalg.norm",
        "mindspore.mint.linalg.matrix_norm",
        "mindspore.mint.nn.functional.normalize",
    ):
        payload = query_explain(api)
        assert "unknown" in payload["warnings"]["summary"]


def test_view_graph_rules_close_exact_rt_nop_and_fallback_cases_without_host_view_overreach():
    expected_yes = {
        "mindspore.mint.reshape": "Reshape",
        "mindspore.mint.squeeze": "Squeeze",
        "mindspore.mint.flatten": "FlattenExt",
    }
    for api, primitive in expected_yes.items():
        payload = query_explain(api)
        assert payload["primitive"] == [primitive]
        assert payload["support_matrix"]["graph_kbk_o0"]["ascend"] == "yes"
    reshape_payload = query_explain("mindspore.mint.reshape")
    reshape_anchors = " ".join(
        item["anchor"]
        for item in reshape_payload["evidence_digest"]
        if item["domain"] == "support" and item["exec_mode"] == "graph_kbk_o0" and item["backend"] == "ascend"
    )
    assert "rt_kernel_ops:Reshape" in reshape_anchors
    assert "IsNopNode:Reshape" in reshape_anchors
    assert "nop_op_to_memcpy_/MemoryCopyAsync" in reshape_anchors

    squeeze_payload = query_explain("mindspore.mint.squeeze")
    squeeze_anchors = " ".join(
        item["anchor"]
        for item in squeeze_payload["evidence_digest"]
        if item["domain"] == "support" and item["exec_mode"] == "graph_kbk_o0" and item["backend"] == "ascend"
    )
    assert "rt_kernel_ops:Squeeze" in squeeze_anchors
    assert "IsNopNode:Squeeze" in squeeze_anchors

    flatten_payload = query_explain("mindspore.mint.flatten")
    flatten_anchors = " ".join(
        item["anchor"]
        for item in flatten_payload["evidence_digest"]
        if item["domain"] == "support" and item["exec_mode"] == "graph_kbk_o0" and item["backend"] == "ascend"
    )
    assert 'REG_FALLBACK_BUILDER("FlattenExt") -> ib->Reshape' in flatten_anchors
    assert "rt_kernel_ops:Reshape" in flatten_anchors

    expected_unknown = {
        "mindspore.mint.meshgrid": "Meshgrid",
    }
    for api, primitive in expected_unknown.items():
        payload = query_explain(api)
        assert payload["primitive"] == [primitive]
        assert payload["support_matrix"]["graph_kbk_o0"]["ascend"] == "unknown"


def test_flatten_pynative_cpu_gpu_closes_through_view_impl_not_exact_kernel():
    payload = query_explain("mindspore.mint.flatten")
    assert payload["primitive"] == ["FlattenExt"]
    assert payload["support_matrix"]["pynative"]["cpu"] == "yes"
    assert payload["support_matrix"]["pynative"]["gpu"] == "yes"
    for backend in ("cpu", "gpu"):
        paths = set(_mixed_paths(payload["path_hints"]["kernel_paths"]["pynative"][backend]))
        assert "mindspore/ccsrc/pynative/utils/pyboost/functions/customize/flatten_ext_impl.cc" in paths
        assert "mindspore/ccsrc/pynative/utils/pyboost/functions/customize/reshape_impl.cc" in paths
        assert all(not path.endswith(".yaml") and not path.endswith(".py") for path in paths)


def test_adaptive_avg_pool2d_shape_prelude_does_not_block_ascend_support():
    for api in ("mindspore.mint.nn.functional.adaptive_avg_pool2d", "mindspore.mint.nn.AdaptiveAvgPool2d"):
        payload = query_explain(api)
        entry = query_api(api)
        assert payload["primitive"] == ["AdaptiveAvgPool2DExt"]
        assert {item["primitive"] for item in payload["support_targets"]} == {"AdaptiveAvgPool2DExt"}
        assert "Shape" not in payload["grad"]["backward_primitives"]
        assert payload["support_matrix"]["pynative"]["ascend"] == "yes"
        assert payload["support_matrix"]["graph_kbk_o0"]["ascend"] == "yes"
        assert entry["support_matrix"]["pynative"]["cpu"] == "no"
        assert entry["support_matrix"]["pynative"]["gpu"] == "no"
        assert "mindspore/ops/op_def/yaml/shape_op.yaml" not in _mixed_paths(payload["path_hints"]["op_def_paths"])

        pynative_ascend_paths = set(_mixed_paths(payload["path_hints"]["kernel_paths"]["pynative"]["ascend"]))
        graph_ascend_paths = set(_mixed_paths(payload["path_hints"]["kernel_paths"]["graph_kbk_o0"]["ascend"]))
        assert "mindspore/ops/kernel/ascend/aclnn/pyboost_impl/auto_generate/pyboost_ascend_ops_2.cc" in pynative_ascend_paths
        assert (
            "mindspore/ops/kernel/ascend/aclnn/kernel_mod_impl/aclnn_auto_gen/adaptive_avg_pool2d_ext_aclnn_kernel.cc"
            in graph_ascend_paths
        )
        graph_ascend_anchors = " ".join(
            item.get("anchor", "")
            for item in payload["path_hints"]["kernel_paths"]["graph_kbk_o0"]["ascend"]
            if isinstance(item, dict)
        )
        assert "MS_ACLNN_KERNEL_FACTORY_REG(AdaptiveAvgPool2DExt" in graph_ascend_anchors

        evidence = " ".join(
            f"{item.get('anchor', '')} {item.get('summary', '')}"
            for item in payload["evidence_digest"]
            if item["domain"] == "support" and item["backend"] == "ascend"
        )
        assert "AdaptiveAvgPool2DExtAscend::Call" in evidence or "aclnnAdaptiveAvgPool2d" in evidence


def test_pure_python_prelude_does_not_block_adaptive_avg_pool3d_support():
    payload = query_explain("mindspore.mint.nn.functional.adaptive_avg_pool3d")
    entry = query_api("mindspore.mint.nn.functional.adaptive_avg_pool3d")
    assert payload["primitive"] == ["AdaptiveAvgPool3DExt"]
    assert {item["primitive"] for item in payload["support_targets"]} == {"AdaptiveAvgPool3DExt"}
    assert entry["unknown_reason"] == ""
    assert entry["support_matrix"]["pynative"]["ascend"] == "yes"
    assert entry["support_matrix"]["graph_kbk_o0"]["ascend"] == "yes"
    assert entry["support_matrix"]["pynative"]["cpu"] == "no"
    assert entry["support_matrix"]["pynative"]["gpu"] == "no"

    graph_ascend_paths = payload["path_hints"]["kernel_paths"]["graph_kbk_o0"]["ascend"]
    graph_ascend_anchors = " ".join(item.get("anchor", "") for item in graph_ascend_paths if isinstance(item, dict))
    assert "MS_ACLNN_KERNEL_FACTORY_REG(AdaptiveAvgPool3DExt" in graph_ascend_anchors
    evidence = " ".join(
        f"{item.get('anchor', '')} {item.get('summary', '')}"
        for item in payload["evidence_digest"]
        if item["domain"] == "support" and item["backend"] == "ascend"
    )
    assert "AdaptiveAvgPool3DExtAscend::Call" in evidence or "aclnnAdaptiveAvgPool3d" in evidence


def test_unique_frontend_guards_do_not_block_branch_support():
    payload = query_explain("mindspore.mint.unique")
    entry = query_api("mindspore.mint.unique")
    assert payload["primitive"] == ["Unique2", "UniqueDim"]
    assert {item["primitive"] for item in payload["support_targets"]} == {"Unique2", "UniqueDim"}
    assert entry["unknown_reason"] == ""
    assert entry["support_reason_kind"] == "direct_support"
    assert entry["support_matrix"]["pynative"] == {"ascend": "yes", "cpu": "no", "gpu": "no"}
    assert entry["support_matrix"]["graph_kbk_o0"] == {"ascend": "yes", "cpu": "no", "gpu": "no"}
    target_text = json.dumps(payload["support_targets"])
    assert "isconstant" not in target_text
    assert "check_value_type" not in target_text
    assert {item["path"] for item in payload["path_hints"]["op_def_paths"]} == {
        "mindspore/ops/op_def/yaml/unique2_op.yaml",
        "mindspore/ops/op_def/yaml/unique_dim_op.yaml",
    }


def test_class_constructor_setup_does_not_block_construct_support():
    expected = {
        "mindspore.mint.nn.Embedding": {"primitives": {"Embedding"}, "reason": "fallback_closure"},
        "mindspore.mint.nn.GroupNorm": {"primitives": {"GroupNorm"}, "reason": "direct_support"},
        "mindspore.mint.nn.Linear": {"primitives": {"Dense"}, "reason": "fallback_closure"},
    }
    for api, spec in expected.items():
        payload = query_explain(api)
        entry = query_api(api)
        assert entry["unknown_reason"] == ""
        assert entry["support_reason_kind"] == spec["reason"]
        assert {item["primitive"] for item in payload["support_targets"]} == spec["primitives"]
        assert set(payload["primitive"]) == spec["primitives"]
        support_target_text = json.dumps(payload["support_targets"])
        for setup_symbol in ("Parameter", "Tensor", "initializer", "HeUniform", "Uniform", "Constant"):
            assert setup_symbol not in support_target_text


def test_sync_batch_norm_preserves_real_composite_inner_chain_targets():
    payload = query_explain("mindspore.mint.nn.SyncBatchNorm")
    assert payload["api_core"]["call_chain_kind"] == "construct_mapped"
    assert payload["api_core"]["trust_level"] == "weak"
    assert payload["api_core"]["support_reason_kind"] == "unknown"
    core_prims = {"BatchNorm", "BatchNormElemt", "BatchNormGatherStatsWithCounts", "BatchNormStats"}
    assert core_prims <= set(payload["primitive"]), (
        f"SyncBatchNorm should contain core BatchNorm primitives, got {payload['primitive']}"
    )
    target_prims = {t["primitive"] for t in payload["support_targets"]}
    assert core_prims <= target_prims, (
        f"support_targets should contain core BatchNorm primitives, got {target_prims}"
    )
    assert "BatchNormExt" not in payload["primitive"]
    assert payload["support_matrix"]["pynative"]["ascend"] == "unknown"
    assert payload["support_matrix"]["graph_kbk_o0"]["cpu"] == "unknown"
    assert payload["grad"]["mode"] == "explicit_bprop"
    assert payload["grad"]["differentiable"] == "yes"
    assert "BatchNormReduceGrad" in payload["grad"]["backward_primitives"]
    assert "BatchNormElemtGrad" in payload["grad"]["backward_primitives"]
    assert any(item["kind"] == "python_cell_bprop" for item in payload["grad"]["impl"])


def test_mint_api_count_is_preserved_after_real_chain_rebuild():
    meta = query_meta()
    assert meta["api_count"] == 379


def test_doc_registered_flatten_is_indexed():
    entry = query_api("mindspore.mint.flatten")
    assert entry["trust_level"] == "certain"
    assert entry["call_chain_kind"] == "func_op"
    assert entry["support_reason_kind"] == "func_dispatch"
    assert entry["primitive"] == ["FlattenExt"]


def test_einsum_closes_through_functional_overload_func_op_bridge_on_ascend():
    entry = query_api("mindspore.mint.einsum")
    result = subprocess.run(
        [
            sys.executable,
            str(QUERY_SCRIPT),
            "--db",
            str(MINT_DB),
            "--evidence-limit",
            "100",
            "explain",
            "--name",
            "mindspore.mint.einsum",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert payload["ok"] is True

    assert entry["unknown_reason"] == ""
    assert entry["primitive"] == ["EinsumExt"]
    assert entry["func_op_expands_to"] == ["EinsumExt"]
    assert entry["support_reason_kind"] == "func_dispatch"
    assert entry["support_matrix"]["pynative"]["ascend"] == "yes"
    assert entry["support_matrix"]["graph_kbk_o0"]["ascend"] == "yes"
    assert entry["support_matrix"]["pynative"]["cpu"] == "unknown"
    assert entry["support_matrix"]["pynative"]["gpu"] == "unknown"
    assert entry["support_matrix"]["graph_kbk_o0"]["cpu"] == "unknown"
    assert entry["support_matrix"]["graph_kbk_o0"]["gpu"] == "unknown"

    op_def_paths = set(_mixed_paths(entry["path_hints"]["op_def_paths"]))
    assert "mindspore/ops/op_def/func_op/einsum_ext_op.yaml" in op_def_paths
    dispatch_paths = set(_mixed_paths(entry["path_hints"]["dispatch_paths"]))
    assert "mindspore/python/mindspore/ops/functional_overload.py" in dispatch_paths
    assert "mindspore/ccsrc/frontend/operator/composite/auto_generate/functional_map.cc" in dispatch_paths
    pynative_ascend_paths = set(_mixed_paths(entry["path_hints"]["kernel_paths"]["pynative"]["ascend"]))
    assert "mindspore/ops/kernel/ascend/aclnn/pyboost_impl/auto_generate/pyboost_ascend_ops_2.cc" in pynative_ascend_paths
    assert "mindspore/ops/kernel/ascend/aclnn/pyboost_impl/customize/einsum_ext.cc" in pynative_ascend_paths

    graph_ascend_evidence = " ".join(
        f"{item.get('path', '')} {item.get('anchor', '')} {item.get('summary', '')}"
        for item in payload["evidence_digest"]
        if item["domain"] == "support" and item["exec_mode"] == "graph_kbk_o0" and item["backend"] == "ascend"
    )
    pynative_ascend_evidence = " ".join(
        f"{item.get('path', '')} {item.get('anchor', '')} {item.get('summary', '')}"
        for item in payload["evidence_digest"]
        if item["domain"] == "support" and item["exec_mode"] == "pynative" and item["backend"] == "ascend"
    )
    assert '"einsum" -> prim::kPrimEinsumExt' in graph_ascend_evidence
    assert "EinsumExtAscend::Call" in pynative_ascend_evidence or "MS_REG_PYBOOST_OP(Ascend, EinsumExt)" in pynative_ascend_evidence


def test_get_cache_prim_wrappers_resolve_to_static_primitives():
    expected = {
        "mindspore.mint.isclose": "IsClose",
        "mindspore.mint.nn.functional.glu": "GLU",
        "mindspore.mint.nn.functional.log_softmax": "LogSoftmaxExt",
        "mindspore.mint.special.log_softmax": "LogSoftmaxExt",
    }
    for api_name, primitive in expected.items():
        entry = query_api(api_name)
        assert entry["primitive"] == [primitive]
        assert entry["unknown_reason"] != "terminal_symbol_unresolved"
        assert entry["call_chain_kind"] != "unresolved"
        assert entry["path_hints"]["op_def_paths"]


def test_remaining_plain_mint_apis_resolve_terminals_and_support():
    expected = {
        "mindspore.mint.gt": ("Greater", "mindspore/ops/op_def/yaml/greater_op.yaml"),
        "mindspore.mint.tile": ("Tile", "mindspore/ops/op_def/yaml/tile_op.yaml"),
        "mindspore.mint.searchsorted": ("SearchSorted", "mindspore/ops/op_def/yaml/searchsorted_op.yaml"),
    }
    for api_name, (primitive, op_def_path) in expected.items():
        entry = query_api(api_name)
        assert entry["primitive"] == [primitive]
        assert entry["unknown_reason"] == ""
        assert entry["call_chain_kind"] != "unresolved"
        assert entry["support_reason_kind"] == "direct_support"
        assert entry["support_matrix"]["pynative"] == {"ascend": "yes", "cpu": "yes", "gpu": "yes"}
        assert entry["support_matrix"]["graph_kbk_o0"] == {"ascend": "yes", "cpu": "yes", "gpu": "yes"}
        assert op_def_path in _mixed_paths(entry["path_hints"]["op_def_paths"])


def test_float_power_resolves_real_scenario_branches():
    entry = query_api("mindspore.mint.float_power")
    assert entry["primitive"] == []
    assert entry["unknown_reason"] == ""
    assert entry["call_chain_kind"] == "python_composite_wrapper"
    assert entry["support_reason_kind"] == "scenario_dependent"
    assert set(entry["possible_primitives"]) == {"Cast", "Pow", "PowTensorScalar", "PowScalarTensor"}
    assert entry["support_matrix"]["pynative"] == {"ascend": "yes", "cpu": "yes", "gpu": "yes"}
    assert entry["support_matrix"]["graph_kbk_o0"] == {"ascend": "yes", "cpu": "yes", "gpu": "yes"}
    op_def_paths = set(_mixed_paths(entry["path_hints"]["op_def_paths"]))
    assert {
        "mindspore/ops/op_def/yaml/cast_op.yaml",
        "mindspore/ops/op_def/yaml/pow_op.yaml",
        "mindspore/ops/op_def/yaml/pow_tensor_scalar_op.yaml",
        "mindspore/ops/op_def/yaml/pow_scalar_tensor_op.yaml",
    }.issubset(op_def_paths)


def test_conv2d_overload_bridge_uses_ext_primitives_without_legacy_kernel_borrowing():
    entry = query_api("mindspore.mint.nn.functional.conv2d")
    assert entry["primitive"] == []
    assert entry["unknown_reason"] == ""
    assert set(entry["possible_primitives"]) == {"Conv2DExt", "Conv2DPadding"}
    assert entry["support_matrix"]["pynative"] == {"ascend": "yes", "cpu": "no", "gpu": "no"}
    assert entry["support_matrix"]["graph_kbk_o0"] == {"ascend": "yes", "cpu": "no", "gpu": "no"}

    dispatch_paths = set(_mixed_paths(entry["path_hints"]["dispatch_paths"]))
    assert "mindspore/python/mindspore/ops/functional_overload.py" in dispatch_paths
    assert "mindspore/ccsrc/frontend/operator/composite/auto_generate/functional_map.cc" in dispatch_paths
    op_def_paths = set(_mixed_paths(entry["path_hints"]["op_def_paths"]))
    assert {
        "mindspore/ops/op_def/yaml/conv2d_ext_op.yaml",
        "mindspore/ops/op_def/yaml/conv2d_padding_op.yaml",
    }.issubset(op_def_paths)
    kernel_paths = " ".join(
        _mixed_paths(entry["path_hints"]["kernel_paths"]["pynative"]["cpu"])
        + _mixed_paths(entry["path_hints"]["kernel_paths"]["pynative"]["gpu"])
        + _mixed_paths(entry["path_hints"]["kernel_paths"]["graph_kbk_o0"]["cpu"])
        + _mixed_paths(entry["path_hints"]["kernel_paths"]["graph_kbk_o0"]["gpu"])
    )
    assert "conv2d_cpu_kernel" not in kernel_paths
    assert "conv2d_gpu_kernel" not in kernel_paths


def test_identity_module_is_python_pass_through_not_identity_primitive():
    entry = query_api("mindspore.mint.nn.Identity")
    assert entry["primitive"] == []
    assert entry["possible_primitives"] == []
    assert entry["unknown_reason"] == ""
    assert entry["call_chain_kind"] == "construct_mapped"
    assert entry["resolution_kind"] == "real_terminal"
    assert entry["implementation_type"] == "python_pass_through"
    assert entry["support_reason_kind"] == "direct_support"
    assert entry["support_matrix"]["pynative"] == {"ascend": "yes", "cpu": "yes", "gpu": "yes"}
    assert entry["support_matrix"]["graph_kbk_o0"] == {"ascend": "yes", "cpu": "yes", "gpu": "yes"}
    assert "python pass-through" in entry["summary"]
    assert "mindspore/ops/op_def/yaml/identity_op.yaml" not in _mixed_paths(entry["path_hints"]["op_def_paths"])


def test_get_cache_prim_ast_rule_handles_nested_assignment_and_rejects_dynamic_expr():
    builder = load_builder_module()
    module = builder.ModuleInfo(
        module_name="test_module",
        path=Path("test_module.py"),
        imports={
            "P": builder.ImportBinding("P", "mindspore.ops.operations", "", "from mindspore.ops import operations as P"),
            "_get_cache_prim": builder.ImportBinding(
                "_get_cache_prim",
                "mindspore.ops._primitive_cache",
                "_get_cache_prim",
                "from mindspore.ops._primitive_cache import _get_cache_prim",
            ),
        },
    )

    direct_expr = ast.parse("_get_cache_prim(P.GLU)(axis=dim)(input)", mode="eval").body
    assert builder.extract_cached_primitive_symbol(direct_expr, module) == "mindspore.ops.operations.GLU"

    fn = ast.parse(
        """
def f(input, other):
    is_close = _get_cache_prim(P.IsClose)(rtol=rtol, atol=atol, equal_nan=equal_nan)
    return is_close(input, other)
"""
    ).body[0]
    calls, _, _ = builder.extract_return_calls_with_metadata(fn, module)
    assert calls == ["mindspore.ops.operations.IsClose"]

    dynamic_expr = ast.parse("_get_cache_prim(dynamic_expr)(input)", mode="eval").body
    assert builder.extract_cached_primitive_symbol(dynamic_expr, module) is None


def test_non_documented_flatten_module_alias_is_not_indexed():
    result = subprocess.run(
        [sys.executable, str(QUERY_SCRIPT), "--db", str(MINT_DB), "api", "--name", "mindspore.mint.Flatten"],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert payload["error"] == "api not found"


def test_xlogy_is_resolved_as_functional_overload_dispatch():
    entry = query_api("mindspore.mint.xlogy")
    assert entry["trust_level"] == "conditional"
    assert entry["call_chain_kind"] == "functional_overload"
    assert entry["resolution_kind"] == "overload_dispatch"
    assert entry["primitive"] == []
    assert entry["possible_primitives"] == ["XLogYScalarOther", "XLogYScalarSelf", "Xlogy"]
    assert entry["unknown_reason"] == ""


def test_sub_closes_pynative_cpu_gpu_from_py_method_fallback():
    entry = query_api("mindspore.mint.sub")
    assert entry["call_chain_kind"] == "functional_overload"
    assert entry["possible_primitives"] == ["SubExt", "SubScalar"]
    assert entry["support_matrix"]["pynative"]["ascend"] == "yes"
    assert entry["support_matrix"]["pynative"]["cpu"] == "yes"
    assert entry["support_matrix"]["pynative"]["gpu"] == "yes"
    assert entry["support_matrix"]["graph_kbk_o0"]["cpu"] == "yes"
    assert entry["support_matrix"]["graph_kbk_o0"]["gpu"] == "yes"
    assert "aclnnSub" in entry["summary"]
    assert "aclnnSubs" in entry["summary"]
    assert _mixed_paths(entry["path_hints"]["api_def_paths"]) == ["mindspore/ops/api_def/sub.yaml"]
    assert set(_mixed_paths(entry["path_hints"]["op_def_paths"])) == {
        "mindspore/ops/op_def/yaml/sub_scalar_op.yaml",
        "mindspore/ops/op_def/yaml/sub_ext_op.yaml",
    }
    dispatch = _mixed_paths(entry["path_hints"]["dispatch_paths"])
    assert "mindspore/python/mindspore/ops/functional_overload.py" in dispatch
    assert "mindspore/ccsrc/frontend/operator/composite/auto_generate/functional_map.cc" in dispatch
    assert "mindspore/ccsrc/pynative/forward/pyboost/auto_generate/pyboost_overload_functions.cc" in dispatch
    impl = _mixed_paths(entry["path_hints"]["implementation_paths"])
    assert "mindspore/python/mindspore/ops/auto_generate/gen_ops_def.py" in impl
    assert "mindspore/python/mindspore/ops/tensor_method.py" in impl
    assert "mindspore/python/mindspore/ops/functional_overload.py" not in impl
    assert "grad_paths" not in entry["path_hints"]
    grad_impl_paths = [str(item.get("path", "")) for item in entry.get("grad", {}).get("impl", [])]
    assert "mindspore/ccsrc/frontend/expander/grad/grad_math_ops.cc" in grad_impl_paths
    assert entry["path_hints"]["kernel_paths"]["pynative"]["cpu"]
    assert entry["path_hints"]["kernel_paths"]["graph_kbk_o0"]["ascend"]
    cpu_kernel = _mixed_paths(entry["path_hints"]["kernel_paths"]["pynative"]["cpu"])
    for p in cpu_kernel:
        assert not p.endswith(".py"), f"Python file {p} should not appear in kernel_paths"


def test_sub_aggregates_functional_overload_aclnn_interfaces():
    entry = query_api("mindspore.mint.sub")
    mode, path_kind, direct, effective = query_aclnn("mindspore.mint.sub")
    assert entry["call_chain_kind"] == "functional_overload"
    assert entry["primitive"] == []
    assert entry["possible_primitives"] == ["SubExt", "SubScalar"]
    assert mode != "unknown"
    assert set(effective) == {"aclnnSub", "aclnnSubs"}
    assert path_kind in {"customize_to_aclnn", "composite_to_aclnn", "direct_aclnn"}
    assert set(direct).issubset({"aclnnSub", "aclnnSubs"})


def test_matmul_aclnn_interfaces_are_scoped_to_matmul_ext():
    payload = query_explain("mindspore.mint.matmul")
    mode, path_kind, direct, effective = query_aclnn("mindspore.mint.matmul")
    assert payload["primitive"] == ["MatMulExt"]
    assert mode == "direct"
    assert path_kind == "direct_aclnn"
    assert direct == ["aclnnMatmul"]
    assert effective == ["aclnnMatmul"]
    assert "aclnnMm" not in effective
    aclnn_paths = [
        str(item.get("path", ""))
        for item in payload.get("evidence_digest", [])
        if item.get("domain") == "aclnn"
    ]
    graph_ascend_support_anchors = [
        str(item.get("anchor", ""))
        for item in payload.get("evidence_digest", [])
        if item.get("domain") == "support" and item.get("exec_mode") == "graph_kbk_o0" and item.get("backend") == "ascend"
    ]
    assert "mindspore/ops/kernel/ascend/aclnn/pyboost_impl/customize/matmul_ext.cc" in aclnn_paths
    assert "mindspore/ops/kernel/ascend/aclnn/pyboost_impl/customize/matmul.cc" not in aclnn_paths
    assert "MS_ACLNN_KERNEL_FACTORY_REG(MatMulExt, MMExtAclnnKernelMod)" in graph_ascend_support_anchors
    assert "MS_ACLNN_KERNEL_FACTORY_REG(MatMul, MatMulAclnnKernelMod)" not in graph_ascend_support_anchors
    cpu_anchors = [item["anchor"] for item in payload["path_hints"]["kernel_paths"]["pynative"]["cpu"]]
    gpu_anchors = [item["anchor"] for item in payload["path_hints"]["kernel_paths"]["pynative"]["gpu"]]
    assert payload["path_hints"]["api_def_paths"] == []
    assert "MatMulExtCPUCustomize(" in cpu_anchors
    assert "MatMulExtGPUCustomize(" in gpu_anchors
    assert payload["support_matrix"]["pynative"] == {"ascend": "yes", "cpu": "yes", "gpu": "yes"}
    assert payload["support_matrix"]["graph_kbk_o0"] == {"ascend": "yes", "cpu": "yes", "gpu": "yes"}


def test_sub_grad_collapses_to_explicit_bprop_when_all_function_branches_close():
    payload = query_explain("mindspore.mint.sub")
    assert payload["grad"]["mode"] == "explicit_bprop"
    assert payload["grad"]["differentiable"] == "yes"
    assert set(payload["grad"]["backward_primitives"]) == {"SubExt", "SubScalar"}
    assert {item["scope_kind"] for item in payload["grad"]["impl"]} == {"overload_branch"}


def test_functional_overload_aclnn_keeps_known_interfaces_when_some_branches_unresolved():
    builder = load_builder_module()
    index = builder.SourceIndex(local_mindspore_repo())
    aclnn_info, evidence = builder.analyze_functional_overload_aclnn(
        index,
        [
            {"primitive": "SubExt", "op_yaml": "sub_ext_op.yaml", "op_def_path": "", "origin_kind": "overload_branch"},
            {"primitive": "MissingBranchPrimitive", "op_yaml": "missing.yaml", "op_def_path": "", "origin_kind": "overload_branch"},
        ],
    )
    assert aclnn_info["mode"] == "unknown"
    assert "aclnnSub" in aclnn_info["effective_interfaces"]
    assert evidence


def test_py_method_raise_stub_is_classified_as_no():
    builder = load_builder_module()
    index = builder.SourceIndex(local_mindspore_repo())
    state, evidence = builder.resolve_py_method_backend_support(index, "tensor_repeat", "pynative", "cpu")
    assert state == "no"
    assert any("raise stub" in item["summary"] for item in evidence)


def test_pixel_shuffle_closes_graph_support_from_func_dispatch_plus_meta_dsl():
    entry = query_api("mindspore.mint.nn.PixelShuffle")
    assert entry["call_chain_kind"] == "construct_mapped"
    assert entry["resolution_kind"] == "func_expansion"
    assert entry["possible_primitives"] == ["PixelShuffle"]
    assert entry["func_op_expands_to"] == []
    assert entry["support_reason_kind"] == "func_dispatch"
    assert entry["support_matrix"]["pynative"]["ascend"] == "yes"
    assert entry["support_matrix"]["graph_kbk_o0"]["ascend"] == "yes"
    assert entry["support_matrix"]["graph_kbk_o0"]["cpu"] == "yes"
    assert entry["support_matrix"]["graph_kbk_o0"]["gpu"] == "yes"
    assert "via func dispatch" in entry["summary"]


def test_module_upsample_keeps_construct_mapping_but_uses_composite_wrapper_targets():
    payload = query_explain("mindspore.mint.nn.Upsample")
    assert payload["api_core"]["call_chain_kind"] == "construct_mapped"
    assert payload["api_core"]["resolution_kind"] == "scenario_candidates"
    upsample_prims = {
        "UpsampleBicubic2D",
        "UpsampleBilinear2D",
        "UpsampleLinear1D",
        "UpsampleNearest1D",
        "UpsampleNearest2D",
        "UpsampleNearest3D",
        "UpsampleTrilinear3D",
    }
    actual_prims = set(payload["primitive"]) | set(payload["possible_primitives"])
    assert upsample_prims <= actual_prims, f"expected upsample prims present, got {actual_prims}"
    assert len(payload["support_targets"]) >= 7


def test_where_grad_remains_branch_scoped_and_unknown():
    payload = query_explain("mindspore.mint.where")
    assert payload["grad"]["mode"] == "unknown"
    assert payload["grad"]["differentiable"] == "unknown"


def test_xlogy_grad_remains_branch_scoped_and_unknown():
    payload = query_explain("mindspore.mint.xlogy")
    assert payload["grad"]["mode"] == "explicit_bprop"
    assert payload["grad"]["differentiable"] == "yes"
    assert set(payload["grad"]["backward_primitives"]) == {"Xlogy", "XLogYScalarSelf", "XLogYScalarOther"}


def test_dropout_func_expansion_detects_aclnn_from_expanded_primitives():
    entry = query_api("mindspore.mint.nn.Dropout")
    assert entry["call_chain_kind"] == "construct_mapped"
    mode, path_kind, direct, effective = query_aclnn("mindspore.mint.nn.Dropout")
    assert mode != "none", f"func_expansion should detect aclnn via expanded primitives, got mode={mode}"
    assert effective, "should have aclnn effective_interfaces"
    assert path_kind == "composite_to_aclnn"


def test_dropout_functional_also_detects_aclnn():
    mode, path_kind, direct, effective = query_aclnn("mindspore.mint.nn.functional.dropout")
    assert mode != "none", f"functional dropout should detect aclnn, got mode={mode}"
    assert effective, "should have aclnn effective_interfaces"


def test_dropout_class_grad_impl_matches_functional_dropout():
    """Class Dropout should have the same grad.impl as functional.dropout,
    since both resolve to the same func_expansion primitives."""
    class_entry = query_explain("mindspore.mint.nn.Dropout")
    func_entry = query_explain("mindspore.mint.nn.functional.dropout")
    class_grad = class_entry.get("grad", {})
    func_grad = func_entry.get("grad", {})
    assert class_grad["impl"], "nn.Dropout grad.impl should not be empty"
    assert class_grad["backward_primitives"], "nn.Dropout backward_primitives should not be empty"
    class_prims = {item["primitive"] for item in class_grad["impl"]}
    func_prims = {item["primitive"] for item in func_grad["impl"]}
    assert class_prims == func_prims, f"grad.impl primitives should match: class={class_prims}, func={func_prims}"


def test_cross_entropy_loss_inherits_from_functional_cross_entropy():
    """CrossEntropyLoss construct delegates to cross_entropy_ext, which is the
    impl_symbol of functional.cross_entropy. The class should inherit primitive
    and grad information from that functional API."""
    entry = query_api("mindspore.mint.nn.CrossEntropyLoss")
    assert entry["primitive"], "CrossEntropyLoss should inherit primitives from functional.cross_entropy"
    func_entry = query_api("mindspore.mint.nn.functional.cross_entropy")
    assert set(entry["primitive"]) == set(func_entry["primitive"])
    class_grad = entry.get("grad", {})
    func_grad = func_entry.get("grad", {})
    if func_grad.get("impl"):
        assert class_grad["impl"], "CrossEntropyLoss should inherit grad.impl from functional"


def test_abs_cpu_gpu_support_detected_via_wrapper_macro():
    """Abs is registered via wrapper macros (ARITHMETIC_SELF_CPU_REGISTER and
    MS_ELEWISE_KERNEL_FACTORY_REG_BY_CREATOR). The index should detect CPU and
    GPU support for both pynative and graph modes."""
    entry = query_api("mindspore.mint.abs")
    sm = entry["support_matrix"]
    assert sm["pynative"]["cpu"] == "yes", f"pynative cpu should be yes, got {sm['pynative']['cpu']}"
    assert sm["pynative"]["gpu"] == "yes", f"pynative gpu should be yes, got {sm['pynative']['gpu']}"
    assert sm["graph_kbk_o0"]["cpu"] == "yes", f"graph cpu should be yes, got {sm['graph_kbk_o0']['cpu']}"
    assert sm["graph_kbk_o0"]["gpu"] == "yes", f"graph gpu should be yes, got {sm['graph_kbk_o0']['gpu']}"
    kp = entry["path_hints"]["kernel_paths"]
    assert kp["pynative"]["cpu"], "pynative cpu kernel_paths should not be empty"
    assert kp["pynative"]["gpu"], "pynative gpu kernel_paths should not be empty"


def test_op_def_paths_anchor_uses_yaml_root_key():
    """Verify op_def_paths anchors use the YAML document root key (e.g. 'sub_ext')
    rather than the filename stem (e.g. 'sub_ext_op')."""
    entry = query_api("mindspore.mint.sub")
    for item in entry["path_hints"]["op_def_paths"]:
        if isinstance(item, dict):
            assert not item["anchor"].endswith("_op"), (
                f"anchor should be YAML root key, not filename stem: {item['anchor']}"
            )


def test_where_path_hints_keep_overload_yaml_and_support_buckets():
    payload = query_explain("mindspore.mint.where")
    assert _mixed_paths(payload["path_hints"]["api_def_paths"]) == ["mindspore/ops/api_def/where.yaml"]
    assert _mixed_paths(payload["path_hints"]["op_def_paths"]) == [
        "mindspore/ops/op_def/yaml/non_zero_ext_op.yaml",
        "mindspore/ops/op_def/yaml/select_op.yaml",
    ]
    dispatch = _mixed_paths(payload["path_hints"]["dispatch_paths"])
    assert "mindspore/ccsrc/frontend/operator/composite/auto_generate/functional_map.cc" in dispatch
    assert "mindspore/ccsrc/pynative/forward/pyboost/auto_generate/pyboost_overload_functions.cc" in dispatch


def test_index_select_infer_paths_capture_real_infer_files():
    payload = query_explain("mindspore.mint.index_select")
    infer_paths = set(_mixed_paths(payload["path_hints"]["infer_paths"]))
    assert "mindspore/ops/infer/ops_func_impl/index_select.cc" in infer_paths
    assert "mindspore/ops/infer/symbol_ops_impl/index_select.cc" in infer_paths


def test_query_script_reports_missing_index_without_yaml_fallback(tmp_path: Path):
    missing_db = tmp_path / "missing.db"
    result = subprocess.run(
        [sys.executable, str(QUERY_SCRIPT), "--db", str(missing_db), "api", "--name", "mindspore.mint.sum"],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert payload["error"] == "index unavailable"


def test_generated_db_contains_deterministic_meta_and_support_rows():
    conn = sqlite3.connect(MINT_DB)
    try:
        row = conn.execute("SELECT generated_at, api_count FROM schema_meta WHERE id = 1").fetchone()
        assert row == ("1970-01-01T00:00:00+00:00", 379)
        count = conn.execute(
            """
            SELECT COUNT(*)
            FROM api_support s
            JOIN api a ON a.api_id = s.api_id
            WHERE a.api_name = 'mindspore.mint.nn.functional.dropout'
            """
        ).fetchone()[0]
        assert count == 6
    finally:
        conn.close()


def test_generated_mint_methodology_describes_execution_chain_closure_rules():
    text = MINT_METHODOLOGY.read_text(encoding="utf-8")
    assert "MindSpore Mint API Index Builder Methodology" in text
    assert "generate_mindspore_failure_index.py" in text
    assert "Resolve the real execution chain" in text
    assert "every expanded primitive closes" in text
    assert "implicit `Ext -> base primitive` fallback" in text
    assert "call_chain_kind" in text
    assert "functional_overload" in text
    assert "`py_method` = Python fallback" in text
    assert "raise/error stub -> `no`" in text
    assert "Single-primitive view ops are a special PYNATIVE case" in text
    assert "pyboost_api.cc -> pyboost_core.cc -> kernel::pyboost::<view_op>() -> *_view_impl" in text
    assert "FlattenExt -> flatten_ext_impl -> reshape_impl" in text
    assert "Single-primitive `graph_view` ops are a special GRAPH case" in text
    assert "MS_HOST_REG_KERNEL(<Primitive>, ...)" in text
    assert "RT_KERNEL + IsNopNode + nop_op_to_memcpy_/MemoryCopyAsync" in text
    assert "`functional_overload` ACLNN facts must also be aggregated branch by branch" in text
    assert "`aclnn.effective_interfaces` means the final aclnn interfaces actually reached by the execution chain" in text
    assert "`path_hints` is grouped for LLM retrieval" in text
    assert "`dispatch_paths` stores branch selection and overload routing files" in text
    assert "`implementation_paths` stores the real Python or C++ implementation entry files" in text
    assert "`kernel_paths` are split by exec mode and backend" in text
    assert "`infer_paths` records real InferShape / InferType implementation files" in text


def _path_strs(entries):
    """Extract path strings from a list of PathEntry objects."""
    return [e.path for e in entries]


def _mixed_paths(items):
    """Extract path strings from a query-output mixed list (str | dict)."""
    return [item["path"] if isinstance(item, dict) else item for item in items]


def test_collect_kernel_path_hints_tighten_replaces_polluted_ascend_paths():
    """Verify that the tighten phase in collect_kernel_path_hints replaces
    suffix-stripping-polluted Ascend evidence paths with precise content-based
    map entries keyed by actual primitive names."""
    builder = load_builder_module()
    PE = builder.PathEntry

    class MockIndex:
        pyboost_ascend_impl_map = {
            "SubExt": [PE("ops/kernel/ascend/aclnn/pyboost_impl/auto_generate/sub_ext.h", "SubExtAscend::Call(")],
            "SubScalar": [
                PE("ops/kernel/ascend/aclnn/pyboost_impl/customize/sub_scalar.cc", "SubScalarAscend::Call("),
                PE("ops/kernel/ascend/aclnn/pyboost_impl/customize/sub_scalar.h", "SubScalarAscend::Call("),
            ],
            "Sub": [PE("ops/kernel/ascend/aclnn/pyboost_impl/customize/sub.cc", "SubAscend::Call(")],
        }
        ascend_kernel_impl_map = {
            "SubExt": [PE("ops/kernel/ascend/aclnn/kernel_mod_impl/customize/sub_ext_aclnn_kernel.cc", "class SubExtAscend")],
            "SubScalar": [PE("ops/kernel/ascend/aclnn/kernel_mod_impl/customize/sub_scalar_aclnn_kernel.cc", "class SubScalarAscend")],
            "Sub": [PE("ops/kernel/ascend/aclnn/kernel_mod_impl/customize/sub_aclnn_kernel.cc", "class SubAscend")],
        }

    support_targets = [
        {"primitive": "SubExt", "op_yaml": "sub_ext_op.yaml"},
        {"primitive": "SubScalar", "op_yaml": "sub_scalar_op.yaml"},
    ]
    polluted_evidence = {
        "pynative": {
            "ascend": [
                {"path": "ops/kernel/ascend/aclnn/pyboost_impl/customize/sub.cc"},
                {"path": "ops/kernel/ascend/aclnn/pyboost_impl/auto_generate/sub_ext.h"},
            ],
            "cpu": [{"path": "ops/kernel/cpu/native/arithmetic_ext_cpu_kernel.cc"}],
            "gpu": [],
        },
        "graph_kbk_o0": {
            "ascend": [
                {"path": "ops/kernel/ascend/aclnn/kernel_mod_impl/customize/sub_aclnn_kernel.cc"},
                {"path": "ops/kernel/ascend/aclnn/kernel_mod_impl/customize/sub_ext_aclnn_kernel.cc"},
            ],
            "cpu": [],
            "gpu": [],
        },
    }
    polluted_aclnn = [
        {"path": "ops/kernel/ascend/aclnn/pyboost_impl/customize/sub.cc"},
    ]
    result = builder.collect_kernel_path_hints(
        MockIndex(), support_targets, polluted_evidence, set(), aclnn_evidence=polluted_aclnn,
    )
    pynative_ascend_paths = _path_strs(result["pynative"]["ascend"])
    graph_ascend_paths = _path_strs(result["graph_kbk_o0"]["ascend"])
    assert "ops/kernel/ascend/aclnn/pyboost_impl/customize/sub.cc" not in pynative_ascend_paths
    assert "ops/kernel/ascend/aclnn/kernel_mod_impl/customize/sub_aclnn_kernel.cc" not in graph_ascend_paths
    assert "ops/kernel/ascend/aclnn/pyboost_impl/auto_generate/sub_ext.h" in pynative_ascend_paths
    assert "ops/kernel/ascend/aclnn/pyboost_impl/customize/sub_scalar.cc" in pynative_ascend_paths
    assert "ops/kernel/ascend/aclnn/kernel_mod_impl/customize/sub_ext_aclnn_kernel.cc" in graph_ascend_paths
    assert "ops/kernel/ascend/aclnn/kernel_mod_impl/customize/sub_scalar_aclnn_kernel.cc" in graph_ascend_paths
    cpu_paths = _path_strs(result["pynative"]["cpu"])
    assert cpu_paths == ["ops/kernel/cpu/native/arithmetic_ext_cpu_kernel.cc"]
    for entry in result["pynative"]["ascend"]:
        assert entry.anchor, f"Ascend kernel entry {entry.path} should have an anchor"


def test_collect_kernel_path_hints_tighten_preserves_non_ascend_paths():
    """Verify the tighten phase only replaces Ascend paths and leaves
    CPU/GPU kernel paths from evidence untouched."""
    builder = load_builder_module()
    PE = builder.PathEntry

    class MockIndex:
        pyboost_ascend_impl_map = {"Abs": [PE("ascend/abs.h", "AbsAscend::Call(")]}
        ascend_kernel_impl_map = {"Abs": [PE("ascend/abs_kernel.cc", "class AbsAscend")]}

    support_targets = [{"primitive": "Abs"}]
    evidence = {
        "pynative": {
            "ascend": [{"path": "ascend/old_abs.cc"}],
            "cpu": [{"path": "cpu/abs_kernel.cc"}],
            "gpu": [{"path": "gpu/abs_kernel.cu"}],
        },
        "graph_kbk_o0": {
            "ascend": [{"path": "ascend/old_abs_kbk.cc"}],
            "cpu": [{"path": "cpu/abs_fallback.cc"}],
            "gpu": [{"path": "gpu/abs_fallback.cu"}],
        },
    }
    result = builder.collect_kernel_path_hints(MockIndex(), support_targets, evidence, set())
    assert _path_strs(result["pynative"]["ascend"]) == ["ascend/abs.h"]
    assert _path_strs(result["graph_kbk_o0"]["ascend"]) == ["ascend/abs_kernel.cc"]
    assert _path_strs(result["pynative"]["cpu"]) == ["cpu/abs_kernel.cc"]
    assert _path_strs(result["pynative"]["gpu"]) == ["gpu/abs_kernel.cu"]
    assert _path_strs(result["graph_kbk_o0"]["cpu"]) == ["cpu/abs_fallback.cc"]
    assert _path_strs(result["graph_kbk_o0"]["gpu"]) == ["gpu/abs_fallback.cu"]
    assert result["pynative"]["ascend"][0].anchor == "AbsAscend::Call("
    assert result["graph_kbk_o0"]["ascend"][0].anchor == "class AbsAscend"


def test_collect_kernel_path_hints_filters_python_paths():
    """Verify that Python dispatch/binding files are excluded from kernel_paths."""
    builder = load_builder_module()
    PE = builder.PathEntry

    class MockIndex:
        pyboost_ascend_impl_map = {}
        ascend_kernel_impl_map = {}

    evidence = {
        "pynative": {
            "ascend": [],
            "cpu": [
                {"path": "python/mindspore/ops/tensor_method.py", "anchor": "def tensor_sub_ext"},
                {"path": "python/mindspore/ops/auto_generate/gen_ops_def.py", "anchor": "sub_ext"},
                {"path": "ops/kernel/cpu/native/arithmetic_ext_cpu_kernel.cc", "anchor": "MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, SubExt)"},
            ],
            "gpu": [
                {"path": "python/mindspore/ops/tensor_method.py", "anchor": "def tensor_sub_ext"},
                {"path": "ops/kernel/gpu/math/arithmetic_ext_gpu_kernel.cc", "anchor": "MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, SubExt)"},
            ],
        },
        "graph_kbk_o0": {"ascend": [], "cpu": [], "gpu": []},
    }
    result = builder.collect_kernel_path_hints(MockIndex(), [], evidence, set())
    cpu_paths = _path_strs(result["pynative"]["cpu"])
    gpu_paths = _path_strs(result["pynative"]["gpu"])
    for p in cpu_paths + gpu_paths:
        assert not p.endswith(".py"), f"Python file {p} should not appear in kernel_paths"
    assert "ops/kernel/cpu/native/arithmetic_ext_cpu_kernel.cc" in cpu_paths
    assert "ops/kernel/gpu/math/arithmetic_ext_gpu_kernel.cc" in gpu_paths


def test_primitive_kernel_candidates_ext_stripping_produces_base_name():
    """Document that primitive_kernel_candidates strips Ext suffix.
    The tighten phase in collect_kernel_path_hints overrides file-name matches
    with content-based maps, so this stripping no longer pollutes kernel_paths."""
    builder = load_builder_module()
    candidates = builder.primitive_kernel_candidates("SubExt")
    assert "SubExt" in candidates
    assert "Sub" in candidates
    no_strip = builder.primitive_kernel_candidates("SubExt", allow_suffix_stripping=False)
    assert "Sub" not in no_strip


def test_path_entry_anchor_roundtrips_through_db(tmp_path: Path):
    """Verify that PathEntry anchors survive write→read through SQLite."""
    builder = load_builder_module()
    PE = builder.PathEntry
    db_file = tmp_path / "anchor_test.db"
    conn = sqlite3.connect(db_file)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("""
        CREATE TABLE api (api_id INTEGER PRIMARY KEY, api_name TEXT NOT NULL UNIQUE,
        category TEXT, api_level TEXT, trust_level TEXT, fact_origin TEXT,
        call_chain_kind TEXT, resolution_kind TEXT, implementation_type TEXT,
        support_reason_kind TEXT, alias_of TEXT DEFAULT '', unknown_reason TEXT DEFAULT '',
        grad_mode TEXT, grad_differentiable TEXT, aclnn_mode TEXT, aclnn_path_kind TEXT,
        terminal_symbol TEXT DEFAULT '', terminal_kind TEXT DEFAULT '',
        execution_entry TEXT DEFAULT '', func_op_is_func_op INTEGER DEFAULT 0,
        summary TEXT)
    """)
    conn.execute("""
        CREATE TABLE api_path (api_id INTEGER NOT NULL, path_kind TEXT NOT NULL,
        ordinal INTEGER NOT NULL, path TEXT NOT NULL, anchor TEXT NOT NULL DEFAULT '',
        PRIMARY KEY (api_id, path_kind, ordinal),
        FOREIGN KEY (api_id) REFERENCES api(api_id) ON DELETE CASCADE)
    """)
    conn.execute("INSERT INTO api(api_id, api_name, category, api_level, trust_level, fact_origin, "
                 "call_chain_kind, resolution_kind, implementation_type, support_reason_kind, "
                 "grad_mode, grad_differentiable, aclnn_mode, aclnn_path_kind, summary) "
                 "VALUES (1, 'test.api', 'mint', 'operator_api', 'certain', 'direct', "
                 "'generated_binding', 'real_terminal', 'single_op', 'direct_support', "
                 "'unknown', 'unknown', 'unknown', 'unknown', 'test')")
    entries = [
        PE("kernel/ascend/sub_ext.h", "SubExtAscend::Call("),
        PE("kernel/cpu/arith.cc", ""),
    ]
    builder._insert_path_rows(conn, 1, "kernel_path_pynative_ascend", entries)
    conn.commit()
    rows = conn.execute(
        "SELECT path, anchor FROM api_path WHERE api_id = 1 ORDER BY ordinal"
    ).fetchall()
    assert rows[0] == ("kernel/ascend/sub_ext.h", "SubExtAscend::Call(")
    assert rows[1] == ("kernel/cpu/arith.cc", "")
    loaded = builder._path_list_from_db(
        conn,
        "SELECT path, anchor FROM api_path WHERE api_id = ? AND path_kind = 'kernel_path_pynative_ascend' ORDER BY ordinal",
        (1,),
    )
    assert loaded[0] == {"path": "kernel/ascend/sub_ext.h", "anchor": "SubExtAscend::Call("}
    assert loaded[1] == "kernel/cpu/arith.cc"
    conn.close()


def test_pyboost_ascend_impl_map_returns_path_entries_with_anchors():
    """Verify that _load_pyboost_ascend_impl_map produces PathEntry with
    XxxAscend::Call( anchors rather than plain strings."""
    builder = load_builder_module()
    PE = builder.PathEntry
    index = builder.SourceIndex(local_mindspore_repo())
    sub_ext_entries = index.pyboost_ascend_impl_map.get("SubExt", [])
    assert sub_ext_entries, "SubExt should exist in pyboost_ascend_impl_map"
    for entry in sub_ext_entries:
        assert isinstance(entry, PE), f"Expected PathEntry, got {type(entry)}"
        assert entry.anchor == "SubExtAscend::Call("


def test_ascend_kernel_impl_map_returns_path_entries_with_anchors():
    """Verify that _load_ascend_kernel_impl_map produces PathEntry with
    class or macro anchors."""
    builder = load_builder_module()
    PE = builder.PathEntry
    index = builder.SourceIndex(local_mindspore_repo())
    sub_ext_entries = index.ascend_kernel_impl_map.get("SubExt", [])
    assert sub_ext_entries, "SubExt should exist in ascend_kernel_impl_map"
    for entry in sub_ext_entries:
        assert isinstance(entry, PE), f"Expected PathEntry, got {type(entry)}"
        assert entry.anchor, f"anchor should be non-empty for {entry.path}"
        assert "SubExt" in entry.anchor


def test_infer_path_map_returns_path_entries_with_anchors():
    """Verify that _load_infer_path_map produces PathEntry with content-derived
    anchors (FuncImpl class or REG_SYMBOL_OP_BUILDER) instead of plain stems."""
    builder = load_builder_module()
    PE = builder.PathEntry
    index = builder.SourceIndex(local_mindspore_repo())
    entries = index.infer_path_map.get("index_select", [])
    assert entries, "index_select should exist in infer_path_map"
    anchors = {entry.anchor for entry in entries}
    assert "IndexSelectFuncImpl" in anchors, f"Expected FuncImpl anchor, got {anchors}"
    assert 'REG_SYMBOL_OP_BUILDER("IndexSelect")' in anchors, f"Expected REG_SYMBOL anchor, got {anchors}"
    for entry in entries:
        assert isinstance(entry, PE), f"Expected PathEntry, got {type(entry)}"
        assert entry.anchor != "index_select", f"anchor should not be plain stem: {entry.anchor}"


def test_exact_terminal_backend_rule_requires_exact_kernel_or_custom_kernel_match():
    builder = load_builder_module()
    index = builder.SourceIndex(local_mindspore_repo())

    assert builder.analyze_pynative_exact_terminal_backend(index, "BitwiseXorTensor", "cpu")[0] == "no"
    assert builder.analyze_pynative_exact_terminal_backend(index, "BitwiseXorTensor", "gpu")[0] == "no"
    assert builder.analyze_pynative_exact_terminal_backend(index, "BitwiseAndTensor", "cpu")[0] == "yes"
    assert builder.analyze_pynative_exact_terminal_backend(index, "BitwiseAndTensor", "gpu")[0] == "yes"


def test_certain_strong_apis_no_needs_manual_review():
    """APIs with trust_level certain or strong should not carry needs_manual_review."""
    certain_strong_apis = [
        "mindspore.mint.allclose",
        "mindspore.mint.cat",
        "mindspore.mint.equal",
        "mindspore.mint.nn.functional.elu",
    ]
    for api_name in certain_strong_apis:
        entry = query_api(api_name)
        assert entry["trust_level"] in ("certain", "strong"), (
            f"{api_name} expected certain/strong, got {entry['trust_level']}"
        )
        assert "needs_manual_review" not in entry["flags"], (
            f"{api_name} (trust={entry['trust_level']}) should not have needs_manual_review"
        )


def test_weak_api_keeps_needs_manual_review():
    """Direct support closure should clear weak/manual-review status."""
    entry = query_api("mindspore.mint.cdist")
    assert entry["trust_level"] == "certain", (
        f"cdist expected certain trust, got {entry['trust_level']}"
    )
    assert "needs_manual_review" not in entry["flags"]


def test_reshape_resolves_primitive_and_support_via_module_import():
    """mint.reshape uses `return mindspore.ops.function.array_func.reshape(...)` which
    requires recognizing `import mindspore` as a valid base for attribute chains."""
    entry = query_api("mindspore.mint.reshape")
    assert entry["primitive"] == ["Reshape"], (
        f"reshape should resolve to Reshape primitive, got {entry['primitive']}"
    )
    assert entry["call_chain_kind"] != "unresolved", (
        f"reshape call_chain should be resolved, got {entry['call_chain_kind']}"
    )
    assert entry["unknown_reason"] != "terminal_symbol_unresolved", (
        f"reshape should not have terminal_symbol_unresolved, got {entry['unknown_reason']}"
    )
    sm = entry["support_matrix"]
    assert sm["pynative"]["cpu"] == "yes", f"reshape pynative cpu should be yes, got {sm['pynative']['cpu']}"
    assert sm["pynative"]["gpu"] == "yes", f"reshape pynative gpu should be yes, got {sm['pynative']['gpu']}"


def test_t_resolves_primitive_via_module_import():
    """mint.t uses `return mindspore.ops.auto_generate.t_ext(...)` — same module-import pattern."""
    entry = query_api("mindspore.mint.t")
    assert entry["primitive"] == ["TExt"], (
        f"t should resolve to TExt primitive, got {entry['primitive']}"
    )
    assert entry["call_chain_kind"] != "unresolved", (
        f"t call_chain should be resolved, got {entry['call_chain_kind']}"
    )
