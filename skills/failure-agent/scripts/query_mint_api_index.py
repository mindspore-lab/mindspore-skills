#!/usr/bin/env python3
"""Read-only query helper for the mint API SQLite index."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import sqlite3
except ImportError:  # pragma: no cover - tested via helper branch
    sqlite3 = None


SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_ROOT = SCRIPT_DIR.parent
DEFAULT_DB = SKILL_ROOT / "reference" / "index" / "mint_api_index.db"


def json_text(data: dict, pretty: bool = False) -> str:
    indent = 2 if pretty else None
    return json.dumps(data, ensure_ascii=False, sort_keys=False, indent=indent)


def extract_output_flags(argv: list[str]) -> tuple[bool, list[str]]:
    pretty = False
    remaining: list[str] = []
    for item in argv:
        if item in {"--pretty", "-p"}:
            pretty = True
            continue
        remaining.append(item)
    return pretty, remaining


def unavailable_payload(kind: str, db_path: Path) -> dict:
    return {"ok": False, "error": kind, "db_path": str(db_path)}


def examples() -> list[str]:
    return [
        "python scripts/query_mint_api_index.py sum",
        "python scripts/query_mint_api_index.py mint.sum",
        "python scripts/query_mint_api_index.py --name mindspore.mint.sum",
        "python scripts/query_mint_api_index.py explain sum",
        "python scripts/query_mint_api_index.py explain mint.sum",
        "python scripts/query_mint_api_index.py search \"reduce sum\"",
        "python scripts/query_mint_api_index.py api --name mindspore.mint.sum",
        "python scripts/query_mint_api_index.py api --name sum --module mint",
        "python scripts/query_mint_api_index.py api --name relu --module nn.functional",
        "python scripts/query_mint_api_index.py api --name log_softmax --module special",
        "python scripts/query_mint_api_index.py explain --name mindspore.mint.sum",
    ]


def cli_error_payload(error: str, hint: str, **extra: object) -> dict:
    payload = {"ok": False, "error": error, "hint": hint, "examples": examples()}
    payload.update(extra)
    return payload


def check_index_available(db_path: Path) -> dict | None:
    if sqlite3 is None:
        return unavailable_payload("sqlite unavailable", db_path)
    if not db_path.exists():
        return unavailable_payload("index unavailable", db_path)
    return None


def scalar_list(conn: sqlite3.Connection, query: str, params: tuple) -> list[str]:
    return [str(row[0]) for row in conn.execute(query, params).fetchall()]


def path_list(conn: sqlite3.Connection, query: str, params: tuple) -> list:
    """Read path+anchor rows.  Return ``str`` when anchor is empty, ``dict`` otherwise."""
    result: list = []
    for row in conn.execute(query, params).fetchall():
        path = str(row[0])
        anchor = str(row[1]) if len(row) > 1 and row[1] else ""
        if anchor:
            result.append({"path": path, "anchor": anchor})
        else:
            result.append(path)
    return result


def path_hints(conn: sqlite3.Connection, api_id: int) -> dict:
    result = {
        "api_def_paths": path_list(
            conn,
            "SELECT path, anchor FROM api_path WHERE api_id = ? AND path_kind = 'api_def_path' ORDER BY ordinal",
            (api_id,),
        ),
        "dispatch_paths": path_list(
            conn,
            "SELECT path, anchor FROM api_path WHERE api_id = ? AND path_kind = 'dispatch_path' ORDER BY ordinal",
            (api_id,),
        ),
        "implementation_paths": path_list(
            conn,
            "SELECT path, anchor FROM api_path WHERE api_id = ? AND path_kind = 'implementation_path' ORDER BY ordinal",
            (api_id,),
        ),
        "op_def_paths": path_list(
            conn,
            "SELECT path, anchor FROM api_path WHERE api_id = ? AND path_kind = 'op_def_path' ORDER BY ordinal",
            (api_id,),
        ),
        "kernel_paths": {
            "pynative": {"ascend": [], "cpu": [], "gpu": []},
            "graph_kbk_o0": {"ascend": [], "cpu": [], "gpu": []},
        },
        "infer_paths": path_list(
            conn,
            "SELECT path, anchor FROM api_path WHERE api_id = ? AND path_kind = 'infer_path' ORDER BY ordinal",
            (api_id,),
        ),
    }
    for exec_mode in ("pynative", "graph_kbk_o0"):
        for backend in ("ascend", "cpu", "gpu"):
            result["kernel_paths"][exec_mode][backend] = path_list(
                conn,
                f"SELECT path, anchor FROM api_path WHERE api_id = ? AND path_kind = 'kernel_path_{exec_mode}_{backend}' ORDER BY ordinal",
                (api_id,),
            )
    return result


def support_matrix(conn: sqlite3.Connection, api_id: int) -> dict[str, dict[str, str]]:
    result = {
        "pynative": {"ascend": "unknown", "cpu": "unknown", "gpu": "unknown"},
        "graph_kbk_o0": {"ascend": "unknown", "cpu": "unknown", "gpu": "unknown"},
    }
    for row in conn.execute(
        "SELECT exec_mode, backend, support_state FROM api_support WHERE api_id = ?",
        (api_id,),
    ).fetchall():
        result[str(row["exec_mode"])][str(row["backend"])] = str(row["support_state"])
    return result


def aclnn_info(conn: sqlite3.Connection, api_id: int, row) -> dict[str, object]:
    return {
        "mode": str(row["aclnn_mode"]),
        "interfaces": scalar_list(
            conn,
            "SELECT interface_name FROM api_aclnn_interface WHERE api_id = ? AND role = 'direct' ORDER BY ordinal",
            (api_id,),
        ),
        "effective_interfaces": scalar_list(
            conn,
            "SELECT interface_name FROM api_aclnn_interface WHERE api_id = ? AND role = 'effective' ORDER BY ordinal",
            (api_id,),
        ),
        "path_kind": str(row["aclnn_path_kind"]),
    }


def evidence_digest(conn: sqlite3.Connection, api_id: int, limit: int) -> list[dict[str, str]]:
    rows = conn.execute(
        """
        SELECT domain, exec_mode, backend, primitive_name, path, kind, anchor, summary
        FROM api_evidence
        WHERE api_id = ?
        ORDER BY domain, exec_mode, backend, ordinal
        LIMIT ?
        """,
        (api_id, limit),
    ).fetchall()
    return [
        {
            "domain": str(row["domain"]),
            "exec_mode": "" if row["exec_mode"] is None else str(row["exec_mode"]),
            "backend": "" if row["backend"] is None else str(row["backend"]),
            "primitive": str(row["primitive_name"]),
            "path": str(row["path"]),
            "kind": str(row["kind"]),
            "anchor": str(row["anchor"]),
            "summary": str(row["summary"]),
        }
        for row in rows
    ]


def fetch_api_row(conn: sqlite3.Connection, api_name: str):
    return conn.execute("SELECT * FROM api WHERE api_name = ?", (api_name,)).fetchone()


def normalize_mint_module(module: str) -> str:
    module = module.strip(".")
    if not module:
        return ""
    if module == "mindspore.mint":
        return "mint"
    if module.startswith("mindspore.mint."):
        return module.removeprefix("mindspore.")
    if module == "mint" or module.startswith("mint."):
        return module
    return f"mint.{module}"


def resolve_api_name(conn: sqlite3.Connection, name: str, module: str = "") -> tuple[str, list[str]]:
    candidates = [name]
    normalized_module = normalize_mint_module(module)
    if normalized_module:
        candidates.append(f"mindspore.{normalized_module}.{name}")
    elif name.startswith("mint."):
        candidates.append(f"mindspore.{name}")
    elif "." not in name:
        candidates.append(f"mindspore.mint.{name}")
        candidates.append(f"mindspore.mint.nn.{name}")
        candidates.append(f"mindspore.mint.nn.functional.{name}")
        candidates.append(f"mindspore.mint.special.{name}")
    for candidate in candidates:
        if fetch_api_row(conn, candidate) is not None:
            return candidate, candidates
    return name, candidates


def build_meta_payload(conn: sqlite3.Connection) -> dict:
    row = conn.execute("SELECT * FROM schema_meta WHERE id = 1").fetchone()
    if row is None:
        return {"ok": False, "error": "schema_meta missing"}
    return {
        "ok": True,
        "meta": {
            "schema_version": str(row["schema_version"]),
            "generator_name": str(row["generator_name"]),
            "generator_version": str(row["generator_version"]),
            "generated_at": str(row["generated_at"]),
            "source_mode": str(row["source_mode"]),
            "source_repo_url": str(row["source_repo_url"]),
            "source_branch": str(row["source_branch"]),
            "source_commit": str(row["source_commit"]),
            "mindspore_version_hint": str(row["mindspore_version_hint"]),
            "generated_after_gen_ops": bool(row["generated_after_gen_ops"]),
            "repo_root_hint": str(row["repo_root_hint"]),
            "api_count": int(row["api_count"]),
            "source_repositories": [
                {
                    "name": str(item["name"]),
                    "repo_url": str(item["repo_url"]),
                    "branch": str(item["branch"]),
                    "commit": str(item["commit_hash"]),
                    "source_type": str(item["source_type"]),
                }
                for item in conn.execute(
                    """
                    SELECT name, repo_url, branch, commit_hash, source_type
                    FROM source_repository
                    ORDER BY repo_id
                    """
                ).fetchall()
            ],
        },
    }


def build_api_payload(conn: sqlite3.Connection, api_name: str, module: str = "") -> dict:
    resolved_name, candidates = resolve_api_name(conn, api_name, module)
    row = fetch_api_row(conn, resolved_name)
    if row is None:
        return {"ok": False, "error": "api not found", "api_name": api_name, "tried": candidates}
    api_id = int(row["api_id"])
    return {
        "ok": True,
        "api": {
            "api": str(row["api_name"]),
            "category": str(row["category"]),
            "api_level": str(row["api_level"]),
            "trust_level": str(row["trust_level"]),
            "fact_origin": str(row["fact_origin"]),
            "call_chain_kind": str(row["call_chain_kind"]),
            "resolution_kind": str(row["resolution_kind"]),
            "implementation_type": str(row["implementation_type"]),
            "support_reason_kind": str(row["support_reason_kind"]),
            "alias_of": str(row["alias_of"]),
            "unknown_reason": str(row["unknown_reason"]),
            "summary": str(row["summary"]),
            "primitive": scalar_list(
                conn,
                "SELECT primitive_name FROM api_primitive WHERE api_id = ? AND role = 'support_terminal' ORDER BY ordinal",
                (api_id,),
            ),
            "possible_primitives": scalar_list(
                conn,
                "SELECT primitive_name FROM api_primitive WHERE api_id = ? AND role = 'possible' ORDER BY ordinal",
                (api_id,),
            ),
            "func_op_expands_to": scalar_list(
                conn,
                "SELECT primitive_name FROM api_primitive WHERE api_id = ? AND role = 'func_expanded' ORDER BY ordinal",
                (api_id,),
            ),
            "support_matrix": support_matrix(conn, api_id),
            "flags": scalar_list(conn, "SELECT flag FROM api_flag WHERE api_id = ? ORDER BY ordinal", (api_id,)),
            "path_hints": path_hints(conn, api_id),
            "aclnn": aclnn_info(conn, api_id, row),
            "grad": {
                "mode": str(row["grad_mode"]),
                "differentiable": str(row["grad_differentiable"]),
                "backward_primitives": scalar_list(
                    conn,
                    "SELECT primitive_name FROM api_grad_primitive WHERE api_id = ? ORDER BY ordinal",
                    (api_id,),
                ),
                "impl": [
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
                ],
            },
        },
    }


def build_explain_payload(conn: sqlite3.Connection, api_name: str, evidence_limit: int, module: str = "") -> dict:
    resolved_name, candidates = resolve_api_name(conn, api_name, module)
    row = fetch_api_row(conn, resolved_name)
    if row is None:
        return {"ok": False, "error": "api not found", "api_name": api_name, "tried": candidates}
    api_id = int(row["api_id"])
    return {
        "ok": True,
        "api_core": {
            "api_name": str(row["api_name"]),
            "implementation_type": str(row["implementation_type"]),
            "trust_level": str(row["trust_level"]),
            "fact_origin": str(row["fact_origin"]),
            "call_chain_kind": str(row["call_chain_kind"]),
            "resolution_kind": str(row["resolution_kind"]),
            "support_reason_kind": str(row["support_reason_kind"]),
        },
        "support_matrix": support_matrix(conn, api_id),
        "primitive": scalar_list(
            conn,
            "SELECT primitive_name FROM api_primitive WHERE api_id = ? AND role = 'support_terminal' ORDER BY ordinal",
            (api_id,),
        ),
        "possible_primitives": scalar_list(
            conn,
            "SELECT primitive_name FROM api_primitive WHERE api_id = ? AND role = 'possible' ORDER BY ordinal",
            (api_id,),
        ),
        "func_op_expands_to": scalar_list(
            conn,
            "SELECT primitive_name FROM api_primitive WHERE api_id = ? AND role = 'func_expanded' ORDER BY ordinal",
            (api_id,),
        ),
        "support_targets": [
            {
                "primitive": str(item["primitive_name"]),
                "api_def": str(item["api_def_name"]),
                "op_yaml": str(item["op_yaml"]),
                "op_def_path": str(item["op_def_path"]),
                "origin_kind": str(item["origin_kind"]),
            }
            for item in conn.execute(
                """
                SELECT primitive_name, api_def_name, op_yaml, op_def_path, origin_kind
                FROM api_support_target
                WHERE api_id = ?
                ORDER BY ordinal
                """,
                (api_id,),
            ).fetchall()
        ],
        "grad": {
            "mode": str(row["grad_mode"]),
            "differentiable": str(row["grad_differentiable"]),
            "backward_primitives": scalar_list(
                conn,
                "SELECT primitive_name FROM api_grad_primitive WHERE api_id = ? ORDER BY ordinal",
                (api_id,),
            ),
            "impl": [
                {
                    "primitive": str(item["primitive_name"]),
                    "kind": str(item["kind"]),
                    "path": str(item["path"]),
                    "anchor": str(item["anchor"]),
                    "scope_kind": str(item["scope_kind"]),
                }
                for item in conn.execute(
                    """
                    SELECT primitive_name, kind, path, anchor, scope_kind
                    FROM api_grad_impl
                    WHERE api_id = ?
                    ORDER BY ordinal
                    """,
                    (api_id,),
                ).fetchall()
            ],
        },
        "aclnn": aclnn_info(conn, api_id, row),
        "flags": scalar_list(conn, "SELECT flag FROM api_flag WHERE api_id = ? ORDER BY ordinal", (api_id,)),
        "path_hints": path_hints(conn, api_id),
        "warnings": {"summary": str(row["summary"])},
        "evidence_digest": evidence_digest(conn, api_id, evidence_limit),
    }


def build_search_payload(conn: sqlite3.Connection, text: str, limit: int = 50) -> dict:
    tokens = [item for item in text.lower().split() if item]
    if not tokens:
        return {"ok": False, "error": "missing text", "text": text}
    clauses = []
    params: list[object] = []
    for token in tokens:
        pattern = f"%{token}%"
        clauses.append(
            """
            (lower(api_name) LIKE ?
             OR lower(summary) LIKE ?
             OR lower(call_chain_kind) LIKE ?
             OR lower(resolution_kind) LIKE ?
             OR lower(support_reason_kind) LIKE ?)
            """
        )
        params.extend([pattern, pattern, pattern, pattern, pattern])
    params.append(limit)
    rows = conn.execute(
        f"""
        SELECT api_name, category, trust_level, call_chain_kind, resolution_kind, support_reason_kind, summary
        FROM api
        WHERE {" OR ".join(clauses)}
        ORDER BY api_name
        LIMIT ?
        """,
        tuple(params),
    ).fetchall()
    return {
        "ok": True,
        "results": [
            {
                "api": str(row["api_name"]),
                "category": str(row["category"]),
                "trust_level": str(row["trust_level"]),
                "call_chain_kind": str(row["call_chain_kind"]),
                "resolution_kind": str(row["resolution_kind"]),
                "support_reason_kind": str(row["support_reason_kind"]),
                "summary": str(row["summary"]),
            }
            for row in rows
        ],
    }


def parse_args(argv: list[str]) -> argparse.Namespace | dict:
    db_value = str(DEFAULT_DB)
    evidence_limit = 12
    remaining: list[str] = []
    idx = 0
    while idx < len(argv):
        item = argv[idx]
        if item == "--db":
            if idx + 1 >= len(argv):
                return cli_error_payload("missing db", "Provide --db followed by mint_api_index.db.")
            db_value = argv[idx + 1]
            idx += 2
            continue
        if item.startswith("--db="):
            db_value = item.split("=", 1)[1]
            idx += 1
            continue
        if item == "--evidence-limit":
            if idx + 1 >= len(argv):
                return cli_error_payload("missing evidence limit", "Provide --evidence-limit followed by an integer.")
            try:
                evidence_limit = int(argv[idx + 1])
            except ValueError:
                return cli_error_payload("invalid evidence limit", "--evidence-limit must be an integer.")
            idx += 2
            continue
        if item.startswith("--evidence-limit="):
            try:
                evidence_limit = int(item.split("=", 1)[1])
            except ValueError:
                return cli_error_payload("invalid evidence limit", "--evidence-limit must be an integer.")
            idx += 1
            continue
        remaining.append(item)
        idx += 1

    if not remaining:
        return cli_error_payload("missing command", "Use one of: meta, api, explain, search.", db_path=db_value)
    command = remaining[0]
    values = remaining[1:]
    if command not in {"meta", "api", "explain", "search"}:
        if len(remaining) == 1:
            values = [command]
            command = "api"
        elif command in {"--name", "--module"} or any(item.startswith("--name=") or item.startswith("--module=") for item in remaining):
            values = remaining
            command = "api"
        else:
            return cli_error_payload("unknown command", "Use one of: meta, api, explain, search.", command=command, db_path=db_value)
    if command not in {"meta", "api", "explain", "search"}:
        return cli_error_payload("unknown command", "Use one of: meta, api, explain, search.", command=command, db_path=db_value)

    parser = argparse.ArgumentParser(description="Query the mint API SQLite index.")
    parser.add_argument("--name", default="")
    parser.add_argument("--module", default="")
    parser.add_argument("--text", default="")
    parser.add_argument("positional", nargs="*")
    try:
        parsed = parser.parse_args(values)
    except SystemExit:
        return cli_error_payload("invalid arguments", "Use --name for api/explain or --text for search.", command=command)

    args = argparse.Namespace(
        db=db_value,
        evidence_limit=evidence_limit,
        command=command,
        name=parsed.name,
        module=parsed.module,
        text=parsed.text,
    )
    if command in {"api", "explain"} and not args.name and parsed.positional:
        args.name = parsed.positional[0]
    if command == "explain" and not args.name and args.text:
        args.command = "search"
    if command == "search" and not args.text and parsed.positional:
        args.text = " ".join(parsed.positional)
    return args


def missing_required_arg_payload(args: argparse.Namespace) -> dict | None:
    if args.command in {"api", "explain"} and not args.name:
        return cli_error_payload("missing name", "Use api --name <api> or explain --name <api>.", command=args.command)
    if args.command == "search" and not args.text:
        return cli_error_payload("missing text", "Use search --text <query text>.", command=args.command)
    return None


def main() -> None:
    pretty, argv = extract_output_flags(sys.argv[1:])
    args = parse_args(argv)
    if isinstance(args, dict):
        print(json_text(args, pretty))
        return
    missing_arg = missing_required_arg_payload(args)
    if missing_arg is not None:
        print(json_text(missing_arg, pretty))
        return
    db_path = Path(args.db).resolve()
    unavailable = check_index_available(db_path)
    if unavailable is not None:
        print(json_text(unavailable, pretty))
        return

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        if args.command == "meta":
            payload = build_meta_payload(conn)
        elif args.command == "api":
            payload = build_api_payload(conn, args.name, args.module)
        elif args.command == "explain":
            payload = build_explain_payload(conn, args.name, args.evidence_limit, args.module)
        else:
            payload = build_search_payload(conn, args.text)
    finally:
        conn.close()
    print(json_text(payload, pretty))


if __name__ == "__main__":
    main()
