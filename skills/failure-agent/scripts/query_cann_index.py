#!/usr/bin/env python3
"""Read-only query helper for CANN SQLite indexes."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

try:
    import sqlite3
except ImportError:  # pragma: no cover
    sqlite3 = None


SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_ROOT = SCRIPT_DIR.parent
DEFAULT_ERROR_DB = SKILL_ROOT / "reference" / "index" / "cann_error_index.db"
DEFAULT_ACLNN_DB = SKILL_ROOT / "reference" / "index" / "cann_aclnn_api_index.db"


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


def check_index_available(db_path: Path) -> dict | None:
    if sqlite3 is None:
        return unavailable_payload("sqlite unavailable", db_path)
    if not db_path.exists():
        return unavailable_payload("index unavailable", db_path)
    return None


def has_table(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    return row is not None


def detect_db_kind(conn: sqlite3.Connection) -> str:
    if has_table(conn, "error_entry"):
        return "error_index"
    if has_table(conn, "api") and has_table(conn, "api_param"):
        return "aclnn_index"
    return "unsupported"


def build_meta_payload(conn: sqlite3.Connection, db_kind: str) -> dict:
    row = conn.execute("SELECT * FROM schema_meta WHERE id = 1").fetchone()
    if row is None:
        return {"ok": False, "error": "schema_meta missing", "db_kind": db_kind}
    meta = {
        "schema_version": str(row["schema_version"]),
        "generator_name": str(row["generator_name"]),
        "generator_version": str(row["generator_version"]),
        "generated_at": str(row["generated_at"]),
        "source_mode": str(row["source_mode"]),
        "source_repo_url": str(row["source_repo_url"]),
        "source_branch": str(row["source_branch"]),
        "source_commit": str(row["source_commit"]),
        "source_repository_count": int(row["source_repository_count"]),
        "source_repositories": [
            {
                "name": str(item["name"]),
                "repo_url": str(item["repo_url"]),
                "branch": str(item["branch"]),
                "commit": str(item["commit_hash"]),
                "source_type": str(item["source_type"]),
            }
            for item in conn.execute(
                "SELECT name, repo_url, branch, commit_hash, source_type FROM source_repository ORDER BY repo_id"
            ).fetchall()
        ],
    }
    if db_kind == "error_index":
        meta["entry_count"] = int(row["entry_count"])
    else:
        meta["api_count"] = int(row["api_count"])
    return {"ok": True, "db_kind": db_kind, "meta": meta}


def mismatch_payload(db_kind: str, command: str) -> dict:
    return {"ok": False, "error": "command/db mismatch", "db_kind": db_kind, "command": command}


def examples() -> list[str]:
    return [
        "python scripts/query_cann_index.py 561003",
        "python scripts/query_cann_index.py EZ1001",
        "python scripts/query_cann_index.py aclnnReduceSum",
        "python scripts/query_cann_index.py search --text \"parameter invalid\"",
        "python scripts/query_cann_index.py --db reference/index/cann_error_index.db code --value 561003",
        "python scripts/query_cann_index.py --db reference/index/cann_aclnn_api_index.db api --name aclnnAbs",
    ]


def cli_error_payload(error: str, hint: str, **extra: object) -> dict:
    payload = {"ok": False, "error": error, "hint": hint, "examples": examples()}
    payload.update(extra)
    return payload


def normalize_error_code_value(value: str) -> tuple[str, str]:
    match = re.search(r"\d+", value)
    return (match.group(0), value) if match else (value, value)


def looks_like_error_code(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z]{0,3}\d{3,}", value))


def looks_like_aclnn_symbol(value: str) -> bool:
    return value.startswith("aclnn")


def default_db_for_command(command: str) -> Path:
    if command in {"api", "resolve"}:
        return DEFAULT_ACLNN_DB
    return DEFAULT_ERROR_DB


def build_error_code_payload(conn: sqlite3.Connection, value: str) -> dict:
    normalized_value, original_value = normalize_error_code_value(value)
    try:
        numeric_value = int(normalized_value)
    except ValueError:
        return {"ok": False, "error": "invalid code", "value": value}
    row = conn.execute(
        "SELECT scope, code, name, meaning, solution FROM error_entry WHERE code = ? ORDER BY ordinal LIMIT 1",
        (numeric_value,),
    ).fetchone()
    if row is None:
        return {
            "ok": False,
            "error": "code not found",
            "value": original_value,
            "normalized_value": normalized_value,
        }
    return {
        "ok": True,
        "db_kind": "error_index",
        "entry": {
            "scope": str(row["scope"]),
            "code": int(row["code"]),
            "name": str(row["name"]),
            "meaning": str(row["meaning"]),
            "solution": str(row["solution"]),
        },
    }


def parse_args(argv: list[str]) -> argparse.Namespace | dict:
    db_value = None
    remaining: list[str] = []
    idx = 0
    while idx < len(argv):
        item = argv[idx]
        if item == "--db":
            if idx + 1 >= len(argv):
                return cli_error_payload("missing db", "Provide --db followed by an index DB path.")
            db_value = argv[idx + 1]
            idx += 2
            continue
        if item.startswith("--db="):
            db_value = item.split("=", 1)[1]
            idx += 1
            continue
        remaining.append(item)
        idx += 1
    if not remaining:
        return cli_error_payload("missing command", "Use one of: meta, code, error, search, api, resolve.")

    command = remaining[0]
    values = remaining[1:]
    if command not in {"meta", "code", "error", "search", "api", "resolve"} and len(remaining) == 1:
        if looks_like_aclnn_symbol(command):
            values = [command]
            command = "api"
        elif looks_like_error_code(command):
            values = [command]
            command = "code"
    command = "code" if command == "error" else command
    if command not in {"meta", "code", "search", "api", "resolve"}:
        return cli_error_payload(
            "unknown command",
            "Use one of: meta, code, error, search, api, resolve.",
            command=remaining[0],
            db_path=db_value,
        )
    if not db_value:
        db_value = str(default_db_for_command(command))

    parser = argparse.ArgumentParser(description="Query CANN SQLite indexes.")
    parser.add_argument("--value", default="")
    parser.add_argument("--text", default="")
    parser.add_argument("--name", default="")
    parser.add_argument("--symbol", default="")
    parser.add_argument("positional", nargs="*")
    try:
        parsed = parser.parse_args(values)
    except SystemExit:
        return cli_error_payload("invalid arguments", "Use the documented flags for the selected query command.", command=command)

    args = argparse.Namespace(
        db=db_value,
        command=command,
        value=parsed.value,
        text=parsed.text,
        name=parsed.name,
        symbol=parsed.symbol,
    )
    if command == "code" and not args.value and parsed.positional:
        args.value = parsed.positional[0]
    if command == "search" and not args.text and parsed.positional:
        args.text = " ".join(parsed.positional)
    if command == "api" and not args.name and parsed.positional:
        args.name = parsed.positional[0]
    if command == "resolve" and not args.symbol and parsed.positional:
        args.symbol = parsed.positional[0]
    return args


def missing_required_arg_payload(args: argparse.Namespace) -> dict | None:
    command = args.command
    if command == "code" and not args.value:
        return cli_error_payload("missing value", "Use code --value <numeric-code> or code <numeric-code>.", command=command)
    if command == "search" and not args.text:
        return cli_error_payload("missing text", "Use search --text <query text>.", command=command)
    if command == "api" and not args.name:
        return cli_error_payload("missing name", "Use api --name <aclnnApiName>.", command=command)
    if command == "resolve" and not args.symbol:
        return cli_error_payload("missing symbol", "Use resolve --symbol <aclnnApiName or workspace symbol>.", command=command)
    return None


def build_error_search_payload(conn: sqlite3.Connection, text: str) -> dict:
    pattern = f"%{text.lower()}%"
    rows = conn.execute(
        """
        SELECT scope, code, name, meaning, solution
        FROM error_entry
        WHERE lower(name) LIKE ? OR lower(meaning) LIKE ? OR lower(solution) LIKE ? OR lower(scope) LIKE ?
        ORDER BY ordinal
        LIMIT 50
        """,
        (pattern, pattern, pattern, pattern),
    ).fetchall()
    return {
        "ok": True,
        "db_kind": "error_index",
        "results": [
            {
                "scope": str(row["scope"]),
                "code": int(row["code"]),
                "name": str(row["name"]),
                "meaning": str(row["meaning"]),
                "solution": str(row["solution"]),
            }
            for row in rows
        ],
    }


def scalar_list(conn: sqlite3.Connection, query: str, params: tuple[object, ...]) -> list[str]:
    return [str(row[0]) for row in conn.execute(query, params).fetchall()]


def fetch_api_row(conn: sqlite3.Connection, api_name: str):
    return conn.execute("SELECT * FROM api WHERE api_name = ?", (api_name,)).fetchone()


def build_aclnn_api_payload(conn: sqlite3.Connection, api_name: str) -> dict:
    row = fetch_api_row(conn, api_name)
    if row is None:
        return {"ok": False, "error": "api not found", "api_name": api_name}
    api_id = int(row["api_id"])
    record: dict[str, object] = {
        "api_name": str(row["api_name"]),
        "doc_names": scalar_list(conn, "SELECT doc_name FROM api_doc_name WHERE api_id = ? ORDER BY ordinal", (api_id,)),
        "source_repo": str(row["source_repo"]),
        "workspace_api": str(row["workspace_api"]),
        "execute_api": str(row["execute_api"]),
        "summary": str(row["summary"]),
        "constraints": scalar_list(conn, "SELECT value_text FROM api_constraint WHERE api_id = ? ORDER BY ordinal", (api_id,)),
        "error_conditions": scalar_list(conn, "SELECT value_text FROM api_error_condition WHERE api_id = ? ORDER BY ordinal", (api_id,)),
        "return_codes": [
            {
                "name": str(item["name"]),
                "code": str(item["code"]),
                "description": str(item["description"]),
            }
            for item in conn.execute(
                "SELECT name, code, description FROM api_return_code WHERE api_id = ? ORDER BY ordinal",
                (api_id,),
            ).fetchall()
        ],
        "evidence": scalar_list(conn, "SELECT value_text FROM api_evidence WHERE api_id = ? ORDER BY ordinal", (api_id,)),
        "extraction_status": str(row["extraction_status"]),
    }
    for param_kind, field_name in (("input", "inputs"), ("output", "outputs")):
        params = []
        for item in conn.execute(
            """
            SELECT name, role, description, dtype, noncontiguous, tensor_rank,
                   layout_templates_json, shape_constraints_json, value_constraints_json,
                   optional_semantics_json, output_relation_json
            FROM api_param
            WHERE api_id = ? AND param_kind = ?
            ORDER BY ordinal
            """,
            (api_id, param_kind),
        ).fetchall():
            param = {
                "name": str(item["name"]),
                "role": str(item["role"]),
            }
            for column in ("description", "dtype", "noncontiguous", "tensor_rank"):
                if item[column]:
                    param[column] = str(item[column])
            for json_column, output_key in (
                ("layout_templates_json", "layout_templates"),
                ("shape_constraints_json", "shape_constraints"),
                ("value_constraints_json", "value_constraints"),
                ("optional_semantics_json", "optional_semantics"),
                ("output_relation_json", "output_relation"),
            ):
                value = json.loads(str(item[json_column]))
                if value:
                    param[output_key] = value
            params.append(param)
        if params:
            record[field_name] = params
    return {"ok": True, "db_kind": "aclnn_index", "api": record}


def build_aclnn_resolve_payload(conn: sqlite3.Connection, symbol: str) -> dict:
    row = conn.execute(
        """
        SELECT api_name
        FROM api
        WHERE api_name = ? OR workspace_api = ? OR execute_api = ?
        ORDER BY api_name
        LIMIT 1
        """,
        (symbol, symbol, symbol),
    ).fetchone()
    if row is None:
        return {"ok": False, "error": "symbol not found", "symbol": symbol}
    payload = build_aclnn_api_payload(conn, str(row["api_name"]))
    if payload.get("ok"):
        payload["resolved_symbol"] = symbol
    return payload


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
        db_kind = detect_db_kind(conn)
        if db_kind == "unsupported":
            payload = {"ok": False, "error": "unsupported schema", "db_path": str(db_path)}
        elif args.command == "meta":
            payload = build_meta_payload(conn, db_kind)
        elif args.command in {"code", "search"}:
            if db_kind != "error_index":
                payload = mismatch_payload(db_kind, args.command)
            elif args.command == "code":
                payload = build_error_code_payload(conn, args.value)
            else:
                payload = build_error_search_payload(conn, args.text)
        else:
            if db_kind != "aclnn_index":
                payload = mismatch_payload(db_kind, args.command)
            elif args.command == "api":
                payload = build_aclnn_api_payload(conn, args.name)
            else:
                payload = build_aclnn_resolve_payload(conn, args.symbol)
    finally:
        conn.close()
    print(json_text(payload, pretty))


if __name__ == "__main__":
    main()
