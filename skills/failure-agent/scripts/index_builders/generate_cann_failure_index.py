"""Generate minimal CANN failure indexes from source repositories.

Default manual command:
    Run from the failure-agent directory. By default, the script clones the
    configured CANN runtime and ops repositories into a temporary workspace and
    writes SQLite indexes under reference/index.

    python scripts/index_builders/generate_cann_failure_index.py \
        --deterministic \
        --with-error-yaml \
        --with-aclnn-yaml

Local source command:
    Use local checkouts to avoid network clone. Pass exactly one runtime repo
    and one or more ops repos. Repeat --local-ops-repo for each ops repository.

    python scripts/index_builders/generate_cann_failure_index.py \
        --local-runtime-repo D:/path/to/runtime \
        --local-ops-repo D:/path/to/ops-nn \
        --local-ops-repo D:/path/to/ops-math \
        --deterministic \
        --with-error-yaml \
        --with-aclnn-yaml

Parameters:
    --workspace-root DIR
        Temporary clone workspace used for remote builds. Default:
        scripts/index_builders/.tmp/cann.
    --out DIR
        Output directory. Default: reference/index under failure-agent.
    --local-runtime-repo DIR
        Use a local CANN runtime repository instead of cloning the default
        runtime remote. Default: unset.
    --local-ops-repo DIR
        Use a local CANN ops repository instead of cloning default ops remotes.
        Can be passed multiple times. Default: unset.
    --keep-workspace
        Keep the temporary remote clone after the build. Default: delete it.
    --with-error-yaml
        Also write cann_error_index.yaml. Default: disabled.
    --with-aclnn-yaml
        Also write cann_aclnn_api_index.yaml. Default: disabled.
    --with-source-docs
        Also write source markdown documents copied/extracted from repos.
        Default: disabled.
    --with-compact
        Also write aclnn_api_compact.md. Default: disabled.
    --deterministic
        Use deterministic metadata timestamps for reproducible outputs. Default:
        use current UTC timestamp.

Outputs:
    Always writes cann_error_index.db and cann_aclnn_api_index.db. Optional
    flags can additionally write YAML, source docs, and compact review artifacts
    into --out.
"""

from __future__ import annotations

import argparse
import datetime
import html
import json
import os
import re
import shutil
import sqlite3
import stat
import subprocess
import sys
import time
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
FAILURE_AGENT_ROOT = SCRIPT_DIR.parents[1]
REFERENCE_DIR = FAILURE_AGENT_ROOT / "reference" / "index"
ACL_DOC_PATH = REFERENCE_DIR / "aclError.md"
ACLNN_DOC_PATH = REFERENCE_DIR / "aclnnApiError.md"
ERROR_INDEX_PATH = REFERENCE_DIR / "cann_error_index.yaml"
ERROR_DB_PATH = REFERENCE_DIR / "cann_error_index.db"
LEGACY_API_DOCS_DIR = REFERENCE_DIR / "aclnn_api_docs"
INDEX_FILE = REFERENCE_DIR / "cann_aclnn_api_index.yaml"
INDEX_DB_FILE = REFERENCE_DIR / "cann_aclnn_api_index.db"
COMPACT_FILE = REFERENCE_DIR / "aclnn_api_compact.md"
WORKSPACE_DEFAULT = SCRIPT_DIR / ".tmp" / "cann"
MANIFEST_NAME = "sources.json"
GENERATOR_NAME = "generate_cann_failure_index.py"
GENERATOR_VERSION = "1.0.0"
INDEX_SCHEMA_VERSION = "1.1"
DETERMINISTIC_TIMESTAMP = "1970-01-01T00:00:00+00:00"
RUNTIME_REPO = {"name": "runtime", "https": "https://gitcode.com/cann/runtime.git", "ssh": "git@gitcode.com:cann/runtime.git"}
OPS_REPOS = [
    {"name": "ops-nn", "https": "https://gitcode.com/cann/ops-nn.git", "ssh": "git@gitcode.com:cann/ops-nn.git"},
    {"name": "ops-math", "https": "https://gitcode.com/cann/ops-math.git", "ssh": "git@gitcode.com:cann/ops-math.git"},
    {"name": "ops-transformer", "https": "https://gitcode.com/cann/ops-transformer.git", "ssh": "git@gitcode.com:cann/ops-transformer.git"},
    {"name": "ops-cv", "https": "https://gitcode.com/cann/ops-cv.git", "ssh": "git@gitcode.com:cann/ops-cv.git"},
]
SPECIAL_RENAMES: dict[str, tuple[Path, str]] = {
    "aclnnApiError.md": (REFERENCE_DIR, "aclnnApiError.md"),
    "aclnn返回码.md": (REFERENCE_DIR, "aclnnApiError.md"),
    "aclnn返回码.md": (REFERENCE_DIR, "aclnnApiError.md"),
    "aclnn杩斿洖鐮?md": (REFERENCE_DIR, "aclnnApiError.md"),
}
ACL_ERROR_DOC_NAME = "aclError.md"
INVALID_SOLUTION_PATTERNS = [
    "联系技术支持",
    "Link",
    "获取日志后",
    "日志的详细介绍",
    "日志参考",
]


def load_workspace_manifest(workspace_root: Path) -> dict[str, object]:
    manifest_path = workspace_root / MANIFEST_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"Workspace manifest not found: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _force_remove_readonly(func, path, _exc_info):
    os.chmod(path, stat.S_IWRITE)
    for _ in range(5):
        try:
            func(path)
            return
        except PermissionError:
            time.sleep(0.2)
    func(path)


def safe_rmtree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path, onerror=_force_remove_readonly)


def safe_unlink(path: Path) -> None:
    if not path.exists():
        return
    for _ in range(5):
        try:
            path.unlink()
            return
        except PermissionError:
            time.sleep(0.2)
        except FileNotFoundError:
            return
    try:
        path.unlink()
    except (PermissionError, FileNotFoundError):
        return


def prune_empty_parents(path: Path, *, stop_at: Path) -> None:
    current = path
    stop_at = stop_at.resolve()
    while current.exists() and current.resolve() != stop_at:
        try:
            current.rmdir()
        except OSError:
            break
        current = current.parent


def run_workspace_root(base_root: Path, *, keep_workspace: bool) -> Path:
    if keep_workspace:
        return base_root
    return base_root / f"run-{int(time.time() * 1000)}"


def current_timestamp(*, deterministic: bool = False) -> str:
    if deterministic:
        return DETERMINISTIC_TIMESTAMP
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def json_dump_text(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=False)


def json_load_list(value: str | None) -> list[object]:
    if not value:
        return []
    loaded = json.loads(value)
    return list(loaded) if isinstance(loaded, list) else []


def initialize_sqlite_pragmas(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        PRAGMA journal_mode = OFF;
        PRAGMA synchronous = OFF;
        PRAGMA temp_store = MEMORY;
        PRAGMA foreign_keys = OFF;
        """
    )


def normalize_sqlite_header_for_determinism(db_path: Path) -> None:
    fixed_value = (1).to_bytes(4, "big")
    with db_path.open("r+b") as handle:
        for offset in (24, 40, 92):
            handle.seek(offset)
            handle.write(fixed_value)


def run_git(args: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git"] + args,
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def resolve_local_repo(path: str) -> Path | None:
    repo_path = Path(path).resolve()
    if not repo_path.is_dir():
        print(f"[ERROR] Directory does not exist: {repo_path}")
        return None
    print(f"[OK] Using local repo: {repo_path}")
    return repo_path


def get_branch_name(repo_path: Path) -> str:
    result = run_git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_path)
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def get_commit_id(repo_path: Path) -> str:
    result = run_git(["rev-parse", "HEAD"], cwd=repo_path)
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def clone_or_update_repo(repo: dict[str, str], workspace_root: Path) -> Path | None:
    workspace_root.mkdir(parents=True, exist_ok=True)
    dest = workspace_root / repo["name"]
    safe_rmtree(dest)
    result = run_git(["-c", "core.longpaths=true", "clone", "--depth", "1", "--no-tags", repo["https"], str(dest)])
    if result.returncode == 0:
        print(f"[OK] {repo['name']} cloned via HTTPS.")
        return dest
    print(f"[WARN] HTTPS clone failed for {repo['name']}: {result.stderr.strip()}")
    safe_rmtree(dest)
    result = run_git(["-c", "core.longpaths=true", "clone", "--depth", "1", "--no-tags", repo["ssh"], str(dest)])
    if result.returncode == 0:
        print(f"[OK] {repo['name']} cloned via SSH.")
        return dest
    safe_rmtree(dest)
    print(f"[ERROR] Failed to clone {repo['name']}: {result.stderr.strip()}")
    return None


def infer_repo_name(repo_path: Path) -> str:
    return repo_path.name


def resolve_local_ops(local_paths: list[str] | None) -> dict[str, Path]:
    resolved: dict[str, Path] = {}
    if not local_paths:
        return resolved
    expected = {repo["name"] for repo in OPS_REPOS}
    for raw_path in local_paths:
        repo_path = resolve_local_repo(raw_path)
        if repo_path is None:
            continue
        name = infer_repo_name(repo_path)
        if name not in expected:
            print(f"[WARN] Skip unsupported ops repo: {repo_path}")
            continue
        resolved[name] = repo_path
    return resolved


def build_manifest_entry(name: str, repo_path: Path, source_type: str, repo_url: str = "") -> dict[str, str]:
    return {
        "name": name,
        "path": str(repo_path),
        "branch": get_branch_name(repo_path),
        "commit": get_commit_id(repo_path),
        "source_type": source_type,
        "repo_url": repo_url,
    }


def write_manifest(workspace_root: Path, payload: dict[str, object]) -> Path:
    workspace_root.mkdir(parents=True, exist_ok=True)
    manifest_path = workspace_root / MANIFEST_NAME
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def cleanup(remove_legacy_docs: bool = True) -> None:
    if remove_legacy_docs:
        safe_rmtree(LEGACY_API_DOCS_DIR)


def write_source_markdown(src: Path, dest: Path, repo_name: str, branch: str, commit: str) -> None:
    today = datetime.date.today().isoformat()
    metadata_header = (
        f"<!-- Source: {repo_name} | Branch: {branch} | Commit: {commit} | Last updated: {today} -->\n\n"
    )
    content = metadata_header + src.read_text(encoding="utf-8")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.read_text(encoding="utf-8") == content:
        return
    dest.write_text(content, encoding="utf-8")


def find_first_doc(repo_path: Path, target_filename: str) -> Path:
    matches = sorted(repo_path.rglob(target_filename))
    if not matches:
        raise FileNotFoundError(f"{target_filename} not found in {repo_path}")
    return matches[0]


def prepare_input_docs(
    repos: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, str]]]:
    docs: list[dict[str, object]] = []
    repo_infos: list[dict[str, str]] = []

    for repo in sorted(repos, key=lambda item: str(item["name"])):
        repo_name = str(repo["name"])
        repo_path = Path(str(repo["path"]))
        branch = str(repo.get("branch", "unknown"))
        commit = str(repo.get("commit", "unknown"))
        repo_infos.append({"name": repo_name, "branch": branch, "commit": commit})
        for src in sorted(repo_path.rglob("aclnn*.md")):
            rename_target = None
            if src.name == "aclnn返回码.md":
                rename_target = "aclnnApiError.md"
            elif src.name in SPECIAL_RENAMES:
                configured = SPECIAL_RENAMES[src.name]
                if isinstance(configured, tuple):
                    rename_target = str(configured[1])
                else:
                    rename_target = str(configured)
            if rename_target:
                write_source_markdown(src, REFERENCE_DIR / rename_target, repo_name, branch, commit)
                continue
            docs.append({"path": src, "doc_name": src.name, "source_repo": repo_name})
    return docs, repo_infos


def prepare_runtime_error_doc(runtime_info: dict[str, object]) -> None:
    repo_path = Path(str(runtime_info["path"]))
    src = find_first_doc(repo_path, ACL_ERROR_DOC_NAME)
    write_source_markdown(
        src,
        ACL_DOC_PATH,
        str(runtime_info["name"]),
        str(runtime_info.get("branch", "unknown")),
        str(runtime_info.get("commit", "unknown")),
    )


def split_api_names(doc_name: str) -> list[str]:
    stem = Path(doc_name).stem
    return [part.strip() for part in re.split(r"[&,，、]", stem) if part.strip()]


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = text.replace("\u00a0", " ")
    text = re.sub(r'<a name="[^"]+"></a>', "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    return text


def clean_fragment(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(r"<br\s*/?>", "\n", text)
    text = re.sub(r"</?(?:p|term|strong|code|span|div|thead|tbody|colgroup|col|ul|li)[^>]*>", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("&nbsp;", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"。{2,}", "。", text)
    return text.strip(" \n|")


def canonical_param_name(name: str) -> str:
    cleaned = clean_fragment(name)
    cleaned = re.sub(r"[（(].*$", "", cleaned).strip()
    return cleaned


def is_runtime_param(name: str) -> bool:
    return canonical_param_name(name) in {"workspace", "workspaceSize", "executor", "stream"}


def split_sections(text: str) -> dict[str, str]:
    pattern = re.compile(r"^##\s+(.+?)\n", re.M)
    matches = list(pattern.finditer(text))
    sections: dict[str, str] = {}
    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        sections[match.group(1).strip()] = text[start:end].strip()
    return sections


def extract_code_blocks(text: str) -> list[str]:
    return [m.group(1).strip() for m in re.finditer(r"```[^\n]*\n(.*?)```", text, re.S)]


def extract_prototypes(text: str) -> list[dict[str, str]]:
    prototypes: list[dict[str, str]] = []
    bullet_patterns = [
        r"`(aclnn\w+\([^`]+\))`",
        r"-\s*(aclnn\w+\([^)\n]+\))",
    ]
    for block in extract_code_blocks(text):
        signature = clean_fragment(block)
        match = re.search(r"\b(aclnn\w+)\s*\(", signature)
        if match:
            prototypes.append({"name": match.group(1), "signature": signature})
    for pattern in bullet_patterns:
        for match in re.finditer(pattern, text):
            signature = clean_fragment(match.group(1))
            name_match = re.search(r"\b(aclnn\w+)\s*\(", signature)
            if name_match:
                prototypes.append({"name": name_match.group(1), "signature": signature})
    deduped: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in prototypes:
        if item["name"] not in seen:
            deduped.append(item)
            seen.add(item["name"])
    return deduped


def parse_markdown_tables(text: str) -> list[dict[str, object]]:
    tables: list[dict[str, object]] = []
    current: list[str] = []

    def flush() -> None:
        nonlocal current
        if len(current) < 2:
            current = []
            return
        lines = [line.strip() for line in current if line.strip()]
        if len(lines) < 2:
            current = []
            return
        header = [cell.strip() for cell in lines[0].strip("|").split("|")]
        rows = [[cell.strip() for cell in line.strip("|").split("|")] for line in lines[2:]]
        tables.append({"headers": header, "rows": rows})
        current = []

    for line in text.splitlines():
        if line.strip().startswith("|"):
            current.append(line)
        else:
            flush()
    flush()
    return tables


def parse_html_tables(text: str) -> list[dict[str, object]]:
    tables: list[dict[str, object]] = []
    for table_match in re.finditer(r"<table.*?>.*?</table>", text, re.S):
        table_html = table_match.group(0)
        # Some upstream docs miss the opening <tr> for a body row such as the `out` row
        # in addmm-style docs. Recover these rows before extracting cells.
        table_html = re.sub(r"(</tr>\s*)(<td\b)", r"\1<tr>\2", table_html)
        table_html = re.sub(r"(<tbody[^>]*>\s*)(<td\b)", r"\1<tr>\2", table_html)
        rows: list[list[str]] = []
        for row_match in re.finditer(r"<tr.*?>(.*?)</tr>", table_html, re.S):
            cells = [clean_fragment(cell) for cell in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row_match.group(1), re.S)]
            if cells:
                rows.append(cells)
        if rows:
            tables.append({"headers": rows[0], "rows": rows[1:]})
    return tables


def parse_tables(text: str) -> list[dict[str, object]]:
    return parse_markdown_tables(text) + parse_html_tables(text)


def normalize_sentence(text: str) -> str:
    cleaned = clean_fragment(text).strip()
    if not cleaned:
        return ""
    return cleaned.rstrip("。") + "。"


def parse_source_metadata(text: str, default_document: str) -> dict[str, str]:
    match = re.search(
        r"<!--\s*Source:\s*(?P<repo>[^|]+)\|\s*Branch:\s*(?P<branch>[^|]+)\|\s*Commit:\s*(?P<commit>[^|]+)\|\s*Last updated:\s*(?P<date>[^>]+)\s*-->",
        text,
    )
    if not match:
        return {
            "repo": "unknown",
            "branch": "unknown",
            "commit": "unknown",
            "document": default_document,
            "last_updated": datetime.date.today().isoformat(),
        }
    info = {key: value.strip() for key, value in match.groupdict().items()}
    return {
        "repo": info["repo"],
        "branch": info["branch"],
        "commit": info["commit"],
        "document": default_document,
        "last_updated": info["date"],
    }


def normalize_detail_lines(text: str) -> list[str]:
    cleaned = clean_fragment(text)
    if not cleaned or cleaned == "-":
        return []
    parts: list[str] = []
    normalized = cleaned.replace("\n- ", "\n").replace("。 - ", "。\n").replace("； - ", "；\n")
    for fragment in re.split(r"\n+", normalized):
        piece = clean_fragment(fragment.lstrip("- ").strip())
        if not piece:
            continue
        if piece.startswith("须知："):
            continue
        parts.append(piece.rstrip("。") + "。")
    return parts


def first_non_empty_line(text: str) -> str:
    for line in text.splitlines():
        cleaned = clean_fragment(line.lstrip("-* ").strip())
        if cleaned:
            return cleaned
    return ""


def extract_bullets(text: str) -> list[str]:
    bullets: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(("- ", "* ")):
            cleaned = clean_fragment(stripped[2:])
            if cleaned:
                bullets.append(cleaned)
    return bullets


def table_rows_to_param_entries(table: dict[str, object] | None) -> list[dict[str, str]]:
    if table is None:
        return []
    entries: list[dict[str, str]] = []
    for row in table.get("rows", []):
        if not isinstance(row, list) or len(row) < 3:
            continue
        entry = {
            "name": clean_fragment(row[0]),
            "direction": clean_fragment(row[1]) if len(row) > 1 else "",
            "description": clean_fragment(row[2]) if len(row) > 2 else "",
            "usage": clean_fragment(row[3]) if len(row) > 3 else "",
            "dtype": clean_fragment(row[4]) if len(row) > 4 else "",
            "format": clean_fragment(row[5]) if len(row) > 5 else "",
            "shape": clean_fragment(row[6]) if len(row) > 6 else "",
            "noncontiguous": clean_fragment(row[7]) if len(row) > 7 else "",
        }
        if entry["name"] and not is_runtime_param(entry["name"]):
            entries.append(entry)
    return entries


def _append_text(entry: dict[str, str], key: str, value: str) -> None:
    value = clean_fragment(value)
    if not value or value == "-":
        return
    if entry.get(key):
        entry[key] = f"{entry[key]} {value}"
    else:
        entry[key] = value


def _extract_param_header(line: str) -> tuple[str, str, str] | None:
    line = line.strip()
    if not line.startswith(("- ", "* ")):
        return None
    body = clean_fragment(line[2:])
    match = re.match(r"^([A-Za-z0-9_]+)\s*[（(](.*?)[)）]\s*[:：]\s*(.+)$", body)
    if not match:
        return None
    return match.group(1), match.group(2), match.group(3)


def _direction_from_meta(meta: str) -> str:
    if "输入输出" in meta:
        return "输入输出"
    if "可选输入" in meta:
        return "可选输入"
    if "可选输出" in meta:
        return "可选输出"
    if "输出" in meta or "出参" in meta:
        return "输出"
    if "输入" in meta or "入参" in meta:
        return "输入"
    return ""


def _extract_dtype(text: str) -> str:
    patterns = [
        r"数据类型支持([A-Z0-9_、,/().\-]+)",
        r"数据类型(?:与.*?一致)?(?:需要是|为)?([A-Z0-9_、,/().\-]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            value = clean_fragment(match.group(1))
            if value:
                return value
    return ""


def _extract_format(text: str) -> str:
    match = re.search(r"数据格式(?:支持)?([A-Z0-9_/、,]+)", text)
    return clean_fragment(match.group(1)) if match else ""


def _extract_noncontiguous(text: str) -> str:
    if "不支持非连续" in text:
        return "no"
    if "支持非连续" in text or "非连续Tensor" in text or "非连续的Tensor" in text:
        return "yes"
    return "unknown"


def parse_bullet_param_entries(text: str) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    for raw_line in text.splitlines():
        parsed = _extract_param_header(raw_line)
        if parsed:
            name, meta, desc = parsed
            if is_runtime_param(name):
                current = None
                continue
            current = {
                "name": name,
                "direction": _direction_from_meta(meta),
                "description": clean_fragment(desc),
                "usage": clean_fragment(desc),
                "dtype": _extract_dtype(desc),
                "format": _extract_format(desc),
                "shape": "",
                "noncontiguous": _extract_noncontiguous(desc),
            }
            entries.append(current)
            continue
        if current is None:
            continue
        stripped = raw_line.strip()
        if stripped.startswith(("* ", "- ")):
            note = clean_fragment(stripped[2:])
            _append_text(current, "usage", note)
            _append_text(current, "description", note)
            if not current["dtype"]:
                current["dtype"] = _extract_dtype(note)
            if not current["format"]:
                current["format"] = _extract_format(note)
            if not current["noncontiguous"]:
                current["noncontiguous"] = _extract_noncontiguous(note)
    return entries


def classify_ios(entries: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    inputs: list[dict[str, str]] = []
    outputs: list[dict[str, str]] = []
    for entry in entries:
        if is_runtime_param(entry.get("name", "")):
            continue
        payload = {k: v for k, v in entry.items() if v and v != "-"}
        direction = entry.get("direction", "")
        name = canonical_param_name(entry.get("name", ""))
        description = clean_fragment(entry.get("description", ""))
        role = direction_to_role(direction)
        explicit_input = role in {"input", "optional_input", "inout"}
        explicit_output = role in {"output", "optional_output", "inout"}
        protected_input_names = {"out", "dOut", "softmaxMax", "softmaxSum"}
        if explicit_input and not explicit_output:
            inputs.append(payload)
            continue
        if explicit_output and not explicit_input:
            outputs.append(payload)
            continue
        looks_like_output = (
            role == "inout"
            or (
                not direction
                and name not in protected_input_names
                and not name.endswith("Optional")
                and (
                    "计算输出" in description
                    or "输出tensor" in description
                    or "公式中的输出" in description
                )
            )
        )
        if looks_like_output:
            outputs.append(payload)
        else:
            inputs.append(payload)
    return inputs, outputs


def direction_to_role(direction: str) -> str:
    if "输入输出" in direction:
        return "inout"
    if "可选输入" in direction:
        return "optional_input"
    if "可选输出" in direction:
        return "optional_output"
    if "输出" in direction:
        return "output"
    return "input"


def extract_layout_templates(entry: dict[str, str]) -> list[str]:
    templates: list[str] = []
    shape = entry.get("shape", "")
    usage = entry.get("usage", "")
    name = canonical_param_name(entry.get("name", ""))
    for match in re.findall(r"\[([A-Z]{3,})\]", shape):
        if match not in templates:
            templates.append(match)
    if "inputLayout" in usage or "数据排布格式" in usage or "格式下" in usage:
        for match in re.findall(r"\b([A-Z]{3,})\b", usage):
            if match not in templates:
                templates.append(match)
    if name.lower().startswith("layout"):
        templates = normalize_templates(templates + re.findall(r"\b([A-Z_]{3,})\b", usage))
    return templates


def _rank_text_to_int_list(value: str) -> list[int]:
    digits = re.findall(r"\d+", value)
    return [int(item) for item in digits]


def extract_tensor_rank(entry: dict[str, str], layout_templates: list[str]) -> str:
    shape = clean_fragment(entry.get("shape", ""))
    if re.fullmatch(r"[0-9,\-、 ]+", shape):
        return shape.replace("、", "|").replace(",", "|").replace("-", "..").replace(" ", "")
    if "维" in shape:
        if match := re.search(r"(\d+)\s*维", shape):
            return match.group(1)
        return shape.replace("支持", "").strip()
    if layout_templates:
        ranks = sorted({len(item) for item in layout_templates})
        if len(ranks) == 1:
            return str(ranks[0])
        return "|".join(str(item) for item in ranks)
    return ""


def extract_shape_constraints(entry: dict[str, str], layout_templates: list[str], tensor_rank: str) -> list[str]:
    constraints: list[str] = []
    shape = clean_fragment(entry.get("shape", ""))
    usage = clean_fragment(entry.get("usage", ""))
    if shape and shape != tensor_rank and shape not in layout_templates and not re.fullmatch(r"[0-9,\-、 ]+", shape):
        if (
            "shape" in shape
            or "形如" in shape
            or re.search(r"\[[A-Z](?:,[A-Z0-9]+)+\]", shape)
            or re.fullmatch(r"\([^)]*\)", shape)
        ):
            constraints.append(f"{entry['name']}: {shape}。")
    for sentence in re.split(r"[。；]", usage):
        sentence = clean_fragment(sentence)
        if not sentence:
            continue
        if any(token in sentence for token in ["None", "必须传入", "必须传", "可传入默认值", "默认值", "不支持空tensor", "不支持空Tensor"]):
            continue
        if "shape" in sentence or "形如" in sentence or re.search(r"支持\[[A-Z](?:,[A-Z0-9]+)+\]", sentence):
            normalized = sentence
            if "，" in normalized and "数据类型" in normalized:
                normalized = normalized.split("，", 1)[0]
            if normalized not in constraints:
                constraints.append(f"{entry['name']}: {normalized}。")
    return constraints


def extract_value_constraints(entry: dict[str, str]) -> list[str]:
    usage = clean_fragment(entry.get("usage", ""))
    constraints: list[str] = []
    for sentence in re.split(r"[。；]", usage):
        sentence = clean_fragment(sentence)
        if not sentence:
            continue
        if any(token in sentence for token in ["None", "默认值", "可选项", "必须传入", "必须传", "不支持空tensor", "不支持空Tensor"]):
            continue
        if "shape" in sentence or "数据类型" in sentence or "数据格式" in sentence or "非连续" in sentence:
            continue
        if (
            "取值" in sentence
            or "必须" in sentence
            or "需要" in sentence
            or "大于" in sentence
            or "小于" in sentence
            or "一致" in sentence
            or "暂未使用" in sentence
        ):
            constraints.append(f"{entry['name']}: {sentence}。")
    return constraints


def extract_optional_semantics(entry: dict[str, str]) -> list[str]:
    usage = clean_fragment(entry.get("usage", ""))
    semantics: list[str] = []
    for sentence in re.split(r"[。；]", usage):
        sentence = clean_fragment(sentence)
        if not sentence:
            continue
        if any(token in sentence for token in ["可选", "为空", "空Tensor", "暂未使用", "预留参数", "None", "默认值", "必须传入", "必须传"]):
            semantics.append(f"{entry['name']}: {sentence}。")
    return semantics


def extract_output_relation(entry: dict[str, str]) -> list[str]:
    usage = clean_fragment(entry.get("usage", ""))
    relations: list[str] = []
    for sentence in re.split(r"[。；]", usage):
        sentence = clean_fragment(sentence)
        if not sentence:
            continue
        if "shape与" in sentence or "数据类型与" in sentence or "dtype" in sentence.lower():
            relations.append(f"{entry['name']}: {sentence}。")
    return relations


def normalize_noncontiguous(value: str) -> str:
    value = clean_fragment(value)
    if value in {"√", "支持", "yes"}:
        return "yes"
    if value in {"×", "不支持", "no"}:
        return "no"
    return "unknown"


def normalize_templates(templates: list[str]) -> list[str]:
    allowed = {"TND", "BSND", "BNSD", "BSH", "SBH", "PA_BSND"}
    ordered: list[str] = []
    for item in templates:
        if item in allowed and item not in ordered:
            ordered.append(item)
    return ordered


def template_rank(template: str) -> int | None:
    mapping = {
        "TND": 3,
        "BSH": 3,
        "SBH": 3,
        "BSND": 4,
        "BNSD": 4,
        "PA_BSND": 4,
    }
    return mapping.get(template)


def is_high_value_description(text: str) -> bool:
    return any(token in text for token in ["复数", "可选", "optional", "空Tensor", "暂未使用", "输出"])


def infer_when_useful(api_name: str, record: dict[str, object]) -> str:
    joined_errors = " ".join(str(item.get("description", "")) for item in record.get("return_codes", []))
    if "161002" in joined_errors or "参数" in joined_errors or "数据类型" in joined_errors:
        return "Use only when diagnosing ACLNN parameter, dtype, or shape-layout contract mismatches."
    if any(code.get("code") == "561001" for code in record.get("return_codes", [])):
        return "Use only when diagnosing ACLNN shape inference or output shape expectation failures."
    if any(code.get("code") in {"561002", "561003", "561112"} for code in record.get("return_codes", [])):
        return "Use only when contract limits or kernel availability need to be verified against the ACLNN interface."
    if "FlashAttention" in api_name or "GroupedMatmul" in api_name:
        return "Use only when verifying layout templates, cross-parameter consistency, or optional parameter semantics."
    return "Use only when the log already points to this ACLNN API and generic diagnosis references are insufficient."


def augment_attention_contracts(
    api_name: str,
    inputs: list[dict[str, object]],
    outputs: list[dict[str, object]],
    constraints: list[str],
    workspace_section: str,
) -> None:
    if "FlashAttention" not in api_name and "PromptFlashAttention" not in api_name and "SparseFlashAttention" not in api_name:
        return
    text = " ".join(constraints) + " " + workspace_section
    templates = normalize_templates(re.findall(r"(TND|BSND|BNSD|BSH|SBH)", text))
    target_names = {"query", "key", "keyIn", "value", "dy", "dqOut", "dkOut", "dvOut"}
    for param in inputs + outputs:
        if param.get("name") not in target_names:
            continue
        existing = list(param.get("layout_templates", []))
        merged = normalize_templates(existing + templates)
        if merged:
            param["layout_templates"] = merged
        if merged and not param.get("tensor_rank"):
            ranks = sorted({rank for item in merged if (rank := template_rank(item)) is not None})
            param["tensor_rank"] = str(ranks[0]) if len(ranks) == 1 else "|".join(str(item) for item in ranks)

    if "SparseFlashAttention" in api_name:
        sparse_patterns = {
            "layout_query": {"query", "sparseIndices", "queryRope", "attentionOut", "softmaxLse"},
            "layout_kv": {"key", "value", "keyRope"},
        }
        discovered: dict[str, list[str]] = {key: [] for key in sparse_patterns}
        for layout_name, param_names in sparse_patterns.items():
            pattern = rf"{layout_name}\s*为\s*([A-Z_]+)\s*时"
            for template in re.findall(pattern, workspace_section):
                normalized = normalize_templates([template])
                if normalized:
                    discovered[layout_name] = normalize_templates(discovered[layout_name] + normalized)
            for param in inputs + outputs:
                if param.get("name") not in param_names:
                    continue
                merged = normalize_templates(list(param.get("layout_templates", [])) + discovered[layout_name])
                if merged:
                    param["layout_templates"] = merged
                ranks = sorted({rank for item in merged if (rank := template_rank(item)) is not None})
                if ranks:
                    param["tensor_rank"] = str(ranks[0]) if len(ranks) == 1 else "|".join(str(item) for item in ranks)


def build_param_payload(entry: dict[str, str]) -> dict[str, object]:
    layout_templates = normalize_templates(extract_layout_templates(entry))
    tensor_rank = extract_tensor_rank(entry, layout_templates)
    payload: dict[str, object] = {
        "name": canonical_param_name(entry.get("name", "")),
        "role": direction_to_role(entry.get("direction", "")),
    }
    description = clean_fragment(str(entry.get("description", "")))
    if description and is_high_value_description(description):
        payload["description"] = description
    dtype = clean_fragment(str(entry.get("dtype", "")))
    if dtype and dtype not in {"-", "ND", "N/A"}:
        payload["dtype"] = dtype
    noncontiguous = normalize_noncontiguous(str(entry.get("noncontiguous", "")))
    if noncontiguous != "unknown":
        payload["noncontiguous"] = noncontiguous
    if tensor_rank:
        payload["tensor_rank"] = tensor_rank
    if layout_templates:
        payload["layout_templates"] = layout_templates
    shape_constraints = extract_shape_constraints(entry, layout_templates, tensor_rank)
    if shape_constraints:
        payload["shape_constraints"] = shape_constraints
    value_constraints = extract_value_constraints(entry)
    if value_constraints:
        payload["value_constraints"] = value_constraints
    optional_semantics = extract_optional_semantics(entry)
    if optional_semantics:
        payload["optional_semantics"] = optional_semantics
    output_relation = extract_output_relation(entry) if payload["role"] in {"output", "optional_output", "inout"} else []
    if output_relation and payload.get("shape_constraints"):
        existing_shapes = {item.split(": ", 1)[-1].rstrip("。") for item in payload["shape_constraints"]}
        output_relation = [item for item in output_relation if item.split(": ", 1)[-1].rstrip("。") not in existing_shapes]
    if output_relation:
        payload["output_relation"] = output_relation
    return payload


def infer_output_names_from_signature(signature: str) -> set[str]:
    names: set[str] = set()
    if not signature:
        return names
    for param in re.split(r",\s*", signature):
        item = clean_fragment(param)
        if "workspaceSize" in item or "executor" in item or "stream" in item or "workspace" in item:
            continue
        if "const aclTensor" in item or "const aclIntArray" in item or "const aclScalar" in item:
            continue
        if ("aclTensor" in item or "aclIntArray" in item or "aclScalar" in item) and "*" in item:
            match = re.search(r"\*\s*([A-Za-z0-9_]+)", item)
            if match:
                names.add(match.group(1))
    return names


def promote_signature_outputs(
    inputs: list[dict[str, str]],
    outputs: list[dict[str, str]],
    signature: str,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    if outputs:
        return inputs, outputs
    output_names = infer_output_names_from_signature(signature)
    if not output_names:
        return inputs, outputs
    protected_names = {"out", "dOut", "softmaxMax", "softmaxSum"}
    remaining_inputs: list[dict[str, str]] = []
    promoted_outputs: list[dict[str, str]] = list(outputs)
    for entry in inputs:
        name = canonical_param_name(entry.get("name", ""))
        direction = entry.get("direction", "")
        if (
            name in output_names
            and not direction
            and name not in protected_names
            and not name.endswith("Out")
        ):
            fixed = dict(entry)
            fixed["direction"] = "输出"
            promoted_outputs.append(fixed)
        else:
            remaining_inputs.append(entry)
    return remaining_inputs, promoted_outputs


def dedupe_text_items(items: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for item in items:
        cleaned = clean_fragment(str(item))
        key = re.sub(r"\s+", "", cleaned.rstrip("。"))
        if not cleaned or key in seen:
            continue
        deduped.append(cleaned if cleaned.endswith("。") else f"{cleaned}。")
        seen.add(key)
    return deduped


def strip_param_prefix(item: str) -> str:
    text = clean_fragment(str(item))
    if ": " in text:
        return text.split(": ", 1)[-1].rstrip("。")
    return text.rstrip("。")


def is_low_value_constraint_text(text: str) -> bool:
    text = clean_fragment(text)
    if not text:
        return True
    if any(token in text for token in ["公式中的", "Device侧的aclTensor", "host侧的aclIntArray"]):
        return True
    if any(token in text for token in ["确定性", "一致性", "图模式", "产品支持", "支持推理场景下使用"]):
        return True
    if any(token in text for token in ["timeout or trap error", "建议做轴切分处理", "实验性功能", "cache缓存能力"]):
        return True
    if any(token in text for token in ["假设真实S长度", "此时若需要传入", "例如[1, 1, 1, 0, 0]"]):
        return True
    if "暂未使用" in text:
        return True
    if text in {"公共约束。", "规格约束。", "Mask。", "入参为空的场景处理：。", "公共约束", "规格约束", "Mask"}:
        return True
    return False


def filter_contract_constraints(api_name: str, constraints: list[str]) -> list[str]:
    filtered: list[str] = []
    for item in constraints:
        text = clean_fragment(item)
        if not text:
            continue
        if is_low_value_constraint_text(text):
            continue
        if text.endswith("：。") or text.endswith(":。"):
            continue
        if text.endswith("：") or text.endswith(":"):
            continue
        filtered.append(text)
    return dedupe_text_items(filtered)


def split_multiline_fact(text: str) -> list[str]:
    parts: list[str] = []
    for fragment in re.split(r"[\n]+", clean_fragment(text)):
        cleaned = clean_fragment(fragment)
        if cleaned:
            parts.append(cleaned.rstrip("。"))
    return parts


def remove_param_field_if_mixed(param: dict[str, object], field: str, banned_tokens: list[str]) -> None:
    values = [str(item) for item in param.get(field, []) or []]
    if any(any(token in item for token in banned_tokens) for item in values):
        param.pop(field, None)


def sanitize_sparse_param(param: dict[str, object]) -> None:
    shape_values = [str(item) for item in param.get("shape_constraints", []) or []]
    cleaned_shapes: list[str] = []
    for item in shape_values:
        parts = split_multiline_fact(item.split(": ", 1)[-1])
        if not parts:
            continue
        if any(any(token in part for token in ["None", "必须传入", "必须传", "默认值", "不支持空tensor", "不支持空Tensor"]) for part in parts):
            continue
        for part in parts:
            if "shape" in part or "形如" in part or re.fullmatch(r"\([^)]*\)", part):
                cleaned_shapes.append(f"{param['name']}: {part}。")
    if cleaned_shapes:
        param["shape_constraints"] = dedupe_text_items(cleaned_shapes)
    else:
        param.pop("shape_constraints", None)

    for field in ("value_constraints", "optional_semantics", "output_relation"):
        values = [str(item) for item in param.get(field, []) or []]
        cleaned: list[str] = []
        for item in values:
            for part in split_multiline_fact(item.split(": ", 1)[-1]):
                cleaned.append(f"{param['name']}: {part}。")
        if cleaned:
            param[field] = dedupe_text_items(cleaned)
        else:
            param.pop(field, None)

    moved_shapes: list[str] = []
    kept_values: list[str] = []
    for item in param.get("value_constraints", []) or []:
        text = str(item).split(": ", 1)[-1]
        if ("Shape" in text and "一致" in text) or "shape与" in text or ("shape" in text.lower() and "一致" in text):
            moved_shapes.append(f"{param['name']}: {text.rstrip('。')}。")
        else:
            kept_values.append(str(item))
    if moved_shapes:
        param["shape_constraints"] = dedupe_text_items(list(param.get("shape_constraints", [])) + moved_shapes)
    if kept_values:
        param["value_constraints"] = dedupe_text_items(kept_values)
    else:
        param.pop("value_constraints", None)

    if param.get("role") == "input" and canonical_param_name(str(param.get("name", ""))).startswith("layout"):
        param.pop("tensor_rank", None)
        param.pop("layout_templates", None)
        if param.get("optional_semantics"):
            kept_optional: list[str] = []
            promoted_values: list[str] = list(param.get("value_constraints", []))
            for item in param["optional_semantics"]:
                text = str(item).split(": ", 1)[-1]
                if "默认值" in text or "supports " in text:
                    promoted_values.append(f"{param['name']}: {text.rstrip('。')}。")
                else:
                    kept_optional.append(str(item))
            if promoted_values:
                param["value_constraints"] = dedupe_text_items(promoted_values)
            if kept_optional:
                param["optional_semantics"] = dedupe_text_items(kept_optional)
            else:
                param.pop("optional_semantics", None)

    if param.get("role") == "input" and canonical_param_name(str(param.get("name", ""))) in {"preTokens", "nextTokens", "returnSoftmaxLse"}:
        promoted_values = list(param.get("value_constraints", []))
        for item in param.get("optional_semantics", []) or []:
            promoted_values.append(str(item))
        if promoted_values:
            param["value_constraints"] = dedupe_text_items(promoted_values)
        param.pop("optional_semantics", None)

    remove_param_field_if_mixed(param, "value_constraints", ["shape", "形如"])
    remove_param_field_if_mixed(param, "shape_constraints", ["None", "必须传入", "必须传", "默认值"])


def sanitize_param_relations(param: dict[str, object]) -> None:
    shape_constraints = [str(item) for item in param.get("shape_constraints", []) or []]
    if not shape_constraints:
        return
    kept_shapes: list[str] = []
    moved_relations: list[str] = list(param.get("output_relation", []) or [])
    for item in shape_constraints:
        text = item.split(": ", 1)[-1]
        if "数据类型" in text:
            for sep in ["，", ","]:
                if sep in text:
                    left, right = text.split(sep, 1)
                    kept_shapes.append(f"{param['name']}: {left.rstrip('。')}。")
                    moved_relations.append(f"{param['name']}: {right.strip().rstrip('。')}。")
                    break
            else:
                moved_relations.append(item)
        else:
            kept_shapes.append(item)
    if kept_shapes:
        param["shape_constraints"] = dedupe_text_items(kept_shapes)
    else:
        param.pop("shape_constraints", None)
    if moved_relations:
        param["output_relation"] = dedupe_text_items(moved_relations)


def is_interface_level_constraint(text: str, param_names: set[str], owner_name: str) -> bool:
    normalized = strip_param_prefix(text)
    if not normalized:
        return False
    mentioned = [name for name in param_names if name != owner_name and re.search(rf"\b{re.escape(name)}\b", normalized)]
    if len(mentioned) >= 1 and any(token in normalized for token in ["一致", "成比例", "broadcast", "Reduce", "相等", "满足", "保持一致"]):
        return True
    if any(token in normalized for token in ["inputLayout", "layout必须一致", "Nq/Nkv", "broadcast关系", "Reduce维度"]):
        return True
    if "、" in normalized and any(token in normalized for token in ["Shape维度保持一致", "shape维度保持一致", "数据类型必须保持一致"]):
        return True
    return False


def move_interface_level_param_constraints(
    inputs: list[dict[str, object]],
    outputs: list[dict[str, object]],
    constraints: list[str],
) -> list[str]:
    param_names = {
        canonical_param_name(str(param.get("name", "")))
        for param in inputs + outputs
        if canonical_param_name(str(param.get("name", "")))
    }
    lifted: list[str] = list(constraints)
    for param in inputs + outputs:
        owner_name = canonical_param_name(str(param.get("name", "")))
        for field in ("shape_constraints", "value_constraints", "optional_semantics", "output_relation"):
            kept: list[str] = []
            for item in param.get(field, []) or []:
                text = str(item)
                if is_interface_level_constraint(text, param_names, owner_name):
                    lifted.append(strip_param_prefix(text))
                else:
                    kept.append(text)
            if kept:
                param[field] = dedupe_text_items(kept)
            else:
                param.pop(field, None)
    return dedupe_text_items(lifted)


def derive_api_constraints(
    inputs: list[dict[str, object]],
    outputs: list[dict[str, object]],
) -> list[str]:
    derived: list[str] = []
    for param in inputs + outputs:
        name = str(param.get("name", ""))
        for field in ("shape_constraints", "optional_semantics", "value_constraints", "output_relation"):
            for item in param.get(field, []) or []:
                text = strip_param_prefix(str(item))
                if not text or is_low_value_constraint_text(text):
                    continue
                score = 0
                if any(token in text for token in ["shape", "broadcast", "Reduce", "一致", "相等"]):
                    score += 3
                if any(token in text for token in ["可选", "为空", "None", "默认值", "必须传"]):
                    score += 2
                if any(token in text for token in ["取值", "大于", "小于", "整除", "范围", "成比例"]):
                    score += 2
                if field == "output_relation":
                    score += 2
                if score <= 0:
                    continue
                derived.append(f"{name}: {text}")
                if len(derived) >= 8:
                    break
            if len(derived) >= 8:
                break
        if len(derived) >= 8:
            break
    return dedupe_text_items(derived)[:4]


def is_attention_api(api_name: str) -> bool:
    return any(
        token in api_name
        for token in (
            "FlashAttentionScore",
            "FlashAttentionUnpaddingScore",
            "FlashAttentionVarLenScore",
            "SparseFlashAttention",
            "FusedInferAttentionScore",
        )
    )


def score_constraint_text(text: str) -> int:
    text = clean_fragment(text)
    if not text or is_low_value_constraint_text(text):
        return -10
    score = 0
    if any(token in text for token in ["layout", "inputLayout", "TND", "BSND", "BNSD", "BSH", "SBH", "PA_BSND"]):
        score += 4
    if any(token in text for token in ["shape", "Shape", "broadcast", "Reduce", "一致", "相等", "成比例"]):
        score += 4
    if any(token in text for token in ["数据类型", "dtype", "TensorList", "一一对应", "标量", "groupType", "transpose", "@"]):
        score += 3
    if any(token in text for token in ["可选", "必须", "prefix", "KeepProb", "headNum"]):
        score += 3
    if any(token in text for token in ["范围", "取值", "整除", "Nq/Nkv", "block_size", "block", "稀疏"]):
        score += 2
    if any(token in text for token in ["Atlas", "训练系列产品", "推理系列产品"]):
        score -= 5
    return score


def compress_attention_constraints(api_name: str, constraints: list[str]) -> list[str]:
    if not is_attention_api(api_name):
        return dedupe_text_items(constraints)
    ranked: list[tuple[int, str]] = []
    for item in dedupe_text_items(constraints):
        score = score_constraint_text(item)
        if score > 0:
            ranked.append((score, clean_fragment(item)))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return dedupe_text_items([text for _, text in ranked[:5]])


def build_template_constraints(api_name: str) -> list[str] | None:
    unary_same_shape_ops = {
        "aclnnBitwiseNot",
        "aclnnCeil",
        "aclnnCosh",
        "aclnnDigamma",
        "aclnnExp2",
        "aclnnHardswish",
        "aclnnInverse",
        "aclnnLgamma",
        "aclnnLog",
        "aclnnLog2",
        "aclnnLogicalNot",
        "aclnnNeg",
        "aclnnReal",
        "aclnnReciprocal",
        "aclnnRsqrt",
        "aclnnSinh",
        "aclnnTan",
        "aclnnTanh",
        "aclnnTrunc",
    }
    reduce_bool_ops = {"aclnnAll", "aclnnAny"}
    compare_tensor_ops = {
        "aclnnGeTensor",
        "aclnnGtTensor",
        "aclnnLtTensor",
        "aclnnLeTensor",
        "aclnnNeTensor",
    }
    compare_scalar_ops = {
        "aclnnGeScalar",
        "aclnnGtScalar",
        "aclnnLtScalar",
        "aclnnLeScalar",
        "aclnnNeScalar",
    }
    broadcast_binary_ops = {
        "aclnnLogAddExp",
        "aclnnLogAddExp2",
        "aclnnRsub",
        "aclnnRsubs",
        "aclnnSubs",
        "aclnnXLogYTensor",
        "aclnnXLogYScalarOther",
        "aclnnXLogYScalarSelf",
    }

    if api_name in unary_same_shape_ops:
        return ["out 的 shape 需要与输入一致。"]
    if api_name == "aclnnBatchNormStats":
        return [
            "mean 和 invstd 的 shape 需要与 input 的 channel 维度一致。",
            "input 的维度需要在支持范围内，且第二维为 channel 轴。",
            "mean 和 invstd 的数据类型为 FLOAT。",
        ]
    if api_name in {"aclnnArgMin"}:
        return [
            "dim 的取值范围为 [-self.dim(), self.dim())。",
            "keepdim 决定指定轴是否在输出 shape 中保留为 1。",
            "out 的数据类型为 INT32 或 INT64。",
        ]
    if api_name in reduce_bool_ops:
        return [
            "dim 的取值范围为 [-self.dim(), self.dim()-1]，支持负数。",
            "keepdim 决定 reduce 维度是否保留在输出 shape 中。",
            "out 的数据类型为 BOOL。",
        ]
    if api_name in {"aclnnClamp", "aclnnClampMin", "aclnnClampMax"}:
        return [
            "out 的 shape 需要与 self 一致。",
            "clipValueMin/clipValueMax 与 self 的数据类型需要满足推导规则。",
            "当 min 或 max 为 None 时，对应边界不生效。",
        ]
    if api_name == "aclnnAtan2":
        return [
            "self 和 other 的数据类型必须在支持范围内。",
            "out 的 shape 需要与 self 和 other broadcast 后的结果一致。",
        ]
    if api_name == "aclnnBitwiseXorTensor":
        return [
            "self 和 other 的数据类型必须在支持范围内，且需要匹配位异或/逻辑异或计算要求。",
            "out 的 shape 需要与 self 和 other broadcast 后的结果一致。",
        ]
    if api_name == "aclnnEqual":
        return [
            "self 和 other 的数据类型需要满足比较计算的推导关系。",
            "out 的数据类型为 BOOL，且输出一个仅包含单个元素的 Tensor。",
        ]
    if api_name in compare_tensor_ops:
        return [
            "self 和 other 的 shape 需要满足 broadcast 关系。",
            "out 的 shape 需要与 self 和 other broadcast 后的结果一致。",
            "out 的数据类型为 BOOL。",
        ]
    if api_name in compare_scalar_ops:
        return [
            "self 和 other 的数据类型必须在支持范围内。",
            "out 的 shape 需要与 self 一致。",
            "out 的数据类型为 BOOL。",
        ]
    if api_name in {"aclnnLogicalOr", "aclnnLogicalAnd", "aclnnLogicalXor"}:
        return [
            "self 和 other 的 shape 需要满足 broadcast 关系。",
            "out 的 shape 需要是 self 和 other broadcast 之后的结果。",
        ]
    if api_name in broadcast_binary_ops:
        return [
            "self 和 other 的 shape 需要满足 broadcast 关系。",
            "out 的 shape 需要与 self 和 other broadcast 后的结果一致。",
        ]
    if api_name == "aclnnCast":
        return [
            "out 的 shape 需要与 self 一致。",
            "out 的数据类型由目标类型参数决定。",
        ]
    if api_name == "aclnnCat":
        return [
            "除拼接维外，其余维度需要保持一致。",
            "out 的 shape 由各输入 tensor 在 dim 维拼接得到。",
        ]
    if api_name == "aclnnComplex":
        return [
            "real 和 imag 的 shape 需要一致。",
            "out 的 shape 需要与 real 和 imag 一致。",
        ]
    if api_name in {"aclnnDiag", "aclnnDiagFlat"}:
        return [
            "offset 控制对角线位置。",
            "out 的 shape 由输入 shape 和 offset 共同决定。",
        ]
    if api_name == "aclnnDropout":
        return [
            "p 控制随机置零概率，取值范围为 [0, 1]。",
            "out 的 shape 需要与 input 一致。",
            "maskOut 的 shape 与对齐后的输入元素个数对应。",
        ]
    if api_name in {"aclnnDropoutGenMask", "aclnnDropoutGenMaskV2"}:
        return [
            "keepProb 控制生成 mask 的保留概率。",
            "mask 的 shape 与目标张量按位对齐后的元素个数对应。",
        ]
    if api_name == "aclnnCircularPad2d":
        return [
            "padding 长度为 4，依次表示左右上下填充值。",
            "padding 的前两个值需小于 self 最后一维大小，后两个值需小于倒数第二维大小。",
            "out 后两维大小由 self 后两维与对应 padding 共同决定。",
        ]
    if api_name == "aclnnCircularPad3d":
        return [
            "padding 长度为 6，分别对应最后三维两侧的填充值。",
            "padding 各值需小于 self 对应维度大小。",
            "out 后三维大小由 self 后三维与对应 padding 共同决定。",
        ]
    if api_name == "aclnnEmbeddingBag":
        return [
            "indices 和 offsets 的数据类型需要为 INT32 或 INT64。",
            "perSampleWeights 仅在 sum 模式下可以传入，其他模式必须为空。",
            "output 的数据类型与 weight 一致，且为 2 维 Tensor。",
        ]
    if api_name == "aclnnEmbeddingRenorm":
        return [
            "selfRef 必须为 2 维 Tensor，indices 的数据类型需要为 INT32 或 INT64。",
            "结果会原地写回 selfRef。",
        ]
    if api_name == "aclnnGatherNd":
        return [
            "indices 的数据类型需要为 INT32 或 INT64。",
            "negativeIndexSupport 控制是否支持负索引。",
            "out 的数据类型需要与 self 一致。",
        ]
    if api_name == "aclnnGlobalMaxPool":
        return [
            "out 的数据类型需要与 self 一致。",
            "out 表示对输入执行全局最大池化后的结果。",
        ]
    if api_name == "aclnnConvertWeightToINT4Pack":
        return [
            "weightInt4Pack 表示将 weight 转换为 INT4 pack 后的结果。",
            "weightInt4Pack 的 shape 由 weight 的打包方式决定。",
        ]
    if api_name == "aclnnIsPosInf":
        return [
            "out 的 shape 需要与 self 一致。",
            "out 的数据类型为 BOOL。",
        ]
    if api_name == "aclnnMuls":
        return [
            "other 为标量乘数。",
            "out 的 shape 需要与 self 一致。",
        ]
    if api_name == "aclnnMv":
        return [
            "self 必须为矩阵，vec 必须为向量，且 Reduce 维度需要匹配。",
            "out 为矩阵向量乘的结果。",
        ]
    if api_name in {"aclnnNormalFloatFloat", "aclnnNormalFloatTensor", "aclnnNormalTensorFloat"}:
        return [
            "mean 和 std 共同决定正态分布参数。",
            "seed 和 offset 控制随机数序列。",
            "out 为生成的随机 Tensor。",
        ]
    if api_name in {"aclnnPdist", "aclnnPdistForward"}:
        return [
            "p 控制距离计算的范数类型。",
            "out 为输入样本两两之间的距离结果。",
        ]
    if api_name == "aclnnPolar":
        return [
            "input 和 angle 的 shape 需要一致。",
            "out 的 shape 需要与 input 和 angle 一致。",
        ]
    if api_name == "aclnnSearchSorteds":
        return [
            "sortedSequence 需要按升序或与 sorter 一致的顺序组织。",
            "out 的 shape 需要与 self 一致。",
            "outInt32 控制输出索引使用 INT32 还是 INT64。",
        ]
    if api_name == "aclnnTrace":
        return [
            "out 为输入矩阵主对角线元素求和的结果。",
            "结果会写入 out。",
        ]
    if api_name == "aclnnUniqueConsecutive":
        return [
            "valueOut 保存去重后的连续元素结果。",
            "returnInverse 和 returnCounts 控制是否输出 inverseOut 与 countsOut。",
            "dim 控制去重执行的维度。",
        ]
    if api_name == "aclnnRoundDecimals":
        return [
            "decimals 控制按小数位进行舍入。",
            "out 的 shape 需要与 self 一致。",
        ]
    if api_name == "aclnnLogSpace":
        return [
            "start、end 和 base 共同定义等比数列范围。",
            "steps 决定输出元素个数。",
        ]
    if api_name == "aclnnMaxDim":
        return [
            "dim 指定取最大值的维度。",
            "keepdim 决定该维是否在输出 shape 中保留为 1。",
            "indices 输出对应最大值的位置索引。",
        ]
    if api_name == "aclnnMaxN":
        return [
            "tensors 中各输入需要逐元素比较最大值。",
            "out 的 shape 需要与参与比较的 Tensor 保持一致。",
        ]
    if api_name in {"aclnnMean", "aclnnProdDim", "aclnnReduceSum"}:
        return [
            "dim/dims 指定 reduce 的维度。",
            "keepDim/keepDims 决定 reduce 维度是否在输出 shape 中保留为 1。",
            "out 的数据类型由 dtype 参数或输入推导决定。",
        ]
    if api_name == "aclnnRepeat":
        return [
            "repeats 指定沿每个维度的重复次数。",
            "out 的 shape 由 self 与 repeats 共同决定。",
        ]
    if api_name == "aclnnSliceV2":
        return [
            "axes 的元素需要落在有效维度范围内且不能重复。",
            "starts、ends、steps 共同决定每个维度的切片范围。",
            "out 的 shape 由切片结果决定。",
        ]
    if api_name == "aclnnSort":
        return [
            "dim 指定排序维度，descending 控制升降序，stable 控制是否稳定排序。",
            "valuesOut 和 indicesOut 的 shape 需要与 self 一致。",
            "indicesOut 的数据类型为索引类型。",
        ]
    if api_name in {"aclnnNonzero", "aclnnNonzeroV2"}:
        return [
            "out 返回输入中非零元素对应的索引位置。",
            "out 的大小取决于非零元素个数与输入维度数。",
        ]
    if api_name == "aclnnSlogdet":
        return [
            "signOut 表示行列式符号，logOut 表示行列式绝对值的对数。",
            "signOut 和 logOut 的 shape 由输入批维决定。",
        ]
    if api_name == "aclnnSignbit":
        return [
            "out 的 shape 需要与 self 一致。",
            "out 的数据类型为 BOOL。",
        ]
    if api_name == "aclnnSignBitsUnpack":
        return [
            "size 和 dtype 共同决定解包后的输出格式。",
            "out 为对 self 中符号位进行解包后的结果。",
        ]
    if api_name == "aclnnSWhere":
        return [
            "condition、self 和 other 的 shape 需要满足 broadcast 关系。",
            "out 的 shape 需要与三者 broadcast 后的结果一致。",
        ]
    if api_name == "aclnnRandperm":
        return [
            "n 决定随机排列的上界和输出长度。",
            "seed 和 offset 控制随机数序列。",
            "out 为 [0, n) 的随机排列结果。",
        ]
    if api_name == "aclnnReflectionPad3d":
        return [
            "padding 决定最后三维的反射填充大小。",
            "out 的 shape 由 self 与 padding 共同决定。",
        ]
    if api_name == "aclnnRoiAlign":
        return [
            "pooledHeight 和 pooledWidth 决定输出特征图大小。",
            "spatialScale 和 samplingRatio 控制 RoI 对齐采样方式。",
            "out 为对输入特征图执行 RoI Align 的结果。",
        ]
    if api_name == "aclnnTril":
        return [
            "diagonal 控制保留的下三角对角线偏移。",
            "结果会写入 out，且 shape 与 self 一致。",
        ]
    if api_name == "aclnnAdaptiveAvgPool2dBackward":
        return [
            "gradOutput 和 self 的数据类型需要一致，且都在支持范围内。",
            "gradOutput 和 self 的维度需要为 3 维或 4 维。",
            "out 的 shape 与数据类型需要与 self 一致。",
        ]
    if api_name == "aclnnAdaptiveAvgPool3dBackward":
        return [
            "gradOutput 和 self 的数据类型需要一致，且都在支持范围内。",
            "gradOutput 和 self 的维度需要为 4 维或 5 维。",
            "out 的 shape 与数据类型需要与 self 一致。",
        ]
    if api_name in {"aclnnAddr", "aclnnAdds"}:
        return [
            "other 会按 alpha 缩放后参与计算。",
            "out 的 shape 需要与 self 和广播结果一致。",
        ]
    if api_name == "aclnnDualLevelQuantMatmulWeightNz":
        return [
            "x1 和 x2 的数据类型为 FLOAT4_E2M1；level0Scale 为 FLOAT32，level1Scale 为 FLOAT8_E8M0。",
            "transposeX1 和 transposeX2 控制矩阵乘法的维度方向。",
            "out 为 2 维 Tensor，数据类型为 FLOAT16 或 BFLOAT16。",
        ]
    return None


def has_diagnostic_param_contracts(inputs: list[dict[str, object]], outputs: list[dict[str, object]]) -> bool:
    strength = 0
    for param in inputs + outputs:
        if param.get("shape_constraints"):
            strength += 2
        if param.get("output_relation"):
            strength += 2
        if param.get("value_constraints"):
            strength += 1
        if param.get("optional_semantics"):
            strength += 1
        if param.get("layout_templates") and (param.get("shape_constraints") or param.get("tensor_rank")):
            strength += 1
        if strength >= 3:
            return True
    return False


def find_param(params: list[dict[str, object]], name: str) -> dict[str, object] | None:
    for param in params:
        if str(param.get("name", "")) == name:
            return param
    return None


def has_strong_param_contracts(inputs: list[dict[str, object]], outputs: list[dict[str, object]]) -> bool:
    return has_diagnostic_param_contracts(inputs, outputs)


def has_high_value_constraints(constraints: list[str]) -> bool:
    for item in constraints:
        text = clean_fragment(item)
        if not text or is_low_value_constraint_text(text):
            continue
        if any(token in text for token in ["shape", "broadcast", "Reduce", "布局", "layout", "数据类型", "dtype", "可选", "必须", "相等", "一致", "成比例", "取值范围", "整除"]):
            return True
    return False


def prune_low_value_param_fields(param: dict[str, object]) -> None:
    for field in ("shape_constraints", "value_constraints", "optional_semantics", "output_relation"):
        kept: list[str] = []
        for item in param.get(field, []) or []:
            text = strip_param_prefix(str(item))
            if text and not is_low_value_constraint_text(text):
                kept.append(str(item))
        if kept:
            param[field] = dedupe_text_items(kept)
        else:
            param.pop(field, None)


def postprocess_record(
    api_name: str,
    inputs: list[dict[str, object]],
    outputs: list[dict[str, object]],
    constraints: list[str],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[str]]:
    constraints = filter_contract_constraints(api_name, constraints)

    if api_name.startswith("aclnnInplace"):
        inplace_self_names = {"selfRef", "self", "inputRef", "selfRet"}
        moved_inputs: list[dict[str, object]] = list(inputs)
        kept_outputs: list[dict[str, object]] = []
        found_selfref = any(str(param.get("name", "")) in inplace_self_names for param in moved_inputs)
        for param in outputs:
            if str(param.get("name", "")) in inplace_self_names:
                promoted = dict(param)
                promoted["role"] = "inout"
                moved_inputs.append(promoted)
                found_selfref = True
            else:
                kept_outputs.append(param)
        inputs = moved_inputs
        outputs = kept_outputs
        for param in inputs:
            if str(param.get("name", "")) in inplace_self_names:
                param["role"] = "inout"
        if found_selfref and not constraints:
            inplace_constraints = ["结果原地写回 selfRef，selfRef 的 shape 保持不变。"]
            param_names = {str(param.get("name", "")) for param in inputs}
            if "other" in param_names:
                inplace_constraints.append("other 参与对 selfRef 的原地更新。")
            elif "value" in param_names:
                inplace_constraints.append("value 用于更新 selfRef。")
            constraints = dedupe_text_items(inplace_constraints)

    template_constraints = build_template_constraints(api_name)
    if template_constraints:
        constraints = dedupe_text_items(template_constraints)

    if api_name == "aclnnAmin" and not outputs:
        moved_inputs: list[dict[str, object]] = []
        for param in inputs:
            if param.get("name") == "out":
                promoted = dict(param)
                promoted["role"] = "output"
                outputs.append(promoted)
            else:
                moved_inputs.append(param)
        inputs = moved_inputs

    if api_name in {"aclnnAmin", "aclnnAmax"}:
        constraints = dedupe_text_items(
            [
                "dim 的取值范围为 [-self.dim(), self.dim()-1]，且元素不能重复。",
                "keepDim 决定 reduce 维度是否保留在输出 shape 中。",
                "out 需要与 self 数据类型一致。",
            ]
        )

    if api_name == "aclnnArgsort":
        constraints = dedupe_text_items(
            [
                "dim 的取值范围为 [-self.dim(), self.dim()-1]。",
                "descending 控制排序方向，True 为降序，False 为升序。",
                "out 的 shape 与数据格式需要与 self 一致，输出数据类型为 INT64。",
            ]
        )

    if api_name in {"aclnnAcosh", "aclnnAtan"} and not constraints:
        input_name = "self" if find_param(inputs, "self") else "input"
        constraints = dedupe_text_items([f"out 的 shape 需要与 {input_name} 一致。"])

    if api_name == "aclnnBitwiseAndTensor":
        constraints = dedupe_text_items(
            [
                "self 和 other 的数据类型必须在支持范围内，且需要匹配位与/逻辑与计算要求。",
                "out 的 shape 需要与 self 和 other broadcast 后的结果一致。",
            ]
        )

    if api_name == "aclnnBitwiseAndScalar":
        constraints = dedupe_text_items(
            [
                "self 和 other 必须为整型或 BOOL。",
                "out 的 shape 需要与 self 一致。",
            ]
        )

    if api_name == "aclnnSub":
        constraints = dedupe_text_items(
            [
                "other 会按 alpha 缩放后参与减法计算。",
                "out 的 shape 需要与 self 和 other broadcast 后的结果一致。",
            ]
        )

    if api_name == "aclnnMaximum":
        constraints = dedupe_text_items(
            [
                "self 和 other 的数据类型必须在支持范围内。",
                "out 的 shape 需要与 self 和 other broadcast 后的结果一致。",
            ]
        )

    if api_name == "aclnnForeachLerpScalar":
        constraints = dedupe_text_items(
            [
                "x1、x2 和 out 需要按位置一一对应。",
                "weight 为标量插值系数。",
                "x1、x2 和 out 支持空 TensorList。",
            ]
        )

    if api_name == "aclnnForeachNonFiniteCheckAndUnscale":
        constraints = dedupe_text_items(
            [
                "scaledGrads 中每个 tensor 会执行反缩放。",
                "foundInf 用于标记是否检测到 Inf 或 NaN。",
                "invScale、scaledGrads 和 foundInf 不支持空 Tensor。",
            ]
        )

    if api_name == "aclnnForeachSqrt":
        constraints = dedupe_text_items(
            [
                "out 中每个 tensor 的 shape 需要与 x 中对应 tensor 一致。",
                "x 与 out 需要按位置一一对应。",
            ]
        )

    if api_name in {"aclnnAddmm", "aclnnInplaceAddmm"}:
        constraints = [item for item in constraints if any(token in item for token in ["broadcast", "数据类型", "Reduce", "shape"])]
        if not constraints:
            constraints = dedupe_text_items(
                [
                    "self 需要与 mat1@mat2 满足 broadcast 关系。",
                    "mat2 的 Reduce 维度需要与 mat1 的对应维度大小相等。",
                    "out 的 shape 需要与 mat1@mat2 一致。",
                ]
            )

    if api_name in {"aclnnMm", "aclnnBatchMatMul", "aclnnBmm"}:
        constraints = dedupe_text_items(
            [
                "mat2 的 Reduce 维度需要与 self 的对应维度大小相等。",
                "out 的 shape 需要与 self @ mat2 的结果一致。",
            ]
        )

    if api_name == "aclnnGroupedMatmulAdd":
        constraints = dedupe_text_items(
            [
                "每组计算满足 yRef_i = x_i @ weight_i + y_i。",
                "仅支持 transposeX=True、transposeWeight=False、groupType=0（K 轴分组）。",
                "x 和 weight 的数据类型需要匹配，yRef 为 FLOAT32。",
            ]
        )

    if api_name == "aclnnGroupedMatMulAllReduce":
        constraints = dedupe_text_items(
            [
                "x 和 weight 中每组 tensor 的 Reduce 维度需要匹配。",
                "splitItem=0 时，x 和 y 支持 2 到 6 维；splitItem=1/2/3 时，x 和 y 仅支持 2 维。",
                "weight 始终为 2 维 Tensor。",
            ]
        )

    if api_name in {"aclnnMatmulCompress", "aclnnMatmulCompressDequant"} and not constraints:
        constraints = dedupe_text_items(
            [
                "out 的 shape 需要与 x @ weight 的结果一致。",
                "bias 存在时需要与输出 shape 匹配。",
            ]
        )

    if api_name in {"aclnnPowTensorScalar", "aclnnInplacePowTensorScalar"}:
        for param in outputs:
            if param.get("name") != "out":
                continue
            refined_shapes: list[str] = []
            refined_relations: list[str] = list(param.get("output_relation", []) or [])
            for item in param.get("shape_constraints", []) or []:
                text = str(item).split(": ", 1)[-1]
                for sep in ["，", ","]:
                    if sep in text:
                        left, right = text.split(sep, 1)
                        refined_shapes.append(f"out: {left.rstrip('。')}。")
                        refined_relations.append(f"out: {right.strip().rstrip('。')}。")
                        break
                else:
                    refined_shapes.append(str(item))
            if refined_shapes:
                param["shape_constraints"] = dedupe_text_items(refined_shapes)
            if refined_relations:
                param["output_relation"] = dedupe_text_items(refined_relations)
        constraints = [
            item
            for item in constraints
            if not any(token in item for token in ["Atlas 训练系列产品", "Atlas 推理系列产品", "保证精度无误差"])
        ]
        if api_name == "aclnnInplacePowTensorScalar" and not constraints:
            constraints = dedupe_text_items(
                [
                    "exponent 为标量指数。",
                    "结果原地写回 selfRef，selfRef 的 shape 保持不变。",
                ]
            )

    if api_name == "aclnnTransMatmulWeight":
        for param in inputs + outputs:
            if param.get("name") == "mmWeightRef":
                param["role"] = "inout"
        if outputs and not inputs:
            inputs = [dict(outputs[0], role="inout")]
            outputs = []

    if api_name == "aclnnForeachZeroInplace":
        if outputs and not inputs:
            promoted = dict(outputs[0])
            promoted["role"] = "inout"
            inputs = [promoted]
            outputs = []
        if not constraints:
            constraints = dedupe_text_items(
                [
                    "x 中每个 tensor 会被原地置零。",
                    "x 中所有 tensor 的数据类型必须保持一致。",
                ]
            )

    if api_name == "aclnnInplaceMuls" and not has_high_value_constraints(constraints):
        constraints = dedupe_text_items(
            [
                "other 为标量乘数。",
                "结果原地写回 selfRef，selfRef 的 shape 保持不变。",
            ]
        )

    if api_name == "aclnnInplaceLeakyRelu" and not has_high_value_constraints(constraints):
        constraints = dedupe_text_items(
            [
                "negativeSlope 控制负半轴斜率。",
                "结果原地写回 selfRef，selfRef 的 shape 保持不变。",
            ]
        )

    if "SparseFlashAttention" in api_name:
        for param in inputs + outputs:
            sanitize_sparse_param(param)
            if param.get("name") in {"layoutQuery", "layoutKv", "layout", "layoutOptional"}:
                param.pop("tensor_rank", None)
                param.pop("layout_templates", None)

    if api_name in {"aclnnFusedInferAttentionScoreV3", "aclnnFusedInferAttentionScoreV4", "aclnnFusedInferAttentionScoreV5"}:
        filtered_attention: list[str] = []
        for item in constraints:
            if any(
                token in item
                for token in [
                    "shape必须为[B,N,Q_S,1]",
                    "keySharedPrefix和valueSharedPrefix都不为空",
                    "PagedAttention的使能必要条件是blocktable存在且有效",
                    "pseShiftOptional不为空",
                ]
            ):
                filtered_attention.append(item)
        if filtered_attention:
            constraints = dedupe_text_items(filtered_attention)

    for param in inputs + outputs:
        sanitize_param_relations(param)
        prune_low_value_param_fields(param)

    constraints = move_interface_level_param_constraints(inputs, outputs, constraints)
    if not constraints:
        constraints = derive_api_constraints(inputs, outputs)
    constraints = compress_attention_constraints(api_name, constraints)
    if api_name in {"aclnnMm", "aclnnBatchMatMul", "aclnnBmm"}:
        constraints = [
            "mat2 的 Reduce 维度需要与 self 的对应维度大小相等。",
            "out 的 shape 需要与 self @ mat2 的结果一致。",
        ]

    return inputs, outputs, constraints


def select_compact_constraints(record: dict[str, object]) -> list[str]:
    max_items = 3 if is_attention_api(str(record.get("api_name", ""))) else 4
    prioritized: list[tuple[int, str]] = []
    for item in record.get("constraints", []) or []:
        text = clean_fragment(str(item))
        if not text:
            continue
        score = score_constraint_text(text)
        if score > 0:
            prioritized.append((score, text))
    prioritized.sort(key=lambda item: (-item[0], item[1]))
    selected = [text for _, text in prioritized[:max_items]]
    if selected:
        return selected
    fallback: list[str] = []
    for param in list(record.get("inputs", []) or []) + list(record.get("outputs", []) or []):
        name = str(param.get("name", ""))
        for key in ("shape_constraints", "value_constraints", "optional_semantics", "output_relation"):
            for item in param.get(key, []) or []:
                text = clean_fragment(str(item).split(": ", 1)[-1])
                if text and score_constraint_text(text) > 0:
                    fallback.append(f"{name}: {text}")
        if len(fallback) >= max_items:
            break
    return dedupe_text_items(fallback)[:max_items]


def compact_param_details(param: dict[str, object]) -> str:
    parts: list[str] = []
    dtype = str(param.get("dtype", ""))
    if dtype and len(dtype) <= 40 and any(token in dtype for token in ["BOOL", "COMPLEX", "FLOAT8", "STRING"]):
        parts.append(f"dtype={param['dtype']}")
    if param.get("layout_templates"):
        parts.append(f"layouts={','.join(str(item) for item in param['layout_templates'])}")
    rank = str(param.get("tensor_rank", ""))
    if rank and rank not in {"0..8", "1..8", "2..8"} and "|" in rank and not param.get("layout_templates"):
        parts.append(f"rank={rank}")
    if param.get("shape_constraints"):
        parts.append("shape=" + " / ".join(str(item).split(": ", 1)[-1] for item in param["shape_constraints"][:1]))
    if param.get("value_constraints"):
        parts.append("value=" + " / ".join(str(item).split(": ", 1)[-1] for item in param["value_constraints"][:1]))
    if param.get("optional_semantics"):
        parts.append("optional=" + " / ".join(str(item).split(": ", 1)[-1] for item in param["optional_semantics"][:1]))
    if param.get("output_relation"):
        parts.append("relation=" + " / ".join(str(item).split(": ", 1)[-1] for item in param["output_relation"][:1]))
    if param.get("noncontiguous") == "no":
        parts.append("noncontiguous=no")
    return "; ".join(parts)


def find_param_table(tables: list[dict[str, object]]) -> dict[str, object] | None:
    for table in tables:
        headers = [str(h) for h in table.get("headers", [])]
        if any("参数名" in h or "参数名称" in h for h in headers):
            return table
    return None


def find_return_code_tables(tables: list[dict[str, object]]) -> list[dict[str, object]]:
    matched: list[dict[str, object]] = []
    for table in tables:
        headers = [str(h) for h in table.get("headers", [])]
        if any("返回值" in h or "返回码" in h or "错误码" in h or "状态码" in h for h in headers):
            matched.append(table)
    return matched


def extract_return_codes(text: str) -> list[dict[str, str]]:
    codes: list[dict[str, str]] = []
    for table in find_return_code_tables(parse_tables(text)):
        for row in table.get("rows", []):
            if not isinstance(row, list) or len(row) < 2:
                continue
            joined = " ".join(clean_fragment(cell) for cell in row)
            if not re.search(r"\b\d{3,6}\b", joined):
                continue
            codes.append(
                {
                    "name": clean_fragment(row[0]) if len(row) > 0 else "",
                    "code": clean_fragment(row[1]) if len(row) > 1 else "",
                    "description": clean_fragment(row[2]) if len(row) > 2 else "",
                }
            )
    deduped: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in codes:
        key = (item["name"], item["code"], item["description"])
        if key not in seen:
            deduped.append(item)
            seen.add(key)
    return deduped


def _shape_rules_from_usage(entry: dict[str, str]) -> list[str]:
    usage = entry.get("usage", "")
    rules: list[str] = []
    for sentence in re.split(r"[。；]", usage):
        sentence = clean_fragment(sentence)
        if not sentence:
            continue
        if "shape" in sentence:
            snippet = sentence
            if "，" in snippet and "数据类型" in snippet:
                snippet = snippet.split("，", 1)[0]
            if "," in snippet and "数据类型" in snippet:
                snippet = snippet.split(",", 1)[0]
            rules.append(f"{entry['name']}: {clean_fragment(snippet)}。")
            continue
        if re.search(r"支持\[[^\]]+\]", sentence):
            rules.append(f"{entry['name']}: {sentence}。")
    return rules


def _rank_rule_from_shape(entry: dict[str, str]) -> str | None:
    value = clean_fragment(entry.get("shape", ""))
    if not value or value == "-":
        return None
    if re.fullmatch(r"[0-9,\-、 ]+", value):
        return f"{entry['name']}: {value}"
    if "维" in value:
        return f"{entry['name']}: {value}"
    return None


def sanitize_entry(entry: dict[str, str]) -> dict[str, str]:
    sanitized = dict(entry)
    if is_runtime_param(sanitized.get("name", "")):
        return {}
    return sanitized


def _layout_rules_from_usage(entry: dict[str, str]) -> list[str]:
    usage = entry.get("usage", "")
    rules: list[str] = []
    for pattern in [r"(inputLayout[^。；]*[。；]?)", r"(布局[^。；]*[。；]?)"]:
        for match in re.finditer(pattern, usage):
            value = clean_fragment(match.group(1))
            if value:
                rules.append(f"{entry['name']}: {value}")
    return rules


def _value_rules_from_usage(entry: dict[str, str]) -> list[str]:
    usage = entry.get("usage", "")
    rules: list[str] = []
    for pattern in [r"(取值[^。；]*[。；]?)", r"(必须[^。；]*[。；]?)", r"(需要[^。；]*[。；]?)"]:
        for match in re.finditer(pattern, usage):
            value = clean_fragment(match.group(1))
            if value:
                rules.append(f"{entry['name']}: {value}")
    return rules


def collect_rule_lines(entries: list[dict[str, str]], key: str, default: str) -> list[str]:
    lines = [f"{entry['name']}: {entry[key]}" for entry in entries if entry.get(key) and entry[key] != "-"]
    return lines or [default]


def collect_special_cases(text: str) -> list[str]:
    cases: list[str] = []
    for bullet in extract_bullets(text):
        if bullet.startswith("**参数说明") or bullet.startswith("**返回值"):
            continue
        if "产品支持" in bullet:
            continue
        if is_runtime_param(bullet):
            continue
        cases.append(bullet)
    return cases or ["unknown special cases"]


def summarize_doc(doc_name: str, source_repo: str, text: str) -> list[dict[str, object]]:
    text = normalize_text(text)
    sections = split_sections(text)
    aliases = split_api_names(doc_name)
    summary = first_non_empty_line(sections.get("功能说明", ""))
    prototypes = extract_prototypes(sections.get("函数原型", ""))
    prototypes_by_name = {item["name"]: item["signature"] for item in prototypes}
    prototype_names = {item["name"] for item in prototypes}
    doc_constraints = extract_bullets(sections.get("约束说明", ""))
    records: list[dict[str, object]] = []

    api_section_titles = [title for title in sections if title.startswith("aclnn")]
    for api_name in aliases:
        workspace_api = f"{api_name}GetWorkspaceSize"
        execute_api = api_name
        workspace_section = sections.get(workspace_api, "")
        execute_section = sections.get(execute_api, "")
        relevant_titles = [title for title in api_section_titles if title in {workspace_api, execute_api}]

        param_table = find_param_table(parse_tables(workspace_section))
        entries = table_rows_to_param_entries(param_table)
        if not entries:
            entries = parse_bullet_param_entries(workspace_section)

        entries = [item for item in (sanitize_entry(entry) for entry in entries) if item]
        raw_inputs, raw_outputs = classify_ios(entries)
        raw_inputs, raw_outputs = promote_signature_outputs(
            raw_inputs,
            raw_outputs,
            prototypes_by_name.get(workspace_api, ""),
        )
        inputs = [dict(build_param_payload(entry), role="input" if direction_to_role(entry.get("direction", "")) == "input" else direction_to_role(entry.get("direction", ""))) for entry in raw_inputs]
        outputs = [dict(build_param_payload(entry), role="output" if direction_to_role(entry.get("direction", "")) in {"input", "output"} else direction_to_role(entry.get("direction", ""))) for entry in raw_outputs]
        constraints = dedupe_text_items([item for item in doc_constraints if "产品支持" not in item])

        has_return_code_table = bool(find_return_code_tables(parse_tables(workspace_section)))
        return_codes = extract_return_codes(workspace_section)
        error_conditions = [item["description"] for item in return_codes if item.get("description")] or []

        augment_attention_contracts(api_name, inputs, outputs, constraints, workspace_section)
        inputs, outputs, constraints = postprocess_record(api_name, inputs, outputs, constraints)

        extraction_status = "complete"
        if not entries:
            extraction_status = "failed"
        elif not inputs and not outputs:
            extraction_status = "partial"
        elif not constraints:
            extraction_status = "partial"
        elif has_return_code_table and not return_codes:
            extraction_status = "partial"
        elif not has_high_value_constraints(constraints) and not has_strong_param_contracts(inputs, outputs):
            extraction_status = "partial"
        elif any(
            any(
                bad in str(item)
                for bad in ["\n ", "。。", "None", "必须传入", "必须传", "默认值"]
            )
            for param in inputs + outputs
            for field in ("shape_constraints",)
            for item in param.get(field, []) or []
        ):
            extraction_status = "partial"

        record: dict[str, object] = {
            "api_name": api_name,
            "doc_names": [doc_name],
            "source_repo": source_repo,
            "workspace_api": workspace_api if workspace_api in prototype_names or workspace_section else workspace_api,
            "execute_api": execute_api if execute_api in prototype_names or execute_section else execute_api,
            "summary": summary,
            "constraints": constraints,
            "error_conditions": dedupe_text_items(error_conditions),
            "return_codes": return_codes,
            "evidence": relevant_titles or ["函数原型"],
            "extraction_status": extraction_status,
        }
        if extraction_status != "failed":
            if inputs:
                record["inputs"] = inputs
            if outputs:
                record["outputs"] = outputs
        records.append(record)
    return records


def yaml_scalar(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(str(value), ensure_ascii=False)


def dump_yaml(value: object, indent: int = 0) -> list[str]:
    prefix = " " * indent
    if isinstance(value, dict):
        if not value:
            return [f"{prefix}{{}}"]
        lines: list[str] = []
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                dumped = dump_yaml(item, indent + 2)
                if dumped == [" " * (indent + 2) + "[]"]:
                    lines.append(f"{prefix}{key}: []")
                elif dumped == [" " * (indent + 2) + "{}"]:
                    lines.append(f"{prefix}{key}: {{}}")
                else:
                    lines.append(f"{prefix}{key}:")
                    lines.extend(dumped)
            else:
                lines.append(f"{prefix}{key}: {yaml_scalar(item)}")
        return lines
    if isinstance(value, list):
        if not value:
            return [f"{prefix}[]"]
        lines: list[str] = []
        for item in value:
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}-")
                lines.extend(dump_yaml(item, indent + 2))
            else:
                lines.append(f"{prefix}- {yaml_scalar(item)}")
        return lines
    return [f"{prefix}{yaml_scalar(value)}"]


def write_yaml(path: Path, payload: dict[str, object]) -> None:
    path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")


def minimal_source_info(path: Path) -> dict[str, str]:
    parsed = parse_source_metadata(path.read_text(encoding="utf-8"), path.name)
    return {
        "repo": parsed.get("repo", "unknown"),
        "branch": parsed.get("branch", "unknown"),
        "commit": parsed.get("commit", "unknown"),
        "document": parsed.get("document", path.name),
    }


def parse_acl_symbol_and_code(cell: str) -> tuple[str, int | None]:
    text = clean_fragment(cell.replace("\n", " "))
    match = re.search(r"([A-Z0-9_]+)\s*=\s*(\d+)", text)
    if not match:
        return "", None
    return match.group(1), int(match.group(2))


def is_invalid_solution(text: str) -> bool:
    if not text:
        return True
    return any(pattern in text for pattern in INVALID_SOLUTION_PATTERNS)


def select_best_solution(candidates: list[str]) -> str:
    for candidate in candidates:
        normalized = normalize_sentence(candidate)
        if normalized and not is_invalid_solution(normalized):
            return normalized
    return ""


def infer_acl_solution(symbol: str, meaning: str, details: list[str]) -> str:
    detail_solution = select_best_solution(details)
    if detail_solution:
        return detail_solution
    specific = {
        "ACL_ERROR_RT_CONTEXT_NULL": "请检查是否调用 aclrtSetCurrentContext 或 aclrtSetDevice。",
        "ACL_ERROR_RT_DEVIDE_OOM": "请检查 Device 内存使用情况并优化内存占用。",
    }
    if symbol in specific:
        return specific[symbol]
    if "参数" in meaning or "无效" in meaning:
        return "请检查接口入参与配置是否正确。"
    if "内存" in meaning or "OOM" in meaning:
        return "请检查设备或主机内存是否充足，并降低资源占用。"
    if "超时" in meaning:
        return "请检查业务负载、算子规模或运行环境是否导致执行超时。"
    return ""


def build_acl_entries(text: str) -> list[dict[str, object]]:
    merged: dict[int, dict[str, object]] = {}
    for table in parse_tables(text):
        for row in table.get("rows", []):
            if not isinstance(row, list) or len(row) < 2:
                continue
            symbol, code = parse_acl_symbol_and_code(row[0])
            if code is None:
                continue
            meaning = normalize_sentence(row[1])
            if not meaning:
                continue
            details = normalize_detail_lines(row[2]) if len(row) > 2 else []
            solution = infer_acl_solution(symbol, meaning, details)
            entry = {
                "scope": "acl",
                "code": code,
                "name": symbol,
                "meaning": meaning,
                "solution": solution,
            }
            existing = merged.get(code)
            if existing is None:
                merged[code] = entry
                continue
            if not existing.get("solution") and solution:
                existing["solution"] = solution
            if not existing.get("meaning") and meaning:
                existing["meaning"] = meaning
            if not existing.get("name") and symbol:
                existing["name"] = symbol
    return [merged[code] for code in sorted(merged)]


def infer_aclnn_solution(code: int, meaning: str) -> str:
    specific = {
        0: "执行成功，无需额外处理。",
        161001: "请检查是否传入非法空指针或遗漏了必需输入。",
        161002: "请检查参数类型、dtype 推导关系、shape 和 layout 约束。",
        361001: "请优先检查 runtime 日志和 Device 状态，再判断是否为接口契约问题。",
        561000: "",
        561001: "请优先检查输出 shape 推导与动态 shape 处理。",
        561002: "请优先检查维度范围、tiling 限制和输入规模。",
        561003: "请检查 OPP 或 kernel 包是否完整安装并成功加载。",
        561107: "请检查 ASCEND_OPP_PATH 是否配置正确。",
        561112: "请检查算子二进制 kernel 包是否存在并已成功加载。",
    }
    if code in specific:
        return specific[code]
    if 161000 <= code < 162000:
        return "请检查接口入参与参数契约是否满足要求。"
    if 361000 <= code < 362000:
        return "请优先检查 runtime 日志和 Device 状态。"
    return ""


def build_aclnn_error_entries(text: str) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    seen: set[int] = set()
    for table in parse_tables(text):
        for row in table.get("rows", []):
            if not isinstance(row, list) or len(row) < 3:
                continue
            code_text = clean_fragment(row[1])
            if not re.fullmatch(r"\d+", code_text):
                continue
            code = int(code_text)
            if code in seen:
                continue
            meaning = normalize_sentence(row[2])
            if not meaning:
                continue
            entries.append(
                {
                    "scope": "aclnn",
                    "code": code,
                    "name": clean_fragment(row[0]),
                    "meaning": meaning,
                    "solution": infer_aclnn_solution(code, meaning),
                }
            )
            seen.add(code)
    entries.sort(key=lambda item: (str(item["scope"]), int(item["code"])))
    return entries


def build_source_repositories(
    runtime_info: dict[str, object],
    ops_infos: list[dict[str, object]],
) -> list[dict[str, str]]:
    repositories = [runtime_info] + ops_infos
    return [
        {
            "name": str(item.get("name", "")),
            "repo_url": str(item.get("repo_url") or item.get("path", "")),
            "branch": str(item.get("branch", "unknown")),
            "commit": str(item.get("commit", "unknown")),
            "source_type": str(item.get("source_type", "unknown")),
        }
        for item in repositories
    ]


def build_index_meta(
    runtime_info: dict[str, object],
    ops_infos: list[dict[str, object]],
    *,
    source_mode: str,
    deterministic: bool,
) -> dict[str, object]:
    source_repositories = build_source_repositories(runtime_info, ops_infos)
    return {
        "generated_at": current_timestamp(deterministic=deterministic),
        "generator_name": GENERATOR_NAME,
        "generator_version": GENERATOR_VERSION,
        "index_schema_version": INDEX_SCHEMA_VERSION,
        "source_mode": source_mode,
        "source_repo_url": str(runtime_info.get("repo_url") or runtime_info.get("path", "")),
        "source_branch": str(runtime_info.get("branch", "unknown")),
        "source_commit": str(runtime_info.get("commit", "unknown")),
        "source_repository_count": len(source_repositories),
        "source_repositories": source_repositories,
    }


def build_cann_error_payload(meta: dict[str, object]) -> dict[str, object]:
    if not ACL_DOC_PATH.exists():
        raise FileNotFoundError(f"Missing source markdown: {ACL_DOC_PATH}")
    if not ACLNN_DOC_PATH.exists():
        raise FileNotFoundError(f"Missing source markdown: {ACLNN_DOC_PATH}")

    acl_text = ACL_DOC_PATH.read_text(encoding="utf-8")
    aclnn_text = ACLNN_DOC_PATH.read_text(encoding="utf-8")
    entries = build_acl_entries(acl_text) + build_aclnn_error_entries(aclnn_text)
    return {
        "meta": {
            **meta,
            "entry_count": len(entries),
        },
        "sources": {
            "acl": minimal_source_info(ACL_DOC_PATH),
            "aclnn": minimal_source_info(ACLNN_DOC_PATH),
        },
        "entries": entries,
    }


def build_aclnn_index_payload(records: list[dict[str, object]], repo_infos: list[dict[str, str]], meta: dict[str, object]) -> dict[str, object]:
    return {
        "meta": {
            **meta,
            "api_count": len(records),
        },
        "repositories": repo_infos,
        "apis": sorted(records, key=lambda item: str(item["api_name"])),
    }


def create_error_index_schema(conn: sqlite3.Connection) -> None:
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
            source_repository_count INTEGER NOT NULL,
            entry_count INTEGER NOT NULL
        );
        CREATE TABLE source_repository (
            repo_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            repo_url TEXT NOT NULL,
            branch TEXT NOT NULL,
            commit_hash TEXT NOT NULL,
            source_type TEXT NOT NULL
        );
        CREATE TABLE error_source (
            source_kind TEXT PRIMARY KEY,
            repo TEXT NOT NULL,
            branch TEXT NOT NULL,
            commit_hash TEXT NOT NULL,
            document_name TEXT NOT NULL
        );
        CREATE TABLE error_entry (
            entry_id INTEGER PRIMARY KEY,
            ordinal INTEGER NOT NULL,
            scope TEXT NOT NULL,
            code INTEGER NOT NULL,
            name TEXT NOT NULL,
            meaning TEXT NOT NULL,
            solution TEXT NOT NULL
        );
        CREATE UNIQUE INDEX idx_error_entry_ordinal ON error_entry(ordinal);
        CREATE INDEX idx_error_entry_code ON error_entry(code);
        CREATE INDEX idx_error_entry_name ON error_entry(name);
        """
    )


def write_error_index_db(db_path: Path, payload: dict[str, object], *, deterministic: bool) -> None:
    conn = sqlite3.connect(db_path)
    try:
        initialize_sqlite_pragmas(conn)
        conn.executescript(
            """
            DROP TABLE IF EXISTS error_entry;
            DROP TABLE IF EXISTS error_source;
            DROP TABLE IF EXISTS source_repository;
            DROP TABLE IF EXISTS schema_meta;
            """
        )
        conn.execute("VACUUM")
        create_error_index_schema(conn)
        meta = payload["meta"]
        conn.execute(
            """
            INSERT INTO schema_meta(
                id, schema_version, generator_name, generator_version, generated_at,
                source_mode, source_repo_url, source_branch, source_commit,
                source_repository_count, entry_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                int(meta["source_repository_count"]),
                int(meta["entry_count"]),
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
        sources = payload.get("sources", {})
        for source_kind in ("acl", "aclnn"):
            item = dict(sources.get(source_kind, {}))
            conn.execute(
                """
                INSERT INTO error_source(source_kind, repo, branch, commit_hash, document_name)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    source_kind,
                    str(item.get("repo", "")),
                    str(item.get("branch", "")),
                    str(item.get("commit", "")),
                    str(item.get("document", "")),
                ),
            )
        conn.executemany(
            """
            INSERT INTO error_entry(entry_id, ordinal, scope, code, name, meaning, solution)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    idx + 1,
                    idx,
                    str(item.get("scope", "")),
                    int(item.get("code", 0)),
                    str(item.get("name", "")),
                    str(item.get("meaning", "")),
                    str(item.get("solution", "")),
                )
                for idx, item in enumerate(payload.get("entries", []))
            ],
        )
        conn.commit()
    finally:
        conn.close()
    if deterministic:
        normalize_sqlite_header_for_determinism(db_path)


def load_error_index_payload_from_db(db_path: Path) -> dict[str, object]:
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
        sources = {
            str(row["source_kind"]): {
                "repo": str(row["repo"]),
                "branch": str(row["branch"]),
                "commit": str(row["commit_hash"]),
                "document": str(row["document_name"]),
            }
            for row in conn.execute("SELECT * FROM error_source ORDER BY source_kind").fetchall()
        }
        entries = [
            {
                "scope": str(row["scope"]),
                "code": int(row["code"]),
                "name": str(row["name"]),
                "meaning": str(row["meaning"]),
                "solution": str(row["solution"]),
            }
            for row in conn.execute("SELECT * FROM error_entry ORDER BY ordinal").fetchall()
        ]
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
                "source_repository_count": int(meta_row["source_repository_count"]),
                "source_repositories": source_repositories,
                "entry_count": int(meta_row["entry_count"]),
            },
            "sources": sources,
            "entries": entries,
        }
    finally:
        conn.close()


def create_aclnn_index_schema(conn: sqlite3.Connection) -> None:
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
            source_repository_count INTEGER NOT NULL,
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
        CREATE TABLE repository_summary (
            repo_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            branch TEXT NOT NULL,
            commit_hash TEXT NOT NULL
        );
        CREATE TABLE api (
            api_id INTEGER PRIMARY KEY,
            api_name TEXT NOT NULL UNIQUE,
            source_repo TEXT NOT NULL,
            workspace_api TEXT NOT NULL,
            execute_api TEXT NOT NULL,
            summary TEXT NOT NULL,
            extraction_status TEXT NOT NULL
        );
        CREATE TABLE api_doc_name (
            api_id INTEGER NOT NULL,
            ordinal INTEGER NOT NULL,
            doc_name TEXT NOT NULL,
            PRIMARY KEY (api_id, ordinal),
            FOREIGN KEY (api_id) REFERENCES api(api_id) ON DELETE CASCADE
        );
        CREATE TABLE api_constraint (
            api_id INTEGER NOT NULL,
            ordinal INTEGER NOT NULL,
            value_text TEXT NOT NULL,
            PRIMARY KEY (api_id, ordinal),
            FOREIGN KEY (api_id) REFERENCES api(api_id) ON DELETE CASCADE
        );
        CREATE TABLE api_error_condition (
            api_id INTEGER NOT NULL,
            ordinal INTEGER NOT NULL,
            value_text TEXT NOT NULL,
            PRIMARY KEY (api_id, ordinal),
            FOREIGN KEY (api_id) REFERENCES api(api_id) ON DELETE CASCADE
        );
        CREATE TABLE api_return_code (
            api_id INTEGER NOT NULL,
            ordinal INTEGER NOT NULL,
            name TEXT NOT NULL,
            code TEXT NOT NULL,
            description TEXT NOT NULL,
            PRIMARY KEY (api_id, ordinal),
            FOREIGN KEY (api_id) REFERENCES api(api_id) ON DELETE CASCADE
        );
        CREATE TABLE api_evidence (
            api_id INTEGER NOT NULL,
            ordinal INTEGER NOT NULL,
            value_text TEXT NOT NULL,
            PRIMARY KEY (api_id, ordinal),
            FOREIGN KEY (api_id) REFERENCES api(api_id) ON DELETE CASCADE
        );
        CREATE TABLE api_param (
            api_id INTEGER NOT NULL,
            param_kind TEXT NOT NULL,
            ordinal INTEGER NOT NULL,
            name TEXT NOT NULL,
            role TEXT NOT NULL,
            description TEXT NOT NULL,
            dtype TEXT NOT NULL,
            noncontiguous TEXT NOT NULL,
            tensor_rank TEXT NOT NULL,
            layout_templates_json TEXT NOT NULL,
            shape_constraints_json TEXT NOT NULL,
            value_constraints_json TEXT NOT NULL,
            optional_semantics_json TEXT NOT NULL,
            output_relation_json TEXT NOT NULL,
            PRIMARY KEY (api_id, param_kind, ordinal),
            FOREIGN KEY (api_id) REFERENCES api(api_id) ON DELETE CASCADE
        );
        CREATE INDEX idx_api_workspace_api ON api(workspace_api);
        CREATE INDEX idx_api_execute_api ON api(execute_api);
        CREATE INDEX idx_api_source_repo ON api(source_repo);
        CREATE INDEX idx_return_code_code ON api_return_code(code);
        """
    )


def _insert_text_rows(conn: sqlite3.Connection, table: str, api_id: int, values: list[object]) -> None:
    if not values:
        return
    conn.executemany(
        f"INSERT INTO {table}(api_id, ordinal, value_text) VALUES (?, ?, ?)",
        [(api_id, ordinal, str(value)) for ordinal, value in enumerate(values)],
    )


def write_aclnn_index_db(db_path: Path, payload: dict[str, object], *, deterministic: bool) -> None:
    conn = sqlite3.connect(db_path)
    try:
        initialize_sqlite_pragmas(conn)
        conn.executescript(
            """
            DROP TABLE IF EXISTS api_param;
            DROP TABLE IF EXISTS api_evidence;
            DROP TABLE IF EXISTS api_return_code;
            DROP TABLE IF EXISTS api_error_condition;
            DROP TABLE IF EXISTS api_constraint;
            DROP TABLE IF EXISTS api_doc_name;
            DROP TABLE IF EXISTS api;
            DROP TABLE IF EXISTS repository_summary;
            DROP TABLE IF EXISTS source_repository;
            DROP TABLE IF EXISTS schema_meta;
            """
        )
        conn.execute("VACUUM")
        create_aclnn_index_schema(conn)
        meta = payload["meta"]
        conn.execute(
            """
            INSERT INTO schema_meta(
                id, schema_version, generator_name, generator_version, generated_at,
                source_mode, source_repo_url, source_branch, source_commit,
                source_repository_count, api_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                int(meta["source_repository_count"]),
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
            "INSERT INTO repository_summary(repo_id, name, branch, commit_hash) VALUES (?, ?, ?, ?)",
            [
                (idx + 1, str(item.get("name", "")), str(item.get("branch", "")), str(item.get("commit", "")))
                for idx, item in enumerate(payload.get("repositories", []))
            ],
        )
        for api_id, record in enumerate(payload.get("apis", []), start=1):
            conn.execute(
                """
                INSERT INTO api(api_id, api_name, source_repo, workspace_api, execute_api, summary, extraction_status)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    api_id,
                    str(record.get("api_name", "")),
                    str(record.get("source_repo", "")),
                    str(record.get("workspace_api", "")),
                    str(record.get("execute_api", "")),
                    str(record.get("summary", "")),
                    str(record.get("extraction_status", "")),
                ),
            )
            conn.executemany(
                "INSERT INTO api_doc_name(api_id, ordinal, doc_name) VALUES (?, ?, ?)",
                [(api_id, ordinal, str(item)) for ordinal, item in enumerate(record.get("doc_names", []))],
            )
            _insert_text_rows(conn, "api_constraint", api_id, list(record.get("constraints", [])))
            _insert_text_rows(conn, "api_error_condition", api_id, list(record.get("error_conditions", [])))
            _insert_text_rows(conn, "api_evidence", api_id, list(record.get("evidence", [])))
            conn.executemany(
                """
                INSERT INTO api_return_code(api_id, ordinal, name, code, description)
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (
                        api_id,
                        ordinal,
                        str(item.get("name", "")),
                        str(item.get("code", "")),
                        str(item.get("description", "")),
                    )
                    for ordinal, item in enumerate(record.get("return_codes", []))
                ],
            )
            for param_kind in ("inputs", "outputs"):
                for ordinal, item in enumerate(record.get(param_kind, []) or []):
                    conn.execute(
                        """
                        INSERT INTO api_param(
                            api_id, param_kind, ordinal, name, role, description, dtype, noncontiguous,
                            tensor_rank, layout_templates_json, shape_constraints_json, value_constraints_json,
                            optional_semantics_json, output_relation_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            api_id,
                            "input" if param_kind == "inputs" else "output",
                            ordinal,
                            str(item.get("name", "")),
                            str(item.get("role", "")),
                            str(item.get("description", "")),
                            str(item.get("dtype", "")),
                            str(item.get("noncontiguous", "")),
                            str(item.get("tensor_rank", "")),
                            json_dump_text(list(item.get("layout_templates", []) or [])),
                            json_dump_text(list(item.get("shape_constraints", []) or [])),
                            json_dump_text(list(item.get("value_constraints", []) or [])),
                            json_dump_text(list(item.get("optional_semantics", []) or [])),
                            json_dump_text(list(item.get("output_relation", []) or [])),
                        ),
                    )
        conn.commit()
    finally:
        conn.close()
    if deterministic:
        normalize_sqlite_header_for_determinism(db_path)


def load_aclnn_index_payload_from_db(db_path: Path) -> dict[str, object]:
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
        repositories = [
            {
                "name": str(row["name"]),
                "branch": str(row["branch"]),
                "commit": str(row["commit_hash"]),
            }
            for row in conn.execute("SELECT * FROM repository_summary ORDER BY repo_id").fetchall()
        ]
        apis = []
        for row in conn.execute("SELECT * FROM api ORDER BY api_name").fetchall():
            api_id = int(row["api_id"])
            record: dict[str, object] = {
                "api_name": str(row["api_name"]),
                "doc_names": [str(item[0]) for item in conn.execute("SELECT doc_name FROM api_doc_name WHERE api_id = ? ORDER BY ordinal", (api_id,)).fetchall()],
                "source_repo": str(row["source_repo"]),
                "workspace_api": str(row["workspace_api"]),
                "execute_api": str(row["execute_api"]),
                "summary": str(row["summary"]),
                "constraints": [str(item[0]) for item in conn.execute("SELECT value_text FROM api_constraint WHERE api_id = ? ORDER BY ordinal", (api_id,)).fetchall()],
                "error_conditions": [str(item[0]) for item in conn.execute("SELECT value_text FROM api_error_condition WHERE api_id = ? ORDER BY ordinal", (api_id,)).fetchall()],
                "return_codes": [
                    {
                        "name": str(item["name"]),
                        "code": str(item["code"]),
                        "description": str(item["description"]),
                    }
                    for item in conn.execute("SELECT name, code, description FROM api_return_code WHERE api_id = ? ORDER BY ordinal", (api_id,)).fetchall()
                ],
                "evidence": [str(item[0]) for item in conn.execute("SELECT value_text FROM api_evidence WHERE api_id = ? ORDER BY ordinal", (api_id,)).fetchall()],
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
                    if item["description"]:
                        param["description"] = str(item["description"])
                    if item["dtype"]:
                        param["dtype"] = str(item["dtype"])
                    if item["noncontiguous"]:
                        param["noncontiguous"] = str(item["noncontiguous"])
                    if item["tensor_rank"]:
                        param["tensor_rank"] = str(item["tensor_rank"])
                    for json_field, output_field in (
                        ("layout_templates_json", "layout_templates"),
                        ("shape_constraints_json", "shape_constraints"),
                        ("value_constraints_json", "value_constraints"),
                        ("optional_semantics_json", "optional_semantics"),
                        ("output_relation_json", "output_relation"),
                    ):
                        loaded = json_load_list(str(item[json_field]))
                        if loaded:
                            param[output_field] = loaded
                    params.append(param)
                if params:
                    record[field_name] = params
            apis.append(record)
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
                "source_repository_count": int(meta_row["source_repository_count"]),
                "source_repositories": source_repositories,
                "api_count": int(meta_row["api_count"]),
            },
            "repositories": repositories,
            "apis": apis,
        }
    finally:
        conn.close()


def write_compact_md(records: list[dict[str, object]], repo_infos: list[dict[str, str]]) -> None:
    lines = [
        "# ACLNN API Compact Reference",
        "",
        f"Last updated: {datetime.date.today().isoformat()}",
        "",
    ]
    for record in sorted(records, key=lambda item: str(item["api_name"])):
        def render_params(title: str, params: list[dict[str, object]]) -> list[str]:
            filtered = list(params)
            if is_attention_api(str(record.get("api_name", ""))) and len(filtered) > 12:
                key_params = {
                    "query",
                    "key",
                    "keyIn",
                    "value",
                    "dy",
                    "dOut",
                    "out",
                    "attentionOut",
                    "softmaxLse",
                    "inputLayout",
                    "layout",
                    "layoutQuery",
                    "layoutKv",
                    "headNum",
                    "numHeads",
                    "numKeyValueHeads",
                }
                filtered = [
                    param
                    for param in filtered
                    if compact_param_details(param) or str(param.get("name", "")) in key_params
                ]
                filtered = filtered[:16]
            block = [f"- {title}:"]
            for param in filtered:
                name = str(param.get("name", ""))
                role = str(param.get("role", ""))
                if not name:
                    continue
                detail = compact_param_details(param)
                if detail:
                    block.append(f"  - `{name}` ({role}): {detail}")
                else:
                    block.append(f"  - `{name}` ({role})")
            return block

        compact_constraints = select_compact_constraints(record)
        failed_like = record.get("extraction_status") == "failed"
        lines.extend(
            [
                f"## {record['api_name']}",
                "",
                f"- Source Repo: `{record['source_repo']}`",
                f"- Doc Names: {', '.join(record['doc_names'])}",
                f"- Workspace API: `{record['workspace_api']}`",
                f"- Execute API: `{record['execute_api']}`",
                f"- When This Doc Is Useful: {infer_when_useful(str(record['api_name']), record)}",
            ]
        )
        if record.get("summary"):
            lines.append(f"- Summary: {record['summary']}")
        if record.get("inputs") and not failed_like:
            lines.extend(render_params("Inputs", list(record["inputs"])))
        if record.get("outputs") and not failed_like:
            lines.extend(render_params("Outputs", list(record["outputs"])))
        lines.append("- Relevant Contract:")
        if compact_constraints:
            for item in compact_constraints:
                lines.append(f"  - {item}")
        else:
            lines.append("  - Information insufficient; rely on explicit parameter table and return codes only.")
        if record.get("error_conditions"):
            lines.append("- Likely Failure Triggers:")
            for item in dedupe_text_items(list(record["error_conditions"])):
                if item and item != "unknown":
                    lines.append(f"  - {item}")
        lines.append("")

    lines.extend(
        [
            "## Repository Info",
            "",
            "| Repository | Branch | Commit |",
            "| --- | --- | --- |",
        ]
    )
    for info in repo_infos:
        lines.append(f"| {info['name']} | {info['branch']} | `{info['commit']}` |")
    lines.append("")
    COMPACT_FILE.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] Wrote {COMPACT_FILE.name}")


def print_generation_stats(records: list[dict[str, object]]) -> None:
    status_counts: dict[str, int] = {"complete": 0, "partial": 0, "failed": 0}
    partial_by_reason: dict[str, int] = {
        "empty_constraints": 0,
        "no_outputs": 0,
        "weak_attention_contract": 0,
        "weak_matmul_contract": 0,
    }
    relevant_contract_total = 0
    attention_card_lines = 0
    attention_card_count = 0

    compact_text = COMPACT_FILE.read_text(encoding="utf-8") if COMPACT_FILE.exists() else ""
    for record in records:
        status = str(record.get("extraction_status", ""))
        if status in status_counts:
            status_counts[status] += 1
        if status == "partial":
            if not record.get("constraints"):
                partial_by_reason["empty_constraints"] += 1
            if not record.get("outputs"):
                partial_by_reason["no_outputs"] += 1
            api_name = str(record.get("api_name", ""))
            if is_attention_api(api_name):
                partial_by_reason["weak_attention_contract"] += 1
            if any(token in api_name for token in ["Matmul", "MatMul", "Mm", "Addmm"]):
                partial_by_reason["weak_matmul_contract"] += 1
        relevant_contract_total += len(select_compact_constraints(record))
        if compact_text and is_attention_api(str(record.get("api_name", ""))):
            marker = f"## {record['api_name']}"
            start = compact_text.find(marker)
            if start != -1:
                next_start = compact_text.find("\n## ", start + 1)
                block = compact_text[start:] if next_start == -1 else compact_text[start:next_start]
                attention_card_lines += len(block.splitlines())
                attention_card_count += 1

    print("[STATS] status:", status_counts)
    print("[STATS] partial_by_reason:", partial_by_reason)
    print("[STATS] empty_constraints:", sum(1 for record in records if not record.get("constraints")))
    print("[STATS] information_insufficient:", compact_text.count("Information insufficient"))
    avg_contracts = round(relevant_contract_total / max(len(records), 1), 2)
    print("[STATS] compact_quality:", {"avg_relevant_contracts": avg_contracts, "attention_avg_lines": round(attention_card_lines / max(attention_card_count, 1), 2)})


def record_quality(record: dict[str, object]) -> tuple[int, int, int]:
    status_order = {"complete": 3, "partial": 2, "failed": 1}
    status_score = status_order.get(str(record.get("extraction_status", "")), 0)
    contract_score = len(record.get("constraints") or [])
    param_score = sum(
        1
        for param in (record.get("inputs") or []) + (record.get("outputs") or [])
        for field in ("layout_templates", "shape_constraints", "value_constraints", "optional_semantics", "output_relation")
        if param.get(field)
    )
    return (status_score, contract_score, param_score)


def consolidate_records(records: list[dict[str, object]]) -> list[dict[str, object]]:
    merged: dict[str, dict[str, object]] = {}
    for record in records:
        name = str(record.get("api_name", ""))
        if not name:
            continue
        existing = merged.get(name)
        if existing is None:
            merged[name] = record
            continue
        better, worse = (record, existing) if record_quality(record) > record_quality(existing) else (existing, record)
        combined = dict(better)
        combined["doc_names"] = sorted(set((better.get("doc_names") or []) + (worse.get("doc_names") or [])))
        combined["evidence"] = dedupe_text_items(list(better.get("evidence") or []) + list(worse.get("evidence") or []))
        if not combined.get("constraints") and worse.get("constraints"):
            combined["constraints"] = worse.get("constraints")
        if combined.get("extraction_status") != "failed":
            if not combined.get("inputs") and worse.get("inputs"):
                combined["inputs"] = worse.get("inputs")
            if not combined.get("outputs") and worse.get("outputs"):
                combined["outputs"] = worse.get("outputs")
        merged[name] = combined
    return list(merged.values())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate final CANN failure indexes from source repositories.")
    parser.add_argument("--workspace-root", metavar="DIR", default=str(WORKSPACE_DEFAULT))
    parser.add_argument("--out", metavar="DIR", default=str(REFERENCE_DIR))
    parser.add_argument("--local-runtime-repo", metavar="DIR")
    parser.add_argument("--local-ops-repo", metavar="DIR", action="append")
    parser.add_argument("--keep-workspace", action="store_true")
    parser.add_argument("--with-error-yaml", action="store_true")
    parser.add_argument("--with-aclnn-yaml", action="store_true")
    parser.add_argument("--with-source-docs", action="store_true")
    parser.add_argument("--with-compact", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    return parser.parse_args()


def remove_legacy_outputs(out_dir: Path) -> None:
    for legacy_name in (
        "cann_error_index.db.tmp",
        "cann_error_index.db.tmp-journal",
        "cann_aclnn_api_index.db.tmp",
        "cann_aclnn_api_index.db.tmp-journal",
    ):
        legacy = out_dir / legacy_name
        if legacy.exists():
            try:
                legacy.unlink()
            except OSError:
                pass


def main() -> None:
    args = parse_args()
    workspace_root = run_workspace_root(Path(args.workspace_root).resolve(), keep_workspace=args.keep_workspace)
    global REFERENCE_DIR, ACL_DOC_PATH, ACLNN_DOC_PATH, ERROR_INDEX_PATH, ERROR_DB_PATH, INDEX_FILE, INDEX_DB_FILE, COMPACT_FILE
    REFERENCE_DIR = Path(args.out).resolve()
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    ACL_DOC_PATH = REFERENCE_DIR / "aclError.md"
    ACLNN_DOC_PATH = REFERENCE_DIR / "aclnnApiError.md"
    ERROR_INDEX_PATH = REFERENCE_DIR / "cann_error_index.yaml"
    ERROR_DB_PATH = REFERENCE_DIR / "cann_error_index.db"
    INDEX_FILE = REFERENCE_DIR / "cann_aclnn_api_index.yaml"
    INDEX_DB_FILE = REFERENCE_DIR / "cann_aclnn_api_index.db"
    COMPACT_FILE = REFERENCE_DIR / "aclnn_api_compact.md"
    remove_legacy_outputs(REFERENCE_DIR)

    source_mode = "remote"
    if args.local_runtime_repo:
        runtime_path = resolve_local_repo(args.local_runtime_repo)
        source_mode = "mixed"
    else:
        runtime_path = clone_or_update_repo(RUNTIME_REPO, workspace_root)
    if runtime_path is None:
        print("[ERROR] Runtime repository unavailable.")
        sys.exit(1)

    local_ops = resolve_local_ops(args.local_ops_repo)
    ops_infos: list[dict[str, str]] = []
    failed_ops: list[str] = []
    for repo in OPS_REPOS:
        repo_path = local_ops.get(repo["name"])
        source_type = "local"
        if repo_path is None:
            source_type = "cloned"
            repo_path = clone_or_update_repo(repo, workspace_root)
        else:
            source_mode = "mixed"
        if repo_path is None:
            failed_ops.append(repo["name"])
            continue
        ops_infos.append(build_manifest_entry(repo["name"], repo_path, source_type, repo.get("https", "")))

    if failed_ops:
        print(f"[ERROR] Required ops repositories unavailable: {', '.join(failed_ops)}")
        sys.exit(1)

    runtime_info = build_manifest_entry(
        RUNTIME_REPO["name"],
        runtime_path,
        "local" if args.local_runtime_repo else "cloned",
        RUNTIME_REPO["https"],
    )
    if args.local_runtime_repo and args.local_ops_repo and len(local_ops) == len(OPS_REPOS):
        source_mode = "local"

    write_manifest(
        workspace_root,
        {
            "last_updated": datetime.date.today().isoformat(),
            "workspace_root": str(workspace_root),
            "runtime": runtime_info,
            "ops": ops_infos,
        },
    )

    try:
        prepare_runtime_error_doc(runtime_info)
        docs, repo_infos = prepare_input_docs([item for item in ops_infos if isinstance(item, dict)])
        if not docs:
            print("[ERROR] No ACLNN markdown documents available.")
            sys.exit(1)
        records: list[dict[str, object]] = []
        for item in docs:
            path = item["path"]
            assert isinstance(path, Path)
            records.extend(
                summarize_doc(
                    doc_name=str(item["doc_name"]),
                    source_repo=str(item["source_repo"]),
                    text=path.read_text(encoding="utf-8"),
                )
            )
        records = consolidate_records(records)
        meta = build_index_meta(
            runtime_info,
            [item for item in ops_infos if isinstance(item, dict)],
            source_mode=source_mode,
            deterministic=args.deterministic,
        )
        error_payload = build_cann_error_payload(meta)
        aclnn_payload = build_aclnn_index_payload(records, repo_infos, meta)
        write_error_index_db(ERROR_DB_PATH, error_payload, deterministic=args.deterministic)
        write_aclnn_index_db(INDEX_DB_FILE, aclnn_payload, deterministic=args.deterministic)
        files = [str(ERROR_DB_PATH), str(INDEX_DB_FILE)]
        if args.with_error_yaml:
            write_yaml(ERROR_INDEX_PATH, load_error_index_payload_from_db(ERROR_DB_PATH))
            print(f"[OK] Wrote {ERROR_INDEX_PATH.name}")
            files.append(str(ERROR_INDEX_PATH))
        if args.with_aclnn_yaml:
            write_yaml(INDEX_FILE, load_aclnn_index_payload_from_db(INDEX_DB_FILE))
            print(f"[OK] Wrote {INDEX_FILE.name}")
            files.append(str(INDEX_FILE))
        if args.with_compact:
            write_compact_md(records, repo_infos)
        print_generation_stats(records)
        if not args.with_source_docs:
            if ACL_DOC_PATH.exists():
                safe_unlink(ACL_DOC_PATH)
            if ACLNN_DOC_PATH.exists():
                safe_unlink(ACLNN_DOC_PATH)
        if not args.with_compact and COMPACT_FILE.exists():
            safe_unlink(COMPACT_FILE)
        if args.with_source_docs:
            if ACL_DOC_PATH.exists():
                files.append(str(ACL_DOC_PATH))
            if ACLNN_DOC_PATH.exists():
                files.append(str(ACLNN_DOC_PATH))
        if args.with_compact and COMPACT_FILE.exists():
            files.append(str(COMPACT_FILE))
        print("files:")
        for item in files:
            print(f"- {item}")
    finally:
        cleanup(remove_legacy_docs=True)
        if not args.keep_workspace:
            try:
                safe_rmtree(workspace_root)
                prune_empty_parents(workspace_root.parent, stop_at=SCRIPT_DIR)
            except PermissionError:
                pass


if __name__ == "__main__":
    main()
