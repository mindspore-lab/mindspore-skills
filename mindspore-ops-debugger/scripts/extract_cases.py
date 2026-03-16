#!/usr/bin/env python3
"""Extract structured data from gitcode issue markdown files.

Parses the 100 gitcode issue files under md_files/gitcode/issues/ and outputs
a JSON array of structured case records.
"""

import json
import os
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Operator name list (derived from YAML filenames)
# ---------------------------------------------------------------------------

YAML_DIR = Path("/Users/claw/work/ms_debug/mindspore/mindspore/ops/op_def/yaml")

def load_op_names(yaml_dir: Path | None = None) -> list[str]:
    """Return sorted list of operator names from YAML filenames."""
    d = yaml_dir or YAML_DIR
    if not d.is_dir():
        return []
    names = []
    for f in d.iterdir():
        if f.suffix == ".yaml" and f.name != "README.md":
            # e.g. adam_weight_decay_op.yaml -> adam_weight_decay
            name = f.stem
            if name.endswith("_op"):
                name = name[:-3]
            names.append(name)
    # Sort longest first so greedy matching prefers longer names
    names.sort(key=lambda n: -len(n))
    return names

# ---------------------------------------------------------------------------
# Section extraction helpers
# ---------------------------------------------------------------------------

def _strip_quote_prefix(text: str) -> str:
    """Remove leading '> ' from each line (quoted template blocks)."""
    return "\n".join(
        line[2:] if line.startswith("> ") else line
        for line in text.splitlines()
    )

def _find_section(text: str, header: str, stop_headers: list[str] | None = None) -> str:
    """Extract text between *header* and the next stop header (or EOF).

    Searches for the LAST non-quoted occurrence of *header* so we skip the
    bot-posted template (which is inside a ``> `` quote block).
    """
    # Find all non-quoted occurrences
    positions = []
    for m in re.finditer(re.escape(header), text):
        # Check if this line starts with '>'
        line_start = text.rfind("\n", 0, m.start()) + 1
        line = text[line_start:m.start()]
        if ">" not in line:
            positions.append(m.end())

    if not positions:
        return ""

    start = positions[-1]  # last non-quoted occurrence

    # Find the end boundary
    if stop_headers:
        end = len(text)
        for sh in stop_headers:
            for m in re.finditer(re.escape(sh), text[start:]):
                line_start = text[start:].rfind("\n", 0, m.start()) + 1
                line = text[start:][line_start:m.start()]
                if ">" not in line:
                    candidate = start + m.start()
                    if candidate < end:
                        end = candidate
                    break
        return text[start:end].strip()
    return text[start:].strip()


def _extract_issue_id(filename: str) -> str:
    """Extract issue number from filename like '...-issues-41932.md'."""
    m = re.search(r"issues-(\d+)\.md$", filename)
    return m.group(1) if m else ""


def _extract_title(text: str) -> str:
    """Extract the [Bug]:... title line."""
    m = re.search(r"\[Bug\]:(.+?)(?:\n|$)", text)
    if m:
        # Clean up trailing noise
        title = m.group(1).strip()
        title = re.sub(r"-org-issues.*$", "", title).strip()
        return title
    # Fallback: first heading
    m = re.search(r"^#\s+(.+)", text, re.MULTILINE)
    return m.group(1).strip() if m else ""


def _extract_env(text: str) -> dict:
    """Extract environment info: device, commit, CANN version."""
    env = {"device": "", "ms_commit": "", "cann_version": ""}

    # Device: 910A, 910B, CPU, GPU
    m = re.search(r"(910[AB]|Ascend|CPU|GPU)", text, re.IGNORECASE)
    if m:
        env["device"] = m.group(1)

    # MindSpore commit
    m = re.search(r"commit[_\s]*id\s*[:：]\s*([0-9a-f]{8,40})", text, re.IGNORECASE)
    if m:
        env["ms_commit"] = m.group(1)

    # CANN version
    m = re.search(r"cann[版本\s]*[:：]\s*(\S+)", text, re.IGNORECASE)
    if m:
        env["cann_version"] = m.group(1)

    return env


def _extract_root_cause(text: str) -> str:
    """Extract the Appearance & Root Cause section."""
    section = _find_section(
        text,
        "Appearance & Root Cause",
        stop_headers=["Fix Solution", "Fix Description"],
    )
    return section


def _extract_fix_solution(text: str) -> str:
    """Extract the Fix Solution section."""
    section = _find_section(
        text,
        "Fix Solution",
        stop_headers=["Fix Description", "Self-test Report"],
    )
    return section


def _extract_fix_pr(text: str) -> str:
    """Extract fix PR link from Fix Description & Test Suggestion."""
    section = _find_section(
        text,
        "Fix Description & Test Suggestion",
        stop_headers=["Self-test Report", "Introduction Analysis"],
    )
    # Find gitee PR links
    prs = re.findall(r"https://gitee\.com/mindspore/mindspore/pulls/\d+", section)
    if prs:
        return prs[0]
    # Fallback: search entire text for non-quoted PR links (skip template)
    all_prs = re.findall(r"https://gitee\.com/mindspore/mindspore/pulls/(\d+)", text)
    # Filter out template example PR (often /pulls/70920 or /pulls/xxx)
    real_prs = [p for p in all_prs if p != "xxx" and p != "70920"]
    if real_prs:
        return f"https://gitee.com/mindspore/mindspore/pulls/{real_prs[0]}"
    return ""


def _extract_introduction_type(text: str) -> str:
    """Extract introduction type from Introduction Analysis."""
    section = _find_section(
        text,
        "Introduction Analysis",
        stop_headers=["Regression Test", "[Image:", "**"],
    )
    m = re.search(r"引入类型\s*[:：]\s*(.+?)(?:\n|$)", section)
    if m:
        val = m.group(1).strip()
        # Skip template placeholder
        if "/" in val and len(val) > 40:
            return ""
        return val
    return ""


def _extract_introduction_pr(text: str) -> str:
    """Extract introduction PR from Introduction Analysis."""
    section = _find_section(
        text,
        "Introduction Analysis",
        stop_headers=["Regression Test", "[Image:", "**"],
    )
    prs = re.findall(r"https://gitee\.com/mindspore/mindspore/pulls/\d+", section)
    return prs[0] if prs else ""


def _extract_status(text: str) -> str:
    """Extract Issue status (DONE/WIP/VALIDATION)."""
    m = re.search(r"#{5}\s*Issue\s*状态\s*\n+\s*(DONE|WIP|VALIDATION|ACCEPTED|OPEN)", text)
    return m.group(1) if m else ""


def _match_op_name(title: str, op_names: list[str], full_text: str = "") -> str:
    """Fuzzy-match an operator name from the title (and optionally full text).

    Strategy:
    1. Look for explicit API patterns: ops.xxx, nn.xxx, mint.xxx, F.xxx
    2. Try exact substring match against known YAML op names (longest first).
    3. Normalize title to snake_case and retry.
    4. Search in full_text for API patterns as fallback.
    """
    title_lower = title.lower()

    # Common non-operator names to exclude
    _EXCLUDE = {"cell", "module", "parameter", "tensor", "context", "cpp",
                "o", "op", "ops", "nn", "functional", "init", "net", "model",
                "operations", "vmap", "jit", "grad", "value_and_grad"}

    # Direct API pattern: ops.xxx / nn.xxx / mint.xxx / F.xxx
    m = re.search(r"(?:ops|nn|mint|F)\.(\w+)", title)
    if m:
        candidate = m.group(1).lower()
        if candidate.isascii() and candidate not in _EXCLUDE:
            return candidate

    # Normalize: CamelCase -> snake_case for matching
    title_snake = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", title).lower()
    # Only keep ASCII alphanumeric and underscores for matching
    title_ascii = re.sub(r"[^a-z0-9_ ]", " ", title_snake)

    for op in op_names:
        op_lower = op.lower()
        # Skip very short op names (<=2 chars) to avoid false positives
        if len(op_lower) <= 2:
            continue
        # Check in ASCII-only title with word boundaries
        if re.search(r"(?<![a-z0-9_])" + re.escape(op_lower) + r"(?![a-z0-9_])", title_ascii):
            return op_lower

    # Fallback: search full text for API patterns
    if full_text:
        for m in re.finditer(r"(?:ops|nn|mint|F)\.(\w+)", full_text[:3000]):
            candidate = m.group(1).lower()
            if candidate.isascii() and len(candidate) > 2 and candidate not in _EXCLUDE:
                return candidate

    return ""


def extract_case(filepath: Path, op_names: list[str]) -> dict:
    """Extract structured data from a single gitcode issue file."""
    text = filepath.read_text(encoding="utf-8")
    filename = filepath.name
    issue_id = _extract_issue_id(filename)
    title = _extract_title(text)

    return {
        "issue_id": issue_id,
        "title": title,
        "op_name": _match_op_name(title, op_names, text),
        "env": _extract_env(text),
        "root_cause": _extract_root_cause(text),
        "fix_solution": _extract_fix_solution(text),
        "fix_pr": _extract_fix_pr(text),
        "introduction_type": _extract_introduction_type(text),
        "introduction_pr": _extract_introduction_pr(text),
        "status": _extract_status(text),
        "source_file": filename,
    }


def main():
    issues_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        Path("/Users/claw/work/ms_debug/md_files/gitcode/issues")
    )
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else (
        Path(__file__).resolve().parent / "gitcode_cases.json"
    )

    if not issues_dir.is_dir():
        print(f"Error: {issues_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    op_names = load_op_names()
    print(f"Loaded {len(op_names)} operator names from YAML definitions")

    files = sorted(issues_dir.glob("*.md"))
    print(f"Processing {len(files)} issue files from {issues_dir}")

    cases = []
    for f in files:
        case = extract_case(f, op_names)
        cases.append(case)

    output_path.write_text(
        json.dumps(cases, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {len(cases)} cases to {output_path}")

    # Summary stats
    with_root_cause = sum(1 for c in cases if c["root_cause"])
    with_fix = sum(1 for c in cases if c["fix_pr"])
    with_op = sum(1 for c in cases if c["op_name"])
    done = sum(1 for c in cases if c["status"] == "DONE")
    print(f"\nStats:")
    print(f"  With root cause: {with_root_cause}/{len(cases)}")
    print(f"  With fix PR:     {with_fix}/{len(cases)}")
    print(f"  With op name:    {with_op}/{len(cases)}")
    print(f"  Status DONE:     {done}/{len(cases)}")


if __name__ == "__main__":
    main()
