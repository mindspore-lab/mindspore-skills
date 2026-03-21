#!/usr/bin/env python3
"""Build a lightweight index of gitee issue markdown files.

Extracts: issue_id, title, status, keywords from 35K+ gitee issue files.
Outputs a JSON index for searchable reference.
"""

import json
import re
import sys
from pathlib import Path

# Keywords relevant to operator issues
OP_KEYWORDS = [
    "算子", "operator", "ops", "kernel", "精度", "precision", "NaN", "Inf",
    "shape", "broadcast", "dtype", "bprop", "grad", "backward", "反向",
    "推导", "infer", "ACLNN", "aclnn", "Ascend", "GPU", "CPU",
    "编译", "compile", "core dump", "segfault", "超时", "timeout",
    "性能", "performance", "内存", "memory", "OOM",
    "动态shape", "dynamic shape", "PyNative", "Graph",
]


def extract_gitee_issue(filepath: Path) -> dict | None:
    """Extract lightweight index data from a gitee issue file."""
    try:
        text = filepath.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None

    # Title: first line (heading)
    title = ""
    m = re.match(r"^#\s+(.+)", text)
    if m:
        title = m.group(1).strip()

    # Issue ID: **Issue ID**: #IXXXXX
    issue_id = ""
    m = re.search(r"\*\*Issue ID\*\*:\s*#?(\S+)", text)
    if m:
        issue_id = m.group(1)
    else:
        # Fallback: extract from filename
        m = re.search(r"issue-([a-z0-9]+)\.md$", filepath.name, re.IGNORECASE)
        if m:
            issue_id = m.group(1).upper()

    # Status
    status = ""
    m = re.search(r"\*\*Status\*\*:\s*(\S+)", text)
    if m:
        status = m.group(1)

    # Match keywords
    text_lower = text.lower()
    matched_kw = [kw for kw in OP_KEYWORDS if kw.lower() in text_lower]

    if not issue_id and not title:
        return None

    return {
        "issue_id": issue_id,
        "title": title,
        "status": status,
        "keywords": matched_kw,
        "source_file": filepath.name,
    }


def main():
    issues_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        Path("/Users/claw/work/ms_debug/md_files/gitee/issues")
    )
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else (
        Path(__file__).resolve().parent / "gitee_index.json"
    )

    if not issues_dir.is_dir():
        print(f"Error: {issues_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    files = sorted(issues_dir.glob("*.md"))
    print(f"Processing {len(files)} gitee issue files...")

    index = []
    skipped = 0
    for i, f in enumerate(files):
        if (i + 1) % 5000 == 0:
            print(f"  ...processed {i + 1}/{len(files)}")
        entry = extract_gitee_issue(f)
        if entry:
            index.append(entry)
        else:
            skipped += 1

    output_path.write_text(
        json.dumps(index, ensure_ascii=False, indent=None),
        encoding="utf-8",
    )
    print(f"Wrote {len(index)} entries to {output_path} (skipped {skipped})")

    # Stats
    with_kw = sum(1 for e in index if e["keywords"])
    statuses = {}
    for e in index:
        s = e["status"]
        statuses[s] = statuses.get(s, 0) + 1
    print(f"\nStats:")
    print(f"  With operator keywords: {with_kw}/{len(index)}")
    print(f"  Status distribution:")
    for s, cnt in sorted(statuses.items(), key=lambda x: -x[1])[:10]:
        print(f"    {s}: {cnt}")


if __name__ == "__main__":
    main()
