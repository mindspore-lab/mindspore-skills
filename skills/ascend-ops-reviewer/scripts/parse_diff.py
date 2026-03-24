#!/usr/bin/env python3
"""
Parse unified diff format and extract structured information.

This module provides functionality to parse git diff output and extract
file-level and hunk-level information for code review purposes.
"""

import re
from typing import Dict, List, Tuple
from pathlib import Path


def classify_file_type(file_path: str) -> str:
    """
    Classify file type based on path and extension.

    Args:
        file_path: Path to the file

    Returns:
        File type: 'python', 'cpp', 'cuda', 'header', 'test', 'doc', 'unknown'
    """
    path = Path(file_path)
    name = path.name.lower()
    suffix = path.suffix.lower()

    # Test files
    if name.startswith('test_') or '/test/' in file_path.lower() or '/tests/' in file_path.lower():
        return 'test'

    # Documentation
    if suffix in ['.md', '.rst', '.txt']:
        return 'doc'

    # Python files
    if suffix == '.py':
        return 'python'

    # CUDA files
    if suffix in ['.cu', '.cuh']:
        return 'cuda'

    # C++ headers
    if suffix in ['.h', '.hpp', '.hh']:
        return 'header'

    # C++ implementation
    if suffix in ['.cpp', '.cc', '.cxx', '.c']:
        return 'cpp'

    return 'unknown'


def parse_hunk_header(header: str) -> Tuple[int, int, int, int]:
    """
    Parse hunk header line.

    Args:
        header: Hunk header line (e.g., "@@ -10,5 +12,7 @@")

    Returns:
        Tuple of (old_start, old_lines, new_start, new_lines)
    """
    match = re.match(r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@', header)
    if not match:
        return (0, 0, 0, 0)

    old_start = int(match.group(1))
    old_lines = int(match.group(2)) if match.group(2) else 1
    new_start = int(match.group(3))
    new_lines = int(match.group(4)) if match.group(4) else 1

    return (old_start, old_lines, new_start, new_lines)


def parse_diff(diff_content: str) -> Dict:
    """
    Parse unified diff content and extract structured information.

    Args:
        diff_content: Content of unified diff

    Returns:
        Dictionary containing:
        - files: List of file changes with hunks
        - total_additions: Total lines added
        - total_deletions: Total lines deleted
        - total_files: Total number of files changed
    """
    files = []
    total_additions = 0
    total_deletions = 0

    # Split by file boundaries
    file_pattern = re.compile(r'^diff --git a/(.*?) b/(.*?)$', re.MULTILINE)
    file_splits = list(file_pattern.finditer(diff_content))

    for i, match in enumerate(file_splits):
        file_path = match.group(2)  # Use 'b/' path (new file path)
        start_pos = match.start()
        end_pos = file_splits[i + 1].start() if i + 1 < len(file_splits) else len(diff_content)
        file_diff = diff_content[start_pos:end_pos]

        # Parse hunks
        hunks = []
        hunk_pattern = re.compile(r'^@@.*?@@.*?$', re.MULTILINE)
        hunk_matches = list(hunk_pattern.finditer(file_diff))

        file_additions = 0
        file_deletions = 0

        for j, hunk_match in enumerate(hunk_matches):
            hunk_header = hunk_match.group(0)
            old_start, old_lines, new_start, new_lines = parse_hunk_header(hunk_header)

            # Extract hunk content
            hunk_start = hunk_match.end()
            hunk_end = hunk_matches[j + 1].start() if j + 1 < len(hunk_matches) else len(file_diff)
            hunk_content = file_diff[hunk_start:hunk_end]

            # Count additions and deletions in this hunk
            for line in hunk_content.split('\n'):
                if line.startswith('+') and not line.startswith('+++'):
                    file_additions += 1
                elif line.startswith('-') and not line.startswith('---'):
                    file_deletions += 1

            hunks.append({
                'old_start': old_start,
                'old_lines': old_lines,
                'new_start': new_start,
                'new_lines': new_lines,
                'content': hunk_content
            })

        files.append({
            'path': file_path,
            'type': classify_file_type(file_path),
            'additions': file_additions,
            'deletions': file_deletions,
            'hunks': hunks
        })

        total_additions += file_additions
        total_deletions += file_deletions

    return {
        'files': files,
        'total_additions': total_additions,
        'total_deletions': total_deletions,
        'total_files': len(files)
    }


if __name__ == '__main__':
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python3 parse_diff.py <diff_file>")
        sys.exit(1)

    diff_file = sys.argv[1]
    with open(diff_file, 'r', encoding='utf-8') as f:
        diff_content = f.read()

    result = parse_diff(diff_content)
    print(json.dumps(result, indent=2, ensure_ascii=False))
