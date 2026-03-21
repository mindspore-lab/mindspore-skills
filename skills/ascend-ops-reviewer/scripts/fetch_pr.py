#!/usr/bin/env python3
"""
Fetch PR diff from GitCode using git commands.

This module provides functionality to fetch PR changes from GitCode
repositories and generate unified diff output.
"""

import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, Tuple
from urllib.parse import urlparse


def parse_pr_url(pr_url: str) -> Tuple[str, str, int]:
    """
    Parse GitCode PR URL to extract repository and PR ID.

    Args:
        pr_url: GitCode PR URL (e.g., https://gitcode.com/mindspore/mindspore/pulls/12345)

    Returns:
        Tuple of (repo_owner, repo_name, pr_id)

    Raises:
        ValueError: If URL format is invalid
    """
    # Support both /pulls/ and /pull/ formats
    pattern = r'gitcode\.com/([^/]+)/([^/]+)/pulls?/(\d+)'
    match = re.search(pattern, pr_url)

    if not match:
        raise ValueError(f"Invalid GitCode PR URL format: {pr_url}")

    repo_owner = match.group(1)
    repo_name = match.group(2)
    pr_id = int(match.group(3))

    return (repo_owner, repo_name, pr_id)


def check_git_repo(repo_path: str) -> bool:
    """
    Check if the given path is a valid git repository.

    Args:
        repo_path: Path to check

    Returns:
        True if valid git repo, False otherwise
    """
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--git-dir'],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=False
        )
        return result.returncode == 0
    except Exception:
        return False


def fetch_pr_diff(pr_url: str, repo_path: str) -> Tuple[str, Dict]:
    """
    Fetch PR diff from GitCode using git commands.

    Args:
        pr_url: GitCode PR URL
        repo_path: Local repository path

    Returns:
        Tuple of (diff_content, metadata)
        metadata contains: pr_id, repo, title, author, base_branch, head_branch

    Raises:
        ValueError: If repo_path is not a git repository
        RuntimeError: If git operations fail
    """
    # Validate repository
    if not check_git_repo(repo_path):
        raise ValueError(f"Not a git repository: {repo_path}")

    # Parse PR URL
    repo_owner, repo_name, pr_id = parse_pr_url(pr_url)
    repo_full_name = f"{repo_owner}/{repo_name}"

    print(f"Fetching PR #{pr_id} from {repo_full_name}...", file=sys.stderr)

    # Fetch PR branch
    pr_branch = f"pr-{pr_id}"
    try:
        # Fetch the PR
        fetch_cmd = ['git', 'fetch', 'origin', f'pull/{pr_id}/head:{pr_branch}']
        result = subprocess.run(
            fetch_cmd,
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True
        )
        print(f"Fetched PR branch: {pr_branch}", file=sys.stderr)

    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to fetch PR: {e.stderr}")

    # Generate diff
    try:
        # Get base branch (usually master or main)
        base_branch = 'origin/master'
        diff_cmd = ['git', 'diff', f'{base_branch}...{pr_branch}']
        result = subprocess.run(
            diff_cmd,
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True
        )
        diff_content = result.stdout

        if not diff_content:
            print("Warning: Empty diff generated", file=sys.stderr)

    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to generate diff: {e.stderr}")

    finally:
        # Clean up temporary branch
        try:
            subprocess.run(
                ['git', 'branch', '-D', pr_branch],
                cwd=repo_path,
                capture_output=True,
                check=False
            )
            print(f"Cleaned up branch: {pr_branch}", file=sys.stderr)
        except Exception:
            pass

    # Build metadata
    metadata = {
        'pr_id': pr_id,
        'repo': repo_full_name,
        'title': f"PR #{pr_id}",  # Could be enhanced with API call
        'author': 'unknown',
        'base_branch': base_branch,
        'head_branch': pr_branch
    }

    return (diff_content, metadata)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python3 fetch_pr.py <pr_url> <repo_path>")
        print("Example: python3 fetch_pr.py https://gitcode.com/mindspore/mindspore/pulls/12345 /path/to/mindspore")
        sys.exit(1)

    pr_url = sys.argv[1]
    repo_path = sys.argv[2]

    try:
        diff_content, metadata = fetch_pr_diff(pr_url, repo_path)
        print(f"\n=== Metadata ===")
        print(f"PR: #{metadata['pr_id']}")
        print(f"Repo: {metadata['repo']}")
        print(f"Base: {metadata['base_branch']}")
        print(f"\n=== Diff ===")
        print(diff_content)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
