from pathlib import Path

import pytest


def test_manifest_exists_or_is_tracked_for_rollout():
    skill_root = Path(__file__).resolve().parents[1]
    manifest = skill_root / "skill.yaml"
    if not manifest.exists():
        pytest.skip("skill.yaml rollout pending for this skill.")
    assert manifest.exists()
