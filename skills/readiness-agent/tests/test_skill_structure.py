from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_expected_layout_exists():
    assert (ROOT / "SKILL.md").exists()
    assert (ROOT / "skill.yaml").exists()
    assert (ROOT / "contract").is_dir()
    assert (ROOT / "references").is_dir()
    assert (ROOT / "scripts").is_dir()
    assert (ROOT / "tests" / "fixtures").is_dir()
