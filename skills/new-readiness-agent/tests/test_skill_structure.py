from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_expected_layout_exists():
    assert (ROOT / "SKILL.md").exists()
    assert (ROOT / "skill.yaml").exists()
    assert (ROOT / "references").is_dir()
    assert (ROOT / "scripts").is_dir()
    assert (ROOT / "contract" / "new-readiness-verdict.schema.json").exists()


def test_expected_reference_files_exist():
    assert (ROOT / "references" / "product-contract.md").exists()
    assert (ROOT / "references" / "decision-rules.md").exists()
    assert (ROOT / "references" / "cache-contract.md").exists()

