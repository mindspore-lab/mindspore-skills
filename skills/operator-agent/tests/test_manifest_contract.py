from pathlib import Path


SKILL_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = SKILL_ROOT / "skill.yaml"
SKILL = SKILL_ROOT / "SKILL.md"


def _manifest_text() -> str:
    return MANIFEST.read_text(encoding="utf-8")


def test_manifest_contract_fields_present():
    text = _manifest_text()
    assert 'name: "operator-agent"' in text
    assert 'display_name: "Op Agent"' in text
    assert 'version: "0.2.0"' in text
    assert 'type: "manual"' in text
    assert 'path: "SKILL.md"' in text
    assert 'filesystem: "workspace-write"' in text


def test_manifest_declares_two_implementation_methods():
    text = _manifest_text()
    assert 'name: "method_preference"' in text
    assert 'choices: ["custom-access", "native-framework"]' in text
    assert 'name: "delivery_goal"' in text
    assert '"operator"' in text
    assert '"report"' in text


def test_skill_describes_four_stage_operator_workflow():
    text = SKILL.read_text(encoding="utf-8")
    assert "# Operator Agent" in text
    assert "1. `operator-analyzer`" in text
    assert "2. `method-selector`" in text
    assert "3. `implementation-builder`" in text
    assert "4. `verification-and-report`" in text
    assert "This skill supports exactly two implementation methods." in text
