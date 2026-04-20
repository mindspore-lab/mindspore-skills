from pathlib import Path


SKILL_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = SKILL_ROOT / "skill.yaml"
SKILL = SKILL_ROOT / "SKILL.md"


def _manifest_text() -> str:
    return MANIFEST.read_text(encoding="utf-8")


def test_manifest_contract_fields_present():
    text = _manifest_text()
    assert 'name: "new-readiness-agent"' in text
    assert 'display_name: "New Readiness Agent"' in text
    assert 'version: "0.1.0"' in text
    assert 'type: "manual"' in text
    assert 'path: "SKILL.md"' in text
    assert 'network: "none"' in text
    assert 'filesystem: "workspace-write"' in text


def test_manifest_declares_read_only_inputs_and_standard_outputs():
    text = _manifest_text()
    for token in (
        'name: "working_dir"',
        'name: "target"',
        'choices: ["training", "inference", "auto"]',
        'name: "framework_hint"',
        'name: "launcher_hint"',
        'name: "selected_python"',
        'name: "selected_env_root"',
        'name: "cann_path"',
        'name: "launch_command"',
        'name: "extra_context"',
        'report_schema',
        'out_dir_layout',
    ):
        assert token in text


def test_skill_describes_read_only_four_stage_workflow():
    text = SKILL.read_text(encoding="utf-8")
    assert text.startswith("---\nname: new-readiness-agent\ndescription:")
    assert "# New Readiness Agent" in text
    assert "1. `workspace-analyzer`" in text
    assert "2. `compatibility-validator`" in text
    assert "3. `snapshot-builder`" in text
    assert "4. `report-builder`" in text
    assert "This skill does not repair anything." in text
    assert "`scripts/run_new_readiness_pipeline.py`" in text

