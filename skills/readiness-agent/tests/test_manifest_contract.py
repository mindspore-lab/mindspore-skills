from pathlib import Path


SKILL_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = SKILL_ROOT / "skill.yaml"
SKILL = SKILL_ROOT / "SKILL.md"


def _manifest_text() -> str:
    return MANIFEST.read_text(encoding="utf-8")


def test_manifest_contract_fields_present():
    text = _manifest_text()
    assert 'name: "readiness-agent"' in text
    assert 'display_name: "Readiness Agent"' in text
    assert 'version: "0.2.0"' in text
    assert 'type: "manual"' in text
    assert 'path: "SKILL.md"' in text
    assert 'network: "optional"' in text
    assert 'filesystem: "workspace-write"' in text


def test_manifest_declares_target_and_fix_inputs():
    text = _manifest_text()
    assert 'name: "working_dir"' in text
    assert 'name: "target"' in text
    assert 'choices: ["training", "inference", "auto"]' in text
    assert 'name: "mode"' in text
    assert 'choices: ["check", "fix", "auto"]' in text
    assert 'name: "allow_network"' in text
    assert 'name: "fix_scope"' in text
    assert 'choices: ["none", "safe-user-space"]' in text
    assert 'name: "task_smoke_cmd"' in text
    assert 'name: "factory_root"' in text
    assert 'report_schema' in text
    assert 'out_dir_layout' in text


def test_skill_describes_certification_workflow():
    text = SKILL.read_text(encoding="utf-8")
    assert "# Readiness Agent" in text
    assert "1. `execution-target-discovery`" in text
    assert "2. `dependency-closure-builder`" in text
    assert "3. `task-smoke-precheck` when a safe explicit smoke command exists" in text
    assert "4. `compatibility-validator`" in text
    assert "5. `blocker-classifier`" in text
    assert "6. `env-fix` when allowed and needed" in text
    assert "7. `revalidator-and-report-builder`" in text
    assert "overall status: `READY`, `WARN`, or `BLOCKED`" in text
