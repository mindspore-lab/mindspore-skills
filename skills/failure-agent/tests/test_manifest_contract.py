from pathlib import Path


SKILL_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = SKILL_ROOT / "skill.yaml"


def _manifest_text() -> str:
    return MANIFEST.read_text(encoding="utf-8")


def test_manifest_contract_fields_present():
    text = _manifest_text()
    assert 'name: "failure-agent"' in text
    assert 'description: "Diagnose MindSpore and PTA (PyTorch + torch_npu) crashes, runtime errors, hangs, and communication failures with evidence-first triage, ordered knowledge lookup, and manual-only report candidates."' in text
    assert 'version: "0.4.0"' in text
    assert 'type: "manual"' in text
    assert 'path: "SKILL.md"' in text
    assert 'python: []' in text
    assert 'network: "none"' in text
    assert 'filesystem: "workspace-write"' in text


def test_manifest_declares_dual_stack_scope_and_honest_dependencies():
    text = _manifest_text()
    assert "MindSpore and PTA" in text
    assert '"bash"' in text
    assert '"rg"' in text
    assert "report_schema" in text
    assert "out_dir_layout" in text


def test_reference_files_exist():
    reference_dir = SKILL_ROOT / "reference"
    for name in [
        "failure-showcase.md",
        "error-codes.md",
        "backend-diagnosis.md",
        "mindspore-api.md",
        "torch-npu-operators.md",
    ]:
        assert (reference_dir / name).exists(), name
