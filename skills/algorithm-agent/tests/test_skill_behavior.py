from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKILL_MD = ROOT / "SKILL.md"
INTAKE_TRIAGE_MD = ROOT / "references" / "intake-prestage-and-triage.md"
INTAKE_VERIFY_MD = ROOT / "references" / "intake-prestage-verification-and-admission.md"
TRANSMLA_CASE_MD = ROOT / "references" / "transmla" / "transmla-case-study.md"


def test_workflow_stages_are_present():
    text = SKILL_MD.read_text(encoding="utf-8")
    assert "1. `feature-analyzer`" in text
    assert "2. `integration-planner`" in text
    assert "3. `patch-builder`" in text
    assert "4. `readiness-handoff-and-report`" in text


def test_route_selection_and_route_packs_are_declared():
    text = SKILL_MD.read_text(encoding="utf-8")
    assert "Choose exactly one integration route:" in text
    assert "- `generic-feature`" in text
    assert "- `mhc`" in text
    assert "- `attnres`" in text
    assert "- `transmla`" in text
    assert "`integration_route`" in text
    assert "`route_evidence`" in text
    assert "`references/mhc/mhc-implementation-pattern.md`" in text
    assert "`references/mhc/mhc-validation-checklist.md`" in text
    assert "`references/mhc/mhc-qwen3-case-study.md`" in text
    assert "`references/attnres/attnres-implementation-pattern.md`" in text
    assert "`references/attnres/attnres-validation-checklist.md`" in text
    assert "`references/attnres/attnres-qwen3-case-study.md`" in text


def test_algorithm_agent_remains_top_level_entry():
    text = SKILL_MD.read_text(encoding="utf-8")
    assert "This skill is the top-level algorithm feature entry." in text
    assert "The user should not need to" in text
    assert "Do not turn route selection into a fifth workflow stage." in text


def test_intake_prestage_pipeline_rules_are_present():
    triage_text = INTAKE_TRIAGE_MD.read_text(encoding="utf-8")
    verify_text = INTAKE_VERIFY_MD.read_text(encoding="utf-8")
    transmla_case_text = TRANSMLA_CASE_MD.read_text(encoding="utf-8")

    assert "DeepXiv as the preferred/default paper-intake assistant" in triage_text
    assert "intake scoring / triage rubric" in triage_text
    assert "Use `TransMLA` as the first worked example" in triage_text
    assert "`qualification_basis`" in triage_text
    assert "`source_status`" in triage_text

    assert "bounded intake pre-stage should default to one combined helper/scaffold" in verify_text
    assert "### Hard blockers" in verify_text
    assert "Allowed status values:" in verify_text
    assert "- `partial`" in verify_text

    assert "intake -> reference-code map -> bounded patch scope -> focused verification" in transmla_case_text
