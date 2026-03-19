from pathlib import Path


SKILL_MD = Path(__file__).resolve().parents[1] / "SKILL.md"


def test_behavior_rules_require_evidence_and_validation():
    text = SKILL_MD.read_text(encoding="utf-8")
    assert "Collect evidence before diagnosis." in text
    assert "State assumptions and unknowns explicitly." in text
    assert "Every root-cause claim must include a validation check." in text
    assert "Do not treat a fix as confirmed until the user verifies it." in text
    assert "do not fabricate Factory lookups when tooling is unavailable" in text
    assert "do not claim a knowledge hit unless the signature actually matches" in text
    assert "do not auto-submit a Factory `report`" in text


def test_mindspore_scoping_and_handoff_rules_present():
    text = SKILL_MD.read_text(encoding="utf-8")
    assert "### MindSpore scoping summary" in text
    assert "`API/mode misuse`, `unsupported/missing op`, `graph compile/frontend issue`, `runtime/backend issue`, `distributed/communication issue`, or `numerical/precision symptom` when it appears as part of a runtime failure rather than standalone accuracy work" in text
    assert "state the selected layer or component" in text
    assert "cite 2-4 supporting facts from the evidence" in text
    assert "hand off to `mindspore-ops-debugger`" in text


def test_dual_stack_routes_and_output_contract_present():
    text = SKILL_MD.read_text(encoding="utf-8")
    assert "`ms`: `Platform -> Scripts -> MindSpore Framework -> Backend`" in text
    assert "`pta`: `Platform -> Scripts -> torch_npu Framework -> CANN`" in text
    assert "knowledge-hit status: `known_failure`, `operator`, or `none`" in text
    assert "12. Knowledge candidate or `report` candidate (optional, manual only)" in text


def test_b12_failure_agent_guidance_is_explicit():
    text = SKILL_MD.read_text(encoding="utf-8")
    assert "full traceback or error log" in text
    assert "Canonical facts to surface before hypotheses:" in text
    assert "search `known_failure` first" in text
    assert "consult `operator`" in text
    assert "propose 1-3 ranked hypotheses tied directly to observed evidence" in text
    assert "manual `report` candidate" in text