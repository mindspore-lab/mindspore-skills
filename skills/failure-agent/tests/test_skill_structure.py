from pathlib import Path


SKILL_MD = Path(__file__).resolve().parents[1] / "SKILL.md"


def test_skill_markers_present():
    text = SKILL_MD.read_text(encoding="utf-8")
    assert "## Golden Rules" in text
    assert "## Stage 0: Gather Context and Detect Stack" in text
    assert "## Stage 1: Scenario Intake" in text
    assert "## Stage 2: Knowledge Lookup First" in text
    assert "## Stage 3: Layered Diagnosis" in text
    assert "## Stage 4: Fix, Verify, and Report Candidate" in text
    assert "stack (`ms` or `pta`)" in text
    assert "## Factory Integration" in text


def test_recommended_use_routes_are_explicit():
    text = SKILL_MD.read_text(encoding="utf-8")
    assert "### training crash" in text
    assert "### runtime failure" in text
    assert "### HCCL / NCCL / device communication issue" in text
    assert "### missing operator / unsupported path" in text
    assert "search `known_failure` first" in text
    assert "consult `operator`" in text
    assert "manual `report` candidate" in text


def test_boundary_and_exclusions_are_present():
    text = SKILL_MD.read_text(encoding="utf-8")
    assert "pure accuracy drift with no runtime failure" in text
    assert "pure throughput, latency, or memory tuning" in text
    assert "environment bootstrapping only" in text
    assert "You MUST keep `failure-agent` at the triage and routing layer." in text
    assert "You MUST stop before source-level investigation, fix implementation, regression validation, or test authoring" in text
