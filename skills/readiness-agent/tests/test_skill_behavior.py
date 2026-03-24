from pathlib import Path


SKILL_MD = Path(__file__).resolve().parents[1] / "SKILL.md"


def test_behavior_rules_require_target_and_revalidation_reasoning():
    text = SKILL_MD.read_text(encoding="utf-8")
    assert "Certification is for a specific intended task" in text
    assert "Only fix what is required by the selected execution target." in text
    assert "After every successful mutation, rerun affected checks before final status." in text
    assert "final `revalidated=true` requires the" in text
    assert "current deterministic pipeline" in text
    assert "`READY` should be reserved for cases where the evidence is strong enough" in text


def test_references_and_scripts_are_declared():
    text = SKILL_MD.read_text(encoding="utf-8")
    assert "`references/product-contract.md`" in text
    assert "`references/execution-target-discovery.md`" in text
    assert "`references/blocker-taxonomy.md`" in text
    assert "`references/dependency-closure.md`" in text
    assert "`references/env-fix-policy.md`" in text
    assert "`scripts/discover_execution_target.py`" in text
    assert "`scripts/build_dependency_closure.py`" in text
    assert "`scripts/collect_readiness_checks.py`" in text
    assert "`scripts/run_task_smoke.py`" in text
    assert "`scripts/normalize_blockers.py`" in text
    assert "`scripts/plan_env_fix.py`" in text
    assert "`scripts/execute_env_fix.py`" in text
    assert "`scripts/build_readiness_report.py`" in text
