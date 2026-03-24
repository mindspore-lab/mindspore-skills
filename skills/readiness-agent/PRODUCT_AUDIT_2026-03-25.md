# Readiness-Agent Product Audit

Date: 2026-03-25

## Audit Verdict

Current state:

- product-grade MVP is achieved
- the skill is credible enough for controlled local use
- the deterministic pipeline, native `env_fix`, revalidation gate, and
  readiness report are all implemented and tested

Recommended readiness level:

- ready for internal alpha / `ms-cli` integration work
- schema strategy is now closed at the skill level through a shared envelope
  plus readiness verdict payload
- not yet fully closed as a final GA skill

## What Is Already Product-Complete

- single-machine readiness certification is centered on one user-facing
  outcome: `READY` / `WARN` / `BLOCKED`
- execution target discovery is explicit and evidence-driven
- dependency closure is target-scoped, not machine-generic
- framework/runtime probing is selected-environment-aware
- framework smoke prerequisite checks exist
- explicit task smoke is supported through `task_smoke_cmd`
- blocker taxonomy is normalized and stable
- native `env_fix` planning and controlled execution exist
- revalidation is enforced from `needs_revalidation` coverage, not from a
  placeholder boolean
- final report preserves internal evidence fields and user-visible summary
- realistic workspace fixtures now cover training-style and inference-style
  workspaces

## Current Implemented Pipeline

1. `discover_execution_target.py`
2. `build_dependency_closure.py`
3. `run_task_smoke.py` when `task_smoke_cmd` exists
4. `collect_readiness_checks.py`
5. `normalize_blockers.py`
6. `plan_env_fix.py`
7. `execute_env_fix.py`
8. rerun required checks when remediation changes the environment
9. `build_readiness_report.py`

## Evidence Of Stability

- local test suite result: `31 passed, 1 warning`
- warning is only pytest cache permission noise
- helper, integration, fixture, manifest, and behavior tests all pass

## Remaining Product Gaps

### 1. `env_fix` Capability Is Still Narrow By Design

The current native remediation scope is intentionally limited to safe
user-space actions. This is correct for product safety, but it means:

- no system-layer repair
- no CANN / driver remediation
- no config mutation
- no dataset repair
- no model code patching

This is not a flaw, but it should remain explicit in integration plans.

### 2. Real Framework Smoke Is Still Minimal

Current framework smoke verifies the minimum import/bootstrap prerequisite,
not a real task-level framework execution path such as:

- one tensor on device
- one forward pass
- one train-step primitive

This is acceptable for now, but stronger task evidence would reduce false
`READY` in edge cases where import succeeds but runtime usage still fails.

### 3. No Formal `ms-cli` Adapter Contract Yet

The skill now has stable internal artifacts, but the adapter layer that maps:

- user request
- `mode`
- `target`
- artifact locations
- final surfaced summary

into `ms-cli` runtime behavior is not yet formalized in this repo.

## Recommended Final Steps

1. Add a small adapter contract for `ms-cli`.
   Define:
   - required inputs
   - output artifact paths
   - surfaced status mapping
   - handoff behavior after `env_fix`

2. Optionally strengthen runtime evidence.
   Add one more controlled smoke tier for frameworks or task kinds where
   minimal runtime execution is safe and deterministic.

## Bottom Line

`readiness-agent` is no longer just a prompt-first concept skill.
It now behaves like a real certification product with deterministic helpers,
native remediation, revalidation, stable artifacts, and realistic fixtures.

The remaining work is mostly contract closure and integration closure, not
core capability invention.
