# 03 Output Protocol

This document is the only output contract of this skill.
Use it after Step 2 decisions are complete.

## Purpose

Ensure every response is:
- traceable to repository evidence
- consistent across cases
- executable as a runbook for next development steps

Field names and meanings are defined in `00_field_dictionary.md`.

## Required Output Sections

Return all sections below in this order.

1. `scope`
2. `mapping`
3. `evidence`
4. `path_decision`
5. `risk_flags`
6. `change_scope`
7. `next_checks`

## Section Definitions

### 1) `scope`

- `user_request`: short request restatement
- `primary_apis`: APIs explicitly requested by user
- `related_apis`: automatically expanded family APIs (for example `add_`)

### 2) `mapping` (array)

Per record:
- `api`
- `op_name`
- `yaml_path`
- `dispatch_ascend`
- `api_def_ascend` (if available)
- `api_def_path` (if available)

### 3) `evidence` (array)

Per operator:
- `op_name`
- `config_evidence`: string[]
- `pyboost_evidence`: string[] (empty allowed)
- `kbk_evidence`: string[] (empty allowed)
- `other_evidence`: string[] (optional)

### 4) `path_decision` (array)

Per operator:
- `op_name`
- `label`: `PATH1_AUTO | PATH2_CUSTOMIZE | UNKNOWN`
- `reason`
- `interface_five_elements`:
  - `function_consistency`
  - `parameter_definition_consistency`
  - `dtype_consistency`
  - `need_new_primitive`
  - `interface_strategy`

### 5) `risk_flags`

Either:
- global `string[]`, or
- per-operator risk list in each decision record

Use explicit, actionable names.

### 6) `change_scope`

Always include 5 buckets:
- `yaml`
- `pyboost`
- `kbk`
- `bprop`
- `tests_docs`

### 7) `next_checks`

1-3 concrete follow-up actions.
No generic wording.

## Case Templates

### Case A: `PATH1_AUTO`

Must show:
- decision signals (`dispatch_ascend = ""`, `api_def_ascend = pyboost`)
- auto-path verification scope
- evidence-gap risks if backend evidence is currently empty

### Case B: `PATH2_CUSTOMIZE`

Must show:
- decision signal (`dispatch_ascend` non-empty, or PTA mismatch override)
- customize-target files for PyBoost and KBK
- whether existing customize evidence already exists

### Case C: `UNKNOWN`

Must show:
- exact missing/conflicting signals
- no direct implementation claim
- next evidence actions first

## Minimal Response Template

```text
[Scope]
- user_request: ...
- primary_apis: [...]
- related_apis: [...]

[Mapping]
- api: ...
  op_name: ...
  yaml_path: ...
  dispatch_ascend: ...
  api_def_ascend: ...
  api_def_path: ...

[Evidence]
- op_name: ...
  config_evidence: [...]
  pyboost_evidence: [...]
  kbk_evidence: [...]
  other_evidence: [...]

[Path Decision]
- op_name: ...
  label: PATH1_AUTO | PATH2_CUSTOMIZE | UNKNOWN
  reason: ...
  interface_five_elements:
    function_consistency: MATCH | MISMATCH | UNKNOWN
    parameter_definition_consistency: MATCH | MISMATCH | UNKNOWN
    dtype_consistency: MATCH | MISMATCH | UNKNOWN
    need_new_primitive: YES | NO | UNKNOWN
    interface_strategy: ...

[Risk Flags]
- ...

[Change Scope]
- yaml: [...]
- pyboost: [...]
- kbk: [...]
- bprop: [...]
- tests_docs: [...]

[Next Checks]
1. ...
2. ...
```

## Mandatory Blocking Checklist (Before Responding)

- [ ] Did I keep `primary_apis` and `related_apis` separate?
  - If NO -> STOP and fix scope labeling.
- [ ] Did each operator have mapping minimum fields (`api/op_name/yaml_path/dispatch_ascend`)?
  - If NO -> STOP and return `UNKNOWN` with missing mapping items.
- [ ] Did I provide concrete evidence paths (not only verbal claims)?
  - If NO -> STOP and add paths.
- [ ] Did I keep empty evidence arrays explicit (`[]`) when nothing was found?
  - If NO -> STOP and make emptiness explicit.
- [ ] Did I apply the Stage D3 rule order exactly as defined in `02_decision_workflow.md`?
  - If NO -> STOP and re-run decision logic.
- [ ] Did every decision include interface five elements?
  - If NO -> STOP and complete all five fields.
- [ ] Did I avoid claiming "already supported" without backend evidence?
  - If NO -> STOP and downgrade claim to evidence-based status.
- [ ] For `UNKNOWN`, did I list exact missing signals and next checks?
  - If NO -> STOP and add missing-signal details.

## Prohibited Patterns

- Hiding family API expansion.
- Returning a path label without evidence or reason.
- Mixing decision logic and evidence collection in one vague paragraph.
- Recommending code edits when decision is `UNKNOWN` and evidence is incomplete.
