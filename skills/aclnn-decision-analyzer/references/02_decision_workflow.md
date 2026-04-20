# 02 Decision Workflow

This is the only decision authority of this skill.
Use it after Step 1 evidence collection.

## Purpose

Convert raw evidence into:
- per-operator path decision (`PATH1_AUTO`, `PATH2_CUSTOMIZE`, `UNKNOWN`)
- explainable decision reasons
- actionable change-scope hints

This document defines all stage terms used below.
Field names and meanings are defined in `00_field_dictionary.md` (single source of truth).

## Required Input Fields (from Step 1)

Per operator candidate, this step consumes:
- `api`
- `op_name`
- `yaml_path`
- `dispatch_ascend`
- `api_def_ascend` (recommended)
- `config_evidence`
- `pyboost_evidence`
- `kbk_evidence`
- optional compare fields:
  - `function_consistency`
  - `parameter_definition_consistency`
  - `dtype_consistency`

## Stage D0: Decision Unit Framing

Decision unit = one mapping record (`api` + `op_name`).

If one API maps to multiple operators, each operator must have its own decision.
Do not merge multiple operators into one label.

## Stage D1: Mapping Validity Gate

For each decision unit, check required mapping fields:
- `api`
- `op_name`
- `yaml_path`
- `dispatch_ascend` (empty string is valid; missing field is invalid)

If any required field is missing:
- set label to `UNKNOWN`
- add risk flag `insufficient_mapping_info`
- stop deeper decision for that unit

Why this gate exists:
- wrong mapping causes wrong path decisions
- later evidence is meaningless without stable mapping

## Stage D2: Evidence Status Summary

Build evidence status flags:
- `has_config_evidence`: `len(config_evidence) > 0`
- `has_pyboost_evidence`: `len(pyboost_evidence) > 0`
- `has_kbk_evidence`: `len(kbk_evidence) > 0`
- `has_backend_evidence`: `has_pyboost_evidence or has_kbk_evidence`

Rules:
- empty evidence is valid and must be kept
- do not infer support from config-only evidence

## Stage D3: Path Decision (Authoritative Rule Order)

Apply rules in this exact order.

### Rule 1: PTA mismatch override (optional input)

If PTA compare exists and any of:
- `function_consistency == MISMATCH`
- `parameter_definition_consistency == MISMATCH`
- `dtype_consistency == MISMATCH`

Then:
- label = `PATH2_CUSTOMIZE`
- reason = `PTA mismatch requires customized adaptation`
- add risk flag `pta_mismatch_requires_customize_or_recheck`

### Rule 2: Dispatch rule

Else if `dispatch_ascend` is non-empty:
- label = `PATH2_CUSTOMIZE`
- reason = `op yaml has dispatch.Ascend, indicating customize path`

### Rule 3: Auto-path rule

Else if `dispatch_ascend` is empty and `api_def_ascend == pyboost`:
- label = `PATH1_AUTO`
- reason = `no dispatch.Ascend and api_def Ascend=pyboost, indicating auto path`

### Rule 4: Fallback

Else:
- label = `UNKNOWN`
- reason = `insufficient or conflicting decision signals`

### Decision notes

- `PATH1_AUTO` means candidate for auto-generated ACLNN path (no customize class required by design).
- `PATH2_CUSTOMIZE` means candidate for manual customize path (preprocess/adaptation required by design).
- This is a planning decision, not runtime proof.

## Required Explanation: Interface Five Elements

Each decision unit must include:
- `function_consistency`: `MATCH | MISMATCH | UNKNOWN`
- `parameter_definition_consistency`: `MATCH | MISMATCH | UNKNOWN`
- `dtype_consistency`: `MATCH | MISMATCH | UNKNOWN`
- `need_new_primitive`: `YES | NO | UNKNOWN`
- `interface_strategy`: short strategy string

Recommended defaulting:
- without PTA compare input, set first three fields to `UNKNOWN`
- if `op_name` exists in current MindSpore YAML, default `need_new_primitive = NO`
- if mapping is incomplete, set `need_new_primitive = UNKNOWN`
- strategy examples:
  - `REUSE_EXISTING_INTERFACE_PATH1_AUTO`
  - `REUSE_EXISTING_INTERFACE_WITH_ASCEND_DISPATCH`
  - `NEW_INTERFACE_OR_EXTEND_REQUIRED`

## Stage D4: Change-Scope Planning Hints

Generate change-scope hints by label:

### If `PATH1_AUTO`
- `yaml`: mapped `api_def` + `op_def` files
- `pyboost`: auto-generate/forward path verification targets
- `kbk`: auto kernel registration verification targets
- `bprop`: existing bprop registration check
- `tests_docs`: UT/ST/doc completion targets

### If `PATH2_CUSTOMIZE`
- `yaml`: mapped `api_def` + `op_def` files, especially `dispatch.Ascend`
- `pyboost`: `ops/kernel/ascend/aclnn/pyboost_impl/customize/<op>.{h,cc}`
- `kbk`: `ops/kernel/ascend/aclnn/kernel_mod_impl/customize/<op>_aclnn_kernel.{h,cc}`
- `bprop`: grad registration/update if needed
- `tests_docs`: UT/ST/doc completion targets

### If `UNKNOWN`
- no direct coding scope
- output missing evidence first
- provide concrete next evidence actions before any file-change recommendation

## Risk Flag Baseline

Use clear, explicit risk flags as needed:
- `insufficient_mapping_info`
- `missing_config_evidence`
- `missing_pyboost_customize_evidence`
- `missing_kbk_kernel_evidence`
- `pta_mismatch_requires_customize_or_recheck`
- `conflicting_decision_signals`

## Example (Minimal)

Input signals:
- `api = mint.add`
- `op_name = add_ext`
- `dispatch_ascend = ""`
- `api_def_ascend = pyboost`

Decision:
- label: `PATH1_AUTO`
- reason: `no dispatch.Ascend and api_def Ascend=pyboost`
- risk flags: include backend evidence gaps if customize evidence is empty
