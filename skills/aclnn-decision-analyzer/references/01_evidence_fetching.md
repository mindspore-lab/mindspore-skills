# 01 Evidence Fetching

This document defines how to collect raw facts.
Do not output `PATH1_AUTO/PATH2_CUSTOMIZE/UNKNOWN` in this step.

## Goal

Build an evidence payload that is sufficient for decision logic.
Field names and meanings are defined in `00_field_dictionary.md` (single source of truth).

## Input Types

At least one of:
- frontend API (`mint.xxx` or `ops.xxx`)
- internal operator name (`xxx_ext`, `xxx`)
- ACLNN symbol (`aclnnXxx`)

## Required Fields for This Step

Populate these fields (definitions are in `00_field_dictionary.md`):
- `scope.primary_apis`
- `scope.related_apis`
- `mapping.api`
- `mapping.op_name`
- `mapping.yaml_path`
- `mapping.dispatch_ascend`
- `mapping.api_def_ascend` (if available)
- `mapping.api_def_path` (if available)
- `evidence.op_name`
- `evidence.config_evidence`
- `evidence.pyboost_evidence`
- `evidence.kbk_evidence`
- `evidence.other_evidence` (optional)
- `mapping.api_def_py_method` (optional)

## A. Reuse api-helper First (API Mapping Hint Source)

Before direct searching, reuse `skills/api-helper` for API call-chain hints.
Then normalize findings into this skill's required payload fields.

## B. Resolve API and Family Variants

Given `mint.xxx`, use token `xxx`.

```bash
# API definitions and common family variant (xxx / xxx_)
rg --files mindspore/mindspore/ops/api_def | rg '/xxx(_)?\.yaml$'

# Inspect API definition fields
rg -n "^xxx:|alias:|op_yaml:|Ascend:|py_method:" mindspore/mindspore/ops/api_def/xxx*.yaml

# Open primary API definition
sed -n '1,260p' mindspore/mindspore/ops/api_def/xxx.yaml
```

What to record:
- list of `op_yaml` entries
- `api_def_ascend` value per entry
- `api_def_py_method` per entry
- related API files (for `related_apis`)

## B2. Resolve Non-API Inputs (op_name / aclnn symbol)

If input starts from internal operator name:

```bash
# op yaml candidates
rg --files mindspore/mindspore/ops/op_def/yaml | rg '/<op>_op\.yaml$'

# back-map to api_def by op_yaml reference
rg -n "<op>_op\.yaml|op_yaml:" mindspore/mindspore/ops/api_def
```

If input starts from ACLNN symbol (for example `aclnnAdd`):

```bash
# search pyboost customize calls
rg -n "aclnnAdd" mindspore/mindspore/ops/kernel/ascend/aclnn/pyboost_impl/customize

# search kbk customize calls
rg -n "aclnnAdd" mindspore/mindspore/ops/kernel/ascend/aclnn/kernel_mod_impl/customize

# search forward traces
rg -n "aclnnAdd" mindspore/mindspore/ccsrc/pynative/forward/pyboost
```

Then back-map to:
- `op_name` from matched files
- `yaml_path` from matched op yaml
- `api` via `api_def` references

## C. Resolve op yaml and dispatch signal

```bash
# Open mapped op yaml
sed -n '1,260p' mindspore/mindspore/ops/op_def/yaml/<op>_op.yaml

# Fast scan for dispatch block
rg -n "dispatch:|Ascend:" mindspore/mindspore/ops/op_def/yaml/<op>_op.yaml
```

What to record:
- `op_name` (top-level key)
- `yaml_path`
- `dispatch_ascend` (empty is valid)

## D. Backend Evidence Search

### PyBoost customize evidence

```bash
rg -n "LAUNCH_ACLNN|CREATE_PYBOOST_OP|aclnn|<OpName>" \
  mindspore/mindspore/ops/kernel/ascend/aclnn/pyboost_impl/customize
```

### KBK customize evidence

```bash
rg -n "AclnnKernelMod|DEFINE_GET_WORKSPACE_FOR_OPS|aclnn|<OpName>" \
  mindspore/mindspore/ops/kernel/ascend/aclnn/kernel_mod_impl/customize
```

### Optional forward traces

```bash
rg -n "aclnn|<OpName>" mindspore/mindspore/ccsrc/pynative/forward/pyboost
```

## E. Optional Coverage Checks

```bash
# bprop
rg -n "REG_BPROP_BUILDER\(\"<OpName>|<OpName>Grad|Emit\(" \
  mindspore/mindspore/ccsrc/frontend/expander/grad

# tests/docs
rg -n "<op_name>|<api_name>" mindspore/tests mindspore/mindspore/ops/op_def/yaml/doc
```

## F. Anti-Traps (Mandatory)

### Trap 1: guessing operator name from API text
- ❌ Wrong: infer `mint.acos` -> `ACos` directly.
- ✅ Correct: confirm import and `api_def`/`op_yaml` chain first.

### Trap 2: ignoring family APIs
- ❌ Wrong: only inspect `mint.add` and skip `mint.add_`.
- ✅ Correct: check same-root family and label as `related_apis`.

### Trap 3: treating empty dispatch as missing data
- ❌ Wrong: `dispatch_ascend == ""` means parsing failed.
- ✅ Correct: empty dispatch is a valid signal for later decision.

### Trap 4: claiming support without backend files
- ❌ Wrong: "already integrated" based only on API yaml.
- ✅ Correct: keep backend evidence explicitly empty if not found.

## G. Exit Criteria for Step 1

Step 1 is complete only when:
- every operator candidate has all required mapping fields
- `config_evidence` is non-empty
- `pyboost_evidence` and `kbk_evidence` are explicitly set (empty or non-empty)
- `primary_apis` and `related_apis` are clearly separated
