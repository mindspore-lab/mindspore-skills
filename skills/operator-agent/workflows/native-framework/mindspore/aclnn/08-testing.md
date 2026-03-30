# Workflow 8: Testing

Reference convention: use current repository UT patterns and final requirements
from `references/verification.md`.

## Goal

Complete the C++ UT and ensure functional and dynamic-shape coverage. This workflow does not cover the creation of ST cases.

## Inputs

- **Operator implementation**: YAML / Infer / PyBoost / KBK / BPROP

## Outputs (Confirm Each Required Test Type)

> **Important**: the test types below are mandatory outputs of Step 8, and each one must be marked explicitly.

| Type | File Location | Requirement | Status |
| --- | --- | --- | --- |
| **C++ UT** | `tests/ut/cpp/ops/test_ops_{op_name}.cc` | `[MUST]` must be newly created | ✅ written / ❌ not written (explain why) |

This workflow does not generate ST cases. ST cases are handled by other tasks.

## Steps

### Step 1: C++ UT - Must Be Newly Created

Typical construction patterns:
- scalar: `ShapeVector{}` + `CreateScalar<T>(value)`
- tuple: `ShapeArray{{}}` + `ValuePtrList{...}`
- `None`: `kMetaTypeNone` + `kNone`
- unknown: `kValueAny`

Refer to existing C++ UT files for similar operators to confirm the testing macros and parameter structures.

---

## 🔒 Mandatory Check Before Marking Step 8 Complete

**Before Step 8 can be marked complete, every item below must be confirmed explicitly:**

Test deliverable checklist:

C++ UT file:
  - File path: `tests/ut/cpp/ops/test_ops_{op_name}.cc`
  - Status: ✅ newly created / ❌ not written (reason: ___)

> If the C++ UT status is ❌, you **must explain why and pause for user confirmation before continuing**.
> Silent skipping is not allowed.

## Success Criteria

- [ ] **The C++ UT file has been produced** (Infer inference covers unknown / `None` / dynamic shape)
- [ ] Covered scenarios include: dynamic shape / static shape / non-contiguous tensors / empty tensors / special values

---
