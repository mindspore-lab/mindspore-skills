# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A Claude Code skill (`SKILL.md`) for diagnosing and fixing MindSpore operator (ops) bugs. It's a documentation-only skill — no build system, no tests, no runtime.

## Skill Activation

The skill triggers on keywords related to MindSpore operator issues: operator bugs, kernel errors, shape inference problems, precision anomalies, ACLNN adaptation, gradient issues, PyBoost errors, etc.

## Repository Structure

```
SKILL.md                        # Skill definition, trigger conditions, and 6-step workflow
references/
  architecture.md               # Source code navigation: operator layer hierarchy and path patterns
  diagnostic_workflow.md        # Detailed 6-step diagnostic process with templates
  issue_patterns.md             # Problem classification (8 categories) and quick decision tree
  fix_patterns.md               # Fix templates by component (precision, shape, kernel, bprop, etc.)
  testing_guide.md              # OpsFactory 2.0 testing framework documentation
  aclnn_guide.md                # Ascend ACLNN operator adaptation workflow
```

## Skill Workflow

The skill guides through 6 steps:
1. **Problem Analysis** — collect error info, environment, reproduction steps
2. **Scoping (定界)** — identify which layer has the issue (API/shape/kernel/bprop/compiler/runtime)
3. **Localization (定位)** — navigate to specific source files using path patterns in `architecture.md`
4. **Fixing** — apply minimal targeted changes using patterns from `fix_patterns.md`
5. **Regression Verification** — run tests to confirm fix and no regressions
6. **Test Supplementation** — add test cases covering the bug scenario

## MindSpore Source Layout (external, referenced by skill)

Operator source lives under `mindspore/mindspore/`:
- `ops/op_def/yaml/` — operator YAML definitions
- `ops/api_def/` — API definitions
- `ops/infer/` — shape/type inference
- `ops/kernel/` — device kernels (cpu, gpu, ascend)
- `ccsrc/frontend/expander/bprop/` — backward propagation
- `python/mindspore/ops/` — Python API layer

## Editing Guidelines

- All reference docs are in Chinese — keep them in Chinese
- When updating `issue_patterns.md`, maintain the decision tree consistency with `diagnostic_workflow.md`
- `fix_patterns.md` patterns should map 1:1 to the problem categories in `issue_patterns.md`
- `SKILL.md` trigger keywords should stay aligned with the 8 problem categories
