# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A Claude Code skill (`SKILL.md`) for diagnosing and fixing torch_npu operator bugs. It's a documentation-only skill — no build system, no tests, no runtime.

## Skill Activation

The skill triggers on keywords related to torch_npu operator issues: NPU operator bugs, ACLNN adaptation errors, precision anomalies, format conversion issues, dispatch key errors, DO_COMPATIBILITY fallback, OpCommand errors, compilation failures, stream sync problems, dtype mismatches, backward propagation issues, HCCL distributed errors, ATB operator issues, version compatibility problems, import failures, triton conflicts, etc.

## Repository Structure

```
SKILL.md                        # Skill definition, trigger conditions, and 6-step workflow
references/
  architecture.md               # Source code navigation: two-layer architecture and path patterns
  issue_patterns.md             # Problem classification (12 categories) and quick decision tree
  fix_patterns.md               # Fix templates by component (ACLNN, format, dtype, registration, ATB, version, etc.)
  debugging_tools.md            # Built-in debugging tools guide (OpHook, NPU Trace, dump, overflow, ATB logs)
scripts/
  sync-to-server.sh             # Rsync script for remote deployment
```

## Skill Workflow

The skill guides through 6 steps:
1. **Problem Analysis** — collect error info, environment, reproduction steps
2. **Scoping (定界)** — identify which layer has the issue (registration/ACLNN/format/runtime)
3. **Localization (定位)** — navigate to specific source files using path patterns in `architecture.md`
4. **Fixing** — apply minimal targeted changes using patterns from `fix_patterns.md`
5. **Regression Verification** — compile and test on remote Ascend server
6. **Test Supplementation** — add test cases covering the bug scenario

## torch_npu Source Layout (external, referenced by skill)

Operator source lives in two layers:
- `torch_npu/csrc/framework/` — OpCommand, FormatHelper, OpHook (framework layer)
- `torch_npu/csrc/core/npu/` — NPUCachingAllocator, NPUStream, ACL interface (runtime layer)
- `third_party/op-plugin/op_plugin/ops/opapi/` — ACLNN operator implementations (new path)
- `third_party/op-plugin/op_plugin/ops/aclops/` — ACL operator implementations (old path)

## Editing Guidelines

- All reference docs are in Chinese — keep them in Chinese
- Code comments in all files must be in English
- When updating `issue_patterns.md`, maintain the decision tree consistency
- `fix_patterns.md` patterns should map to the problem categories in `issue_patterns.md`
- `SKILL.md` trigger keywords should stay aligned with the problem categories
