---
name: aclnn-decision-analyzer
description: Auto-invoked when users ask whether a MindSpore api/op is already supported by Ascend ACLNN, which ACLNN path it uses, or what files should be changed for ACLNN integration. Uses a strict serial pipeline:evidence fetching -> decision workflow -> output protocol.
---

# ACLNN Decision Analyzer

This skill plans ACLNN integration paths from repository evidence.

## When to Use

Use this skill when:
- The user asks whether a MindSpore API/operator is already connected to ACLNN.
- The user asks for ACLNN path decision.
- The user asks what files should be changed for ACLNN integration.

## Scope

- Tool-first evidence analysis with a strict serial workflow.
- Produces path decision and file-level change scope.
- Does not claim runtime support without backend evidence.
- Field meanings are defined in `./references/00_field_dictionary.md`; other docs reference this definition and do not redefine it.

## Decision Labels

- `PATH1_AUTO`: auto-generated ACLNN integration path.
- `PATH2_CUSTOMIZE`: manual customize ACLNN integration path.
- `UNKNOWN`: evidence is insufficient or conflicting for a final path label, but the skill still returns analyzed findings, gaps, and next checks.

## Instructions

This is a strict serial pipeline

### Step 0: Load Field Dictionary

Must read: `./references/00_field_dictionary.md`

- Load field names, meaning, and ownership once before the pipeline.

### Step 1: Collect Evidence

Must read: `./references/01_evidence_fetching.md`

- Collect raw evidence only.
- Reuse `skills/api-helper` first for API-to-operator mapping hints.
- Do not make path decisions in this step.

### Step 2: Decide Path

Must read: `./references/02_decision_workflow.md`

- Apply D0->D4 decision logic using only evidence from Step 1.
- Determine `PATH1_AUTO`, `PATH2_CUSTOMIZE`, or `UNKNOWN` per operator.
- If any rule/case is uncertain, re-check MindSpore source files before deciding.

### Step 3: Format and Validate Output

Must read: `./references/03_output_protocol.md`

- Format the result in the required schema.
- Run the mandatory blocking checklist before responding.
- If a checklist item fails, stop and return to Step 1 or Step 2.
