# Setup-Agent Rework Trace

Date: 2026-03-20

## Purpose

This document records the intent, scope, decisions, and implementation shape of
the `setup-agent` rework completed on 2026-03-20. It is meant to preserve the
reasoning trail so future optimization work can start from the full context
instead of rediscovering why the current structure exists.

## Maintenance Rule

From this point forward, every `setup-agent` change should update this trace.

Each update entry must record:

1. The date of the change
2. The reason or trigger for the change
3. Every file that was added, updated, or deleted
4. A short description of what each changed file is responsible for
5. A short summary of what changed in that file

This rule exists so future refactors can reconstruct the full scope quickly,
instead of relying on scattered commit history or chat context.

## Why This Rework Happened

The original `setup-agent` had several structural problems:

1. It mixed Ascend/NPU and Nvidia/GPU setup in one skill, which made routing
   ambiguous and diluted the main use case.
2. It only modeled `MindSpore + CANN` on Ascend and did not explicitly cover
   the common `PyTorch + torch_npu + CANN` path.
3. It implied system-layer auto-fix behavior for driver and toolkit problems,
   which is too risky and not aligned with the actual safe automation boundary.
4. It did not treat `uv` as the primary Python environment entry point.
5. It was not exposed from the repository root `AGENTS.md`, so it was difficult
   for an agent to trigger reliably.
6. The entry `SKILL.md` was too document-heavy for an execution prompt, making
   it harder for a model to lock onto the main decision path.
7. The tests only checked for file existence and some table shape, not the key
   workflow rules that determine whether the skill behaves correctly.

## Rework Goals

The rework targeted these goals:

1. Make `setup-agent` Ascend-only.
2. Support both framework paths on Ascend:
   - `MindSpore + CANN`
   - `PyTorch + torch_npu + CANN`
3. Limit automation to user-space actions:
   - install user-level `uv`
   - install Python packages inside a user-confirmed `uv` environment
4. Never auto-install:
   - NPU driver
   - firmware
   - CANN toolkit
   - system Python
5. Force the workflow to validate the system layer before doing Python work.
6. Require explicit user confirmation when reusing or creating a `uv`
   environment.
7. Produce outputs consistent with the repository reporting contract.
8. Separate the high-density execution prompt from the longer reference data.

## Files Changed

### Repository routing

- `AGENTS.md`

Changes:
- Added `setup-agent` to the repository skill list
- Added trigger hints for environment setup scenarios
- Clarified boundaries between:
  - `setup-agent`
  - `failure-agent`
  - `performance-agent`

Reason:
- Without this, the skill existed but was not reliably discoverable from the
  repo-level routing rules.

### Skill manifest

- `skills/setup-agent/skill.yaml`

Changes:
- Scope changed from mixed Ascend/GPU wording to local Ascend-only wording
- Added inputs for:
  - `frameworks`
  - `task_type`
  - `uv_env_mode`
  - `python_version`
- Restricted `target` to `local`
- Removed unrelated `composes`
- Changed permissions to require:
  - network
  - workspace write
- Updated tags from generic GPU-oriented tags to Ascend/`torch_npu`/`uv`

Reason:
- The manifest now reflects the actual supported behavior and no longer claims
  a cross-platform scope that the workflow does not support.

### Skill entry prompt

- `skills/setup-agent/SKILL.md`

Changes:
- Rewritten from a long mixed guide into a compact execution-oriented prompt
- Kept only the parts that should steer model behavior directly:
  - scope
  - hard rules
  - workflow
  - compatibility source
  - reporting requirements
  - out-of-scope list
- Explicitly preserved these decision points:
  - system layer must be healthy before Python work
  - `uv` must exist before Python package work
  - existing `uv` environments must be confirmed with the user
  - new `uv` environments require a user-confirmed Python version
  - Python packages must never be installed into the system interpreter

Reason:
- The prompt is now shaped for model execution rather than human reading.
- Long explanations and lookup-heavy detail were moved to `references/`.

### Reference data

- `skills/setup-agent/references/ascend-compat.md`
- Deleted: `skills/setup-agent/references/nvidia-compat.md`

Changes:
- Removed the Nvidia reference because GPU support was intentionally removed
- Restructured the Ascend reference into a retrieval-friendly format:
  - `Quick Use`
  - `Driver / Firmware / CANN Matrix`
  - `MindSpore on Ascend`
  - `PyTorch + torch_npu on Ascend`
  - `Detection Hints`
  - `Official Installation Guides`
  - `Repair Policy`

Reason:
- `SKILL.md` should drive decisions; `references/` should answer lookup
  questions quickly.
- The new structure helps future models find the exact section they need with
  less prompt noise.

### Tests

- `skills/setup-agent/tests/test_references.py`

Changes:
- Replaced weak existence-only tests with behavior-contract checks
- Added coverage for:
  - Ascend-only scope
  - no Nvidia reference usage
  - `uv` as a required gate before Python installs
  - prohibition on driver/CANN auto-install
  - mandatory user confirmation for `uv` environment choice
  - stop behavior when the system layer fails
  - both framework paths being represented
  - standard reporting contract references
  - root `AGENTS.md` exposure

Reason:
- The goal was to guard workflow intent, not just file presence.

## Core Design Decisions

### 1. Ascend-only was a deliberate narrowing, not an omission

The skill originally tried to do too much. Restricting it to Ascend improves:
- routing precision
- prompt clarity
- compatibility reasoning
- maintenance cost

GPU/Nvidia setup should not remain half-supported inside the same skill.

### 2. The system layer is authoritative

The workflow now enforces this order:

1. OS and device visibility
2. driver / firmware / CANN
3. `set_env.sh` sourcing
4. `uv`
5. framework packages
6. model runtime dependencies

This order prevents a common failure mode where Python packages are installed
into an environment that cannot use the NPU anyway.

### 3. `uv` is the Python environment control plane

The rework assumes that model runtime setup should happen inside `uv`, not
through ad hoc global `pip install` commands. This improves:
- reproducibility
- isolation
- lower risk of polluting the host interpreter

### 4. User confirmation is required at the environment boundary

The skill is allowed to automate Python package repair, but not to silently
choose a `uv` environment when multiple plausible environments exist.

This was intentionally kept interactive because:
- the wrong environment is a common source of confusion
- silent selection would create hard-to-debug state divergence

### 5. Reporting contract matters

The skill now explicitly points to the standard output layout and report files.
This matters because later automation or orchestration layers need stable
artifacts, not just console prose.

## What Was Intentionally Not Changed

These items were deliberately left out of scope:

1. Remote SSH support
2. Nvidia/GPU setup support
3. System-level auto-install for driver or CANN
4. Other skills' docs or workflows
5. Repository-wide skill schema updates unrelated to `setup-agent`

This trace is intentionally scoped to `setup-agent` and the root routing entry
needed for `setup-agent` to be discoverable.

## Known Follow-Up Opportunities

These were identified but not implemented in this rework:

1. Add a second reference focused on command recipes, so the main reference can
   stay more decision-oriented.
2. Add richer test coverage around reference structure, for example enforcing
   that `Quick Use`, `Detection Hints`, and `Repair Policy` remain present.
3. Add scripted helpers in the future if the skill evolves from prompt-driven
   workflow to a more tool-backed workflow.
4. Improve version matrices over time as the official compatibility data changes.

## Post-Rework Follow-Up: Streaming Output and Mailbox Summary

After the initial rework, an additional UX requirement was identified:

1. The skill should print progress as a live stream while checks are running.
2. The skill should always finish with a mailbox-style summary, regardless of
   whether the run succeeds or fails.

Why this matters:
- Users need immediate visibility into what the skill is checking right now.
- Early failure is easier to understand when the failed step is printed in-line.
- A final compact summary is better for operator handoff than a long narrative.

What this requirement changed conceptually:
- `SKILL.md` is no longer only a decision workflow; it also defines console UX.
- The console stream now becomes part of the skill contract, not an optional
  presentation detail.
- The final summary must always answer five questions:
  - what is already installed
  - what is missing
  - what was skipped
  - why the run failed, if it failed
  - what should happen next

What should remain true in future edits:
- do not replace the stream with only an end-of-run report
- do not omit the final summary on early failure
- keep per-step lines short and status-first
- keep the final summary compact and operator-readable

## Change Log Updates

### 2026-03-21 - Flow control and install guidance refinement

Trigger:
- The workflow needed to stop earlier when no NPU card is present.
- Missing Ascend driver, framework, or toolkit needed a stronger and more
  explicit redirect to the Huawei Ascend CANN download portal.

Files changed:

1. `skills/setup-agent/SKILL.md`
   - Description: Main execution contract for the skill
   - Change:
     - Added the rule that no NPU card means immediate stop at device
       visibility
     - Added the rule that later driver and CANN checks are skipped in that
       case
     - Added explicit user guidance to `https://www.hiascend.com/cann/download`
       when Ascend driver, framework, or toolkit is missing

2. `skills/setup-agent/references/ascend-compat.md`
   - Description: Compatibility and official installation guidance reference
   - Change:
     - Added the no-NPU early stop rule
     - Added the Ascend CANN download portal as the primary manual recovery
       link

3. `skills/setup-agent/tests/test_references.py`
   - Description: Behavior-contract tests for the skill prompt and references
   - Change:
     - Added tests for no-NPU short-circuit behavior
     - Added tests for the required `https://www.hiascend.com/cann/download`
       guidance

### 2026-03-21 - Python checks moved behind uv and system readiness

Trigger:
- Python environment checks were happening too early in the mental model of the
  workflow.
- The desired contract is: finish NPU-related checks first, then enter `uv`,
  then inspect Python.

Files changed:

1. `skills/setup-agent/SKILL.md`
   - Description: Main execution contract for the skill
   - Change:
     - Removed `python3 --version` from system baseline
     - Moved Python runtime checks to the post-`uv` stage
     - Clarified that Python readiness must not be reported before the system
       layer is healthy and the workflow has entered `uv`

2. `skills/setup-agent/tests/test_references.py`
   - Description: Behavior-contract tests for the skill prompt and references
   - Change:
     - Added a test that Python checks only happen after entering `uv`
     - Added a regression guard to prevent `python3 --version` from returning
       to the system baseline stage

### 2026-03-21 - Commercial-run readiness refactor

Trigger:
- The prompt had accumulated too many responsibilities in one file and was
  becoming harder to maintain as a commercial-grade skill entry.
- The target shape is a compact execution prompt with reference files owning
  compatibility and runtime-output detail.

Files changed:

1. `skills/setup-agent/SKILL.md`
   - Description: Main execution prompt used by the model
   - Change:
     - Refactored into a shorter, execution-oriented contract
     - Kept scope, non-negotiables, workflow, and stop conditions
     - Moved console/reporting detail out to a dedicated reference

2. `skills/setup-agent/references/execution-contract.md`
   - Description: Runtime UX and reporting contract for streaming output,
     report artifacts, and final mailbox summary
   - Change:
     - Added as a new reference file
     - Centralized streaming console output rules
     - Centralized report artifact requirements
     - Centralized final mailbox summary contract

3. `skills/setup-agent/references/ascend-compat.md`
   - Description: Compatibility and official installation guidance reference
   - Change:
     - Retained as the compatibility/install reference after the prompt split
     - Continued to own device/toolkit/framework compatibility rules

4. `skills/setup-agent/tests/test_references.py`
   - Description: Behavior-contract tests for the skill prompt and references
   - Change:
     - Updated to require `references/execution-contract.md`
     - Updated reporting and streaming assertions to target the new reference
     - Relaxed old title-specific assertions to match the refactored structure

### 2026-03-21 - Runtime dependency policy tightened

Trigger:
- `datasets` and `diffusers` should no longer be treated as optional checks.
- The target runtime environment should install and verify the full common model
  dependency set by default.

Files changed:

1. `skills/setup-agent/SKILL.md`
   - Description: Main execution prompt used by the model
   - Change:
     - Updated runtime dependency policy so `datasets` and `diffusers` are
       standard checks alongside `transformers`, `tokenizers`, `accelerate`,
       and `safetensors`

2. `skills/setup-agent/references/execution-contract.md`
   - Description: Runtime UX and reporting contract
   - Change:
     - Expanded the required runtime dependency report list to explicitly
       include `datasets` and `diffusers`

3. `skills/setup-agent/tests/test_references.py`
   - Description: Behavior-contract tests for the skill prompt and references
   - Change:
     - Added regression checks to ensure `datasets` and `diffusers` remain
       standard runtime checks rather than optional branches

### 2026-03-21 - Work dir and artifact validation upgrade

Trigger:
- The current shell path needed to become an explicit default work dir.
- The skill needed a post-dependency workspace validation step for training
  scripts and checkpoint files.

Files changed:

1. `skills/setup-agent/SKILL.md`
   - Description: Main execution prompt used by the model
   - Change:
     - Added the current `pwd` path as the default work dir
     - Added a dedicated work dir phase before `uv` discovery
     - Added a new workdir artifact check phase after runtime dependency checks
     - Defined training script detection with `.py`
     - Defined checkpoint detection with `.ckpt`, `.pt`, `.pth`, `.bin`,
       `.safetensors`
     - Added generic Hugging Face guidance when script or checkpoint artifacts
       are missing from the current work dir

2. `skills/setup-agent/references/execution-contract.md`
   - Description: Runtime UX and reporting contract
   - Change:
     - Added work dir to streaming output requirements
     - Added training script and checkpoint file checks to streaming output
     - Added current work dir and workdir artifact findings to required report
       content
     - Added current work dir and Hugging Face guidance to the final mailbox
       summary contract

3. `skills/setup-agent/tests/test_references.py`
   - Description: Behavior-contract tests for the skill prompt and references
   - Change:
     - Added tests that require `pwd`-based work dir capture
     - Added tests for the new workdir artifact check phase
     - Added tests for supported checkpoint suffixes
     - Added tests requiring Hugging Face guidance when training scripts or
       checkpoints are missing

### 2026-03-21 - Artifact path visibility refinement

Trigger:
- The artifact-check stage needed to expose the exact matched training script
  and checkpoint paths, not just pass/fail states.
- The final mailbox summary also needed to preserve those paths for operator
  handoff.

Files changed:

1. `skills/setup-agent/SKILL.md`
   - Description: Main execution prompt used by the model
   - Change:
     - Added the requirement to print and record matched training script paths
       and checkpoint paths during the workdir artifact phase

2. `skills/setup-agent/references/execution-contract.md`
   - Description: Runtime UX and reporting contract
   - Change:
     - Added streaming examples that print concrete training script and
       checkpoint paths
     - Added matched artifact paths to required report content
     - Added matched artifact paths to the final mailbox summary contract

3. `skills/setup-agent/tests/test_references.py`
   - Description: Behavior-contract tests for the skill prompt and references
   - Change:
     - Added assertions for path-level streaming output and mailbox summary
       content

### 2026-03-21 - uv resolution and model-first workspace workflow

Trigger:
- The `uv` branch was still too weak because it allowed "installed but not
  directly callable" outcomes.
- The workspace phase only looked for scripts and checkpoint files and did not
  support the required "reuse local model directory first, otherwise download
  the requested Hugging Face model" behavior.
- `task_type` existed in the manifest but was not shaping runtime or artifact
  decisions.

Files changed:

1. `skills/setup-agent/SKILL.md`
   - Description: Main execution prompt used by the model
   - Change:
     - Tightened the `uv` gate so health now requires both `command -v uv` and
       `uv --version`
     - Added PATH-persistence remediation with explicit user confirmation
     - Reworked the workspace phase into a model-first flow:
       - discover local model directories first
       - ask the user to choose a local model when candidates exist
       - otherwise ask which Hugging Face model to download
       - use `huggingface_hub.snapshot_download` in the selected `uv`
         environment
     - Added `task_type`-specific artifact classification for training and
       inference cases

2. `skills/setup-agent/references/execution-contract.md`
   - Description: Runtime UX and reporting contract
   - Change:
     - Added streamed steps for local model discovery, model selection, and
       Hugging Face download
     - Added report fields for direct `uv` resolution, PATH update action,
       selected model path, model source, and download/auth failures
     - Added selected-model fields to the final mailbox summary example

3. `skills/setup-agent/references/ascend-compat.md`
   - Description: Compatibility lookup and repair policy
   - Change:
     - Tightened the Python-layer ready signals to require direct `uv`
       resolution from the shell
     - Updated allowed automation to include user-confirmed PATH persistence
     - Added a stop condition when `uv` still cannot be resolved after install
       and PATH remediation

4. `skills/setup-agent/skill.yaml`
   - Description: Skill manifest
   - Change:
     - Updated the skill description to mention direct `uv` resolution and
       model workspace reuse/download
     - Added optional `model_id` and `model_root` inputs
     - Bumped the version to `0.4.0`

5. `skills/setup-agent/tests/test_references.py`
   - Description: Behavior-contract tests for the skill prompt and references
   - Change:
     - Replaced the old generic artifact tests with checks for:
       - direct `uv` resolution after install
       - model-first workspace selection
       - `snapshot_download` usage
       - `task_type`-specific artifact classification
       - selected-model and download/auth reporting requirements

## Validation Performed

The rework was validated with:

```bash
pytest -q skills/setup-agent/tests
```

Result at completion:
- `14 passed`

## Practical Guidance For Future Editors

When updating `setup-agent` later, keep these boundaries unless there is a
deliberate redesign:

1. Keep `SKILL.md` compact and execution-oriented.
2. Move lookup-heavy detail into `references/`.
3. Do not reintroduce GPU scope into this skill.
4. Do not allow system-layer auto-install unless the risk model changes
   explicitly.
5. Preserve `uv` as the environment selection boundary.
6. If adding new behavior, add tests that encode the workflow rule, not just
   file existence.
