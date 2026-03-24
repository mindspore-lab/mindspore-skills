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

### 2026-03-21 - Minimal system checks and final env_summary only

Trigger:
- The baseline system command list still included `npu-smi info -t board`, but
  that extra board query was not required for the setup decision.
- The runtime contract was still asking for intermediate log artifacts even
  though environment setup should only surface the final installation state in
  the summary.

Files changed:

1. `skills/setup-agent/SKILL.md`
   - Description: Main execution prompt used by the model
   - Change:
     - Removed `npu-smi info -t board` from the baseline system checks
     - Added an explicit rule that environment checking should not maintain
       step-by-step run logs
     - Replaced the old final-summary wording with final `env_summary`

2. `skills/setup-agent/references/execution-contract.md`
   - Description: Runtime UX and reporting contract
   - Change:
     - Replaced `logs/run.log` and `logs/verify.log` with `env_summary.md`
     - Clarified that successful installs or repairs should be reflected only
       in the final `env_summary`
     - Renamed the final console summary contract to final `env_summary`

3. `skills/setup-agent/tests/test_references.py`
   - Description: Behavior-contract tests for the skill prompt and references
   - Change:
     - Added assertions that `npu-smi info -t board` is no longer used
     - Added assertions that intermediate run logs are no longer required
     - Updated summary assertions from final summary to final `env_summary`

### 2026-03-21 - Fixed boxed mailbox summary format

Trigger:
- The final summary still had too much formatting freedom, which made it easy
  for the model to drift in layout and field naming.
- The desired output is a short, aligned, boxed mailbox card with a single
  example the model can copy.

Files changed:

1. `skills/setup-agent/SKILL.md`
   - Description: Main execution prompt used by the model
   - Change:
     - Changed the final-output instruction to require the fixed mailbox
       example from the execution contract

2. `skills/setup-agent/references/execution-contract.md`
   - Description: Runtime UX and reporting contract
   - Change:
     - Replaced the free-form final summary contract with a fixed boxed mailbox
       summary contract
     - Defined exact field order, aligned labels, title shape, and one-line
       field rules
     - Added a canonical aligned example for the model to follow

3. `skills/setup-agent/tests/test_references.py`
   - Description: Behavior-contract tests for the skill prompt and references
   - Change:
     - Replaced the old summary-shape assertions with checks for the new boxed
       mailbox format, fixed field order, and aligned labels

### 2026-03-21 - Console-only output and stricter training-script detection

Trigger:
- The skill was still requiring `runs/<run_id>/out/` artifacts even though the
  desired behavior is console-only output.
- Training-script detection was too loose because any `.py` file could be
  treated as a training script.

Files changed:

1. `skills/setup-agent/SKILL.md`
   - Description: Main execution prompt used by the model
   - Change:
     - Removed the remaining `runs/<run_id>/out/` output requirement
     - Changed final output to console-only mailbox summary
     - Replaced the generic `*.py` scan with candidate training entry-script
       detection rules
     - Split script discovery and checkpoint discovery into separate roots

2. `skills/setup-agent/references/execution-contract.md`
   - Description: Runtime UX and reporting contract
   - Change:
     - Replaced artifact-writing requirements with a console-only contract
     - Removed `env_summary.md` and other result-file requirements
     - Clarified that mid-run repairs should appear only in the final mailbox
       summary

3. `skills/setup-agent/skill.yaml`
   - Description: Skill manifest
   - Change:
     - Removed the `outputs` block so the manifest no longer advertises
       `runs/<run_id>/out/` artifacts

4. `skills/setup-agent/tests/test_references.py`
   - Description: Behavior-contract tests for the skill prompt and references
   - Change:
     - Replaced artifact-output assertions with console-only assertions
     - Added checks for candidate training entry-script matching and the
       removal of the old broad `*.py` rule

### 2026-03-23 - Hugging Face mirror fallback for China network paths

Trigger:
- Direct Hugging Face model downloads may fail in mainland China because of
  DNS, timeout, or proxy reachability problems.
- The setup flow needed an explicit, repeatable mirror fallback instead of
  leaving download failures as generic network errors.

Files changed:

1. `skills/setup-agent/SKILL.md`
   - Description: Main execution prompt used by the model
   - Change:
     - Added a retry rule that switches model download to
       `HF_ENDPOINT=https://hf-mirror.com` when direct Hugging Face access fails
       because of network reachability problems
     - Added a mirror retry command example
     - Clarified that auth and permission failures should not trigger the
       mirror fallback

2. `skills/setup-agent/references/execution-contract.md`
   - Description: Runtime UX and reporting contract
   - Change:
     - Added explicit China mirror fallback guidance to the console contract

3. `skills/setup-agent/skill.yaml`
   - Description: Skill manifest
   - Change:
     - Added the optional `hf_endpoint` input with the public Hugging Face
       endpoint as the default value

4. `skills/setup-agent/tests/test_references.py`
   - Description: Behavior-contract tests for the skill prompt and references
   - Change:
     - Added assertions for the mirror fallback rule, `HF_ENDPOINT` example,
       and `hf_endpoint` manifest input

## Validation Performed

The rework was validated with:

```bash
pytest -q skills/setup-agent/tests
```

Result at completion:
- `14 passed`

## Change Log Updates

### 2026-03-24 - CANN-matched framework validation and remediation

Trigger:
- The framework layer needed to be driven by the detected CANN version instead
  of only broad framework-version buckets.
- PTA compatibility needed to stay local-first while allowing remote fallback
  to the upstream `Ascend/pytorch` README for unknown tuples.
- Installed incompatible `mindspore` or PTA packages needed an explicit
  replacement path rather than only pass/fail classification.

Files changed:

1. `skills/setup-agent/SKILL.md`
   - Description: Main execution prompt used by the model
   - Change:
     - Added CANN-first framework resolution flow in Gate 5
     - Added confirmation policy for framework replacement
     - Clarified that compatible replacement versions should be derived before
       replacing installed framework packages

2. `skills/setup-agent/references/ascend-compat.md`
   - Description: Compatibility lookup and repair policy
   - Change:
     - Added `Compatibility Source Policy`
     - Added `Framework Package Remediation Policy`
     - Reworked PTA compatibility from coarse `2.x` rows to exact CANN-keyed
       tuples
     - Added local PTA rows for `2.6.0`, `2.7.1`, `2.8.0`, and `2.9.0`
     - Restricted CANN download guidance to missing driver or toolkit, rather
       than missing framework packages

3. `skills/setup-agent/references/execution-contract.md`
   - Description: Runtime UX and reporting contract
   - Change:
     - Added reporting requirements for detected CANN, framework compatibility
       reasoning, recommended replacement versions, and confirmed replacement
       actions

4. `skills/setup-agent/skill.yaml`
   - Description: Skill manifest
   - Change:
     - Bumped the skill version as the framework compatibility contract
       expanded

5. `skills/setup-agent/tests/test_references.py`
   - Description: Behavior-contract tests for the skill prompt and references
   - Change:
     - Added regression checks for exact PTA compatibility rows, replacement
       guidance, and CANN-first framework resolution

### 2026-03-24 - Direct uv install path for missing frameworks and dependencies

Trigger:
- Missing `mindspore`, `torch`, or `torch_npu` should no longer redirect users
  to the CANN download portal for framework installation.
- When framework import or smoke checks fail because of missing Python
  packages, the skill should repair that state directly inside the selected
  `uv` environment when the missing package can be identified safely.

Files changed:

1. `skills/setup-agent/SKILL.md`
   - Description: Main execution prompt used by the model
   - Change:
     - Changed missing-framework handling to direct `pip install` inside the
       selected `uv` environment
     - Added direct dependency remediation for framework imports and smoke
       tests
     - Clarified that package names must not be guessed when the import error
       is ambiguous

2. `skills/setup-agent/references/execution-contract.md`
   - Description: Runtime UX and reporting contract
   - Change:
     - Added required reporting for direct `pip install` remediation inside the
       selected `uv` environment
     - Added explicit reporting for Python packages installed to recover
       failed framework imports or smoke tests

3. `skills/setup-agent/tests/test_references.py`
   - Description: Behavior-contract tests for the skill prompt and references
   - Change:
     - Added assertions for direct framework installation in `uv`
     - Added assertions for missing dependency remediation and the
       non-guessing rule for ambiguous import failures

### 2026-03-24 - Skill structure refactor to thin SKILL plus references and helper script

Trigger:
- `SKILL.md` had grown too large and was mixing orchestration, lookup-heavy
  rules, and command details in one file.
- The target shape for commercial-grade maintenance was a thin execution prompt
  with deeper references and a deterministic helper for PTA compatibility
  lookup.

Files changed:

1. `skills/setup-agent/SKILL.md`
   - Description: Main execution prompt used by the model
   - Change:
     - Reduced the prompt from 491 lines to about 310 lines
     - Kept scope, confirmation policy, gate ordering, and stop conditions in
       the prompt
     - Moved Gate 5 and Gate 7 detail out to dedicated references
     - Added explicit navigation to the new references and helper script

2. `skills/setup-agent/references/framework-remediation.md`
   - Description: New framework-layer execution reference
   - Change:
     - Added as the owner of framework install, replacement, PTA fallback,
       dependency remediation, and runtime dependency checks

3. `skills/setup-agent/references/workspace-discovery.md`
   - Description: New workspace-layer execution reference
   - Change:
     - Added as the owner of local model discovery, Hugging Face download,
       training script discovery, and checkpoint discovery rules

4. `skills/setup-agent/scripts/pta_compat_lookup.py`
   - Description: Deterministic PTA compatibility lookup helper
   - Change:
     - Added local-table parsing
     - Added optional remote fallback parsing against the upstream
       `Ascend/pytorch` README
     - Added normalized query handling for `torch`, `torch_npu`, and Python
       version filtering

5. `skills/setup-agent/tests/test_references.py`
   - Description: Behavior-contract tests for the skill prompt and references
   - Change:
     - Reoriented assertions so `SKILL.md` is tested as an orchestrator rather
       than the owner of every detailed rule
     - Added structural tests for the new references and helper script
     - Added a size guard to keep `SKILL.md` compact after the refactor

6. `skills/setup-agent/skill.yaml`
   - Description: Skill manifest
   - Change:
     - Bumped the version to `0.6.0` after the structural refactor

7. `skills/setup-agent/doc/rework-trace.md`
   - Description: Long-lived trace for the skill's rework history
   - Change:
     - Renamed from `rework-trace-2026-03-20.md` to `rework-trace.md`
     - Updated to include all 2026-03-24 changes in the running trace

### 2026-03-24 - MindSpore local compatibility policy switched to exact CANN-keyed rows plus official versions confirmation

Trigger:
- The MindSpore compatibility path needed a newer local table covering the
  officially confirmed `2.7.0-rc1` through `2.8.0` rows.
- The previous MindSpore local table structure no longer matched the intended
  source policy once the newer official `versions` pages became the reference
  for unknown tuples.
- After the table was updated, the rest of the skill documentation and tests
  needed to be brought back into alignment with the new exact CANN-keyed
  MindSpore table.

Files changed:

1. `skills/setup-agent/SKILL.md`
   - Description: Main execution prompt used by the model
   - Change:
     - Kept MindSpore compatibility local-first
     - Clarified that unknown local MindSpore tuples should be checked against
       the official `https://www.mindspore.cn/versions` page
     - Clarified that official `versions` lookup is user-confirmed reference
       data, not silent auto-remediation input

2. `skills/setup-agent/references/ascend-compat.md`
   - Description: Compatibility lookup and repair policy
   - Change:
     - Replaced the older MindSpore table shape with an exact
       `CANN | MindSpore | Python | Typical Use` local table
     - Added local MindSpore rows for:
       - `8.5.0 -> 2.8.0`
       - `8.5.0 -> 2.7.2`
       - `8.3.RC1 -> 2.8.0`
       - `8.3.RC1 -> 2.7.1`
       - `8.2.RC1 -> 2.7.0`
       - `8.2.RC1 -> 2.7.0-rc1`
     - Changed MindSpore validation and decision rules to use exact
       CANN-keyed local row matching instead of the old
       `Recommended CANN / Minimum CANN / Recommended Replacement` structure
     - Added official `versions` anchors as the verification source for the
       newer MindSpore rows

3. `skills/setup-agent/references/framework-remediation.md`
   - Description: Framework-layer execution reference
   - Change:
     - Updated MindSpore missing-package and installed-package handling to
       resolve compatibility from the exact local CANN-keyed MindSpore table
     - Kept unknown MindSpore tuple handling on the official `versions` page
       plus user confirmation
     - Preserved `WARN` behavior when official confirmation is still
       incomplete

4. `skills/setup-agent/tests/test_references.py`
   - Description: Behavior-contract tests for the skill prompt and references
   - Change:
     - Replaced the older MindSpore table-shape assertions with checks for the
       new exact CANN-keyed local table
     - Added regression checks that the local-first plus official `versions`
       lookup policy remains documented
     - Added coverage for the new `2.7.0-rc1` through `2.8.0` MindSpore rows

5. `skills/setup-agent/doc/rework-trace.md`
   - Description: Long-lived trace for the skill's rework history
   - Change:
     - Added this entry so the MindSpore compatibility policy and table change
       remains reconstructable without relying on commit history alone

### 2026-03-24 - Driver and firmware matrix expanded through CANN 8.5.0

Trigger:
- The system-layer matrix needed to cover the newer CANN lines already used by
  the MindSpore and PTA compatibility sections.
- The requested scope for the system-layer matrix was `8.0.RC1` through
  `8.5.0`, rather than stopping at `8.1.RC1`.

Files changed:

1. `skills/setup-agent/references/ascend-compat.md`
   - Description: Compatibility lookup and repair policy
   - Change:
     - Added `8.2.RC1`, `8.3.RC1`, and `8.5.0` rows to the
       `Driver / Firmware / CANN Matrix`
     - For newer 25.x lines, documented firmware requirements as "use the
       firmware from the paired Ascend HDK release" because the public source
       is published as an HDK pairing rather than one central firmware minimum
       table
     - Added official verification links for the newer CANN and HDK pairing
       sources

2. `skills/setup-agent/tests/test_references.py`
   - Description: Behavior-contract tests for the skill prompt and references
   - Change:
     - Added a regression test that the system-layer matrix now covers
       `8.0.RC1` through `8.5.0`

3. `skills/setup-agent/doc/rework-trace.md`
   - Description: Long-lived trace for the skill's rework history
   - Change:
     - Added this matrix-expansion entry

### 2026-03-24 - Legacy system-layer rows removed and uv-scoped command contract tightened

Trigger:
- The system-layer matrix should now stop at `8.0.RC1` on the low end rather
  than continuing to legacy `7.3.0` and `7.1.0` CANN rows.
- The execution examples still used bare `python` and `pip install`, which was
  inconsistent with the skill rule that Python work must stay inside the
  user-confirmed `uv` environment.
- The execution-contract examples still used lowercase `passed` and `failed`
  even though the allowed status set was uppercase `PASS` / `FAIL` / `WARN` /
  `SKIP` / `INFO`.

Files changed:

1. `skills/setup-agent/references/ascend-compat.md`
   - Description: Compatibility lookup and repair policy
   - Change:
     - Removed legacy `7.3.0` and `7.1.0` rows from the
       `Driver / Firmware / CANN Matrix`
     - Kept the matrix floor at `8.0.RC1` while preserving coverage through
       `8.5.0`

2. `skills/setup-agent/SKILL.md`
   - Description: Main execution prompt used by the model
   - Change:
     - Replaced bare `python` framework-check examples with
       `uv run --python <selected_python> python ...`
     - Replaced the bare PTA helper invocation with a `uv run`-scoped form

3. `skills/setup-agent/references/framework-remediation.md`
   - Description: Framework-layer execution reference
   - Change:
     - Replaced bare `python` examples with `uv run --python <selected_python>`
     - Replaced bare `pip install` examples with
       `uv pip install --python <selected_python> ...`
     - Kept remediation explicitly tied to the selected `uv` interpreter

4. `skills/setup-agent/references/execution-contract.md`
   - Description: Runtime UX and reporting contract
   - Change:
     - Updated streaming examples to use uppercase status values such as
       `PASS` and `FAIL`
     - Updated runtime install reporting to describe
       `uv pip install --python ...` remediation rather than bare `pip install`

5. `skills/setup-agent/tests/test_references.py`
   - Description: Behavior-contract tests for the skill prompt and references
   - Change:
     - Added regression checks that the matrix no longer includes
       `7.3.0` or `7.1.0`
     - Added regression checks for `uv`-scoped command examples
     - Added regression checks for uppercase streaming status examples

### 2026-03-24 - Selected uv environment contract tightened to absolute interpreter paths

Trigger:
- `uv pip install --python ...` installs into the interpreter it is given, so
  the real failure mode was an ambiguous contract around what the selected
  target actually is.
- The skill still used `<selected_python>` as a placeholder, which could be
  misread as a version string like `3.10` instead of the chosen virtual
  environment interpreter path.
- The workspace download examples also still assumed a fixed `.venv` name
  instead of the user-confirmed environment that may have been reused.

Files changed:

1. `skills/setup-agent/SKILL.md`
   - Description: Main execution prompt used by the model
   - Change:
     - Added an explicit post-selection contract for `selected_env_root` and
       `selected_python_path`
     - Changed Gate 5 wording so Python facts are derived from the selected
       interpreter path rather than from an abstract "selected Python version"
     - Replaced framework and PTA examples with
       `uv run --python <selected_python_path> ...`

2. `skills/setup-agent/references/framework-remediation.md`
   - Description: Framework-layer execution reference
   - Change:
     - Replaced `<selected_python>` with `<selected_python_path>` throughout
     - Required all `uv pip install` and `uv run` operations to target the
       selected interpreter path explicitly

3. `skills/setup-agent/references/workspace-discovery.md`
   - Description: Workspace download and artifact discovery reference
   - Change:
     - Replaced `.venv/bin/python` download examples with
       `<selected_python_path>` so reused environments stay valid

4. `skills/setup-agent/references/execution-contract.md`
   - Description: Runtime UX and reporting contract
   - Change:
     - Added selected environment root and selected interpreter path to the
       required console evidence
     - Tightened install reporting to show the selected interpreter path in
       `uv pip install --python ...` remediation

5. `skills/setup-agent/tests/test_references.py`
   - Description: Behavior-contract tests for the skill prompt and references
   - Change:
     - Added regression checks for `selected_env_root` and
       `selected_python_path`
     - Added negative coverage so `<selected_python>` no longer remains in the
       execution examples

### 2026-03-24 - PTA helper path resolved from the skill root instead of the user workdir

Trigger:
- The PTA fallback helper was still documented as `scripts/pta_compat_lookup.py`,
  which only works if the current working directory happens to be the
  `setup-agent` skill root.
- In real setup-agent runs, the current working directory is the user project
  directory, so the fallback command could fail even when the helper script was
  present in the installed skill.

Files changed:

1. `skills/setup-agent/SKILL.md`
   - Description: Main execution prompt used by the model
   - Change:
     - Changed the PTA helper example to use
       `<setup_agent_skill_root>/scripts/pta_compat_lookup.py`
     - Added an explicit note that the helper path must be resolved from the
       installed skill directory rather than from the user work dir

2. `skills/setup-agent/references/framework-remediation.md`
   - Description: Framework-layer execution reference
   - Change:
     - Changed the bundled PTA helper invocation to use the skill-root-based
       path
     - Added an explicit warning against resolving the helper path relative to
       the user work dir

3. `skills/setup-agent/tests/test_references.py`
   - Description: Behavior-contract tests for the skill prompt and references
   - Change:
     - Added regression checks for the new skill-root-based helper path
     - Added regression checks that the workdir-relative interpretation is
       rejected in the documentation contract

## Latest Validation Snapshot

Validation performed after the latest 2026-03-24 setup-agent consistency
update:

```bash
pytest -q skills/setup-agent/tests/test_references.py
pytest -q skills/setup-agent/tests/test_manifest_contract.py
```

Result:
- `54 passed` in `skills/setup-agent/tests/test_references.py`
- `1 passed` in `skills/setup-agent/tests/test_manifest_contract.py`

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
