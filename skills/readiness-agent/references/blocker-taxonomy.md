# Readiness-Agent Blocker Taxonomy

This reference defines how `readiness-agent` should classify readiness issues
into normalized blocker categories.

Use it whenever the skill needs to:

- classify failed checks
- distinguish blockers from warnings
- decide whether remediation is possible
- decide whether the final result should be `READY`, `WARN`, or `BLOCKED`

## Core Rule

Every failed or uncertain readiness check must map to either:

- a normalized blocker, or
- a normalized warning

Do not leave important readiness issues as unclassified free-form notes.

## Required Blocker Fields

Each blocker should capture:

- `id`
- `category`
- `severity`
- `summary`
- `evidence`
- `remediable`
- `remediation_owner`
- `revalidation_scope`

## Categories

The normalized blocker categories are:

- `system_fatal`
- `env_remediable`
- `framework_remediable`
- `asset_remediable`
- `workspace_manual`
- `unknown`

## `system_fatal`

Use `system_fatal` when the system layer is not healthy enough for user-space
repair to proceed.

Typical examples:

- no Ascend NPU detected
- driver missing or unusable
- firmware mismatch that prevents runtime use
- CANN missing or unusable
- `set_env.sh` missing or cannot be sourced

Rules:

- `system_fatal` is not remediable by `env_fix`
- `remediation_owner` is external or manual system setup
- unresolved `system_fatal` normally implies final `BLOCKED`

## `env_remediable`

Use `env_remediable` when the failure is inside the selected Python execution
environment and can be repaired through safe user-space actions.

Typical examples:

- `uv` missing
- `uv` installed but not visible in PATH
- selected environment missing
- missing Python runtime dependency
- environment metadata inconsistent with the intended task

Rules:

- this category is remediable when the missing piece is explicit enough
- `remediation_owner` is `readiness-agent`
- remediation must stay inside safe user-space boundaries

## `framework_remediable`

Use `framework_remediable` when the framework path is identified, but the
installed framework tuple is missing, incompatible, or unhealthy in a way that
can be repaired inside the selected environment.

Typical examples:

- missing `mindspore`
- missing `torch` or `torch_npu`
- incompatible MindSpore / CANN / Python tuple
- incompatible PTA tuple
- framework import fails because a clear runtime dependency is missing

Rules:

- this category is remediable only when the compatible target package or fix is
  known with sufficient confidence
- `remediation_owner` is `readiness-agent`
- if compatibility remains unresolved, downgrade certainty and avoid silent fix

## `asset_remediable`

Use `asset_remediable` when the execution target is clear but one or more
required assets are missing and can be safely acquired or resolved.

Typical examples:

- required local model is missing but can be downloaded with confirmation
- checkpoint path is missing but expected location is known
- tokenizer files are absent but the selected model source is clear

Rules:

- this category is remediable only when asset acquisition is explicit and
  within allowed workflow scope
- `allow_network` and confirmation policy must be respected
- unresolved required assets normally prevent `READY`

## `workspace_manual`

Use `workspace_manual` when the issue is in the workspace logic, structure, or
task definition and should not be auto-repaired by `env_fix`.

Typical examples:

- missing or invalid training script
- missing or invalid inference entry path
- broken config semantics
- dataset format mismatch
- task logic inconsistent with discovered framework path
- task-smoke script parsing fails for the selected entry script
- explicit task-smoke command fails after prerequisite checks pass

Rules:

- `workspace_manual` is not remediable by `env_fix`
- `remediation_owner` is the user or another skill
- unresolved `workspace_manual` usually implies `BLOCKED` or strong `WARN`
  depending on severity and certainty

## `unknown`

Use `unknown` when evidence is insufficient or contradictory and the skill
cannot safely classify the issue into a stronger category.

Typical examples:

- conflicting framework signals
- multiple plausible targets with no dominant evidence
- compatibility not confirmed from available facts
- missing package name cannot be identified with confidence

Rules:

- `unknown` should usually avoid automatic remediation
- unresolved `unknown` normally implies `WARN`, unless it blocks all practical
  execution judgment

## Warnings

Warnings are not blockers, but they still affect trust in certification.

Use warnings when:

- execution appears possible but not strongly enough proven
- ambiguity does not yet rise to a hard blocker
- non-critical assets or validations remain incomplete

Warnings should still include:

- summary
- evidence
- whether they lower confidence for `READY`

## Severity Guidance

Recommended severities:

- `fatal`
- `high`
- `medium`
- `low`

Interpretation:

- `fatal`: directly prevents certification of runnability
- `high`: likely prevents runnability unless remediated
- `medium`: may prevent strong certification or certain workflows
- `low`: informational or confidence-reducing issue

## Remediable Rule

A blocker is remediable only when all of the following are true:

- the blocker is inside allowed fix scope
- the repair action is sufficiently explicit
- the repair action stays in safe user-space
- required confirmation can be obtained
- revalidation scope is known

If any of these are false:

- do not mark the blocker as safely remediable

## Status Synthesis Guidance

### `READY`

Do not emit `READY` if:

- any unresolved `system_fatal` blocker exists
- any unresolved `workspace_manual` blocker exists for the intended target
- any unresolved high-confidence missing critical asset exists
- target ambiguity materially affects runnability

### `WARN`

Prefer `WARN` when:

- only `unknown` issues remain
- the target is plausible but not fully stable
- a warning lowers confidence below the `READY` threshold
- asset or compatibility evidence is incomplete but not conclusively blocking

### `BLOCKED`

Prefer `BLOCKED` when:

- any hard blocker remains unresolved
- minimum target validation fails
- a required closure element is clearly absent

## Invariants

The classifier must preserve these invariants:

- every important failed check maps to a blocker category
- unresolved `system_fatal` cannot produce `READY`
- unresolved `workspace_manual` cannot silently become `READY`
- ambiguous `unknown` cases must not be auto-fixed by guesswork
- remediation ownership must be explicit
