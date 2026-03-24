# Readiness-Agent Product Contract

This reference defines the output and certification contract for
`readiness-agent`.

Use it whenever the skill:

- decides whether the workspace is `READY`, `WARN`, or `BLOCKED`
- writes report artifacts
- summarizes readiness for the user
- hands machine-consumable results to other skills

The report format is two-layer:

- shared run envelope in `out/report.json`
- readiness verdict payload in `out/meta/readiness-verdict.json`

## Core Product Meaning

`readiness-agent` has one user-facing responsibility:

- certify whether the intended task is runnable

The intended task must resolve to:

- `training`
- `inference`

Do not leave the final result with an unresolved `auto` target.

## Output Layers

The result has two layers:

1. user-visible result
2. internal machine-consumable result

Rule:

- user-visible output is primary
- internal output exists for certification, debugging, and cross-skill use
- the shared report envelope records the run
- the readiness verdict payload records the business decision

## User-Visible Result

The primary user-visible result must contain:

- `status`
- `can_run`
- `target`
- `summary`
- `blockers`
- `warnings`
- `next_action`

### User-visible field meanings

`status`
- enum: `READY` | `WARN` | `BLOCKED`

`can_run`
- boolean
- whether the intended task should be treated as runnable now

`target`
- enum: `training` | `inference`

`summary`
- one-sentence result summary in direct language

`blockers`
- concise blocker statements for the user

`warnings`
- concise warning statements for the user

`next_action`
- the smallest useful next step

## User-Visible Status Rules

`READY`
- enough strong evidence exists to treat the intended task as runnable
- `can_run` must be `true`

`WARN`
- the task may be runnable, but evidence is incomplete, ambiguous, or below
  the threshold for `READY`
- `can_run` may be `true` or `false`, but the summary must make the distinction
  clear
- if `can_run=true`, `next_action` should focus on inspecting residual warnings
  before proceeding
- if `can_run=false`, `next_action` should direct the user to rerun validation,
  complete task smoke, or confirm the target as appropriate

`BLOCKED`
- one or more hard blockers prevent execution, or minimum task validation
  clearly failed
- `can_run` must be `false`
- `next_action` should point to the smallest unblock step, not generic retry

## Internal Result

The internal result should contain at minimum:

- `execution_target`
- `evidence_level`
- `dependency_closure`
- `checks`
- `blockers_detailed`
- `warnings_detailed`
- `fix_applied`
- `revalidated`

### Internal field meanings

`execution_target`
- normalized object describing what the skill believes the user intends to run

`evidence_level`
- enum: `structural` | `import` | `runtime_smoke` | `task_smoke`

`task_smoke_state`
- enum: `not_requested` | `passed` | `failed` | `skipped` | `missing_result` | `unknown`
- records whether an explicit task smoke command was requested and what happened

`dependency_closure`
- normalized view of prerequisites required by the selected target
- the final report should preserve the resolved closure used for certification,
  not an empty placeholder

`checks`
- structured per-check results
- task-smoke related checks should preserve structured execution evidence such
  as `command_preview`, `exit_code`, `stdout_head`, `stderr_head`, and
  `timed_out` when available

`blockers_detailed`
- normalized blocker objects with category, severity, evidence, and
  remediation metadata

`warnings_detailed`
- warning objects with evidence and uncertainty reasoning

`fix_applied`
- remediation actions actually executed
- when remediation execution results exist, preserve the executor result object
  including `results`, `executed_actions`, `failed_actions`, and
  `needs_revalidation`

`revalidated`
- whether affected checks were rerun after successful mutation
- if `fix_applied.executed_actions` is non-empty, compute this from whether all
  `fix_applied.needs_revalidation` scopes are covered by the final checks, not
  from the mere existence of a remediation result

## Simplicity Rule

Do not force the user to understand internal terms such as:

- `evidence_level`
- `dependency_closure`
- `blocker_taxonomy`
- `check_matrix`

These belong in artifacts and internal reasoning, not in the primary user
summary.

## Status Synthesis Rules

### `READY`

Emit `READY` only when:

- no hard blocker remains
- the final interpreted target is stable enough
- required assets for the target are present
- the selected framework path is healthy
- permissions and storage are sufficient
- the internal evidence threshold for `READY` is met

Recommended threshold:

- generally require at least `runtime_smoke`
- prefer `task_smoke` when a target-specific smoke path is reasonably available
- if an explicit `task_smoke_cmd` is present, do not emit `READY` unless that
  task smoke result is present and passed

### `WARN`

Emit `WARN` when:

- no hard blocker is proven, but confidence is insufficient for `READY`
- target discovery remains partially ambiguous
- compatibility is plausible but not fully confirmed
- asset or environment state is likely usable but not strongly enough proven

### `BLOCKED`

Emit `BLOCKED` when:

- a hard blocker remains
- target-specific minimum validation clearly fails
- critical closure elements are missing

## Invariants

The skill must preserve these invariants:

- `READY` never appears together with a hard blocker
- `BLOCKED` always implies `can_run=false`
- `fix_applied` implies `revalidated` must be explicitly set
- internal evidence fields must not contradict the user-visible status
- final `target` must never remain `auto`

## Artifact Expectations

Recommended artifact outputs:

- `out/report.json`
- `out/report.md`
- `out/meta/readiness-verdict.json`
- `out/meta/execution-target.json`
- `out/meta/checks.json`
- `out/meta/blockers.json`
- `out/meta/remediation.json`

`out/report.json` should satisfy the shared report schema.

`out/meta/readiness-verdict.json` should satisfy the readiness-agent verdict
schema and carry the fields defined in this document.

The Markdown report should prioritize the user-visible layer and only include
the highest-value evidence.

## Cross-Skill Use

Other skills may consume:

- `status`
- `can_run`
- `target`
- `execution_target`
- `task_smoke_state`
- `blockers_detailed`
- `fix_applied`
- `revalidated`

This supports questions such as:

- "Did my changes keep the environment runnable?"
- "Which blocker still remains?"
- "Was the workspace revalidated after mutation?"
