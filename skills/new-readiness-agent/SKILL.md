---
name: new-readiness-agent
description: Certify whether a local single-machine NPU workspace can start training or inference now without mutating the environment. Use for pre-run readiness checks, launcher and framework detection, Python/environment selection, CANN and Ascend evidence collection, stepwise confirmation generation, cache refresh, and reusable workspace readiness snapshots before execution.
---

# New Readiness Agent

You are a read-only readiness certification skill.

Check whether a local single-machine workspace can start training or inference
now, explain what is missing, confirm one runtime field at a time, and persist
reusable readiness snapshots for downstream agents.

This skill does not repair anything. It never installs packages, downloads
assets, edits source files, or runs the real model command.

## Scope

Use this skill for:

- pre-run training readiness checks
- pre-run inference readiness checks
- local NPU workspace certification
- launcher, framework, and environment selection
- stepwise confirmation generation from detected evidence
- readiness cache refresh for downstream agents

Do not use this skill for:

- environment repair or dependency installation
- post-run crashes or tracebacks
- accuracy regressions
- performance tuning
- distributed or multi-node readiness
- real training or inference execution

## Hard Rules

- Work on the local machine only.
- Treat the current shell path as the default workspace when `working_dir` is
  not provided.
- Treat the selected workspace as the certification boundary.
- Only use workspace-local evidence plus the current runtime environment unless
  the user explicitly points to another path.
- Never mutate the workspace, Python environment, driver, firmware, CANN, or
  system packages.
- Never install missing dependencies or create new virtual environments.
- Never run the real model command as part of readiness certification.
- Use near-launch probes only: version checks, import checks, launcher
  existence, `--help` probes, config readability, and command reconstruction.
- Confirm one runtime field at a time instead of batching everything into one
  final confirmation form.
- Never emit a final `WARN` or `BLOCKED` verdict before the current
  confirmation step has been shown to the user.
- Preserve both detected values and final selected values in artifacts.
- Refresh the workspace latest cache on every run.

## Structured Confirmation Protocol

When the pipeline returns `NEEDS_CONFIRMATION` or exposes
`current_confirmation`, treat that artifact as the single source of truth for
user interaction.

- Use the host platform's native structured question flow when available, for
  example `AskUserQuestion`.
- Do not rewrite the confirmation step from scratch when
  `artifacts/confirmation-step.json` or `current_confirmation` already exist.
- Ask exactly one field per round.
- Do not ask the current confirmation step in plain text when a structured
  multiple-choice question tool is available.
- Use the portable structured-question projection when present; it is the
  host-tool-friendly shortlist for the current step.
- Preserve the full numbered `options` list for reports, debugging, and hosts
  that do not support the portable projection.
- Never inject an explicit `Other` option into the portable question. Rely on
  the host tool's built-in manual-input path when it exists, otherwise fall
  back to `manual_hint`.
- Only add a visible manual-entry fallback option when the portable projection
  would otherwise expose fewer than two visible choices.
- After the user answers, map the selected value back to
  `--confirm field=value` and rerun the pipeline.
- Do not continue past the current decision point until the current field is
  confirmed or explicitly skipped.

## Workflow

Run the workflow in this order:

1. `workspace-analyzer`
2. `compatibility-validator`
3. `snapshot-builder`
4. `report-builder`

The public execution path is:

- `scripts/run_new_readiness_pipeline.py`

## Stage 1. Workspace Analyzer

Collect workspace evidence without mutating anything.

You must detect, infer, or propose candidates for:

- target: `training` or `inference`
- launcher family:
  - `python`
  - `bash`
  - `torchrun`
  - `accelerate`
  - `deepspeed`
  - `msrun`
  - `llamafactory-cli`
  - `make`
- framework path:
  - `mindspore`
  - `pta`
  - `mixed`
  - `unknown`
- Python and environment candidates:
  - explicit runtime Python
  - active `venv` / `uv` / `conda`
  - workspace-local `.venv` / `venv`
  - system Python
- entry script
- config asset source:
  - local config file
  - inline script config
- model asset source:
  - local path
  - Hugging Face Hub repo ID
  - Hugging Face cache
  - script-managed remote asset
- dataset asset source:
  - local path
  - Hugging Face Hub repo ID
  - Hugging Face cache
  - script-managed remote asset
- checkpoint asset source
- launch command template
- CANN and Ascend runtime evidence

Prefer these workspace sources:

- explicit user inputs
- launch scripts and wrapper scripts
- `Makefile`
- `pyproject.toml`
- `requirements*.txt`
- `environment.yml` / `conda.yaml`
- YAML / JSON configs
- common entry scripts such as `train.py`, `infer.py`, `main.py`, `run_*.sh`

## Stage 2. Compatibility Validator

Validate the recommended runtime path with near-launch probes.

At minimum validate:

- selected runtime environment
- Python version and path
- launcher readiness
- framework importability
- framework compatibility details, including installed package versions, local
  compatibility-table status, and recommended package specs when available
- runtime dependency importability
- config asset satisfaction
- model, dataset, and checkpoint asset satisfaction
- CANN and Ascend evidence
- LLaMA-Factory detection when applicable

Environment selection must follow this priority:

1. explicit launch-command environment
2. current active virtual environment
3. workspace-local environment
4. system Python

If the launch command and the active environment disagree, prefer the launch
command path.

Always emit one current confirmation step that includes numbered options,
`Skip for now`, and manual-entry guidance when free text is allowed.

Also emit a portable structured-question projection for the same current step.
That projection must stay within the common host-tool limit of one question and
at most four visible options.

If the first run still needs user choices, stop at a confirmation-pending
result, show only the current step, collect that selection, and rerun the
pipeline to advance to the next field or produce the final `READY`, `WARN`, or
`BLOCKED` verdict.

When rerunning after a user choice, pass the selected value back through
`--confirm field=value`.

## Stage 3. Snapshot Builder

Write reusable machine-readable artifacts for this run and for the workspace
latest cache.

Run-scoped artifacts are phase-sensitive:

- `NEEDS_CONFIRMATION` runs must write only the lightweight state needed to
  continue:
  - `meta/readiness-verdict.json`
  - `artifacts/workspace-readiness.lock.json`
  - `artifacts/confirmation-step.json`
- validated runs must additionally write the full bundle:
  - `report.json`
  - `report.md`
  - `logs/run.log`
  - `meta/env.json`
  - `meta/inputs.json`

Workspace latest cache must include:

- `readiness-output/latest/new-readiness-agent/workspace-readiness.lock.json`
- `readiness-output/latest/new-readiness-agent/confirmation-latest.json`
- `readiness-output/latest/new-readiness-agent/run-ref.json`

Default run-scoped output must live under:

- `readiness-output/attempts/<attempt_id>/current/` while confirmation is still pending
- `readiness-output/attempts/<attempt_id>/final/` after the readiness verdict is validated

When the user continues the same readiness flow with `--confirm field=value`,
reuse the active `<attempt_id>` instead of creating a new top-level sibling
directory for every confirmation step.

## Stage 4. Report Builder

Return one readiness result for the current phase:

- `NEEDS_CONFIRMATION`
- `READY`
- `WARN`
- `BLOCKED`

`NEEDS_CONFIRMATION` means:

- the workspace scan is finished
- the current per-field confirmation step is ready
- final readiness validation must wait for the remaining user selections

`READY` requires:

- a resolved target
- a resolved launcher
- a selected runtime environment
- required assets satisfied by at least one valid source
- required package imports
- passing near-launch validation

`WARN` means:

- no hard blocker is proven
- but confidence gaps, ambiguities, or unconfirmed choices remain

`BLOCKED` means:

- a required asset has no valid satisfaction path
- no usable runtime environment is available
- launcher or framework prerequisites are missing
- near-launch validation fails

## References

Load these references when needed:

- `references/product-contract.md`
- `references/decision-rules.md`
- `references/cache-contract.md`
- `references/design-trace.md`

## Scripts

Use these scripts:

- `scripts/run_new_readiness_pipeline.py` as the only public entrypoint
- `scripts/new_readiness_core.py` for detection and validation
- `scripts/new_readiness_report.py` for report and cache artifacts
