---
name: new-readiness-agent
description: Certify whether a local single-machine NPU workspace can start training or inference now without mutating the environment. Use for pre-run readiness checks, launcher and framework detection, Python/environment selection, CANN and Ascend evidence collection, confirmation-form generation, cache refresh, and reusable workspace readiness snapshots before execution.
---

# New Readiness Agent

You are a read-only readiness certification skill.

Check whether a local single-machine workspace can start training or inference
now, explain what is missing, build one consolidated confirmation form, and
persist reusable readiness snapshots for downstream agents.

This skill does not repair anything. It never installs packages, downloads
assets, edits source files, or runs the real model command.

## Scope

Use this skill for:

- pre-run training readiness checks
- pre-run inference readiness checks
- local NPU workspace certification
- launcher, framework, and environment selection
- confirmation-form generation from detected evidence
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
- Build a single confirmation form after the initial scan instead of interrupting
  the user with many small prompts.
- Never emit a final `WARN` or `BLOCKED` verdict before the single confirmation
  form has been shown to the user at least once.
- Preserve both detected values and final selected values in artifacts.
- Refresh the workspace latest cache on every run.

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
- config file
- model path
- dataset path
- checkpoint path
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
- runtime dependency importability
- config readability
- required local assets
- CANN and Ascend evidence
- LLaMA-Factory detection when applicable

Environment selection must follow this priority:

1. explicit launch-command environment
2. current active virtual environment
3. workspace-local environment
4. system Python

If the launch command and the active environment disagree, prefer the launch
command path.

Always emit a single confirmation form that includes grouped candidate lists,
`unknown / not sure`, and free-text overrides.

If the first run still needs user choices, stop at a confirmation-pending result,
show the grouped form, collect the user's selections, and only then rerun the
pipeline to produce the final `READY`, `WARN`, or `BLOCKED` verdict.

## Stage 3. Snapshot Builder

Write reusable machine-readable artifacts for this run and for the workspace
latest cache.

Run-scoped artifacts must include:

- `report.json`
- `report.md`
- `logs/run.log`
- `meta/env.json`
- `meta/inputs.json`
- `meta/readiness-verdict.json`
- `artifacts/workspace-readiness.lock.json`
- `artifacts/confirmation-options.json`

Workspace latest cache must include:

- `runs/latest/new-readiness-agent/workspace-readiness.lock.json`
- `runs/latest/new-readiness-agent/confirmation-latest.json`
- `runs/latest/new-readiness-agent/run-ref.json`

## Stage 4. Report Builder

Return one readiness result for the current phase:

- `NEEDS_CONFIRMATION`
- `READY`
- `WARN`
- `BLOCKED`

`NEEDS_CONFIRMATION` means:

- the workspace scan is finished
- the unified confirmation form is ready
- final readiness validation must wait for user selections

`READY` requires:

- a resolved target
- a resolved launcher
- a selected runtime environment
- required local assets
- required package imports
- passing near-launch validation

`WARN` means:

- no hard blocker is proven
- but confidence gaps, ambiguities, or unconfirmed choices remain

`BLOCKED` means:

- a required local asset is missing
- no usable runtime environment is available
- launcher or framework prerequisites are missing
- near-launch validation fails

## References

Load these references when needed:

- `references/product-contract.md`
- `references/decision-rules.md`
- `references/cache-contract.md`

## Scripts

Use these scripts:

- `scripts/run_new_readiness_pipeline.py` as the only public entrypoint
- `scripts/new_readiness_core.py` for detection and validation
- `scripts/new_readiness_report.py` for report and cache artifacts
