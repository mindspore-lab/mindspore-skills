# Readiness-Agent Dependency Closure

This reference defines how `readiness-agent` should build and reason about the
dependency closure for the selected execution target.

Use it whenever the skill needs to answer:

- what "configuration is complete" means for the intended task
- what prerequisites must exist before validation or remediation
- whether a missing element is actually required for this target

## Core Rule

Do not ask whether the whole machine is "fully configured" in the abstract.

Instead ask:

- is the dependency closure for the selected execution target complete enough
  to run?

The closure is target-scoped, not machine-global.

## What Dependency Closure Means

`DependencyClosure` is the set of prerequisites required by the selected
execution path.

It should be derived from:

- the resolved `ExecutionTarget`
- the discovered framework path
- the actual imports and assets needed by that path
- the minimum runnable command shape for the target

## Required Layers

The dependency closure must cover these layers:

1. system layer
2. Python environment layer
3. framework layer
4. runtime dependency layer
5. workspace and asset layer
6. task execution layer

## 1. System Layer

The system layer includes the host prerequisites required before Python-level
validation is meaningful.

Typical elements:

- Ascend NPU visibility
- driver presence and usability
- firmware presence and usability
- CANN presence and usability
- Ascend environment sourcing such as `set_env.sh`

Rules:

- if the selected target requires Ascend and the system layer is unhealthy, the
  closure is incomplete
- unresolved system-layer gaps usually become `system_fatal`

## 2. Python Environment Layer

The Python environment layer includes the execution environment that the target
actually depends on.

Typical elements:

- `uv` availability when the workflow uses it
- selected environment path
- interpreter path
- Python version
- PATH viability for the chosen toolchain

Rules:

- environment checks must align with the selected target, not a random
  interpreter on the machine
- unresolved Python environment gaps usually become `env_remediable`

## 3. Framework Layer

The framework layer includes the target framework stack and its compatibility
requirements.

Typical elements:

- MindSpore package and compatibility
- PTA package set such as `torch` and `torch_npu`
- framework-to-CANN compatibility
- framework importability
- framework smoke readiness

Rules:

- do not validate both framework paths equally when the target clearly selects
  one path first
- unresolved framework gaps usually become `framework_remediable` or `unknown`

## 4. Runtime Dependency Layer

The runtime dependency layer includes Python packages actually required by the
selected script or launch path.

Typical elements:

- direct imports from the selected script
- imports from immediately referenced modules
- tokenizer, dataset, trainer, or accelerator packages required by the target

Rules:

- prefer dependencies proven by code and launch evidence
- do not build this layer from a generic package wishlist
- unresolved runtime dependency gaps usually become `env_remediable` when the
  package name is explicit

## 5. Workspace And Asset Layer

This layer includes the files, directories, and permissions required by the
selected target.

Typical elements:

- entry script
- config path
- model path
- dataset path
- checkpoint path
- output path
- tokenizer files
- storage availability
- path permissions

Rules:

- only require assets that the target truly depends on
- `dataset_path` is generally required for training
- `checkpoint_path` is required only when resume or checkpoint-backed execution
  is intended
- unresolved missing assets usually become `asset_remediable` or
  `workspace_manual`

## 6. Task Execution Layer

This layer captures the minimum runnable path for the intended task.

Typical elements:

- minimum command shape
- required arguments
- target-specific smoke path
- minimum proof that the task can begin execution

Examples:

- training: config parse, dataset openability, model construction, train-step
  smoke path
- inference: model load, tokenizer load, forward or generation smoke path

Rules:

- closure is not complete just because files exist
- closure must be complete enough to support the minimum task proof expected
  for certification

## Closure Construction Rules

Build the closure from strong evidence, in this order:

1. explicit user input
2. launch scripts and config files
3. selected code path and imports
4. model and checkpoint markers
5. local environment facts

Do not add requirements that are not justified by the selected target.

## Completeness Rules

The closure is complete enough for strong certification only when:

- every required layer has been populated for the target
- no critical required element is missing
- blockers and warnings explain every known gap
- the closure supports the minimum validation path required for the target

The closure is incomplete when:

- a required layer is missing evidence
- a critical asset is absent
- environment or framework requirements remain unresolved
- the task smoke path cannot be attempted or justified

## Relationship To Remediation

`env_fix` must use the dependency closure as its scope boundary.

This means:

- fix only the missing pieces that belong to the closure
- do not repair unrelated machine state
- do not install generic packages that are not justified by closure evidence

Important principle:

- "missing whatever is needed" means "missing from the dependency closure of
  the selected execution target"

## Relationship To Status

The closure directly influences final status synthesis.

Typical guidance:

- complete closure with strong proof may support `READY`
- partially complete closure with unresolved uncertainty usually supports
  `WARN`
- missing critical closure elements usually supports `BLOCKED`

## Invariants

The skill must preserve these rules:

- dependency closure is always target-scoped
- closure is built from evidence, not generic assumptions
- critical missing closure elements must become blockers
- `env_fix` must not mutate outside closure scope
- certification must not ignore unresolved closure gaps
