# Agent Architecture Overview

This document describes the current top-level agent architecture of the
`mindspore-skills` repository.

The repository is organized around a small set of user-facing agents. Each
agent owns one problem domain, exposes a stable top-level workflow, and emits
structured outputs that downstream tools such as `mscli` can consume.

## Design Goals

- Keep user entrypoints task-oriented instead of implementation-oriented.
- Give each agent a narrow, defensible responsibility boundary.
- Separate top-level routing from route-specific implementation details.
- Keep outputs reusable by reports, snapshots, tests, and future orchestrators.

## Top-Level Skills

The current top-level skill surface is:

- `readiness-agent`
- `new-readiness-agent`
- `failure-agent`
- `accuracy-agent`
- `algorithm-agent`
- `performance-agent`
- `operator-agent`
- `migrate-agent`

There is also one supporting skill:

- `api-helper`

`api-helper` is not a top-level domain agent. It is a supporting codebase
exploration capability that can help other workflows.

## User-Level Routing

The user should enter by intent, not by low-level implementation detail.

Recommended entry mapping:

- "Can this workspace train?" -> `readiness-agent`
- "Which launcher, Python, and CANN path should this workspace use?" -> `new-readiness-agent`
- "The run failed, why?" -> `failure-agent`
- "The run finishes, but the result is wrong." -> `accuracy-agent`
- "Adapt this paper trick or feature into my current model codebase." -> `algorithm-agent`
- "Add mHC to this llm model." -> `algorithm-agent`
- "The run works, but it is slow." -> `performance-agent`
- "Implement an operator." -> `operator-agent`
- "Migrate this model or repo to MindSpore." -> `migrate-agent`

This keeps product routing stable even when internal builder or route choices
change later.

## Common Agent Shape

Most top-level agents follow the same high-level pattern:

1. analyze the target workspace, run, or task
2. validate or select the most likely route
3. build a snapshot of the evidence and result
4. emit a final report

This repository intentionally keeps that pattern consistent across agents so
documentation, testing, and future orchestration remain predictable.

## Agent Responsibilities

### `readiness-agent`

Purpose:
- validate whether a single-machine training or inference workspace is ready to run

Core workflow:
1. `workspace-analyzer`
2. `compatibility-validator`
3. `snapshot-builder`
4. `report-builder`

What it checks:
- training or inference entrypoints and config
- framework and backend clues
- environment and library compatibility
- dataset, model, checkpoint, and storage readiness

What it does not do:
- system-layer driver, firmware, or CANN repair
- broad machine cleanup outside the selected workspace or environment
- post-run diagnosis
- multi-node distributed checks

### `new-readiness-agent`

Purpose:
- certify whether a single-machine NPU workspace can start training or inference now without mutating the environment

Core workflow:
1. `workspace-analyzer`
2. `compatibility-validator`
3. `snapshot-builder`
4. `report-builder`

What it checks:
- launcher family and launch command shape
- framework path and LLaMA-Factory usage
- Python and environment candidates
- CANN and Ascend runtime evidence
- required local assets and import-level runtime prerequisites
- workspace latest readiness cache for downstream agents

What it does not do:
- environment repair
- dependency installation
- real workload execution
- distributed checks

### `failure-agent`

Purpose:
- diagnose why a run crashed, hung, or failed

Core workflow:
1. `failure-analyzer`
2. `root-cause-validator`
3. `snapshot-builder`
4. `report-builder`

What it focuses on:
- logs, traceback, stderr, run context
- failure stage and failure type
- root-cause candidates with ranked evidence

What it does not do:
- full pre-run readiness from scratch
- performance optimization
- migration work

`failure-agent` can recommend rerunning `readiness-agent` when a failure points
back to a pre-run readiness problem.

### `accuracy-agent`

Purpose:
- diagnose result mismatches, regressions, drift, and wrong-result issues after
  a run completes

Core workflow:
1. `accuracy-analyzer`
2. `consistency-validator`
3. `snapshot-builder`
4. `report-builder`

What it focuses on:
- baseline versus current result comparison
- config, data, model, checkpoint, and precision consistency
- likely numerical or workflow drift

What it does not do:
- runtime crash diagnosis
- readiness checks as a primary workflow
- performance tuning

### `algorithm-agent`

Purpose:
- adapt a paper feature, released reference implementation, or user-described
  algorithm change into an existing model codebase

Core workflow:
1. `feature-analyzer`
2. `integration-planner`
3. `patch-builder`
4. `readiness-handoff-and-report`

What it focuses on:
- extracting a feature or trick from paper text, released code, or user intent
- selecting the correct integration route, including specialized packs such as
  `mhc`, without changing the top-level workflow
- planning how the change should fit into the current codebase
- generating the minimal patch and config delta
- preparing the updated workspace for readiness validation

Important boundary:
- `algorithm-agent` remains the only top-level user entry for algorithm feature
  work
- specialized routes such as `mhc` stay behind the stable
  four-stage workflow instead of becoming separate public skills

What it does not do:
- full model migration
- framework operator development
- post-run failure diagnosis
- post-run accuracy or performance analysis

### `performance-agent`

Purpose:
- diagnose why a run is slow after it is already working

Core workflow:
1. `performance-analyzer`
2. `bottleneck-validator`
3. `snapshot-builder`
4. `report-builder`

What it focuses on:
- throughput and latency
- utilization and memory pressure
- dataloader, host overhead, and communication bottlenecks
- profiler and trace interpretation

What it does not do:
- crash diagnosis
- environment setup
- model migration

### `operator-agent`

Purpose:
- implement framework operators

Core workflow:
1. `operator-analyzer`
2. `method-selector`
3. `implementation-builder`
4. `verification-and-report`

This agent supports exactly two implementation methods:

- `custom-access`
- `native-framework`

Important boundary:
- `operator-agent` owns analysis, route selection, and orchestration
- concrete builder or backend implementation details can evolve behind that
  stable top-level workflow

### `migrate-agent`

Purpose:
- migrate model implementations or repos into the MindSpore ecosystem

Core workflow:
1. `migration-analyzer`
2. `route-selector`
3. `migration-builder`
4. `verification-and-report`

Supported migration routes:

- `hf-transformers`
- `hf-diffusers`
- `generic-pytorch-repo`

Important boundary:
- users enter through `migrate-agent`
- users do not need to decide route details up front

## Relationship Between Agents

These agents are not isolated. They form a layered problem space:

- `readiness-agent` answers whether a workspace should run
- `new-readiness-agent` answers whether a workspace should run when the user wants a read-only certification plus a reusable latest cache
- `failure-agent` answers why a run did not complete
- `accuracy-agent` answers why a completed run produced the wrong result
- `algorithm-agent` adapts new feature ideas into the current codebase
- `performance-agent` answers why a completed run is too slow
- `operator-agent` handles operator implementation work
- `migrate-agent` handles migration work

Useful handoff patterns:

- `failure-agent` may point back to `readiness-agent`
- `accuracy-agent` may point back to `readiness-agent` for missing context
- `algorithm-agent` should hand patched workspaces to `readiness-agent` before
  execution
- `performance-agent` may use environment or profiler context produced by other
  workflows
- `operator-agent` stays separate from diagnosis flows
- `migrate-agent` stays separate from runtime diagnosis flows

## Why This Architecture

This structure replaces older skill surfaces that were organized around
specialized builders or source-specific migration entrypoints. The current
architecture is better because it:

- keeps top-level user entrypoints stable
- avoids exposing internal route choices too early
- gives each agent a clear domain boundary
- makes future `mscli` routing simpler
- keeps route-specific detail inside the correct agent instead of spreading it
  across many top-level skills

## Current Repository Surface

Current top-level `skills/` directories:

- `_shared`
- `accuracy-agent`
- `algorithm-agent`
- `api-helper`
- `failure-agent`
- `migrate-agent`
- `operator-agent`
- `performance-agent`
- `readiness-agent`

Current top-level `commands/` entries:

- `diagnose`
- `fix`
- `migrate`

## Guidance For Future Changes

When adding or changing a skill in this repository:

- prefer adding capability behind an existing top-level agent before creating a
  new top-level agent
- keep top-level naming based on user intent, not implementation detail
- preserve the stable four-stage workflow shape unless there is a strong reason
  not to
- document route-specific logic inside the owning agent rather than exposing it
  as a separate product-level entrypoint
