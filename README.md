# MindSpore Skills

[English](README.md) | [简体中文](README.zh-CN.md)

## Tell the agent what went wrong. It fixes the rest.

**failure-agent**

:x: **Problem:** Qwen3 training crashed on Ascend 910B with gradient checkpointing error

:keyboard: **Type:** `/fix "according to the error, fix the issue"`

:white_check_mark: **Result:** failure-agent reads the logs, locates the root cause, and applies the fix automatically

<img src="docs/assets/failure_agent.gif" width="720" />

**accuracy-agent**

:x: **Problem:** Qwen3 inference output has precision errors after switching to Ascend

:keyboard: **Type:** `/fix "qwen3 infer has accuracy issue, check run_01.log"`

:white_check_mark: **Result:** accuracy-agent compares results, traces the numerical drift, and fixes it automatically

<img src="docs/assets/accuracy_agent.gif" width="720" />

**algorithm-agent**

:x: **Problem:** Need to integrate MHC (Manifold-constrained Hyperparameters) into Qwen3 on Ascend 910B

:keyboard: **Type:** `/algorithm-agent "integrate MHC feature into Qwen3 model"`

:white_check_mark: **Result:** algorithm-agent analyzes the model structure, integrates MHC into the decoder layer, updates config, adds tests, and verifies the result

<img src="docs/assets/algorithm_agent.gif" width="720" />

---

MindSpore Skills for **AI infra and model training workflows**.

MindSpore Skills focuses on the high-frequency tasks around model training and migration, including **environment readiness, failure diagnosis, accuracy analysis, performance analysis, model migration, operator work, and algorithm adaptation**.

Unlike general-purpose coding agent extensions, MindSpore Skills is designed to help developers and algorithm engineers **get training jobs running, locate failures, align results, and improve performance**.

MindSpore Skills can be used in two ways:

- as the capability layer behind **MindSpore CLI**
- as reusable domain skills loaded by other **CLI agents**

---

## 1. Why MindSpore Skills

Model training is not just a coding task.

Real-world training workflows are often blocked by a mix of environment issues, runtime failures, accuracy mismatch, profiling noise, framework behavior, operator gaps, and migration work. These tasks are high-frequency, cross-layer, and highly experience-driven.

What slows teams down is often not the idea itself, but the repeated engineering-heavy work around it: checking environments, reading logs, narrowing root causes, comparing results, collecting profiling evidence, and iterating toward a stable run.

MindSpore Skills is built to turn those repeated, messy, engineering-heavy tasks into reusable skills and guided workflows.

In short:

- general coding agents focus on **how to write the code**
- MindSpore Skills focuses on **how to get the training task running, running correctly, and running efficiently**

---

## 2. What it solves

MindSpore Skills is built for tasks such as:

- checking whether a training workspace is ready
- diagnosing training and runtime failures
- analyzing accuracy mismatch and regression
- identifying common performance bottlenecks
- assisting model migration and operator adaptation
- routing algorithm feature adaptation into an existing model codebase

These are common in AI infra and model training workflows, especially when work crosses:

- framework and runtime behavior
- model code and training scripts
- data preprocessing and result validation
- operator support and backend differences
- profiling signals and performance bottlenecks

---

## 3. What it is not

MindSpore Skills is **not** a general-purpose coding assistant package.

It is not centered on:

- generic code generation
- repository-wide refactoring
- broad software engineering tasks unrelated to training workflows

It is a domain skill layer focused on **AI infra and model training tasks**.

---

## 4. Current focus

MindSpore Skills currently focuses on the most common and highest-value tasks in the **model training workflow**, especially in single-machine or otherwise controlled environments:

- preflight readiness checks
- training/runtime failure diagnosis
- accuracy debugging
- performance analysis
- model migration and operator-related work

This repository does **not** try to cover every large-scale training scenario at once.

The current priority is to make core training workflows deeper, more reliable, and more reusable.

---

## 5. Skill map

A simple way to understand the repository:

### Before training
Use **Readiness** skills to check whether the workspace, dependencies, and execution environment are ready.

### When training fails
Use **Failure** skills to analyze runtime and startup failures, narrow the root cause, and identify the responsible layer or component.

### When results are wrong
Use **Accuracy** skills to investigate mismatch, drift, regression, or wrong results after execution succeeds.

### When performance is poor
Use **Performance** skills to inspect throughput, latency, memory, dataloader, utilization, host/device behavior, and other bottlenecks.

### When adaptation is needed
Use **Migration**, **Operator**, and **Algorithm** skills to route model migration, operator implementation, and feature adaptation work.

---

## 6. Available skills

### Diagnosis and optimization

| Skill | Description |
|---|---|
| `readiness-agent` | Check whether a local single-machine workspace is ready to train or run inference before execution |
| `failure-agent` | Diagnose MindSpore and PTA (`torch_npu`) training and runtime failures with evidence-backed root-cause validation |
| `accuracy-agent` | Diagnose accuracy regressions, drift, wrong results, and cross-platform mismatch after successful execution |
| `performance-agent` | Diagnose throughput, latency, memory, utilization, dataloader, and communication bottlenecks after the workload already runs |

### Migration and adaptation

| Skill | Description |
|---|---|
| `migrate-agent` | Top-level model migration entry that analyzes the source repo, selects the correct migration route, and verifies the result |
| `algorithm-agent` | Adapt a paper feature or released reference implementation into an existing model codebase and prepare the patch for readiness validation |

### Operator and API work

| Skill | Description |
|---|---|
| `operator-agent` | Route and build `torch` or `mindspore` operators through custom-access or native-framework integration, with MindSpore API-resolution and `op_info` verification support |
| `api-helper` | Find API call chains and operator wiring in the MindSpore codebase |

---

## 7. Available commands

The public slash command surface is intentionally small.

Specialist capabilities such as operator work, readiness checks, accuracy analysis, and algorithm adaptation remain available as skills and routed workflows rather than becoming public top-level commands.

| Command | Description |
|---|---|
| `/diagnose` | Top-level symptom router for failure, accuracy, and performance diagnosis |
| `/fix` | Top-level symptom router for diagnose + propose + confirm + apply + verify |
| `/migrate` | Migration router for HF / third-party model migration, with readiness follow-up when the user wants to run locally |

---

## 8. Quick examples

### Diagnose a training problem

```bash
/diagnose my qwen3 lora run crashes with operator not implemented on ascend
```

### Diagnose and fix a training problem

```bash
/fix throughput is only 340 tok/s on npu, expected 520
```

### Route a migration request

```bash
/migrate port this HuggingFace qwen2 repo to MindSpore
```

### Migrate and make it runnable locally

```bash
/migrate I want to run this Hugging Face Transformers Qwen3 model in MindSpore on this machine
```

### Diagnose an accuracy issue

```bash
/diagnose training succeeds but final loss diverges from the torch baseline after step 1200
```

### Diagnose a performance issue

```bash
/diagnose dataloader is slow and device utilization stays low during qwen finetuning
```

Other capabilities can also be triggered by describing the task directly, for example operator implementation, readiness validation, or algorithm changes.

---

## 9. How to use

MindSpore Skills can be used in two main ways.

### 9.1 With MindSpore CLI

MindSpore CLI provides the official end-to-end entrypoint for integrating these skills into model training workflows.

Use this path when you want a unified workflow for:

- readiness checks
- diagnosis
- result analysis
- follow-up actions
- training-task-oriented interaction

### 9.2 With other CLI agents

MindSpore Skills can also be loaded by other CLI agents as reusable domain skills for MindSpore, Ascend, and model training tasks.

Use this path when you already work inside another agent host and want to add specialized AI infra capabilities.

---

## 10. Installation

### 10.1 Claude Code

Register the marketplace and install:

```bash
/plugin marketplace add mindspore-lab/mindspore-skills
/plugin install ms@mindspore-skills
```

Then use slash commands:

```bash
/ms:diagnose
/ms:fix
/ms:migrate
```

### 10.2 OpenCode

OpenCode loads custom commands from `commands/` and skills from `skills/`.

For a project-local install:

```bash
git clone https://github.com/mindspore-lab/mindspore-skills.git .opencode
```

This gives OpenCode the expected layout:

```text
.opencode/commands/
.opencode/skills/
```

For a global install, copy or symlink the contents into your existing OpenCode directories instead of replacing the whole directory:

```bash
git clone https://github.com/mindspore-lab/mindspore-skills.git ~/.config/opencode/mindspore-skills
mkdir -p ~/.config/opencode/skills ~/.config/opencode/commands
ln -s ~/.config/opencode/mindspore-skills/skills/* ~/.config/opencode/skills/
ln -s ~/.config/opencode/mindspore-skills/commands/* ~/.config/opencode/commands/
```

Then in OpenCode:

```bash
/diagnose
/fix
/migrate
```

### 10.3 Gemini CLI

Install as an extension:

```bash
gemini extensions install https://github.com/mindspore-lab/mindspore-skills.git --consent
```

Or from a local clone:

```bash
git clone https://github.com/mindspore-lab/mindspore-skills.git
gemini extensions install ./mindspore-skills --consent
```

### 10.4 Codex

Codex does not install slash commands from this repository. It follows the instructions it discovers from `AGENTS.md` files in the active project.

If you are working in this repository, no extra install step is needed. Open the repo in Codex and it will read `AGENTS.md`.

If you want to reuse this guidance in another project, copy or adapt the relevant sections into that project's `AGENTS.md`.

---

## 11. Repository layout

```text
mindspore-skills/
├── .claude-plugin/     # Claude plugin metadata
├── .github/workflows/  # CI workflows
├── commands/           # public slash command surface
├── docs/               # concepts, contracts, and architecture notes
├── examples/           # examples and demos
├── githooks/           # git hook content
├── hooks/              # repository hook scripts
├── skills/             # reusable domain skills
├── tests/contract/     # cross-skill contract tests
├── tools/              # helper and consistency scripts
├── AGENTS.md           # Codex instruction entry
├── CLAUDE.md           # Claude-facing repository notes
├── Makefile
├── README.md
└── gemini-extension.json
```

### Directory roles

- `commands/` keeps the public command surface intentionally small
- `skills/` is the core capability layer
- `docs/` stores concepts, contracts, and architecture notes
- `examples/` shows usage and scenario coverage
- `tests/contract/` validates cross-skill contracts and repository expectations
- `tools/` helps maintain consistency across skills and integrations

---

## 12. Architecture notes

MindSpore Skills is the capability layer.

- MindSpore Agent covers broader agent directions such as scientific computing, operator generation, model workflows, and lite-side scenarios
- MindSpore CLI provides the official end-to-end interaction surface for model-training-oriented tasks
- MindSpore Skills provides reusable, domain-specific training capabilities

This separation allows the same skills to be reused across different agent hosts while still supporting an official integrated workflow.

---

## 13. Examples

See `examples/README.md` for the current example inventory and status.

Example categories should gradually cover:

- readiness before first run
- failure after training start
- accuracy mismatch after successful execution
- performance bottleneck analysis
- migration routing and adaptation

---

## 14. Docs

Key concept and contract docs:

- `docs/concepts/agent-architecture-overview.md`
- `docs/concepts/skills-contract.md`
- `docs/concepts/artifacts-and-reporting.md`

Recommended reading order:

1. architecture overview
2. skills contract
3. artifacts and reporting
4. examples

---

## 15. Contributing

We currently encourage two kinds of contribution.

### 15.1 Content contribution

Contribute to `mindspore-skills` by improving:

- skills
- workflows
- examples
- docs
- diagnose patterns
- prompts, rules, and recipes

This is the main path for turning repeated training experience into reusable public capabilities.

### 15.2 Problem collaboration

Help improve issue quality and closure efficiency by contributing:

- reproducible issue reports
- environment details
- logs and command traces
- narrowing hints
- verification feedback
- regression checks

This is the main path for turning user problems into clearer, actionable inputs.

### Adding a new skill

When adding a new skill, update the repository consistently:

- add `skills/<skill-name>/SKILL.md`
- add a public slash command in `commands/` only if it belongs to the small public command surface
- update `AGENTS.md` if global guidance changes
- update `README.md`
- update `gemini-extension.json` if needed
- update `CLAUDE.md` if needed

### Consistency checks

```bash
python tools/check_consistency.py
```

Optional local setup:

```bash
python tools/install_git_hooks.py
make hooks
```

---

## 16. Recommended next docs

The repository already includes strong technical building blocks, but the public contribution surface will be clearer with:

- `CONTRIBUTING.md`
- issue templates for bug reports and skill proposals
- a short roadmap doc
- a use-cases doc organized by training lifecycle

---

## 17. License

Apache 2.0
