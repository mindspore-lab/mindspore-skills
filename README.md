# MindSpore Model Agent

[English](README.md) | [简体中文](README.zh-CN.md)

MindSpore Model Agent is a training-focused AI agent solution for the MindSpore ecosystem. It is designed for the high-frequency engineering work around model training, where users need more than general code generation and need help with domain-specific training tasks.

It is built on two closely related parts:

- `mindspore-skills`: the domain capability layer for model training and debugging tasks. It provides reusable skills for readiness checking, failure diagnosis, accuracy analysis, performance analysis, model migration, algorithm adaptation, and operator implementation. These skills can work not only with MindSpore Model Agent, but also with other agentic CLI environments such as Claude Code, OpenCode, and Codex.
- [`mindspore-cli`](https://github.com/mindspore-lab/mindspore-cli): the official CLI of MindSpore Model Agent. It provides better integration with related skills and is optimized for model training use cases, offering a more unified end-to-end experience for training-oriented workflows.

## What's New in v0.1.0

- Initial public release of MindSpore Model Agent
- Introduced `mindspore-skills` as the reusable capability layer for model training and debugging tasks
- Introduced `mindspore-cli` as the official CLI optimized for model training workflows

## Demos

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

:keyboard: **Type:** `/integrate "add MHC feature into Qwen3 model"`

:white_check_mark: **Result:** algorithm-agent analyzes the model structure, integrates MHC into the decoder layer, updates config, adds tests, and verifies the result

<img src="docs/assets/algorithm_agent.gif" width="720" />

## Key Capabilities

- workspace readiness checking
- failure diagnosis
- accuracy analysis
- performance analysis
- model migration
- algorithm adaptation
- operator implementation

## Commands

MindSpore Model Agent exposes a small set of slash commands as the public interface. Each command does lightweight routing and hands off to one or more specialist agents.

| Command | Routes to | Mode |
|---------|-----------|------|
| `/preflight` | readiness-agent | environment check |
| `/diagnose` | failure-agent · accuracy-agent · performance-agent | diagnose only |
| `/fix` | failure-agent · accuracy-agent · performance-agent | diagnose + fix |
| `/migrate` | readiness-agent · migrate-agent | environment check + migration |
| `/integrate` | algorithm-agent · operator-agent | algorithm feature or operator integration |

## Repositories

- `mindspore-cli`: the official CLI interface for MindSpore Model Agent
- `mindspore-skills`: the reusable skill layer behind MindSpore Model Agent

## Installation

### mindspore-cli

`mindspore-cli` bundles all skills in the binary — no separate skill installation is required. See the [mindspore-cli repository](https://github.com/mindspore-lab/mindspore-cli) for more details.

Launch and use slash commands directly:

```bash
/preflight
/diagnose
/fix
/migrate
/integrate
```

### Claude Code

Register the marketplace and install:

```bash
/plugin marketplace add mindspore-lab/mindspore-skills
/plugin install ms@mindspore-skills
```

Then use slash commands:

```bash
/ms:preflight
/ms:diagnose
/ms:fix
/ms:migrate
/ms:integrate
```

### OpenCode

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
/preflight
/diagnose
/fix
/migrate
/integrate
```

### Gemini CLI

Install as an extension:

```bash
gemini extensions install https://github.com/mindspore-lab/mindspore-skills.git --consent
```

Or from a local clone:

```bash
git clone https://github.com/mindspore-lab/mindspore-skills.git
gemini extensions install ./mindspore-skills --consent
```

### Codex

Codex does not install slash commands from this repository. It follows the instructions it discovers from `AGENTS.md` files in the active project.

If you are working in this repository, no extra install step is needed. Open the repo in Codex and it will read `AGENTS.md`.

If you want to reuse this guidance in another project, copy or adapt the relevant sections into that project's `AGENTS.md`.

## License

Apache 2.0
