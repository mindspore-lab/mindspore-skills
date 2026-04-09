# MindSpore Model Agent

[English](README.md) | [简体中文](README.zh-CN.md)

MindSpore Model Agent 是一个面向 MindSpore 生态的、聚焦模型训练场景的 AI agent solution。它面向模型训练周边高频工程工作而设计，适用于那些仅靠通用代码生成还不够、还需要训练领域专项能力支持的场景。

它由两个紧密相关的部分组成：

- `mindspore-skills`：面向模型训练与调试任务的领域能力层，提供可复用的技能，包括 readiness 检查、failure diagnosis、accuracy analysis、performance analysis、model migration、algorithm adaptation 和 operator implementation。这些 skills 不仅可用于 MindSpore Model Agent，也可以与 Claude Code、OpenCode、Codex 等其他 agentic CLI 环境配合使用。
- [`mindspore-cli`](https://github.com/mindspore-lab/mindspore-cli)：MindSpore Model Agent 的官方 CLI。它与相关 skills 有更好的集成，并针对模型训练场景进行了优化，提供更统一的端到端训练任务交互体验。

## v0.1.0 新增内容

- MindSpore Model Agent 首个公开版本发布
- 引入 `mindspore-skills` 作为面向模型训练与调试任务的可复用能力层
- 引入 `mindspore-cli` 作为面向模型训练工作流优化的官方 CLI

## Demos

**failure-agent**

:x: **问题：** Qwen3 在昇腾 910B 上训练崩溃，报 gradient checkpointing 错误

:keyboard: **输入：** `/fix "according to the error, fix the issue"`

:white_check_mark: **结果：** failure-agent 自动读取日志、定位根因并修复

<img src="docs/assets/failure_agent.gif" width="720" />

**accuracy-agent**

:x: **问题：** Qwen3 推理输出在切换到昇腾后出现精度误差

:keyboard: **输入：** `/fix "qwen3 infer has accuracy issue, check run_01.log"`

:white_check_mark: **结果：** accuracy-agent 自动比对结果、追踪数值偏差并修复

<img src="docs/assets/accuracy_agent.gif" width="720" />

**algorithm-agent**

:x: **问题：** 需要将 MHC（流形约束超参数）特性集成到昇腾 910B 上的 Qwen3 模型中

:keyboard: **输入：** `/integrate "add MHC feature into Qwen3 model"`

:white_check_mark: **结果：** algorithm-agent 自动分析模型结构，将 MHC 集成到 decoder layer，更新配置，添加测试并验证结果

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

MindSpore Model Agent 通过一组 slash commands 作为对外接口。每个命令负责轻量路由，并将任务交给一个或多个专项 agent 处理。

| 命令         | 路由至                                             | 模式               |
| ------------ | -------------------------------------------------- | ------------------ |
| `/preflight` | readiness-agent                                    | 环境检查           |
| `/diagnose`  | failure-agent · accuracy-agent · performance-agent | 仅定位问题         |
| `/fix`       | failure-agent · accuracy-agent · performance-agent | 定位问题 + 修复    |
| `/migrate`   | readiness-agent · migrate-agent.                   | 环境检查 + 迁移    |
| `/integrate` | algorithm-agent · operator-agent                   | 算法特性或算子集成 |

## Repositories

- `mindspore-cli`：MindSpore Model Agent 的官方 CLI 接口
- `mindspore-skills`：MindSpore Model Agent 背后的可复用 skill layer

## Installation

### mindspore-cli

`mindspore-cli` 将所有 skills 内置于二进制文件中，无需单独安装。mindspore-cli 介绍详见 [mindspore-cli 仓库](https://github.com/mindspore-lab/mindspore-cli)。

启动后直接使用 slash commands：

```bash
/preflight
/diagnose
/fix
/migrate
/integrate
```

### Claude Code

注册 marketplace 并安装：

```bash
/plugin marketplace add mindspore-lab/mindspore-skills
/plugin install ms@mindspore-skills
```

然后使用以下 slash commands：

```bash
/ms:preflight
/ms:diagnose
/ms:fix
/ms:migrate
/ms:integrate
```

### OpenCode

OpenCode 会从 `commands/` 加载自定义命令，并从 `skills/` 加载技能。

项目级安装：

```bash
git clone https://github.com/mindspore-lab/mindspore-skills.git .opencode
```

这样会得到 OpenCode 期望的目录结构：

```text
.opencode/commands/
.opencode/skills/
```

全局安装时，请把内容复制或软链接到现有 OpenCode 目录，而不是整体替换：

```bash
git clone https://github.com/mindspore-lab/mindspore-skills.git ~/.config/opencode/mindspore-skills
mkdir -p ~/.config/opencode/skills ~/.config/opencode/commands
ln -s ~/.config/opencode/mindspore-skills/skills/* ~/.config/opencode/skills/
ln -s ~/.config/opencode/mindspore-skills/commands/* ~/.config/opencode/commands/
```

然后在 OpenCode 中使用：

```bash
/preflight
/diagnose
/fix
/migrate
/integrate
```

### Gemini CLI

作为扩展安装：

```bash
gemini extensions install https://github.com/mindspore-lab/mindspore-skills.git --consent
```

或从本地克隆安装：

```bash
git clone https://github.com/mindspore-lab/mindspore-skills.git
gemini extensions install ./mindspore-skills --consent
```

### Codex

Codex 不会从该仓库安装 slash command。它会读取当前项目中的 `AGENTS.md` 指令文件。

如果你正在这个仓库内工作，不需要额外安装步骤。直接在 Codex 中打开该仓库，它会读取 `AGENTS.md`。

如果你想在其他项目中复用这些指导，可以把相关内容复制或改写到目标项目的 `AGENTS.md` 中。

## License

Apache 2.0
