# MindSpore Skills

[English](README.md) | [简体中文](README.zh-CN.md)

面向 **AI 基础设施与模型训练工作流** 的 MindSpore Skills。

MindSpore Skills 聚焦于模型训练与迁移周边的高频任务，包括 **环境就绪性检查、故障诊断、精度分析、性能分析、模型迁移、算子相关工作与算法适配**。

与通用型编码代理扩展不同，MindSpore Skills 的目标是帮助开发者和算法工程师 **把训练任务跑起来、定位失败原因、对齐结果并改善性能**。

MindSpore Skills 主要有两种使用方式：

- 作为 **MindSpore CLI** 背后的能力层
- 作为其他 **CLI agent** 可加载的可复用领域技能

## Demo

<table>
  <tr>
    <td align="center"><b>failure-agent demo</b></td>
    <td align="center"><b>acc agent demo</b></td>
  </tr>
  <tr>
    <td><img src="docs/assets/faliure_agent.gif" width="420" /></td>
    <td><img src="docs/assets/accuracy_agent.gif" width="420" /></td>
  </tr>
</table>

---

## 1. 为什么需要 MindSpore Skills

模型训练不只是写代码。

真实训练工作流经常会被多种问题同时卡住：环境问题、运行时失败、精度不一致、profiling 噪声、框架行为差异、算子缺失、迁移工作等。这些任务高频、跨层，而且高度依赖经验。

真正拖慢团队的，往往不是想法本身，而是这些反复出现、工程性很重的工作：检查环境、读日志、收窄根因、比对结果、收集 profiling 证据，并反复迭代直到任务稳定运行。

MindSpore Skills 的目标，就是把这些重复、复杂、工程性很强的任务沉淀成可复用技能与引导式工作流。

简而言之：

- 通用 coding agent 更关注 **代码怎么写**
- MindSpore Skills 更关注 **训练任务如何跑起来、跑正确、跑高效**

---

## 2. 它解决什么问题

MindSpore Skills 面向的典型任务包括：

- 检查训练工作区是否已就绪
- 诊断训练与运行时失败
- 分析精度不一致与回归
- 识别常见性能瓶颈
- 辅助模型迁移与算子适配
- 将算法特性适配路由到已有模型代码库中

这些问题在 AI 基础设施与模型训练工作流中很常见，尤其当问题横跨以下层面时：

- 框架与运行时行为
- 模型代码与训练脚本
- 数据预处理与结果校验
- 算子支持情况与后端差异
- profiling 信号与性能瓶颈

---

## 3. 它不是什么

MindSpore Skills **不是** 一个通用型编码助手包。

它不以以下方向为中心：

- 通用代码生成
- 仓库级重构
- 与训练工作流无关的大范围软件工程任务

它是一个面向 **AI 基础设施与模型训练任务** 的领域技能层。

---

## 4. 当前重点

MindSpore Skills 当前聚焦于 **模型训练工作流** 中最常见、价值最高的任务，尤其是在单机或其他相对可控环境下：

- 运行前就绪性检查
- 训练/运行时失败诊断
- 精度问题排查
- 性能分析
- 模型迁移与算子相关工作

这个仓库 **并不** 试图一次性覆盖所有大规模训练场景。

当前优先级是把核心训练工作流做得更深、更稳、更可复用。

---

## 5. 技能地图

可以这样理解这个仓库：

### 训练前
使用 **Readiness** 技能检查工作区、依赖和执行环境是否准备完成。

### 训练失败时
使用 **Failure** 技能分析启动失败或运行时失败，收窄根因，并定位责任层或责任组件。

### 结果不对时
使用 **Accuracy** 技能调查执行成功后的结果不一致、漂移、回归或错误结果。

### 性能差时
使用 **Performance** 技能检查吞吐、时延、内存、dataloader、利用率、host/device 行为以及其他瓶颈。

### 需要适配时
使用 **Migration**、**Operator** 和 **Algorithm** 技能来路由模型迁移、算子实现和特性适配工作。

---

## 6. 当前技能

### 诊断与优化

| Skill | Description |
|---|---|
| `readiness-agent` | 检查本地单机工作区在执行前是否已具备训练或推理条件 |
| `failure-agent` | 基于证据诊断 MindSpore 与 PTA (`torch_npu`) 的训练及运行时失败 |
| `accuracy-agent` | 诊断执行成功后的精度回归、漂移、错误结果与跨平台不一致 |
| `performance-agent` | 诊断任务已运行后的吞吐、时延、内存、利用率、dataloader 与通信瓶颈 |

### 迁移与适配

| Skill | Description |
|---|---|
| `migrate-agent` | 顶层模型迁移入口，分析源仓库并选择正确迁移路径，随后完成验证 |
| `algorithm-agent` | 将论文特性或参考实现适配进现有模型代码库，并为就绪性验证准备补丁 |

### 算子与 API 工作

| Skill | Description |
|---|---|
| `operator-agent` | 通过 custom-access 或 native-framework integration 路由和构建 `torch` 或 `mindspore` 算子，并支持 MindSpore API 解析与 `op_info` 校验 |
| `api-helper` | 在 MindSpore 代码库中查找 API 调用链与算子接线关系 |

---

## 7. 可用命令

对外公开的 slash command 面刻意保持很小。

算子工作、readiness 检查、精度分析、算法适配等专用能力仍然作为 skills 与内部路由工作流存在，而不是暴露成顶层公共命令。

| Command | Description |
|---|---|
| `/diagnose` | failure、accuracy、performance 诊断的顶层症状路由入口 |
| `/fix` | diagnose + propose + confirm + apply + verify 的顶层路由入口 |
| `/migrate` | 面向 HF / 第三方模型迁移的顶层路由入口；如果用户目标是本地运行，也会继续进入 readiness 检查 |

---

## 8. 快速示例

### 诊断训练问题

```bash
/diagnose my qwen3 lora run crashes with operator not implemented on ascend
```

### 诊断并修复训练问题

```bash
/fix throughput is only 340 tok/s on npu, expected 520
```

### 路由迁移请求

```bash
/migrate port this HuggingFace qwen2 repo to MindSpore
```

### 迁移并让它在本机可运行

```bash
/migrate I want to run this Hugging Face Transformers Qwen3 model in MindSpore on this machine
```

### 诊断精度问题

```bash
/diagnose training succeeds but final loss diverges from the torch baseline after step 1200
```

### 诊断性能问题

```bash
/diagnose dataloader is slow and device utilization stays low during qwen finetuning
```

也可以通过直接描述任务来触发其他能力，例如算子实现、readiness 校验或算法变更。

---

## 9. 如何使用

MindSpore Skills 主要有两种使用方式。

### 9.1 配合 MindSpore CLI 使用

MindSpore CLI 是把这些技能集成进模型训练工作流的官方端到端入口。

适合以下场景：

- readiness 检查
- 问题诊断
- 结果分析
- 后续动作执行
- 面向训练任务的统一交互

### 9.2 配合其他 CLI agent 使用

MindSpore Skills 也可以作为可复用领域技能，被其他 CLI agent 加载，用于 MindSpore、Ascend 和模型训练相关任务。

适合你已经在其他 agent host 中工作、但希望补充 AI infra 专项能力的场景。

---

## 10. 安装

### 10.1 Claude Code

注册 marketplace 并安装：

```bash
/plugin marketplace add mindspore-lab/mindspore-skills
/plugin install ms@mindspore-skills
```

然后使用以下 slash commands：

```bash
/ms:diagnose
/ms:fix
/ms:migrate
```

### 10.2 OpenCode

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
/diagnose
/fix
/migrate
```

### 10.3 Gemini CLI

作为扩展安装：

```bash
gemini extensions install https://github.com/mindspore-lab/mindspore-skills.git --consent
```

或从本地克隆安装：

```bash
git clone https://github.com/mindspore-lab/mindspore-skills.git
gemini extensions install ./mindspore-skills --consent
```

### 10.4 Codex

Codex 不会从该仓库安装 slash command。它会读取当前项目中的 `AGENTS.md` 指令文件。

如果你正在这个仓库内工作，不需要额外安装步骤。直接在 Codex 中打开该仓库，它会读取 `AGENTS.md`。

如果你想在其他项目中复用这些指导，可以把相关内容复制或改写到目标项目的 `AGENTS.md` 中。

---

## 11. 仓库结构

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

### 目录职责

- `commands/` 保持小而清晰的公共命令面
- `skills/` 是核心能力层
- `docs/` 存放概念、契约与架构说明
- `examples/` 展示使用方式与场景覆盖
- `tests/contract/` 校验跨 skill 契约和仓库级约束
- `tools/` 用于保持 skills 与多种集成的一致性

---

## 12. 架构说明

MindSpore Skills 是能力层。

- MindSpore Agent 覆盖更广的 agent 方向，例如科学计算、算子生成、模型工作流与 lite 侧场景
- MindSpore CLI 提供面向模型训练任务的官方端到端交互入口
- MindSpore Skills 提供可复用、领域化的训练能力

这种分层让同一套 skills 可以在不同 agent host 之间复用，同时仍然支持官方集成工作流。

---

## 13. 示例

当前示例清单与状态见 `examples/README.md`。

示例类别会逐步覆盖：

- 首次运行前的 readiness
- 训练开始后的 failure
- 成功执行后的精度不一致
- 性能瓶颈分析
- 迁移路由与适配

---

## 14. 文档

关键概念与契约文档：

- `docs/concepts/agent-architecture-overview.md`
- `docs/concepts/skills-contract.md`
- `docs/concepts/artifacts-and-reporting.md`

推荐阅读顺序：

1. architecture overview
2. skills contract
3. artifacts and reporting
4. examples

---

## 15. 贡献

当前主要鼓励两类贡献。

### 15.1 内容贡献

欢迎通过以下方式改进 `mindspore-skills`：

- skills
- workflows
- examples
- docs
- diagnose patterns
- prompts、rules 和 recipes

这是把反复出现的训练经验沉淀为公共能力的主要路径。

### 15.2 问题协作

也欢迎通过以下方式提升 issue 质量与闭环效率：

- 可复现的问题报告
- 环境细节
- 日志与命令轨迹
- 收窄问题的线索
- 验证反馈
- 回归检查

这是把用户问题转化为更清晰、可操作输入的主要路径。

### 新增一个 skill

新增 skill 时，请同步更新以下内容：

- 添加 `skills/<skill-name>/SKILL.md`
- 只有当它属于小型公共命令面时，才在 `commands/` 中新增公开 slash command
- 如全局指导发生变化，更新 `AGENTS.md`
- 更新 `README.md`
- 必要时更新 `gemini-extension.json`
- 必要时更新 `CLAUDE.md`

### 一致性检查

```bash
python tools/check_consistency.py
```

可选本地设置：

```bash
python tools/install_git_hooks.py
make hooks
```

---

## 16. 建议补充的文档

仓库已经具备较强的技术基础，但以下文档会让公共贡献面更清晰：

- `CONTRIBUTING.md`
- bug report / skill proposal issue templates
- 简短 roadmap
- 按训练生命周期组织的 use-cases 文档

---

## 17. License

Apache 2.0
