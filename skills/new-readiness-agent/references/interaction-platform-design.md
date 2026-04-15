# New-Readiness-Agent Host 交互适配设计

## 1. 文档目的

这份文档讨论的是：

- `new-readiness-agent` 如何在不同宿主里接入宿主原生交互能力
- 当前参考实现为什么先以 **Claude Code + AskUserQuestion** 为例
- 同时如何保证 skill 主体对未来其他工具或平台保持开放

这份文档是维护者设计说明，不是运行时 skill 指令。

## 2. 范围与非目标

本设计只覆盖：

- `NEEDS_CONFIRMATION` 阶段如何向用户提问
- 如何把 `current_confirmation` 映射成宿主原生结构化提问工具
- 当前参考实现中，Claude Code 如何映射到 `AskUserQuestion`
- 其他宿主如何继续回退到文本流

本设计不覆盖：

- Python pipeline 的产物结构重写
- artifacts / cache contract 改造
- Codex `request_user_input` 集成
- 长列表分页式结构化交互

## 3. 核心原则

### 3.1 Python pipeline 不变

当前公开入口仍然是：

- [`C:\workspace\mindspore-skills\skills\new-readiness-agent\scripts\run_new_readiness_pipeline.py`](/C:/workspace/mindspore-skills/skills/new-readiness-agent/scripts/run_new_readiness_pipeline.py)

它继续只负责：

- 运行 readiness pipeline
- 输出 `current_confirmation`
- 接收 `--confirm field=value`
- 输出最终 verdict

它不直接感知：

- Claude Code
- `AskUserQuestion`
- 任何宿主 UI tool

### 3.2 AskUserQuestion 属于 host agent 层

宿主原生提问工具不是 Python subprocess 能直接调用的能力。

因此分层必须保持为：

- readiness core 负责“**问什么**”
- host adapter 负责“**怎么问**”

也就是说：

- `current_confirmation` 是统一交互描述
- 宿主 adapter 把它渲染成具体 tool
- 其他宿主继续渲染成文本流

## 4. 现有可复用交互描述

当前 pipeline 已经输出：

- `phase`
- `status`
- `confirmation_required`
- `pending_confirmation_fields`
- `current_confirmation`

其中 `current_confirmation` 已经足够支撑宿主适配：

- `field`
- `label`
- `prompt`
- `step_number`
- `total_steps`
- `recommended_value`
- `allow_free_text`
- `manual_hint`
- `options`

因此 v1 不建议新增第二套交互 schema。

## 5. 当前参考实现：Claude Code

当前最明确的宿主原生提问工具是 Claude Code 的 `AskUserQuestion`。

依据 Claude Code 源码：

- tool 名称是 `AskUserQuestion`
- schema 在 [`C:\workspace\claude-code\src\tools\AskUserQuestionTool\AskUserQuestionTool.tsx`](/C:/workspace/claude-code/src/tools/AskUserQuestionTool/AskUserQuestionTool.tsx)
- prompt 在 [`C:\workspace\claude-code\src\tools\AskUserQuestionTool\prompt.ts`](/C:/workspace/claude-code/src/tools/AskUserQuestionTool/prompt.ts)

它的关键限制是：

- 每次最多 `1-4` 个问题
- 每题必须是 `2-4` 个选项
- 用户端自动提供 `Other`
- 支持 preview，但 v1 不需要用

因此：

- 并不是所有 readiness confirmation step 都适合 `AskUserQuestion`
- 像长 checkpoint 列表、长环境列表，不能强行映射

未来如果接入别的宿主工具，应沿用相同判断原则：

- 先看 option 数量限制
- 再看是否自带 free-text fallback
- 再决定是否使用结构化提问或回退文本流

## 6. 推荐的适配策略

### 6.1 总体策略

当 `new-readiness-agent` 返回 `NEEDS_CONFIRMATION` 时：

1. 宿主 agent 读取 `current_confirmation`
2. 若当前宿主可用原生结构化提问工具，且当前步骤适合结构化提问，则调用该工具
3. 否则，沿用当前文本流确认
4. 将用户回答映射回 `--confirm field=value`
5. 重新运行 pipeline

### 6.2 何时用 AskUserQuestion

建议 v1 只在以下条件同时满足时使用宿主结构化提问：

- 当前宿主提供可用的结构化提问 tool
- 当前只有一个 active confirmation step
- 当前步骤的可选择项可以压缩进该 tool 的上限

推荐优先覆盖：

- `target`
- `launcher`
- `framework`
- 小规模的 `runtime_environment`
- 小规模的 `config_asset`

推荐先不要覆盖：

- `checkpoint_asset` 的长列表
- 候选很多的 `model_asset`
- 候选很多的 `dataset_asset`
- 候选很多的环境列表

这些步骤继续使用文本流。

当前的参考实现是：

- Claude Code 中用 `AskUserQuestion`

## 7. 映射规则

### 7.1 从 current_confirmation 到宿主结构化提问

映射规则建议为：

- `header`
  - 使用 `current_confirmation.label`
  - 必要时裁剪为短标题
- `question`
  - 使用 `current_confirmation.prompt`
- `options`
  - 使用当前 concrete options
  - 保留 `skip check for now`
  - 不把 “enter a custom value manually” 作为显式 option
  - 前提是宿主工具已提供等价的 free-text fallback

### 7.2 从宿主提问结果回到 pipeline

用户回答后：

- 若选择已有 option
  - 回填 `--confirm field=<option.value>`
- 若选择 `Other`
  - 回填 `--confirm field=<free text>`
- 若选择 `skip check for now`
  - 回填 `--confirm field=__unknown__`

## 8. 为什么不做长列表分页

长列表步骤理论上可以做分页式宿主提问：

- 第一轮先问“推荐 / 展开更多 / skip”
- 第二轮再问具体候选

但 v1 不建议这样做，因为：

- 会引入 host-specific 的分页状态
- 会让一个 `current_confirmation.field` 对应多轮额外交互
- 会增加 artifacts 与真实确认过程对齐的复杂度

因此推荐策略是：

- **结构化工具只覆盖小选项步骤**
- **长列表步骤统一回退到文本流**

## 9. 对 SKILL.md 的影响

`SKILL.md` 应只增加平台中立的运行时指令：

- 如果宿主结构化提问工具可用且步骤适合，就优先调用它
- 否则回退文本流
- 不要求运行时去读本设计文档

`SKILL.md` 不应再把这份文档作为 runtime reference 强制加载。

## 10. 对代码的影响

v1 不需要改 Python pipeline contract。

推荐保持不变：

- `run_new_readiness_pipeline.py`
- `new_readiness_core.py`
- artifacts / cache schema

也就是说，宿主交互适配首先是 **skill orchestration 层改动**，不是 pipeline 内核改动。

## 11. 一句话总结

`new-readiness-agent` 应保持平台中立：Python pipeline 继续只负责 readiness 判定和确认状态输出；宿主 adapter 再把 `current_confirmation` 映射成具体平台的结构化提问工具。当前参考实现是 Claude Code 的 `AskUserQuestion`，其他宿主暂时继续保留文本流回退。
