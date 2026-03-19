# MindSpore 与 torch_npu AI Skills 集合

面向工程落地的 AI 技能仓库，覆盖算子调试、代码检视、网页抓取与
MHTML 转 Markdown 等高频场景。

该仓库重点服务 MindSpore、torch_npu 与 Ascend 相关开发流程。每个 skill
目录均包含可直接复用的方法论、参考资料与辅助脚本。

[English README](./README.md)

## 仓库内容

- **MindSpore 算子问题调试流程**：从问题分析到修复回归的端到端闭环
- **算子代码检视流程**：支持 torch_npu / MindSpore 的 diff 与 PR 检视
- **网页登录态抓取流程**：基于 Playwright 的多格式网页留档
- **MHTML 转 Markdown 流程**：支持 OCR 提取图片文字

## Skill 清单

| Skill | 作用 | 典型触发场景 |
|---|---|---|
| [`mindspore-ops-debugger`](./mindspore-ops-debugger/SKILL.md) | MindSpore 算子问题端到端定位与修复指引 | kernel 报错、shape 推导错误、精度异常、梯度/bprop 问题、ACLNN 适配 |
| [`ascend-ops-reviewer`](./ascend-ops-reviewer/SKILL.md) | torch_npu/MindSpore 算子改动的结构化检视 | review PR、本地 diff 检视、算子代码审查 |
| [`web-fetch`](./web-fetch/SKILL.md) | 支持登录态的网页抓取，输出 MHTML/PDF/PNG/HTML | 抓取受保护页面、批量归档 URL、留存网页证据 |
| [`mhtml-to-md`](./mhtml-to-md/SKILL.md) | 将 `.mhtml` 档案转换为 Markdown，支持 OCR | 网页归档转技术文档、问题单内容提取 |

## 目录结构

```text
skills/
├── mindspore-ops-debugger/
├── ascend-ops-reviewer/
├── web-fetch/
└── mhtml-to-md/
```

大多数 skill 目录包含：
- `SKILL.md`：触发条件与执行工作流
- `references/`：检查清单、模式库、排障资料
- `scripts/`：解析、抓取、转换等辅助脚本

## 快速开始

### 1）克隆仓库

```bash
git clone <your-repo-url> skills
cd skills
```

### 2）按需使用 skill

打开目标 `SKILL.md`，按其中步骤在 AI 编程助手中执行。

示例：
- MindSpore 算子问题定位：`mindspore-ops-debugger/SKILL.md`
- 基于 diff 的算子检视：`ascend-ops-reviewer/SKILL.md`
- 登录态网页抓取：`web-fetch/SKILL.md`
- MHTML 转 Markdown：`mhtml-to-md/SKILL.md`

### 3）安装可选依赖（按 skill）

部分 skill 带有可执行脚本，需要额外运行时依赖：
- `web-fetch`：Python 依赖 + Playwright 浏览器
- `mhtml-to-md`：Python 依赖 + Ollama（OCR 使用 `glm-ocr:bf16`）

请以各 skill 的 `SKILL.md` 中命令为准。

## 适用人群

- MindSpore / torch_npu 算子开发与维护工程师
- 负责问题定界、回归分析的 AI 框架工程师
- 需要抓取并沉淀网页问题单资料的团队
- 希望构建可复用调试/检视流程的工程团队

## 贡献建议

欢迎提交改进，建议方向：
- 提升 `SKILL.md` 的流程清晰度与可执行性
- 补充高价值案例与模式库
- 增强脚本能力与使用示例
- 持续完善中英文文档一致性

建议优先提交可复现、可验证、贴近真实工程场景的改动。

## 文档语言

- 英文：`README.md`
- 中文：`README.zh-CN.md`

