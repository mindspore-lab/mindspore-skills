---
name: MindSpore算子Skill创建
overview: 使用 /skill-creator 创建一个端到端定位并解决 MindSpore 算子问题的 Skill，将 md_files 中的问题单定位经验、算子开发文档、源码结构知识固化为可复用的自动化诊断与修复流程。
todos:
  - id: read-skill-creator
    content: 阅读 skill-creator SKILL.md，按照其流程创建 skill
    status: completed
  - id: write-architecture-ref
    content: 编写 references/architecture.md - 从源码分析中固化算子架构知识与源码导航地图
    status: completed
  - id: write-issue-patterns-ref
    content: 编写 references/issue_patterns.md - 从 md_files 问题单中归纳6大类根因模式及诊断特征
    status: completed
  - id: write-diagnostic-workflow-ref
    content: 编写 references/diagnostic_workflow.md - 固化定界→定位→修复的标准工作流
    status: completed
  - id: write-fix-patterns-ref
    content: 编写 references/fix_patterns.md - 从已解决问题中提取常见修复代码模式
    status: completed
  - id: write-testing-guide-ref
    content: 编写 references/testing_guide.md - 测试框架2.0使用指南与用例编写模板
    status: completed
  - id: write-aclnn-guide-ref
    content: 编写 references/aclnn_guide.md - ACLNN 算子适配专项指导
    status: completed
  - id: write-skill-md
    content: 编写 SKILL.md 主指令文件 - 定义触发条件、端到端流程、references 引用策略
    status: completed
  - id: test-and-iterate
    content: 用 2-3 个真实算子问题测试 skill 效果，迭代改进
    status: completed
isProject: false
---

# MindSpore 算子问题端到端解决 Skill 创建计划

## 项目资源概况

当前 `/Users/claw/work/ms_debug` 下包含：

- **md_files/** - 7632 个算子问题单 (100 gitcode + 7532 gitee)
- **mindspore/** - MindSpore 2.9.0 完整源码
- **operator_data/** / **operator_data2/** - 算子开发指导文档 (ACLNN适配、测试框架2.0、View算子、C++拼接组合等)
- **MindSporeTest/** - 测试套件

## Skill 设计

### 核心能力

Skill 需要覆盖算子问题处理的完整流程：

1. **问题分析 (Issue Analysis)** - 解析问题单，提取关键信息（环境、复现步骤、错误日志、关联组件）
2. **定界 (Fault Boundary)** - 判断问题属于哪个组件层（op_def / infer / kernel / bprop / API / compiler / runtime）
3. **定位 (Root Cause Location)** - 根据定界结果导航到相关源码位置
4. **修复 (Fix)** - 基于根因生成修复代码
5. **回归验证 (Regression)** - 运行相关测试验证修复
6. **测试补充 (Test)** - 补充缺失的测试用例

### Skill 文件结构

```
mindspore-ops-debugger/
├── SKILL.md                      # 主指令文件
└── references/
    ├── architecture.md           # MindSpore 算子架构与源码导航
    ├── issue_patterns.md         # 常见问题模式与根因分类
    ├── diagnostic_workflow.md    # 诊断工作流（定界→定位→修复）
    ├── fix_patterns.md           # 常见修复模式与代码模板
    ├── testing_guide.md          # 测试框架使用与用例编写
    └── aclnn_guide.md            # ACLNN 算子适配专项
```

### SKILL.md 主体结构

- **触发条件**：当用户提到 MindSpore 算子问题、ops bug、kernel error、shape inference 错误、精度问题等
- **步骤1：问题分析** - 解析问题描述，提取环境信息、错误类型、复现步骤
- **步骤2：定界** - 基于错误特征分类到具体组件层，读取 `references/issue_patterns.md`
- **步骤3：定位** - 根据组件层导航源码，读取 `references/architecture.md`
- **步骤4：修复** - 参考同类修复模式，读取 `references/fix_patterns.md`
- **步骤5：验证与测试** - 读取 `references/testing_guide.md`

### references 内容来源


| 参考文档                   | 信息来源                                           |
| ---------------------- | ---------------------------------------------- |
| architecture.md        | mindspore 源码结构分析 + operator_data 开发指导          |
| issue_patterns.md      | md_files 中的问题单归纳（6大类根因 + 诊断特征）                 |
| diagnostic_workflow.md | 问题单中的定位过程 + operator_data 调试指南                 |
| fix_patterns.md        | 问题单中的修复方案 + 源码中的代码模式                           |
| testing_guide.md       | operator_data2/9.4 测试框架 2.0 + MindSporeTest 结构 |
| aclnn_guide.md         | operator_data/ACLNN 算子适配指导                     |


### 关键知识点固化

**根因分类体系**（从 md_files 中归纳）：

- **精度/数值问题** - Ascend fp16、dtype 处理、随机数、loss scaling
- **API/签名不一致** - 位置参数 vs 关键字参数、参数校验、错误类型
- **Shape 推导/广播** - 动态 shape、broadcast 规则、AbstractProblem
- **编译器/IR** - DeadNode、控制流、pass 缺失、关键字参数处理
- **Kernel 实现** - ACLNN 适配、GPU/CPU 特化、动态 shape 支持
- **反向传播** - bprop 注册、不可导输入处理、inplace 操作

**源码导航地图**（从源码分析）：

- YAML 定义: `ops/op_def/yaml/`, `core/ops/ops_def/`
- Infer: `ops/infer/ops_func_impl/`, `core/ops/ops_func_impl/`
- Kernel: `ops/kernel/{cpu,gpu,ascend}/`
- Bprop: `ccsrc/frontend/expander/bprop/grad_ops/`
- Python API: `python/mindspore/ops/`, `ops/api_def/`
- 测试: `tests/st/ops/share/`

## 执行步骤

详见下方 Todos。

## 需要用户后续补充的数据

1. **更多已解决的问题单**（md_files 持续补充）- 尤其需要包含完整定位过程和修复 PR 链接的问题
2. **典型修复 PR 的 diff** - 有助于学习具体的代码修改模式
3. **常见误判案例** - 帮助 skill 避免错误定界
4. **特定算子类型的专项知识** - 如通信算子、自定义算子等，如果需要覆盖

