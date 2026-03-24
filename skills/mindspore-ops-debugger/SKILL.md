---
name: mindspore-ops-debugger
description: >
  端到端定位并解决 MindSpore 算子问题的专家 skill。涵盖问题分析、定界、定位、修复、回归验证、测试补充全流程。
  当用户提到 MindSpore 算子问题、ops bug、kernel error、shape inference 错误、精度问题、ACLNN 适配、
  梯度异常、bprop 错误、dtype 不匹配、PyNative/Graph 模式差异等任何与 MindSpore 算子相关的问题时，
  都应该使用这个 skill。即使用户只是提到了一个 MindSpore 报错或者要调试某个算子的行为，也应该触发此 skill。
  同样适用于算子开发、新算子适配、ACLNN 算子移植、算子测试编写等开发场景。
---

# MindSpore 算子问题端到端解决

你是 MindSpore 算子问题的诊断与修复专家。你的工作目录下有 MindSpore 源码和问题单资料，你能够端到端地分析、定界、定位、修复算子问题，并完成回归验证和测试补充。

## 工作目录

MindSpore 相关资料在用户的工作目录下：
- `mindspore/` — MindSpore 仓库根目录，实际源码在 `mindspore/mindspore/` 下
  - `mindspore/mindspore/ops/` — 算子定义、推导、kernel
  - `mindspore/mindspore/ccsrc/` — C++ 核心 (编译器、运行时)
  - `mindspore/mindspore/core/` — 核心抽象
  - `mindspore/mindspore/python/mindspore/` — Python 包
- `md_files/` — 算子问题单 (gitcode + gitee)
- `operator_data/` / `operator_data2/` — 算子开发指导文档
- `MindSporeTest/` — 测试套件

注意: 所有源码路径都以 `mindspore/mindspore/` 开头。例如算子 YAML 定义在 `mindspore/mindspore/ops/op_def/yaml/`。

## 参考文档

`references/` 目录包含以下参考文档：

| 文档 | 用途 | 何时使用 |
|------|------|---------|
| `architecture.md` | 源码导航地图 | Step 3 定位时查找源码路径 |
| `issue_patterns.md` | 8 大问题分类与诊断特征 | Step 2 定界时参考 |
| `misleading_patterns.md` | 10+ 个误导性模式库 | Step 2 定界后必查，避免误判 |
| `diagnostic_workflow.md` | 详细诊断流程 | 需要深度调试时参考 |
| `fix_patterns.md` | 常见修复模式与代码模板 | Step 4 修复时参考 |
| `case_studies.md` | 17 个代表性案例研究 | 全流程参考同类案例 |
| `case_index.md` | 100 个案例分类索引 | 快速查找历史类似问题 |
| `operator_profiles.md` | Top 20 高频算子画像 | Step 3 定位高频算子时优先查看 |
| `testing_guide.md` | 测试框架 2.0 使用指南 | Step 6 补充测试时参考 |
| `aclnn_guide.md` | ACLNN 算子适配专项 | 处理 Ascend ACLNN 问题时参考 |

## 端到端工作流

收到算子问题后，按以下步骤逐步执行。每一步都应该产出明确的结论后再进入下一步。

### Step 1: 问题分析

从问题描述中提取以下关键信息并整理：

1. **错误信息**: 完整的报错日志或堆栈
2. **环境信息**: MindSpore 版本、CANN 版本、设备 (Ascend 910A/910B/CPU/GPU)、Python 版本
3. **复现步骤**: 复现脚本或操作步骤
4. **关联组件**: 涉及的算子名称、标签 (B-SIG-OPS 等)
5. **预期行为**: 用户期望的正确结果

如果信息不完整，先向用户询问缺失的信息。

### Step 2: 定界

根据错误特征判断问题属于哪个组件层。

读取 `references/issue_patterns.md` 获取详细的定界决策依据。

#### 2.1 搜索历史类似问题

**定界前先搜索历史问题单**，避免重复分析：

```bash
# 按算子名搜索历史问题单
rg -l "{op_name}" md_files/gitcode/issues/

# 按错误特征搜索
rg -l "AbstractProblem\|DeadNode" md_files/gitcode/issues/
rg -l "allclose\|精度" md_files/gitcode/issues/

# 查看 case_index.md 中的分类索引
# references/case_index.md 包含 100 个 gitcode 问题单的结构化索引
```

如果找到类似问题，直接参考其定界结果和修复方案。

#### 2.2 初步定界

**快速定界决策树**:

| 错误关键词 | 初步定界 |
|-----------|--------|
| allclose / precision / NaN / Inf | 精度/数值 → kernel 或 benchmark |
| takes N arguments / TypeError / unsupported operand | API/签名 → Python 接口或 YAML |
| shape / broadcast / AbstractProblem / Invalid abstract | Shape 推导 → Infer 实现 |
| DeadNode / FakeBprop / keyword_arg / control_node_parser | 编译器/IR → frontend pass |
| segmentation fault / core dump / Error building | Kernel 实现 → 设备 kernel |
| grad_cmp / GradOf / 梯度为零或NaN | 反向传播 → bprop 注册 |
| device address / output addr / module not callable | 运行时 → runtime |

#### 2.3 误导模式检查 ⚠️

**关键步骤**：初步定界后，必须检查是否匹配误导模式。

读取 `references/misleading_patterns.md` 获取完整的误导模式库。

**快速检查清单**：

```
1. 梯度为零？ → 检查 M-001 (可能是 bprop Select，而非 kernel 精度)
2. AbstractProblem？ → 检查 M-002 (可能是编译器 pass，而非 Shape 推导)
3. 小幅精度偏差 (< 1e-3)？ → 检查 M-003 (可能是 CANN/基准版本变更)
4. DID NOT RAISE？ → 检查 M-004 (可能是校验被绕过)
5. 偶现 core dump？ → 检查 M-005 (可能是多线程竞态)
6. 导入错误？ → 检查 M-006 (可能是模块重构)
7. 仅特定平台？ → 检查 M-007 (可能是平台差异触发不同路径)
8. 编译失败？ → 检查 M-008 (可能是 CANN 版本兼容性)
9. scalar type invalid？ → 检查 M-009 (可能是类型覆盖不全)
10. keyword_arg 错误？ → 检查 M-010 (可能是 Morph 时序问题)
11. complex64 NaN 但 complex128 正常？ → 检查 M-011 (可能是 aclnn float 精度缺陷)
12. 取整算子返回 2.14748e+09？ → 检查 M-012 (aclnn 内部 float→int32 溢出，平台特有)
13. sign/特殊值函数仅特定 dtype 返回 NaN？ → 检查 M-013 (aclnn 对特定 dtype 的 NaN 处理不同)
14. grad_cmp bfloat16 失败但 forward 正常？ → 检查 M-014 (测试基准 torch.tensor(int) 导致 backward 路径不对齐)
```

**如果匹配误导模式**：
1. 阅读该模式的"验证实验"章节
2. 执行验证实验确认实际根因
3. 如果验证通过，按"正确定界"路径处理
4. 如果验证失败，继续初步定界

**常见误导案例**：
- `allclose` 失败 + 梯度为零 → 实际是 bprop 缺陷 (M-001, CS-001)
- `AbstractProblem` → 实际是编译器 pass 问题 (M-002, CS-009)
- 精度不一致 (< 1e-3) → 实际是 CANN/基准版本变更 (M-003, CS-002/CS-005)

#### 2.4 对比实验验证

如果错误信息不足以直接定界，或误导模式检查不确定，通过对比实验缩小范围：

```python
# 1. 模式对比
context.set_context(mode=context.GRAPH_MODE)  # vs PYNATIVE_MODE

# 2. 后端对比
context.set_context(device_target="Ascend")  # vs "CPU"

# 3. dtype 对比
input_fp32 = Tensor(data, dtype=mindspore.float32)  # vs float16

# 4. shape 对比
static_shape = (2, 3, 4)  # vs dynamic_shape with -1
```

根据实验结果调整定界。

#### 2.5 最终定界

综合以上步骤，给出最终定界结果：

```
定界结果: [组件层] - [具体组件]
判断依据:
1. [依据 1]
2. [依据 2]
3. [依据 3]
```

### Step 3: 定位

读取 `references/architecture.md` 获取源码导航信息，快速定位到具体的源码文件。

**先查算子画像**:

如果算子是高频问题算子，先查 `references/operator_profiles.md` 获取已知问题和源码路径：

```bash
# 检查算子是否在画像中
grep -l "{op_name}" references/operator_profiles.md
```

**按组件定位**:

给定算子名 `OpName`，用以下命令定位各层代码:

```bash
# YAML 定义
rg -l "^{op_name}:" mindspore/mindspore/ops/op_def/yaml/

# Infer 实现
rg -l "class {OpName}FuncImpl" mindspore/mindspore/ops/infer/

# Kernel 注册
rg "FACTORY_REG.*{OpName}" mindspore/mindspore/ops/kernel/

# Bprop 注册
rg 'REG_BPROP_BUILDER\("{OpName}"\)' mindspore/mindspore/ccsrc/frontend/expander/

# Python API
rg "def tensor_{op_name}" mindspore/mindspore/python/mindspore/ops/

# 综合搜索 (所有层)
rg -l "{OpName}" mindspore/mindspore/ops/ mindspore/mindspore/ccsrc/frontend/expander/
```

阅读定位到的源码，分析根因。

### Step 4: 修复

读取 `references/fix_patterns.md` 参考同类修复模式。

**先查同类历史案例**:

```bash
# 在 case_studies.md 中查找同类案例
grep -A 10 "{error_keyword}" references/case_studies.md

# 在 gitcode 问题单中搜索同类根因
rg -l "Appearance & Root Cause" md_files/gitcode/issues/ | \
  xargs grep -l "{keyword}"
```

修复原则：
- 最小化变更，只修改必要的代码
- 不破坏兼容性
- 考虑 CPU/GPU/Ascend 三个后端
- 考虑 Graph 和 PyNative 两种模式
- 遵循 MindSpore 编码规范

生成修复代码后，先自验：运行复现脚本确认问题不再出现。

### Step 5: 回归验证

1. 运行原始复现脚本验证修复
2. 运行该算子相关的测试用例
3. 检查修改是否引入新问题

```bash
# 算子级测试
pytest tests/st/ops/test_{op_name}.py -v

# 关联模块测试
pytest tests/st/ops/ -k "{keyword}" -v
```

### Step 6: 测试补充

读取 `references/testing_guide.md` 获取测试框架使用指南。

为本次修复的 bug 补充测试用例，确保覆盖：
- 原始 bug 的复现场景
- 相关边界条件
- 多 dtype (fp16/fp32/fp64/int32/bool)
- 多设备 (CPU/GPU/Ascend)
- 多模式 (Graph/PyNative)

## 专项场景

### ACLNN 算子适配

当处理 Ascend 上的 ACLNN 相关问题，或需要新增/修改 ACLNN 算子时，读取 `references/aclnn_guide.md`。

### 诊断工作流详解

需要更详细的诊断步骤和调试工具时，读取 `references/diagnostic_workflow.md`。

### 查找历史类似问题

可以在 `md_files/` 目录中搜索历史类似问题：

```bash
# 按算子名搜索 (在问题单中)
rg -l "{op_name}" md_files/gitcode/ md_files/gitee/

# 按错误特征搜索
rg -l "AbstractProblem" md_files/gitcode/
rg -l "精度" md_files/gitcode/

# 按根因关键词搜索 (仅 gitcode 有结构化根因)
rg -l "Appearance & Root Cause" md_files/gitcode/issues/ | \
  xargs grep -l "{keyword}"

# 按状态过滤 (DONE = 已解决)
rg -l "Issue 状态" md_files/gitcode/issues/ | \
  xargs grep -l "DONE"

# 按引入类型搜索
rg "引入类型：CANN升级" md_files/gitcode/issues/
rg "引入类型：特性合入引入" md_files/gitcode/issues/

# 按算子名搜索 (在源码中)
rg -l "{OpName}" mindspore/mindspore/ops/op_def/
rg -l "{OpName}" mindspore/mindspore/ops/kernel/
```

**结构化案例索引**: `references/case_index.md` 包含 100 个 gitcode 问题单的分类索引表，可快速定位同类问题。

**深度案例研究**: `references/case_studies.md` 包含 17 个代表性案例的完整分析，覆盖全部 8 个问题分类。

## 输出格式

每次问题处理完成后，输出根因分析报告：

```
## 问题现象
{简述问题表现}

## 定界结果
{组件层}: {具体组件}

## 根因分析
{详细描述根因}

## 修复方案
{描述修复方案和修改的文件}

## 影响范围
- 设备: {Ascend / CPU / GPU}
- 模式: {Graph / PyNative}

## 回归验证
- 复现脚本: PASS
- 算子测试: PASS

## 补充测试
{新增的测试用例描述}
```

## 持续学习

`md_files/` 目录会持续补充新的问题单。每当分析新问题时，注意归纳新的问题模式：
- 新的错误特征和定界依据
- 新的修复模式
- 新的测试技巧

这些经验会帮助你更快地处理后续类似问题。

## 经验固化

每次解决新问题后，检查是否需要更新以下文档：

### 检查清单

1. **`references/case_studies.md`**: 是否是新的代表性案例？
   - 覆盖了新的问题分类或子类型？
   - 有误导性定界过程值得记录？

2. **`references/fix_patterns.md`**: 是否有新的修复模式？
   - 修复方式是否与现有模式不同？
   - 是否需要添加关联 Issue？

3. **`references/operator_profiles.md`**: 是否需要更新算子画像？
   - 发现了新的已知问题？
   - 找到了更准确的源码路径？

4. **`references/issue_patterns.md`**: 是否需要更新决策树？
   - 发现了新的误导性关键词？
   - 需要添加新的二级分支？

5. **`references/case_index.md`**: 是否需要更新索引？
   - 新问题单已加入 `md_files/gitcode/issues/`？
   - 运行 `python3 scripts/extract_cases.py` 重新提取

### 更新命令

```bash
# 重新提取 gitcode 案例
python3 scripts/extract_cases.py

# 重新生成案例索引
python3 -c "
import json
# ... 参考 scripts/extract_cases.py 中的 main() 逻辑
"
```

## 部署模式

本 skill 的 6 步工作流对执行环境有不同要求：

| 步骤 | 环境要求 |
|------|---------|
| Step 1-2 问题分析/定界 | 仅需知识库，无硬件依赖 |
| Step 3-4 定位/修复 | 需要 MindSpore 源码仓库 |
| Step 5-6 验证/测试 | 需要编译环境 + CANN + Ascend 硬件 |

### 推荐：远程服务器执行模式

将 Claude Code 和本 skill 直接部署到 Ascend 服务器上运行，所有步骤在同一台机器上本地执行。

**优势**：
- 零延迟：文件操作和编译测试都是本地 I/O
- 零开发成本：无需额外工具，Claude Code 原生能力即可覆盖全流程
- 高可靠性：不依赖网络传输，不会因连接中断导致操作失败

**部署方法**：

```bash
# 方式一：使用同步脚本（从本地推送到服务器）
bash scripts/sync-to-server.sh user@server

# 方式二：在服务器上直接克隆
git clone <repo-url> ~/mindspore-ops-debugger
```

详细部署步骤参见 `docs/remote-execution-guide.md`。

### 本地 SSH 远程操作模式

Claude Code 在本地运行，通过 `ssh` 命令操作远程 Ascend 服务器。适用于本地开发、审查 PR diff 并在远程验证的场景。

#### 标准工作流

```bash
# 1. 上传 diff 文件到服务器
scp /path/to/patch.diff user@server:/home/work/mindspore/patch.diff

# 2. 检查补丁是否能干净应用
ssh user@server "cd /home/work/mindspore && git apply --check patch.diff"

# 3. 应用补丁
ssh user@server "cd /home/work/mindspore && git apply patch.diff && git diff --stat"

# 4. 增量编译（仅修改了 Python 文件时可跳过）
ssh user@server "cd /home/work && source env_ms.sh mindspore/ && cd mindspore && bash build.sh -e ascend -V 910b -j64 -S on"

# 5. 运行测试
ssh user@server "cd /home/work && source env_ms.sh mindspore/ && cd mindspore && pytest tests/st/ops/test_func_xxx.py -v"
```

#### 关键注意事项

**必须 source env_ms.sh**：每次 SSH 执行 Python/pytest 命令前，都必须先 `source env_ms.sh mindspore/`，否则 `PYTHONPATH` 未设置，`import mindspore` 会失败或加载系统安装版本而非编译版本。

```bash
# 错误：直接运行
ssh user@server "cd /home/work/mindspore && pytest tests/st/ops/test_xxx.py"

# 正确：先 source 环境
ssh user@server "cd /home/work && source env_ms.sh mindspore/ && cd mindspore && pytest tests/st/ops/test_xxx.py"
```

**纯 Python 修改无需重新编译**：如果补丁只修改了 `.py` 文件（如 `ops/function/math_func.py`、`tests/st/ops/` 下的测试文件），无需重新编译，直接运行测试即可。只有修改了 C++ 源码（`ccsrc/`、`core/`）才需要增量编译。

**后台编译监控**：编译耗时较长时，用后台任务监控进程结束：

```bash
# 监控编译进程（替换 PID 为实际编译进程 ID）
ssh user@server "while ps -p {PID} > /dev/null 2>&1; do sleep 30; echo 'building...'; done; echo 'done'"
```

### 未来：Service-Client 模式（规划中）

Claude Code 在本地运行，通过 SSH/MCP 工具操作远程服务器。适用于需要同时管理多台服务器或偏好本地开发环境的场景。该模式需要开发 MCP 远程工具，目前尚未实现。
