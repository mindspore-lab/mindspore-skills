# MindSpore 算子问题诊断工作流

## 目录

1. [端到端流程概览](#1-端到端流程概览)
2. [Step 1: 问题分析](#2-step-1-问题分析)
3. [Step 2: 定界](#3-step-2-定界)
4. [Step 3: 定位](#4-step-3-定位)
5. [Step 4: 修复](#5-step-4-修复)
6. [Step 5: 回归验证](#6-step-5-回归验证)
7. [Step 6: 测试补充](#7-step-6-测试补充)
8. [根因分析报告模板](#8-根因分析报告模板)
9. [环境信息收集](#9-环境信息收集)
10. [调试工具与命令](#10-调试工具与命令)
11. [多步推理示例](#11-多步推理示例定界被误导后如何修正)

---

## 1. 端到端流程概览

```
问题报告 (Issue / 复现脚本)
    │
    ▼
Step 1: 问题分析
    │  ─ 提取: 环境、错误信息、复现步骤、关联组件
    ▼
Step 2: 定界
    │  ─ 判定: 属于哪个组件层 (Infer / Kernel / Bprop / API / Compiler / Runtime)
    ▼
Step 3: 定位
    │  ─ 导航: 定位到具体源码文件和函数
    │  ─ 根因: 分析代码逻辑缺陷
    ▼
Step 4: 修复
    │  ─ 编码: 修改代码
    │  ─ 自验: 本地验证修复
    ▼
Step 5: 回归验证
    │  ─ 运行: 关联测试用例
    │  ─ 确认: 问题不再复现，无副作用
    ▼
Step 6: 测试补充
    │  ─ 添加: 覆盖该 bug 的测试用例
    ▼
完成 (PR 提交)
```

---

## 2. Step 1: 问题分析

### 从问题单中提取关键信息

| 信息 | 来源 | 用途 |
|------|------|------|
| **错误信息** | 日志、堆栈 | 定界的首要依据 |
| **复现脚本** | 问题描述、附件 | 本地复现 |
| **环境信息** | 问题模板 | 确认设备、版本 |
| **关联标签** | B-SIG-OPS 等 | 辅助定界 |
| **关联 PR** | 评论区 | 了解历史修复 |
| **评论讨论** | 评论区 | 定位过程记录 |

### 错误信息分类

快速扫描错误信息中的关键词进行初步分类：

```python
keywords_to_category = {
    # 精度问题
    ("allclose", "precision", "NaN", "Inf", "rtol", "atol",
     "data_expected_std", "data_me_error", "loss_count"): "精度/数值",

    # API 问题
    ("takes", "positional arguments", "unsupported operand",
     "unexpected keyword"): "API/签名",

    # Shape 问题
    ("broadcast", "shape", "AbstractProblem", "Invalid abstract",
     "product of shape"): "Shape推导",

    # 编译器问题
    ("DeadNode", "FakeBprop", "keyword_arg", "control_node_parser",
     "make_keyword_arg"): "编译器/IR",

    # Kernel 问题
    ("segmentation fault", "core dump", "Error building",
     "FAILED:", "std::_Rb_tree"): "Kernel实现",

    # 梯度问题
    ("grad_cmp", "GradOfAllInputs", "GradOfFirstInput",
     "backward"): "反向传播",

    # 运行时问题
    ("device address", "output addr", "module not callable"): "运行时",
}
```

### 环境信息标准化

```
- MindSpore 版本: (commit hash)
- CANN 版本:
- Python 版本:
- 设备: Ascend 910A / 910B / CPU / GPU
- 操作系统:
- 执行模式: Graph / PyNative
- 编译优化: O0 / O1 / O2
```

---

## 3. Step 2: 定界

### 定界策略

通过系统性排除确定问题所在层：

#### 策略 1: 错误信息直接定界

很多错误信息直接指向组件层，参考 `issue_patterns.md` 的决策树。

#### 策略 2: 对比实验定界

| 对比实验 | 目的 | 结论 |
|---------|------|------|
| Graph vs PyNative | 区分编译器问题 vs 运行时问题 | Graph 出错 → 编译器/IR; PyNative 出错 → 运行时/kernel |
| Ascend vs CPU | 区分后端问题 vs 前端问题 | 仅 Ascend 出错 → ACLNN kernel; CPU 也出错 → 通用逻辑 |
| fp32 vs fp16 | 区分精度问题 vs 逻辑问题 | 仅 fp16 出错 → 精度; 两者都出错 → 逻辑缺陷 |
| 静态 shape vs 动态 shape | 区分动态 shape 支持 | 仅动态出错 → Infer 或 Kernel 的动态 shape 处理 |
| MS vs TF/Torch | 确认预期行为 | MS 独有 → MS bug; 框架一致 → 用户理解有误 |
| 有 Morph vs 无 Morph | 区分编译器变换问题 | 有 Morph 出错 → 编译器 Morph pass |

#### 策略 3: IR 分析定界

```python
import mindspore as ms
ms.set_context(save_graphs=True, save_graphs_path='./ir_debug')
```

关键 IR 文件:
- `*_validate.ir` — 验证阶段的 IR
- `*_execute_*.ir` — 执行前的最终 IR (含反向)
- `*_optimize.ir` — 优化后的 IR

检查点:
- 搜索 `DeadNode`、`ValueProblem`
- 检查算子输入/输出 shape 与 type 是否正确
- 确认 bprop 展开是否正确

#### 策略 4: 堆栈分析定界

C++ 堆栈关键帧映射:

| 堆栈关键帧 | 组件 |
|-----------|------|
| `ops_func_impl` / `InferShape` / `InferType` | Shape/Type 推导 |
| `NativeCpuKernelMod` / `NativeGpuKernelMod` | CPU/GPU Kernel |
| `AclnnKernelMod` / `LAUNCH_ACLNN` | Ascend ACLNN Kernel |
| `BpropBuilder` / `grad_ops` | 反向传播 |
| `control_node_parser` / `graph_scheduler` | 运行时调度 |
| `FrontendOptimize` / `AbstractSpecialize` | 编译器前端 |
| `PyBoost` / `OpRunner` | PyBoost 执行 |

---

## 4. Step 3: 定位

### 按组件定位策略

#### 精度问题定位

1. **确认算子**: 从错误日志中提取出错的算子名
2. **定位 Kernel**: 参考 `architecture.md` 的导航表定位对应设备的 kernel 源码
3. **对比基准**: 对比 PyTorch 的实现逻辑
4. **检查 dtype 传递**: 从 Python API → op_def → kernel 整条链路
5. **检查 CANN 接口**: 如为 ACLNN，检查 aclnn 接口文档

```bash
# 定位某算子的 Ascend kernel (路径前缀 mindspore/mindspore/)
rg "LAUNCH_ACLNN.*aclnn{OpName}" mindspore/mindspore/ops/kernel/ascend/
rg "FACTORY_REG.*{OpName}" mindspore/mindspore/ops/kernel/ascend/
```

#### Shape 推导问题定位

1. **定位 Infer 实现**:
```bash
rg -l "class {OpName}FuncImpl" mindspore/mindspore/ops/infer/
rg -l "class {OpName}FuncImpl" mindspore/mindspore/core/ops/
```
2. **检查 InferShape 逻辑**: 维度校验、广播规则、动态 shape 处理
3. **检查 InferType 逻辑**: dtype 提升规则

#### API 问题定位

1. **检查 YAML 定义**:
```bash
rg -l "{op_name}" mindspore/mindspore/ops/op_def/yaml/
rg -l "{op_name}" mindspore/mindspore/ops/api_def/
```
2. **检查 Python 函数**: `mindspore/mindspore/python/mindspore/ops/tensor_method.py`, `functional_overload.py`
3. **对比文档**: `mindspore/mindspore/ops/api_def/function_doc/`, `method_doc/`

#### Kernel 问题定位

1. **从堆栈定位文件和行号**
2. **检查 Init/Resize/Launch 三个阶段**
3. **检查线程安全**: 全局变量、static 变量
4. **检查 CANN 兼容**: `#ifdef` 宏保护

#### Bprop 问题定位

1. **定位 bprop 注册**:
```bash
rg 'REG_BPROP_BUILDER\("{OpName}"\)' mindspore/mindspore/ccsrc/frontend/expander/
```
2. **检查 SetUnusedInputs**: 确认没有误标记反向需要的输入
3. **检查反向 IR**: 第 13 步 `execute_*.ir`
4. **检查 Python 实验性 bprop**: `ops/_grad_experimental/`

#### 编译器问题定位

1. **导出完整 IR**: `save_graphs=True`
2. **定位问题 pass**: 从 IR 中异常节点倒推触发的 pass
3. **检查 frontend 代码**: `ccsrc/frontend/`
4. **检查 control flow**: `ccsrc/frontend/operator/`, `control_node_parser.cc`

---

## 5. Step 4: 修复

### 修复原则

1. **最小化变更**: 只修改必要的代码
2. **不破坏兼容性**: 检查修改是否影响其他算子
3. **参考同类修复**: 查看 `fix_patterns.md` 中的模式
4. **考虑多设备**: 确认修复在 CPU/GPU/Ascend 上都正确
5. **考虑多模式**: Graph 和 PyNative 两种模式

### 修复后自验

```bash
# 运行复现脚本
python repro_script.py

# 运行算子级单元测试
pytest tests/st/ops/test_{op_name}.py -v

# 精度对比
python -c "
import mindspore as ms
import numpy as np
# ... 精度对比代码
"
```

---

## 6. Step 5: 回归验证

### 验证范围

| 级别 | 范围 | 命令 |
|------|------|------|
| 最小 | 原始复现脚本 | `python repro_script.py` |
| 算子级 | 该算子的所有测试 | `pytest tests/st/ops/test_{op_name}.py` |
| 模块级 | 关联模块测试 | `pytest tests/st/ops/ -k "{keyword}"` |
| 系统级 | CI 全量测试 | 提交 PR 触发 CI |

### 回归报告模板

```
回归验证:
- 版本: {commit_hash}
- 环境: {device} / {cann_version}
- 步骤: {复现步骤}
- 结果: PASS / FAIL
- 截图/日志: {附件}
```

---

## 7. Step 6: 测试补充

参考 `testing_guide.md` 编写测试用例，确保覆盖：

1. **原始 bug 场景**: 直接复现原问题的测试
2. **边界条件**: 空 tensor、标量、高维 tensor
3. **dtype 覆盖**: fp16、fp32、fp64、int32、bool
4. **多设备**: CPU、GPU、Ascend
5. **多模式**: Graph、PyNative
6. **动态 shape**: 如果算子支持

---

## 8. 根因分析报告模板

问题单定位完成后，使用以下模板记录：

```markdown
## 问题现象
{描述问题表现}

## 根因分析
{描述根本原因}

## 修复方案
{描述修复方案}

## 引入分析
- 引入类型: 新功能引入 / 代码变更引入 / 环境变更引入
- 引入 PR: {PR 链接}
- 引入时间: {时间}

## 影响范围
- 影响设备: {Ascend / CPU / GPU}
- 影响模式: {Graph / PyNative}
- 影响版本: {版本范围}

## 修复 PR
{PR 链接}

## 回归验证
- 版本: {commit}
- 结果: PASS
```

---

## 9. 环境信息收集

### Python 环境

```python
import mindspore as ms
print(ms.__version__)
print(ms.run_check())

import mindspore.context as context
print(context.get_context("device_target"))
print(context.get_context("mode"))
```

### 系统环境

```bash
# CANN 版本
cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg

# 设备信息
npu-smi info

# Python 版本
python --version

# MindSpore commit
python -c "import mindspore; print(mindspore.__version__)"
```

---

## 10. 调试工具与命令

### IR 导出

```python
ms.set_context(save_graphs=True, save_graphs_path='./ir_debug')
```

### 禁止 Kernel Fallback

```bash
export MS_DISABLE_KERNEL_BACKOFF=1
```

### 日志级别

```bash
export GLOG_v=1          # INFO
export GLOG_v=0          # DEBUG
export MS_SUBMODULE_LOG_v="{SUBMODULE}=0"  # 子模块 DEBUG
```

### 精度对比

```python
import numpy as np

def compare_precision(ms_output, ref_output, rtol=1e-3, atol=1e-3):
    ms_np = ms_output.asnumpy()
    ref_np = ref_output if isinstance(ref_output, np.ndarray) else ref_output.numpy()
    is_close = np.allclose(ms_np, ref_np, rtol=rtol, atol=atol)
    if not is_close:
        diff = np.abs(ms_np - ref_np)
        print(f"Max diff: {diff.max()}")
        print(f"Mean diff: {diff.mean()}")
        print(f"Mismatched elements: {(diff > atol + rtol * np.abs(ref_np)).sum()}/{diff.size}")
    return is_close
```

### 搜索算子相关代码

```bash
# 搜索所有与某算子相关的文件
op_name="Sigmoid"
rg -l "$op_name" mindspore/mindspore/ops/
rg -l "$op_name" mindspore/mindspore/ccsrc/frontend/expander/
rg -l "$op_name" mindspore/mindspore/python/mindspore/ops/
```

---

## 11. 多步推理示例：定界被误导后如何修正

真实案例中，初始定界方向经常被错误信息误导。以下示例展示如何通过二次判断修正定界。

### 示例 A: ops.pow 精度失败 → 实际是 bprop 缺陷 (CS-001, #41932)

**初始信息**:
```
AssertionError: data_expected_std:[14.697707] data_me_error:[0.] loss:[14.697707]
```

**第一次定界（错误）**: 包含 `allclose` → 精度/数值问题 → 检查 kernel 精度

**发现矛盾**: 正向精度正常，仅反向梯度为零 → 精度问题不在 kernel

**修正定界**: 梯度为零 → 反向传播 → 检查 bprop 实现

**根因**: 反向图中 Select 操作在 GE 上导致内存踩踏，梯度被覆盖为零

**教训**: `allclose` 失败不一定是 kernel 精度问题。先区分正向/反向，再定界。

---

### 示例 B: PixelShuffle AbstractProblem → 实际是编译器 pass 缺失 (CS-009, #41973)

**初始信息**:
```
RuntimeError: Invalid abstract;AbstractProblem(Value: DeadNode, ...)
at control_node_parser.cc:362 FetchOutputSizeByNode
```

**第一次定界（错误）**: 包含 `AbstractProblem` → Shape 推导问题 → 检查 InferShape

**发现矛盾**: 静态 shape 正常，仅动态 shape 出错；InferShape 逻辑无问题

**修正定界**: 错误信息含 `DeadNode` → 编译器 IR 问题 → 导出 IR 检查

**根因**: 前端脚本变化产生 DeadNode，缺少 switch_simplify pass 清理

**教训**: `AbstractProblem` 可能是 Shape 推导，也可能是编译器 pass 遗留的 DeadNode。先导出 IR 确认。

---

### 示例 C: nn.Adam 精度不一致 → 实际是基准环境差异 (CS-002, #41934)

**初始信息**: MindSpore 与 TensorFlow 结果不一致

**第一次定界（错误）**: 精度不一致 → MindSpore kernel 精度问题 → 检查 Adam kernel

**发现矛盾**: 对比 TF 2.15 结果正常，仅 TF 2.18 不一致

**修正定界**: 框架版本差异 → 基准环境问题 → 检查 TF 代码

**根因**: TF 2.18 Dense 层未传 dtype 时自动升精度，基准代码需显式指定 dtype

**教训**: 精度不一致时，先确认基准框架版本是否变化，再排查 MindSpore 侧。

---

### 示例 D: 多线程 core dump → 双重根因 (CS-013, #41935)

**初始信息**: `SIGSEGV in std::_Rb_tree_insert_and_rebalance`

**第一次定界**: 堆栈指向 set 插入 → 线程安全问题 → 加锁

**发现矛盾**: 加锁后仍偶现，且仅在训练场景出现

**修正定界**: 训练场景误入推理路径 → optional 判断错误 → 双重根因

**根因**: ① optional 判断错误导致训练误入推理路径；② set 无锁并发写

**教训**: 偶现 core dump 可能有多个根因叠加，修复一个后需继续验证。
