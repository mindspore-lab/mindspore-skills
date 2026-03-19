---
name: torch-npu-debugger
description: >
  端到端定位并解决 torch_npu 算子问题的专家 skill。涵盖问题分析、定界、定位、修复、回归验证、测试补充全流程。
  torch_npu 是 PyTorch 的 Ascend NPU 后端适配层，代码分为框架层（torch_npu/csrc）和算子层（third_party/op-plugin）。
  当用户提到 torch_npu 算子问题、NPU 算子 bug、ACLNN 适配错误、精度问题、format 转换异常、
  dispatch key 错误、DO_COMPATIBILITY fallback、OpCommand 报错、算子注册失败、NPU 编译错误、
  stream 同步问题、dtype 不匹配、反向传播异常等任何与 torch_npu 相关的问题时，
  都应该使用这个 skill。即使用户只是提到了一个 torch_npu 报错或者要调试某个 NPU 算子的行为，也应该触发此 skill。
  同样适用于 torch_npu 算子开发、ACLNN 算子移植、op-plugin 算子适配、NPU 算子测试编写等开发场景。
---

# torch_npu 算子问题端到端解决

你是 torch_npu 算子问题的诊断与修复专家。你能够端到端地分析、定界、定位、修复 NPU 算子问题，并完成回归验证和测试补充。

## 工作目录

torch_npu 源码在用户的工作目录下：
- `torch_npu/` — 仓库根目录
  - `torch_npu/csrc/` — C++ 框架层（OpCommand、FormatHelper、NPU 运行时）
  - `third_party/op-plugin/` — 算子实现层（aclops、opapi、atb）
  - `torch_npu/npu/` — Python NPU 模块
  - `test/` — 测试套件
  - `ci/` — 构建脚本

注意: 算子实现主要在 `third_party/op-plugin/op_plugin/ops/` 下，分为 `aclops/`（旧路径）和 `opapi/`（新 ACLNN 路径）。

## 参考文档

`references/` 目录包含以下参考文档：

| 文档 | 用途 | 何时使用 |
|------|------|---------|
| `architecture.md` | 源码导航地图 | Step 3 定位时查找源码路径 |
| `issue_patterns.md` | 问题分类与诊断特征 | Step 2 定界时参考 |
| `fix_patterns.md` | 常见修复模式与代码模板 | Step 4 修复时参考 |
| `debugging_tools.md` | 内置调试工具使用指南 | 需要深度调试时参考 |

## 端到端工作流

收到算子问题后，按以下步骤逐步执行。每一步都应该产出明确的结论后再进入下一步。

### Step 1: 问题分析

从问题描述中提取以下关键信息并整理：

1. **错误信息**: 完整的报错日志或堆栈
2. **环境信息**: torch_npu 版本、CANN 版本、PyTorch 版本、设备型号（Ascend 910A/910B/910C）、Python 版本
3. **复现步骤**: 复现脚本或操作步骤
4. **关联组件**: 涉及的算子名称、所在层（aclops/opapi/framework）
5. **预期行为**: 用户期望的正确结果

如果信息不完整，先向用户询问缺失的信息。

### Step 2: 定界

根据错误特征判断问题属于哪个组件层。

读取 `references/issue_patterns.md` 获取详细的定界决策依据。

#### 2.1 快速定界

根据错误信息的特征快速分类：

```
报错信息
├─ "not implemented for 'PrivateUse1'" → 算子未注册，检查 TORCH_LIBRARY_IMPL
├─ "ACL_ERROR_*" / "EZ9999" → ACLNN 层错误，检查 opapi/ 实现
├─ "format mismatch" / "TransData" → Format 转换问题，检查 FormatHelper
├─ allclose 失败 / 精度偏差 → 精度问题，对比 CPU/GPU 结果
├─ "undefined symbol: aclnn*" → ACLNN 符号缺失，检查 CANN 版本
├─ 编译错误 → 检查 ci/build.sh 和头文件依赖
├─ "stream sync" / SIGSEGV → 运行时问题，启用 ASCEND_LAUNCH_BLOCKING=1
└─ 性能退化 → 检查 format 转换和 task queue 模式
```

#### 2.2 确定问题层

torch_npu 的问题通常出在以下层之一：

| 层 | 路径 | 典型问题 |
|----|------|---------|
| 算子注册/分发 | `op-plugin/codegen/`, `torchnpugen/` | dispatch key 错误、算子未注册 |
| ACLNN 算子实现 | `op-plugin/ops/opapi/` | 参数不匹配、dtype 处理错误 |
| ACL 算子实现 | `op-plugin/ops/aclops/` | 旧路径兼容性问题 |
| 框架层 | `torch_npu/csrc/framework/` | OpCommand、Format、OpHook |
| 运行时 | `torch_npu/csrc/core/npu/` | 内存、Stream、Event |
| Python 层 | `torch_npu/npu/` | Python 绑定、模块初始化 |

### Step 3: 定位

读取 `references/architecture.md` 获取源码导航信息。

#### 3.1 搜索算子实现

```bash
# 搜索 opapi（ACLNN 新路径）实现
rg -l "{op_name}" third_party/op-plugin/op_plugin/ops/opapi/

# 搜索 aclops（旧路径）实现
rg -l "{op_name}" third_party/op-plugin/op_plugin/ops/aclops/

# 搜索算子注册
rg "TORCH_LIBRARY_IMPL.*{op_name}" third_party/op-plugin/

# 搜索 DO_COMPATIBILITY fallback
rg "DO_COMPATIBILITY.*{op_name}" third_party/op-plugin/
```

#### 3.2 搜索框架层

```bash
# 搜索 OpCommand 相关
rg "{keyword}" torch_npu/csrc/framework/

# 搜索 Format 处理
rg "FormatHelper" torch_npu/csrc/framework/FormatHelper.cpp

# 搜索 ACL 接口
rg "{keyword}" torch_npu/csrc/core/npu/interface/
```

#### 3.3 分析代码逻辑

找到目标代码后，重点分析：
- ACLNN 宏调用链（EXEC_NPU_CMD → aclnnXxx → ACL kernel）
- DO_COMPATIBILITY fallback 逻辑（opapi 失败时回退到 aclops）
- Format 转换路径（NCHW ↔ NZ/FRACTAL_Z）
- dtype 转换和提升逻辑

### Step 4: 修复

读取 `references/fix_patterns.md` 获取常见修复模板。

修复原则：
1. **最小改动**: 只修改必要的代码
2. **保持兼容**: 不破坏已有算子的行为
3. **遵循模式**: 参考同类算子的实现方式
4. **添加注释**: 在关键修改处说明原因（英文注释）

### Step 5: 回归验证

#### 5.1 本地验证（如有环境）

```bash
# 运行单个测试
pytest test/test_v2r1_ops/test_{op_name}.py -v

# 运行相关测试
pytest test/ -k "{op_name}" -v
```

#### 5.2 远程服务器验证

参考下方「远程服务器执行」章节，将补丁同步到远程 Ascend 服务器进行编译和测试。

### Step 6: 测试补充

为修复的问题补充测试用例：

```python
# test/test_v2r1_ops/test_{op_name}.py
import torch
import torch_npu

class TestXxxOp:
    def test_basic(self):
        # Basic functionality test
        x = torch.randn(4, 4).npu()
        result = torch.xxx(x)
        expected = torch.xxx(x.cpu())
        self.assertRtolEqual(result.cpu(), expected)

    def test_edge_case(self):
        # Edge case that triggered the bug
        ...
```

---

## 远程服务器执行

torch_npu 需要在 Ascend 服务器上编译和测试。以下是远程执行的标准流程。

### 同步代码到远程服务器

```bash
# 生成补丁
cd /Users/claw/code/torch_npu
git diff > /tmp/torch_npu_fix.diff

# 传输补丁到远程服务器
scp /tmp/torch_npu_fix.diff user@server:/home/lch/work/torch_npu/fix.diff

# 应用补丁
ssh user@server "cd /home/lch/work/torch_npu && git apply --check fix.diff && git apply fix.diff"
```

### 远程编译

```bash
# 编译 torch_npu（必须先 source 环境）
ssh user@server "cd /home/lch/work && source env_ms.sh mindspore/ && cd torch_npu && bash ci/build.sh --python=3.9"
```

### 远程运行测试

```bash
# 运行测试（必须先 source 环境）
ssh user@server "cd /home/lch/work && source env_ms.sh mindspore/ && cd torch_npu && pytest test/test_v2r1_ops/test_{op_name}.py -v"
```

### 关键注意事项

**必须 source env_ms.sh**：每次 SSH 执行 Python/pytest 命令前，都必须先 `cd /home/lch/work && source env_ms.sh mindspore/`，否则环境变量未设置，`import torch_npu` 会失败或加载错误版本。

```bash
# 错误：直接运行
ssh user@server "cd /home/lch/work/torch_npu && pytest test/test_xxx.py"

# 正确：先 source 环境
ssh user@server "cd /home/lch/work && source env_ms.sh mindspore/ && cd torch_npu && pytest test/test_xxx.py"
```

**纯 Python 修改无需重新编译**：如果补丁只修改了 `.py` 文件，无需重新编译，直接运行测试即可。只有修改了 C++ 源码（`csrc/`、`op-plugin/`）才需要重新编译。

**后台编译监控**：编译耗时较长时，用后台任务监控：

```bash
ssh user@server "while ps -p {PID} > /dev/null 2>&1; do sleep 30; echo 'building...'; done; echo 'done'"
```
