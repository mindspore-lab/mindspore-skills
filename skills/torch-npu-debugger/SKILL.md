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
├─ import 失败
│  ├─ "undefined symbol" 含 cxx11 → GCC ABI 不匹配
│  ├─ `import torch_npu` / `_c10d_npu_init` 直接 SIGSEGV
│  │  ├─ 同环境旧包正常、重编包崩溃 → 优先比较二进制编译器版本
│  │  ├─ `readelf -p .comment` 显示 torch/libpython/good 包为 GCC 11.x，bad 包为 GCC 10.x
│  │  │  → Python / torch / torch_npu 工具链不兼容
│  │  └─ 服务器存在 gcc-toolset-11 但宿主机默认 gcc 为 10.x
│  │     → 可考虑使用专用的编译容器进行编译。
│  ├─ "Unsupported soc version" → torch_npu 版本过旧
│  ├─ 调用栈含 triton → 卸载 triton 或设 TORCH_DEVICE_BACKEND_AUTOLOAD=0
│  └─ _lazy_init 卡死 → 驱动异常，lspci | grep ascend
├─ 编译错误
│  ├─ "undefined reference to op_api::*" → op-plugin submodule 不对齐
│  ├─ "is not a member of OpCommand" → op-plugin 比 torch_npu 新
│  ├─ error code 500002 → GE 图编译失败，查 plog
│  ├─ "ld: cannot find -lstdc++fs" → GCC 版本/ABI 不兼容
│  └─ "ld: cannot find -ltorch_npu" → 先检查链接路径、构建产物和安装结果
├─ "not implemented for 'PrivateUse1'" → 算子未注册，检查 TORCH_LIBRARY_IMPL
├─ "ACL_ERROR_*" / "EZ9999" → ACLNN 层错误，检查 opapi/ 实现
│  ├─ 161002 + CheckAxisRange → opapi infershape 错误
│  ├─ 0x800000 MTE 越界 → 检查 gen_opapi out dtype
│  └─ CPU 正常 + ACLOP 正常 + ACLNN 异常 → 优先判定 ACLNN 与 ACLOP 语义不一致
├─ ATB "setup failed" → 开启 ASDOPS_LOG_LEVEL=INFO 查看 dtype 组合
├─ "format mismatch" / "TransData" → Format 转换问题，检查 FormatHelper
├─ allclose 失败 / 精度偏差 → 精度问题，对比 CPU/GPU 结果
│  ├─ matmul 相关 → 试 CLOSE_MATMUL_K_SHIFT=1
│  └─ 非连续 tensor → 检查 out 参数写入逻辑
├─ "undefined symbol: aclnn*" → ACLNN 符号缺失，检查 CANN 版本
├─ EJ0001 HCCL 失败 → 检查残留进程，kill 后等 10 秒
├─ "stream sync" / SIGSEGV → 运行时问题，启用 ASCEND_LAUNCH_BLOCKING=1
├─ KeyError on state_dict["state"][0] → optimizer state 为空（如 SGD momentum=0），添加非空守卫
└─ 性能退化 → 检查 format 转换、task queue、MLIR fallback
```

#### 2.2 确定问题层

torch_npu 的问题通常出在以下层之一：

| 层 | 路径 | 典型问题 |
|----|------|---------|
| 算子注册/分发 | `op-plugin/codegen/`, `torchnpugen/` | dispatch key 错误、算子未注册 |
| ACLNN 算子实现 | `op-plugin/ops/opapi/` | 参数不匹配、dtype 处理错误、infershape 错误 |
| ACL 算子实现 | `op-plugin/ops/aclops/` | 旧路径兼容性问题 |
| 自定义算子/ATB | `op-plugin/ops/atb/` | ATB dtype 约束、gen_opapi 配置错误 |
| 框架层 | `torch_npu/csrc/framework/` | OpCommand、Format、OpHook |
| 运行时 | `torch_npu/csrc/core/npu/` | 内存、Stream、Event |
| 分布式 | `torch_npu/csrc/distributed/` | HCCL 通信域、集合通信 |
| Python 层 | `torch_npu/npu/` | Python 绑定、模块初始化 |
| 编译/版本配套 | `ci/`, `CMakeLists.txt`, `setup.py` | 版本三元组不匹配、GCC 兼容性 |

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

#### 3.4 语义对齐类问题的专项检查

当问题表现为 **shape 正常性、keepdim 语义、dim 缺省语义、dtype 语义或边界输入行为** 与 PyTorch CPU/GPU 不一致时，必须补做以下对比：

1. **先确认 PyTorch 语义**
   - 直接在 CPU 上构造最小样例验证期望输出
   - 必要时同时对比 GPU（若环境可用）
2. **区分 ACLNN 与 ACLOP 路径**
   - 分别检查 `opapi/` 与 `aclops/` 的同名算子实现
   - 重点比较：输入 reshape、real dim、keepdim、output_size 推导、dtype 转换
3. **建立三方对比结论**
   - CPU 正常 + ACLOP 正常 + ACLNN 异常 → 优先判定为 ACLNN 实现缺陷或语义不一致
   - CPU 正常 + ACLNN/ACLOP 都异常 → 优先判定为 torch_npu 适配层公共逻辑问题
   - CPU/上游本身异常 → 不能直接归因 torch_npu
4. **对 reduce 类算子额外检查**
   - `dim=None` 时是否先扁平化执行
   - 扁平化后是否错误沿用了扁平 tensor 的输出 shape 推导
   - `keepdim=True` 时是否应保留原始 rank

这类问题不要只盯报错本身；要重点核对“语义是否与 PyTorch 对齐”，以及“ACLNN 与 ACLOP 是否一致”。

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

### Step 7: 交付件归档

问题定位完成后，除了代码修改和验证外，还应按需保存以下交付件，便于复盘、提单和后续接手：

1. **定位总结（必选）**
   - 使用中文书写
   - 至少包含：问题现象、复现方式、定界结论、根因分析、修复/规避方案、验证结果、最终结论
2. **代码 diff（推荐）**
   - 保存本次改动的补丁或 diff 文件
   - 便于远端同步、代码复查、后续 cherry-pick 或回溯
3. **问题单材料（当发现底层缺陷时必选）**
   - 若最终结论指向 ACLNN / CANN / 驱动等底层问题，应额外整理：
     - 详细问题单模板
     - 简版摘要（适合 GitCode / Jira）
   - 内容应强调对比证据：CPU、ACLOP、ACLNN 三方行为是否一致
   - **必须详细描述复现过程**，至少写清：
     - 复现环境（Python / PyTorch / torch_npu / CANN / 设备型号）
     - 复现命令或执行步骤
     - 最小复现脚本
     - 实际结果 / 报错
     - 期望结果
   - 如果当前版本已加 fallback 或 workaround，需额外说明：
     - 用户态默认是否已经被规避
     - 如需复现底层原始问题，是否需要临时关闭 fallback 或强制走 ACLNN 路径
   - **建议按以下一级标题组织详细问题单**：
     - 环境信息
     - 复现步骤
     - 最小复现脚本
     - 实际结果
     - 期望结果
     - 对比结论
     - torch_npu 当前规避方式
4. **验证证据（推荐）**
   - 保存关键命令输出、报错片段、shape 对比结果、wheel 安装路径或运行时加载路径
   - 如果自动测试受环境阻塞，也要明确记录“阻塞点在环境，不在算子逻辑”

交付件建议默认保存在用户提供的 issue 目录或同级调试目录下，并使用清晰命名，例如：

- `定位总结.md`
- `code_changes.diff`
- `aclnn_xxx_issue_template.md`
- `aclnn_xxx_issue_summary_short.md`

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
# 编译 torch_npu（必须先 source 环境，并与运行时 Python 版本保持一致）
ssh user@server "cd /home/lch/work && source env_ms.sh mindspore/ && cd torch_npu && bash ci/build.sh --python=<python-version>"
```

### 远程运行测试

```bash
# 运行测试（必须先 source 环境）
ssh user@server "cd /home/lch/work && source env_ms.sh mindspore/ && cd torch_npu && pytest test/test_v2r1_ops/test_{op_name}.py -v"
```

### 关键注意事项

**必须 source env_ms.sh**：每次 SSH 执行 Python/pytest 命令前，都必须先 `cd /home/lch/work && source env_ms.sh mindspore/`，否则环境变量未设置，`import torch_npu` 会失败或加载错误版本。

**遇到 import 阶段 native crash 时优先排查工具链**：若 `import torch_npu`、`_c10d_npu_init` 或 pybind 绑定阶段直接 `SIGSEGV`，且同一 Python 环境下旧包正常、重编包崩溃，不要先改源码。优先比较 `libpython`、`torch/_C`、good/bad `torch_npu` 的 `.comment`。其中 `<good-torch-npu-path>` / `<bad-torch-npu-path>` 应替换为现场实际的好/坏包路径（例如本次问题中的 `torch_npu--` 与 `torch_npu`）：

```bash
readelf -p .comment /root/miniconda3/envs/lch_py310/lib/libpython3.10.so
readelf -p .comment /root/miniconda3/envs/lch_py310/lib/python3.10/site-packages/torch/_C.cpython-310-aarch64-linux-gnu.so
readelf -p .comment <good-torch-npu-path>/_C.cpython-310-aarch64-linux-gnu.so
readelf -p .comment <bad-torch-npu-path>/_C.cpython-310-aarch64-linux-gnu.so
```

若 Python / torch / good 包为 GCC 11.x，而 bad 包为 GCC 10.x，应优先判定为构建工具链不兼容。

**优先考虑回到原容器内做 clean rebuild，再验证是否消除崩溃。**

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
