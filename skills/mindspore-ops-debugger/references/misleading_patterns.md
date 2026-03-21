# MindSpore 算子问题误导性模式库

当错误信息的表面现象与实际根因不一致时，容易导致定界失误。本文档记录已识别的误导性模式，帮助快速识别并避免误判。

---

## 使用方法

在 Step 2 定界后，检查是否匹配以下误导模式：
1. 如果匹配，执行"验证实验"确认实际根因
2. 如果验证通过，按"正确定界"路径处理
3. 记录误判案例，持续完善此库

---

## M-001: allclose 失败 + 梯度为零

### 表面现象
```
AssertionError in allclose_nparray
data_expected_std:[14.697707]
data_me_error:[0.]
```

### 常见误判
定界到 **Kernel 精度问题** 或 **数值计算问题**

### 正确定界
**反向传播 (bprop)** - 反向图结构问题

### 判断依据
1. **前向精度正常**，仅反向异常 → bprop 而非 kernel
2. **梯度为零**而非小幅偏差 → 图结构问题而非精度累积
3. **部分梯度为零**，其他正常 → 特定分支的图结构缺陷

### 验证实验
```python
# 1. 检查前向精度
fact.forward_mindspore_impl()  # 应该 PASS

# 2. 导出反向 IR
context.set_context(save_graphs=True)
# 检查 IR 中是否有 Select / DeadNode

# 3. 对比 PyTorch 反向
# 如果 PyTorch 梯度正常，确认是 MindSpore bprop 问题
```

### 根因类型
- **Select 操作在 GE 上内存踩踏** (最常见)
- **DeadNode 残留导致梯度传播中断**
- **SetUnusedInputs 过度释放**

### 修复模式
- 消除 Select，改用 Mul + Cast
- 新增编译器 pass 清理 DeadNode
- 调整 SetUnusedInputs 标记

### 参考案例
- **CS-001 (#41932)**: ops.pow 反向梯度为零 - Select 在 GE 上内存踩踏

---

## M-002: AbstractProblem / Invalid abstract

### 表面现象
```
ValueError: For 'PixelShuffle', condition.shape and input.shape need to broadcast
RuntimeError: Invalid abstract; AbstractProblem
```

### 常见误判
定界到 **Shape 推导问题** 或 **Infer 实现缺陷**

### 正确定界
**编译器/IR 问题** - 编译器 pass 缺失导致 DeadNode 残留

### 判断依据
1. **错误信息提到 shape/broadcast**，但实际是 IR 节点异常
2. **仅在特定优化级别出现** (O1/O2) → 编译器 pass 问题
3. **静态 shape 也报错** → 不是动态 shape 推导问题

### 验证实验
```python
# 1. 导出 IR
context.set_context(save_graphs=True, save_graphs_path="./ir_dump")

# 2. 检查 execute_*.ir 文件
grep -n "DeadNode\|ValueProblem\|AbstractProblem" ir_dump/*.ir

# 3. 如果发现 DeadNode，确认是编译器问题
```

### 根因类型
- **编译器 pass 缺失**，未清理 DeadNode
- **pass 执行顺序错误**，导致 IR 节点残留
- **控制流优化不完整**

### 修复模式
- 新增 switch_simplify / dead_node_eliminate pass
- 调整 pass 执行顺序
- 在 Infer 阶段提前检测并报错

### 参考案例
- **CS-009 (#41973)**: PixelShuffle AbstractProblem - 需新增 switch_simplify pass

---

## M-003: 精度不一致 (小幅偏差)

### 表面现象
```
allclose_nparray 失败
data_expected: [1.234567]
data_me: [1.234560]
偏差 < 1e-3
```

### 常见误判
定界到 **Kernel 精度问题** 或 **算子实现缺陷**

### 正确定界
**环境问题** - 基准框架版本差异 或 **CANN 版本变更**

### 判断依据
1. **偏差很小** (< 1e-3) → 不是逻辑错误
2. **最近 CANN 升级** → CANN 内核优化导致
3. **基准框架版本变化** → dtype 自动提升行为变化

### 验证实验
```python
# 1. 检查 CANN 版本
cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg

# 2. 对比不同 CANN 版本结果
# 如果结果一致变化，确认是 CANN 变更

# 3. 检查基准框架版本
import tensorflow as tf
print(tf.__version__)  # 2.15 vs 2.18 行为可能不同
```

### 根因类型
- **CANN matmul/conv 内核优化**
- **TensorFlow/PyTorch 版本升级改变 dtype 处理**
- **累加顺序变化导致浮点误差**

### 修复模式
- 重新基线 (更新 baseline 数据)
- 调整 tolerance (rtol/atol)
- 在基准代码中显式指定 dtype

### 参考案例
- **CS-002 (#41934)**: nn.Adam 与 TF 不一致 - TF 2.18 dtype 自动提升
- **CS-005 (#41977)**: DeepSeek loss 偏差 - CANN matmul 变更

---

## M-004: TypeError / 参数错误但未抛出

### 表面现象
```
pytest 报错: DID NOT RAISE <class 'TypeError'>
测试期望抛出异常，但实际未抛出
```

### 常见误判
定界到 **测试用例问题** 或 **API 文档错误**

### 正确定界
**API/签名问题** - 参数校验被绕过

### 判断依据
1. **Graph 模式正常抛出**，PyNative 模式不抛出 → 校验路径不同
2. **自动类型转换** (list→tuple) 绕过了校验
3. **deprecated 分支**意外放宽了约束

### 验证实验
```python
# 1. 对比 Graph vs PyNative
context.set_context(mode=context.GRAPH_MODE)  # 应该抛出异常
context.set_context(mode=context.PYNATIVE_MODE)  # 未抛出

# 2. 检查 ConvertSequence 是否自动转换
# 在 PyNative 分支打断点，查看参数类型

# 3. 检查是否走了 deprecated 分支
```

### 根因类型
- **ConvertSequence 自动转换 list→tuple**
- **PyNative 分支缺少参数校验**
- **deprecated 接口约束放宽**

### 修复模式
- 在 PyNative 分支补充校验逻辑
- 禁止 ConvertSequence 在特定场景自动转换
- 收紧 deprecated 分支的约束

### 参考案例
- **CS-006 (#41971)**: scatternd PyNative 参数校验丢失 - ConvertSequence 绕过

---

## M-005: 偶现 core dump

### 表面现象
```
segmentation fault (core dumped)
堆栈: std::_Rb_tree / std::map / std::set
偶现，无法稳定复现
```

### 常见误判
定界到 **Kernel 实现缺陷** 或 **内存泄漏**

### 正确定界
**Kernel 实现 - 多线程竞态** 或 **运行时 - 线程安全问题**

### 判断依据
1. **偶现** → 多线程竞态条件
2. **堆栈中有 STL 容器** (map/set/vector) → 并发写入
3. **仅在多卡或高并发场景出现** → 通信/并行相关

### 验证实验
```python
# 1. 单线程测试
export OMP_NUM_THREADS=1
# 如果不再出现，确认是多线程问题

# 2. 使用 ThreadSanitizer
export TSAN_OPTIONS="halt_on_error=1"
# 检测数据竞争

# 3. 检查全局变量访问
# 搜索代码中的 static / global 变量
```

### 根因类型
- **全局容器并发写入** (无锁保护)
- **环境变量多线程读写冲突**
- **通信域异步初始化竞态**

### 修复模式
- 添加 std::mutex 保护全局状态
- 使用线程安全的容器 (concurrent_map)
- 同步通信域初始化

### 参考案例
- **CS-009 (#41952)**: 多线程 core dump - 环境变量并发访问
- **CS-013 (#41935)**: optional 误用 + set 无锁并发写入

---

## M-006: module not callable / 导入错误

### 表面现象
```
TypeError: 'module' object is not callable
ModuleNotFoundError: No module named 'xxx'
```

### 常见误判
定界到 **环境问题** 或 **安装问题**

### 正确定界
**运行时问题** - 模块重构后 import 路径变更

### 判断依据
1. **最近有模块重构** → import 路径变更
2. **仅在特定场景出现** (PyNative/lazy_inline) → 特定代码路径
3. **其他算子正常** → 不是全局环境问题

### 验证实验
```python
# 1. 检查 import 路径
import mindspore
print(mindspore.__file__)

# 2. 搜索旧的 import 路径
rg "from mindspore.ops.composite import" mindspore/

# 3. 检查模块是否存在
python3 -c "from mindspore.ops.composite import lazy_inline"
```

### 根因类型
- **模块重构后 import 路径变更**
- **循环导入**
- **延迟导入失败**

### 修复模式
- 更新 import 路径
- 全局搜索并替换旧路径
- 调整模块导入顺序

### 参考案例
- **CS-016 (#42129)**: lazy_inline 导入错误 - 模块重构后路径变更

---

## M-007: 仅特定平台出现的问题

### 表面现象
```
RuntimeError: device address not exist
仅在 Mac arm / Windows / 特定 Linux 发行版出现
```

### 常见误判
定界到 **平台兼容性问题** 或 **驱动问题**

### 正确定界
**运行时问题** - 平台差异触发不同执行路径

### 判断依据
1. **仅特定平台** → 平台相关的配置或资源
2. **线程数/内存/指令集不同** → 触发不同代码分支
3. **其他平台正常** → 不是通用逻辑错误

### 验证实验
```python
# 1. 检查平台特定配置
import platform
print(platform.machine())  # x86_64 vs arm64

# 2. 检查线程数
import os
print(os.cpu_count())

# 3. 对比执行路径
# 在关键分支打日志，对比不同平台的执行流程
```

### 根因类型
- **线程数差异触发不同执行模式** (kernel actor vs runtime actor)
- **内存布局差异**
- **指令集差异** (AVX vs NEON)

### 修复模式
- 在单线程场景特殊处理
- 统一不同平台的执行路径
- 添加平台特定的兼容代码

### 参考案例
- **CS-010 (#41943)**: Mac arm device address 未创建 - 线程数为 1 触发退化

---

## M-008: CANN 版本不兼容

### 表面现象
```
Error building extension 'my_ops'
undefined reference to `aclrtDevResLimitType`
编译失败
```

### 常见误判
定界到 **Kernel 实现错误** 或 **编译环境问题**

### 正确定界
**Kernel 实现 - CANN 版本兼容性**

### 判断依据
1. **符号缺失** (undefined reference) → 新 API 在老版本不存在
2. **最近 CANN 升级** → 新代码使用了新 API
3. **特定 CANN 版本出现** → 版本相关

### 验证实验
```bash
# 1. 检查 CANN 版本
cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg

# 2. 搜索符号定义
grep -r "aclrtDevResLimitType" /usr/local/Ascend/ascend-toolkit/

# 3. 检查 API 引入版本
# 查看 CANN 文档或 release notes
```

### 根因类型
- **新 API 在老 CANN 版本不存在**
- **API 签名变化**
- **废弃 API 被移除**

### 修复模式
- 添加编译宏条件隔离
  ```cpp
  #if CANN_VERSION >= 8030
    aclrtDevResLimitType limit_type;
  #endif
  ```
- 提供 fallback 实现
- 更新最低 CANN 版本要求

### 参考案例
- **CS-008 (#41948)**: CANN 版本不兼容 - aclrtDevResLimitType 是 8.3 新增

---

## M-009: 梯度计算报 scalar type invalid

### 表面现象
```
RuntimeError: When convert scalar to tensor, the scalar type is invalid
仅在 complex 类型输入时出现
```

### 常见误判
定界到 **反向传播实现错误**

### 正确定界
**反向传播 - 底层工具函数类型覆盖不全**

### 判断依据
1. **仅特定 dtype 出现** (complex/bool) → 类型覆盖问题
2. **错误在工具函数** (ScalarToTensor) → 不是 bprop 逻辑错误
3. **其他 dtype 正常** → 不是通用实现问题

### 验证实验
```python
# 1. 测试不同 dtype
ops.expm1(Tensor([1.0], dtype=mindspore.float32))  # PASS
ops.expm1(Tensor([1.0+2.0j], dtype=mindspore.complex64))  # FAIL

# 2. 检查 ScalarToTensor 实现
# 查看是否支持 complex 类型
```

### 根因类型
- **ScalarToTensor 不支持 complex**
- **Cast 不支持 bool**
- **其他工具函数类型覆盖不全**

### 修复模式
- 在工具函数中添加类型支持
- 在 bprop 中提前转换类型
- 添加类型检查和友好报错

### 参考案例
- **CS-011 (#41954)**: expm1/log1p complex 梯度崩溃 - ScalarToTensor 不支持 complex

---

## M-010: keyword_arg 相关错误

### 表面现象
```
RuntimeError: make_keyword_arg / extract_keyword_arg error
IR 中残留 keyword_arg 节点
```

### 常见误判
定界到 **API 签名问题** 或 **参数处理错误**

### 正确定界
**编译器/IR 问题** - Morph 展开时序问题

### 判断依据
1. **错误信息包含 keyword_arg** → IR 节点残留
2. **函数包含 *args/**kwargs** → Morph 展开相关
3. **仅在 Graph 模式出现** → 编译器 pass 问题

### 验证实验
```python
# 1. 导出 IR
context.set_context(save_graphs=True)

# 2. 搜索 keyword_arg 节点
grep -n "keyword_arg" ir_dump/*.ir

# 3. 检查 Morph 展开是否完整
```

### 根因类型
- **RewriterBeforeOptA 时序问题**
- **keyword_arg 未完全展开**
- **pass 执行顺序错误**

### 修复模式
- 调整 pass 执行顺序
- 确保 keyword_arg 在后续 pass 前完全展开
- 添加 keyword_arg 清理 pass

### 参考案例
- **CS-010 (#41959)**: Morph *args/**kwargs - keyword_arg 残留

---

## 快速检查清单

在定界后，按以下顺序检查误导模式：

```
1. 梯度为零？ → 检查 M-001 (bprop Select)
2. AbstractProblem？ → 检查 M-002 (编译器 pass)
3. 小幅精度偏差？ → 检查 M-003 (CANN/基准版本)
4. DID NOT RAISE？ → 检查 M-004 (校验被绕过)
5. 偶现 core dump？ → 检查 M-005 (多线程竞态)
6. 导入错误？ → 检查 M-006 (模块重构)
7. 仅特定平台？ → 检查 M-007 (平台差异)
8. 编译失败？ → 检查 M-008 (CANN 版本)
9. scalar type invalid？ → 检查 M-009 (类型覆盖)
10. keyword_arg 错误？ → 检查 M-010 (Morph 时序)
```

---

## M-011: complex64 NaN 但 complex128 正常

### 表面现象
```
Tensor.reciprocal([inf+0j, inf+0j], complex64) → [nan+nanj, nan+nanj]
Tensor.reciprocal([inf+0j, inf+0j], complex128) → [0+0j, 0+0j]
```

### 常见误判
定界到 **MindSpore Kernel 实现缺陷** 或 **InferValue 常量折叠错误**

### 正确定界
**CANN aclnn 算子缺陷** — aclnn 内部对 complex64 (float) 的特殊值处理有 bug

### 判断依据
1. **标量 shape 正确，非标量 shape 错误** → 标量走 InferValue 常量折叠（C++ 标准库），非标量走 aclnn kernel
2. **complex128 正确，complex64 错误** → 底层浮点精度相关，float 的 inf² 溢出处理与 double 不同
3. **torch_npu 有完全相同的行为** → 非 MindSpore 问题，是 aclnn 共性缺陷

### 验证实验
```python
# 三方对比: PyTorch CPU vs torch_npu vs MindSpore
# PyTorch CPU (基准): 0+0j
# torch_npu: nan+nanj  ← 与 MindSpore 一致
# MindSpore: nan+nanj
# 结论: aclnn 算子 bug
```

### 根因类型
- **aclnn 算子对 complex64 inf 的内部计算** `conj(x)/|x|²` 中 float 精度 inf²=inf, inf/inf=nan
- **complex128 (double) 路径实现不同**，无此问题

### 修复模式
- 提交 CANN 问题单修复 aclnn 算子
- MindSpore 侧无需修改（与 torch_npu 行为一致）

### 参考案例
- **CS-018 (#42294)**: Reciprocal complex64 inf → NaN

---

## M-012: 取整算子返回 INT32_MAX / INT32_MIN

### 表面现象
```
mint.fix(Tensor(1.e+20, dtype=float32)) → Tensor(2.14748e+09)  # = 2^31-1 = INT32_MAX
mint.trunc(Tensor(-1.e+20, dtype=float32)) → Tensor(-2.14748e+09)  # = -2^31 = INT32_MIN
```

### 常见误判
定界到 **MindSpore 类型推导** 或 **PyBoost 类型转换** — 怀疑 MindSpore 侧做了 float→int32 的错误转换

### 正确定界
**CANN aclnn 算子缺陷（平台特有）** — aclnn 算子内部实现在特定芯片型号上使用了 float→int32→float 的计算路径

### 判断依据
1. **返回值恰好是 2^31-1 或 -2^31** → float→int32 溢出的标志性特征，不可能是精度误差
2. **仅特定芯片型号出现** → 910A 出现但 910B 正常，说明不同芯片的 aclnn 实现不同
3. **MindSpore 侧代码链路无任何类型转换** → PyBoost 直接调用 `LAUNCH_ACLNN(aclnnTrunc, ...)`，无中间转换
4. **CPU kernel 使用 `std::trunc()` 结果正确** → 问题仅在 Ascend 特定平台

### 验证实验
```python
# 1. 检查返回值是否为 INT32 边界值
import numpy as np
result = mint.fix(Tensor(1.e+20, dtype=ms.float32))
assert result.asnumpy() == np.float32(np.iinfo(np.int32).max)  # 2147483647

# 2. CPU 对比
ms.set_context(device_target="CPU")
result_cpu = mint.fix(Tensor(1.e+20, dtype=ms.float32))  # 正确: 1e+20

# 3. torch_npu 对比（同一台机器）
import torch, torch_npu
r = torch.fix(torch.tensor(1.e+20).npu())  # 如果也是 INT32_MAX → aclnn bug
```

### 根因类型
- **aclnn 算子在特定芯片上的实现路径** 使用了 float→int32→float 的截断方式，而非直接在浮点域操作
- **int32 范围仅 ±2.1e9**，远小于 float32 可表示的最大整数 (~3.4e38)

### 修复模式
- 提交 CANN 问题单修复对应芯片的 aclnn 算子实现
- 如需 MindSpore 侧 workaround，可参考 `numpy.fix` 的实现：用 `floor` + `ceil` + `select` 组合替代 `trunc`

### 参考案例
- **CS-019 (#42295)**: mint.fix 大数值返回 INT32_MAX（910A 特有）

---

## M-013: sign/特殊值函数仅特定 dtype 返回 NaN

### 表面现象
```python
mint.sign(Tensor(float('nan'), dtype=ms.float64))  # → nan（错误）
mint.sign(Tensor(float('nan'), dtype=ms.float32))  # → 0（正确）
mint.sign(Tensor(float('nan'), dtype=ms.float16))  # → 0（正确）
```

### 常见误判
定界到 **MindSpore kernel 实现** 或 **dtype 分发逻辑**

### 正确定界
**CANN aclnn 算子缺陷** - aclnn 算子对特定 dtype 的 NaN 处理不符合 IEEE 754 标准

### 判断依据
1. **仅特定 dtype 触发**：float64 异常，float16/float32 正常 → 不是通用逻辑问题
2. **torch_npu 行为一致**：torch_npu.sign(NaN, float64) 同样返回 nan → CANN 共性缺陷
3. **MindSpore 侧无 dtype 分支逻辑**：sign 算子在 MindSpore 中无针对 float64 的特殊处理路径
4. **PyTorch CPU 行为正确**：torch.sign(NaN) 在 CPU 上正确返回 0 → 标准行为明确

### 验证实验
```python
import torch
import mindspore as ms
from mindspore import Tensor, mint
import numpy as np

nan = float('nan')

# 1. MindSpore dtype 对比
for dt in [ms.float16, ms.float32, ms.float64]:
    r = mint.sign(Tensor(nan, dtype=dt))
    print(f"MS sign(NaN, {dt}): {r}")  # fp16/fp32 → 0, fp64 → nan

# 2. PyTorch CPU 基准
for dt in [torch.float16, torch.float32, torch.float64]:
    r = torch.sign(torch.tensor(nan, dtype=dt))
    print(f"Torch CPU sign(NaN, {dt}): {r}")  # 全部 → 0

# 3. torch_npu 对比（如有环境）
import torch_npu
for dt in [torch.float16, torch.float32, torch.float64]:
    r = torch.sign(torch.tensor(nan, dtype=dt).npu())
    print(f"torch_npu sign(NaN, {dt}): {r}")  # fp64 → nan = CANN bug
```

### 根因类型
- **aclnnSign 对 float64 NaN 的特殊值处理缺陷**，未按 IEEE 754 标准将 sign(NaN) 映射为 0

### 修复模式
- 提交 CANN 问题单修复 aclnnSign 的 float64 NaN 处理
- MindSpore 侧无需修改（非 MindSpore 问题）

### 参考案例
- **CS-020 (#42296)**: mint.sign float64 NaN 返回 nan 而非 0

---

## M-014: grad_cmp bfloat16 精度偏差实为测试基准 backward 路径不对齐

### 表面现象
`grad_cmp` 在 bfloat16 下失败，`allclose_nparray` 报告大量元素超出容差 (如 316542/2097152, max_diff=0.0625)，正向精度正常。

### 常见误判
定界到 **精度/数值** — MindSpore 反向 kernel 在 bfloat16 下精度不足

### 正确定界
**测试基准代码缺陷** - PyTorch 基准侧参数类型导致走了不同的 backward 路径

### 判断依据
1. 正向精度正常 (forward_cmp PASS)，仅反向有偏差
2. 偏差在 bfloat16 下出现，float32 下两条路径结果一致
3. 测试基准代码中 `repeats = torch.tensor(self.repeats)` 将 int 包装为 0-d tensor
4. PyTorch 对 tensor repeats 和 int repeats 使用不同的 backward 实现

### 验证实验
```python
import torch

x = torch.randn(2048, 1, 8, 128, dtype=torch.bfloat16, requires_grad=True)
grad_out = torch.randn(2048, 1, 48, 128, dtype=torch.bfloat16)

# Path A: int repeats → reshape + sum backward
y1 = x.repeat_interleave(6, dim=2)
y1.backward(grad_out)
grad_int = x.grad.clone()

x.grad = None

# Path B: tensor repeats → scatter_add backward
y2 = x.repeat_interleave(torch.tensor(6), dim=2)
y2.backward(grad_out)
grad_tensor = x.grad.clone()

# Compare: bfloat16 下两条路径结果不同
diff = (grad_int.float() - grad_tensor.float()).abs()
print(f"max_diff={diff.max()}, mismatch={( diff > 0.004).sum()}/{diff.numel()}")
# max_diff=0.0625, mismatch > 0 → 路径不对齐导致的精度差异
```

### 根因类型
- PyTorch `repeat_interleave` 对 int repeats 和 tensor repeats 使用不同 backward 路径
- int repeats: reshape + sum (与 MindSpore 一致)
- tensor repeats: scatter_add (累加精度在 bfloat16 下不同)
- `torch.tensor(int_value)` 会将 int 转为 0-d tensor，触发 tensor 路径

### 修复模式
- 测试基准代码中确保 `repeats` 参数类型与 MindSpore 侧一致
- `repeats = torch.tensor(self.repeats)` → `repeats = self.repeats`

### 参考案例
- **CS-020 (#1574)**: repeat_interleave bfloat16 梯度精度偏差

---

## 持续更新

每次发现新的误导模式，按以下模板添加：

```markdown
## M-XXX: [模式名称]

### 表面现象
[错误信息]

### 常见误判
定界到 **[错误的分类]**

### 正确定界
**[正确的分类]** - [根因类型]

### 判断依据
1. [判断依据 1]
2. [判断依据 2]
3. [判断依据 3]

### 验证实验
[验证代码或命令]

### 根因类型
- [根因 1]
- [根因 2]

### 修复模式
- [修复方式 1]
- [修复方式 2]

### 参考案例
- **CS-XXX (#issue_id)**: [案例描述]
```
