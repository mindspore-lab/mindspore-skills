# MindSpore 算子常见问题模式与根因分类

## 目录

1. [根因分类总览](#1-根因分类总览)
2. [精度与数值问题](#2-精度与数值问题)
3. [API 与签名不一致](#3-api-与签名不一致)
4. [Shape 推导与广播错误](#4-shape-推导与广播错误)
5. [编译器与 IR 问题](#5-编译器与-ir-问题)
6. [Kernel 实现缺陷](#6-kernel-实现缺陷)
7. [反向传播与梯度问题](#7-反向传播与梯度问题)
8. [运行时与 PyNative 问题](#8-运行时与-pynative-问题)
9. [性能退化](#9-性能退化)
10. [快速定界决策树](#10-快速定界决策树)

---

## 1. 根因分类总览

| 分类 | 占比(估) | 典型标签 | 涉及组件 |
|------|---------|---------|---------|
| 精度/数值 | ~25% | B-SIG-OPS, Ascend | kernel, benchmark, CANN |
| API/签名 | ~15% | B-SIG-FrontEnd | functional_overload, ops/operations |
| Shape 推导 | ~15% | B-SIG-OPS | ops/infer, core/ops |
| 编译器/IR | ~10% | B-SIG-Compiler | ccsrc/frontend, control flow |
| Kernel 实现 | ~15% | B-SIG-OPS, B-SIG-BackendRuntime | kernel/{cpu,gpu,ascend} |
| 反向传播 | ~10% | B-SIG-OPS | bprop/grad_ops |
| 运行时 | ~5% | B-SIG-BackendRuntime | ccsrc/runtime |
| 性能退化 | ~5% | B-ComponentTest | 多组件 |

---

## 2. 精度与数值问题

### 诊断特征

**典型错误信息**:
- `AssertionError` in `allclose_nparray`
- `_count_unequal_element` 统计不匹配元素
- `data_expected_std:[X] data_me_error:[Y]`
- `loss_count / total_count >= rtol`
- 输出全 NaN / Inf / 全零

**触发条件**:
- Ascend fp16 计算
- 对比 TensorFlow / PyTorch 基准
- 特定 CANN 版本更新后

### 常见根因

| 根因 | 说明 | 示例 |
|------|------|------|
| **fp16 精度不足** | Ascend 算子 fp16 计算溢出或精度丢失 | FloorMod fp16 梯度偏差 (#42163) |
| **基准环境差异** | TF/Torch 版本不同导致 dtype 自动提升行为变化 | nn.Adam TF 2.18 vs 2.15 (#41934) |
| **CANN 内核变更** | CANN 升级后 matmul 等算子行为变化 | DeepSeek loss 偏差 (#41977) |
| **反向 Select 缺陷** | backward 中 Select 操作导致 GE 内存问题 | ops.pow 梯度为零 (#41932) |
| **随机数不一致** | 旧模型未使用 mint 接口，随机数种子处理不同 | mindone I2VGenXLUNet NaN (#42126) |
| **dtype 处理错误** | 输入 dtype 未正确处理或传递 | benchmark 忘记指定 dtype |
| **aclnn 特殊值处理缺陷** | aclnn 算子对 inf/nan 等特殊值处理不正确，仅特定 dtype 触发 | Reciprocal complex64 inf → NaN (#42294)；sign float64 NaN → nan (#42296) |
| **测试基准 backward 路径不对齐** | PyTorch 基准代码中参数类型导致走不同 backward 路径，bfloat16 下精度差异被放大 | repeat_interleave `torch.tensor(int)` vs `int` (#1574) |

### 诊断步骤

1. 确认环境: MindSpore 版本、CANN 版本、设备型号 (910A/910B)
2. 对比模式: `forward_cmp()` / `grad_cmp()` 对比基准框架
3. 隔离后端: `export MS_DISABLE_KERNEL_BACKOFF=1` 禁止 kernel fallback
4. 检查 dtype: 输入输出的 dtype 是否预期
5. 检查 CANN: 对比不同 CANN 版本的结果

### 真实案例

- **CS-001 (#41932)**: ops.pow 反向梯度为零 — 反向 Select 操作导致 GE 内存踩踏
- **CS-002 (#41934)**: nn.Adam 与 TF 不一致 — TF 2.18 dtype 自动提升行为变化
- **CS-003 (#41931)**: mint.nn.Linear 偶现精度 — 大 k 轴 MatMul 累加误差
- **CS-004 (#41933)**: CPU trace fp16 精度 — CPU 上 fp16 累加精度不足
- **CS-005 (#41977)**: DeepSeek loss 偏差 — CANN matmul 变更导致
- **CS-020 (#1574)**: repeat_interleave bfloat16 梯度偏差 — 测试基准 torch.tensor(int) 导致 backward 路径不对齐

### 二级定界决策

```
allclose 失败
├─ 输出全零 → 反向图结构问题 (Select/DeadNode)，参考 CS-001
├─ 输出全 NaN → dtype 溢出或未初始化，检查 fp16 计算链路
├─ 小幅偏差 (< 1e-3) → 累加精度或 CANN 变更，参考 CS-003/CS-005
├─ 大幅偏差 → 逻辑错误或基准环境差异，参考 CS-002
└─ 仅 bfloat16 梯度偏差，正向正常 → 检查基准 backward 路径是否对齐，参考 CS-020
```

### ⚠️ 误导性关键词

| 表面现象 | 实际根因 | 参考 |
|---------|---------|------|
| allclose 精度失败 | bprop Select 缺陷 | CS-001 |
| 精度不一致 | 基准框架版本差异 | CS-002 |
| fp16 精度问题 | CPU kernel 未做精度提升 | CS-004 |
| bfloat16 梯度偏差 | 测试基准 backward 路径不对齐 | CS-020 |

---

## 3. API 与签名不一致

### 诊断特征

**典型错误信息**:
- `TypeError: XXX() takes N positional arguments but M were given`
- `TypeError: unsupported operand type(s) for +=`
- `ValueError` vs `RuntimeError` 类型不符
- 文档与实际接口参数不一致

**触发条件**:
- API 升级后签名变更
- 位置参数与关键字参数混淆
- 多参数组合的边界条件

### 常见根因

| 根因 | 说明 | 示例 |
|------|------|------|
| **签名变更** | functional 接口参数数量或顺序变化 | nsa_compress_attention 9 vs 14 参数 (#41955) |
| **选项冲突** | 混合使用不兼容的选项 | FusedAdamW amsgrad=False + add_param_group amsgrad=True (#42227) |
| **参数校验缺失** | list vs tuple 未区分, None 未校验 | scatternd list 自动转 tuple |
| **错误类型不一致** | 通信 API ValueError vs RuntimeError | get_group_size 错误类型 |
| **自动类型转换** | ConvertSequence 过度转换 | PyNative 模式下 list 被转为 tuple |

### 诊断步骤

1. 对比接口签名: 检查 YAML 定义 vs Python API 文档
2. 最小复现: 精简调用参数组合
3. 检查 `functional_overload.py` 中的分发逻辑
4. 检查 `ops/api_def/*.yaml` 中的 `kwonlyargs` 定义

### 真实案例

- **CS-006 (#41971)**: scatternd PyNative 参数校验丢失 — ConvertSequence 自动转换 list→tuple
- **CS-007 (#42116)**: tensor.mul 类型校验丢失 — deprecated 分支意外放宽约束
- **CS-008 (#42227)**: FusedAdamW amsgrad 参数冲突 — 延迟初始化未考虑动态添加

### 二级定界决策

```
TypeError / 参数错误
├─ "DID NOT RAISE" → 校验被绕过，检查 ConvertSequence 和 deprecated 分支
├─ "takes N arguments" → 签名变更，对比 YAML 定义
├─ "unsupported operand" → 内部状态未初始化，检查延迟初始化逻辑
└─ "module not callable" → 导入错误，检查 import 路径
```

---

## 4. Shape 推导与广播错误

### 诊断特征

**典型错误信息**:
- `ValueError: For 'X', condition.shape and input.shape need to broadcast`
- `RuntimeError: Invalid abstract;AbstractProblem`
- `ValueError: For 'Reshape', the product of shape should be equal to ...`
- `ValueError: For 'X', input shapes can not broadcast`

**触发条件**:
- 动态 shape 场景
- jacfwd / jacrev 变换
- 控制流中的 shape 变化

### 常见根因

| 根因 | 说明 | 示例 |
|------|------|------|
| **广播规则错误** | condition 与 input shape 不兼容 | Select shape 广播 (#41936) |
| **DeadNode 残留** | 控制流优化后 IR 中残留无效节点 | PixelShuffle AbstractProblem (#41973) |
| **动态 shape 推导** | -1 维度推导逻辑不完整 | Reshape -1 推导失败 |
| **前端脚本变更** | 前端 pass 修改了 shape 推导路径 | 编译器变更导致 shape 失配 |
| **Rank 不匹配** | 输入 rank 超出算子支持范围 | 高维 tensor 输入 |

### 诊断步骤

1. `context.set_context(save_graphs=True)` 导出 IR
2. 检查 `InferShape` 实现中的维度校验逻辑
3. 测试静态 shape 是否正常，定位是否为动态 shape 特有问题
4. 检查是否在 `jacfwd`/`jacrev` 等变换下出现

### 真实案例

- **CS-009 (#41973)**: PixelShuffle AbstractProblem — 表面是 Shape 错误，实际是编译器 pass 缺失导致 DeadNode 残留

### ⚠️ 误导性关键词

| 表面现象 | 实际根因 | 参考 |
|---------|---------|------|
| AbstractProblem | 编译器 pass 缺失 (DeadNode) | CS-009 |
| Invalid abstract | 可能是 Shape 推导，也可能是 IR 节点残留 | — |

---

## 5. 编译器与 IR 问题

### 诊断特征

**典型错误信息**:
- `make_keyword_arg` / `extract_keyword_arg` 相关错误
- `FakeBprop` 未实现
- `ValueNode<ValueProblem> DeadNode`
- `control_node_parser.cc` 中的断言失败
- `FetchOutputSizeByNode` 错误

**触发条件**:
- `ms.jit(ms.grad(...))` 组合
- Morph / 编译器变换
- *args / **kwargs 传递

### 常见根因

| 根因 | 说明 | 示例 |
|------|------|------|
| **IR 节点缺少 Bprop** | 新 IR 节点没有注册反向 | Morph kwargs (#42145) |
| **编译 Pass 缺失** | 优化 pass 未清理无效节点 | 缺少 switch_simplify (#41973) |
| **关键字参数处理** | make_keyword_arg / extract_keyword_arg 不受支持 | Morph *args/**kwargs |
| **控制流解析** | 控制流节点解析逻辑不完整 | control_node_parser 断言失败 |

### 诊断步骤

1. `save_graphs=True` 导出全量 IR，检查第 13 步 `execute_*.ir`
2. 搜索 IR 中的 `DeadNode`、`ValueProblem` 等异常节点
3. 检查相关 pass 是否覆盖该模式
4. 定位 `ccsrc/frontend/` 中的相关源码

### 真实案例

- **CS-010 (#41959)**: Morph *args/**kwargs — 清理 pass 时序问题，keyword_arg 残留
- **CS-011 (#41967)**: acosh O1 dump 报错 — IR dump 工具未覆盖 fp16 类型
- **CS-009 (#41973)**: PixelShuffle DeadNode — 需新增 switch_simplify pass

---

## 6. Kernel 实现缺陷

### 诊断特征

**典型错误信息**:
- `RuntimeError: Error building extension 'my_ops'`
- `FAILED: *.o` — 编译失败
- `segmentation fault` / core dump
- `std::_Rb_tree` 插入崩溃 (并发问题)
- CANN 符号缺失

**触发条件**:
- 自定义算子编译
- 多线程 / 多卡并行
- CANN 版本不匹配

### 常见根因

| 根因 | 说明 | 示例 |
|------|------|------|
| **CANN 版本不兼容** | 代码引用了新 CANN 符号，老版本缺失 | aclrtDevResLimitType (#41948) |
| **线程安全** | 多线程访问未加锁的数据结构 | SetMsInternalEnableCustomKernelList core (#IDJHH5) |
| **optional 误用** | optional 未检查即访问 | 训练路径错误进入推理分支 |
| **设备地址缺失** | 运行时地址未创建就被访问 | Output_idx 0 addr not exist (#IDILBW) |
| **CUDA 内存** | Launch 中使用 CudaMalloc/CudaFree | 应在 Resize 中预分配 |

### 诊断步骤

1. 检查 CANN 版本与代码兼容性
2. 查看完整堆栈 (stack trace)
3. 检查多线程场景下的数据竞争
4. 对比 CPU/GPU/Ascend 行为是否一致
5. 检查 `Init()`/`Resize()`/`Launch()` 各阶段的资源管理

### 真实案例

- **CS-012 (#41948)**: CANN 版本不兼容 — aclrtDevResLimitType 符号缺失，需编译宏隔离
- **CS-013 (#41935)**: 多线程 core dump — optional 误用 + set 无锁并发写入

---

## 7. 反向传播与梯度问题

### 诊断特征

**典型错误信息**:
- `grad_cmp` 失败
- `GradOfAllInputs` / `GradOfFirstInput` 输出异常
- `FakeBprop` 错误
- 梯度为 NaN / 零 / 与基准不一致

**触发条件**:
- `ms.grad()` 或 `value_and_grad()`
- 特定 dtype (fp16) 下的梯度
- inplace 操作的反向

### 常见根因

| 根因 | 说明 | 示例 |
|------|------|------|
| **bprop 未注册** | 新算子缺少 REG_BPROP_BUILDER | — |
| **反向图中的 Select** | 反向图中 Select 操作导致 GE 内存问题 | ops.pow 梯度为零 (#41932) |
| **fp16 精度** | 反向计算中的 fp16 精度丢失 | FloorMod fp16 梯度 (#42163) |
| **Inplace 处理错误** | 前向 inplace 修改了反向需要的输入 | 未使用 CloneInplaceInput |
| **unused_inputs 设置错误** | 标记了反向实际需要的输入为 unused | 导致前向张量被提前释放 |

### 诊断步骤

1. 检查 `REG_BPROP_BUILDER("OpName")` 是否存在
2. 对比 `ms.grad()` 与 PyTorch `backward()` 结果
3. 检查 `SetUnusedInputs` 是否过度释放
4. 导出反向 IR 检查图结构
5. 测试 fp32 下梯度是否正确，排除精度问题

### 真实案例

- **CS-001 (#41932)**: ops.pow 梯度为零 — 反向 Select 操作在 GE 上内存踩踏
- **CS-014 (#41954)**: expm1/log1p complex 梯度崩溃 — ScalarToTensor 不支持 complex 类型

---

## 8. 运行时与 PyNative 问题

### 诊断特征

**典型错误信息**:
- `TypeError: 'module' object is not callable`
- `RuntimeError: Output_idx 0 of node ... output addr is not exist`
- `device address` 相关错误
- 超时 / OOM

**触发条件**:
- Graph 模式 vs PyNative 模式差异
- Mac ARM 等非主流平台
- 导入错误

### 常见根因

| 根因 | 说明 | 示例 |
|------|------|------|
| **导入错误** | 导入 module 而非 callable | initializer 导入错误 (#42185) |
| **设备地址未创建** | ref 地址在 any-type 输入前检查 | Cell 赋值 (#IDILBW) |
| **执行路径错误** | 训练/推理路径判断错误 | optional 判断分支错误 |
| **平台差异** | Mac ARM runtime threads=1 触发不同路径 | kernel actor 路径差异 |

### 诊断步骤

1. 区分 Graph 模式和 PyNative 模式下行为差异
2. 检查 `context.get_context("mode")` 和 `device_target`
3. 检查堆栈中的 runtime 组件
4. 最小化复现，排除网络复杂度影响

### 真实案例

- **CS-015 (#41943)**: Cell 属性修改 mac arm 报错 — 平台线程数差异触发不同执行路径
- **CS-016 (#42129)**: lazy_inline 导入错误 — 模块重构后 import 路径变更

---

## 9. 性能退化

### 诊断特征

- 用例超时 (timeout / killed)
- 运行时间从 N 分钟增长到 2N 分钟以上
- 无功能错误但吞吐量下降

### 常见根因

| 根因 | 说明 | 示例 |
|------|------|------|
| **动态 shape 循环** | GetNext 动态 shape 性能退化 | (#41951) |
| **编译器变更** | 新优化 pass 引入额外开销 | — |
| **内存分配** | Launch 中频繁 malloc/free | — |
| **CANN 版本** | CANN 升级改变算子执行效率 | matmul 行为变化 |

---

## 10. 快速定界决策树

```
错误信息
│
├─ 包含 "allclose" / "precision" / "NaN" / "Inf"
│   → 【精度/数值】读 §2
│   ├─ 输出全零 → 反向图结构问题 (bprop Select/DeadNode)
│   ├─ 输出全 NaN → dtype 溢出，检查 fp16 计算链路
│   ├─ 小幅偏差 (< 1e-3) → 累加精度或 CANN 变更
│   └─ 大幅偏差 → 逻辑错误或基准环境差异
│
├─ 包含 "takes N arguments" / "TypeError" / "unsupported operand"
│   → 【API/签名】读 §3
│   ├─ "DID NOT RAISE" → 校验被绕过 (ConvertSequence/deprecated 分支)
│   ├─ "takes N arguments" → 签名变更，对比 YAML
│   └─ "unsupported operand" → 内部状态未初始化
│
├─ 包含 "shape" / "broadcast" / "AbstractProblem" / "Invalid abstract"
│   → 【Shape 推导】读 §4
│   ├─ 含 "DeadNode" → 实际是编译器 pass 问题，读 §5
│   ├─ 仅动态 shape → Infer 动态 shape 处理
│   └─ 静态也出错 → 广播规则或 Rank 不匹配
│
├─ 包含 "DeadNode" / "FakeBprop" / "keyword_arg" / "control_node_parser"
│   → 【编译器/IR】读 §5
│   ├─ "keyword_arg" → Morph 展开时序问题
│   ├─ "Unknown scalar type" → IR dump 类型覆盖不全
│   └─ "FakeBprop" → 新 IR 节点缺少 bprop 注册
│
├─ 包含 "segmentation fault" / "core dump" / "Error building" / "FAILED: *.o"
│   → 【Kernel 实现】读 §6
│   ├─ "std::_Rb_tree" → 多线程并发写入，检查锁
│   ├─ "Error building" → CANN 版本兼容性，检查编译宏
│   └─ 偶现 → 线程安全 + 执行路径判断
│
├─ 包含 "grad_cmp" / "GradOf" / 梯度异常
│   → 【反向传播】读 §7
│   ├─ "scalar type invalid" → 底层工具函数类型覆盖不全
│   └─ 梯度为零 → bprop 中的 Select 或 SetUnusedInputs 错误
│
├─ 包含 "device address" / "output addr" / "module not callable"
│   → 【运行时】读 §8
│   ├─ "module not callable" → 导入路径错误
│   ├─ "output addr not exist" → any 类型延迟地址创建
│   └─ 仅特定平台 → 平台差异触发不同执行路径
│
└─ 超时 / 性能指标异常
    → 【性能退化】读 §9
```

### 组件标签映射

| 标签 | 对应问题类型 |
|------|------------|
| B-SIG-OPS | 算子定义、推导、kernel、精度 |
| B-SIG-Compiler | 编译器、IR、pass |
| B-SIG-FrontEnd | 前端、API、接口 |
| B-SIG-BackendRuntime | 运行时、设备管理 |
| B-ComponentTest | 测试用例本身的问题 |
| B-Deploy | 部署、环境 |
| B-MDTest | 模型测试 |
