# 真实案例研究

基于 100 个 gitcode 问题单提取的代表性案例，覆盖全部 8 个问题分类。每个案例包含完整的定界→定位→修复链路，可作为同类问题的参考模板。

## 目录

- [精度/数值](#精度数值)
- [API/签名](#api签名)
- [Shape 推导](#shape-推导)
- [编译器/IR](#编译器ir)
- [Kernel 实现](#kernel-实现)
- [反向传播](#反向传播)
- [运行时](#运行时)
- [性能退化](#性能退化)

---

## 精度/数值

### Case CS-001: ops.pow 反向梯度为零 (#41932)

| 字段 | 内容 |
|------|------|
| 问题 | ops.pow fp32 在 4D broadcast 场景下反向梯度部分为零 |
| 环境 | 910B, CANN 20260105, Graph O2 |
| 错误特征 | `allclose_nparray` 失败，`data_me_error:[0.]`，梯度输出含零值 |
| 定界过程 | 正向精度正常 → 反向精度异常 → 定界到 bprop 层 |
| 根因 | 反向计算使用 Select 操作，A2 后端 GE 场景出现内存踩踏问题 |
| 修复 | 消除 Select 操作，改用 Mul + Cast 替代 ([PR#91490](https://gitee.com/mindspore/mindspore/pulls/91490)) |
| 引入类型 | 特性合入引入 (PR#91387, 2026-01-07) |
| 关键教训 | **错误信息说"精度"但根因在 bprop**。反向图中的 Select 操作在 GE 上有内存安全风险，应优先使用 Mul+Cast 替代 |

### Case CS-002: nn.Adam 与 TF 结果不一致 (#41934)

| 字段 | 内容 |
|------|------|
| 问题 | nn.Adam Ascend 后端 fp16 计算结果与 TensorFlow 标杆不一致 |
| 环境 | 910B, CANN 20260105, Graph O0 |
| 错误特征 | `allclose_nparray` 失败，前向和反向均有偏差 |
| 定界过程 | MS vs TF 对比 → TF 版本差异 → 定界到基准环境 |
| 根因 | TensorFlow 2.18 内部实现与 2.15 有变化，Dense 层未传入 dtype 时自动升精度到 fp32 |
| 修复 | 在 TF 基准代码中显式传入 `dtype=self.dtype` ([PR#91490](https://gitee.com/mindspore/mindspore/pulls/91490)) |
| 引入类型 | 环境问题 (TF 版本升级) |
| 关键教训 | 基准框架版本升级可能改变 dtype 自动提升行为，基准代码应显式指定 dtype |

### Case CS-003: mint.nn.Linear fp32 偶现精度问题 (#41931)

| 字段 | 内容 |
|------|------|
| 问题 | mint.nn.Linear fp32 在 910B 上反向梯度偶现精度超标 |
| 环境 | 910B, Graph O2 |
| 错误特征 | `allclose_nparray` 失败，反向梯度偏差超出 rtol=0.0001 |
| 定界过程 | 正向正常 → 反向偏差 → 特定随机种子必现 → 定界到 MatMul 累加精度 |
| 根因 | 反向 MatMul 的 k 轴为 11907，累加次数过多导致浮点误差积累 |
| 修复 | 根据累加次数放宽反向精度阈值 (0.0001 → 0.0004) |
| 引入类型 | 存量特性 |
| 关键教训 | 大 k 轴 MatMul 的精度阈值应根据累加次数动态设定，而非固定绝对阈值 |

### Case CS-004: CPU trace 算子 fp16 偶现精度异常 (#41933)

| 字段 | 内容 |
|------|------|
| 问题 | CPU 后端 trace 算子 fp16 偶现精度异常 |
| 环境 | CPU, PyNative |
| 错误特征 | `allclose_nparray` 失败，固定 `--randomly-seed=1767963664` 必现 |
| 定界过程 | CPU 独有 → fp16 独有 → 定界到 CPU kernel fp16 累加精度 |
| 根因 | CPU 上 trace 算子以 fp16 执行累加，精度损失显著；TF 标杆以 fp32 计算 |
| 修复 | CPU 内部提升至 fp32 计算再降精度输出 ([PR#91548](https://gitee.com/mindspore/mindspore/pulls/91548)) |
| 引入类型 | 测试新增测试场景 |
| 关键教训 | fp16 算子在 CPU 上做累加运算时应内部提升至 fp32，而非全程 fp16 |

### Case CS-005: DeepSeek loss 偏差 (#41977)

| 字段 | 内容 |
|------|------|
| 问题 | DeepSeek 网络 loss 与以往版本有微弱差异（万分之三） |
| 环境 | Ascend, CANN 1215+ |
| 错误特征 | loss 数值偏差，无报错 |
| 定界过程 | CANN 版本对比 → 1215 前后差异 → 定界到 CANN matmul 变更 |
| 根因 | CANN 包 matmul 变更导致计算结果微调，经评估变更合理 |
| 修复 | 重新基线 loss 数据 |
| 引入类型 | CANN 升级 |
| 关键教训 | CANN 升级后需重新基线，万分之三级别的偏差通常是 CANN 内部优化导致 |

### Case CS-020: mint.sign float64 NaN 返回 nan 而非 0 (#42296)

| 字段 | 内容 |
|------|------|
| 问题 | mint.sign(NaN, float64) 返回 nan，预期应返回 0（IEEE 754 标准：sign(NaN) = 0） |
| 环境 | Ascend 910B, CANN |
| 错误特征 | float64 NaN → nan；float16/float32 NaN → 0（正确） |
| 定界过程 | dtype 对比（fp16/fp32 正常，fp64 异常）→ torch_npu 对比（行为一致）→ 定界到 CANN aclnnSign |
| 根因 | CANN aclnnSign 算子对 float64 的 NaN 特殊值处理缺陷，未按 IEEE 754 标准返回 0 |
| 修复 | 提交 CANN 问题单，MindSpore 侧无需修改 |
| 引入类型 | CANN 算子缺陷 |
| 关键教训 | **仅特定 dtype 异常时，优先做三方对比（PyTorch CPU vs torch_npu vs MindSpore）**。torch_npu 行为一致说明是 CANN 共性缺陷而非 MindSpore 问题 |

---

## API/签名

### Case CS-006: ops.scatternd PyNative 参数校验丢失 (#41971)

| 字段 | 内容 |
|------|------|
| 问题 | ops.scatternd 在 PyNative 模式下传入 list 类型 shape 未报 TypeError |
| 环境 | PyNative 模式 |
| 错误特征 | `Failed: DID NOT RAISE <class 'TypeError'>` |
| 定界过程 | Graph 模式正常报错 → PyNative 不报错 → 定界到 ConvertSequence |
| 根因 | `ConvertSequence` 将 list 自动转为 tuple，绕过了类型校验 |
| 修复 | 修改 ConvertSequence 不再自动转换 list→tuple ([PR#91430](https://gitee.com/mindspore/mindspore/pulls/91430)) |
| 引入类型 | 特性合入引入 (PR#89799) |
| 关键教训 | 自动类型转换会静默掩盖参数校验，引入此类转换时需评估对异常场景的影响 |

### Case CS-007: tensor.mul 类型校验丢失 (#42116)

| 字段 | 内容 |
|------|------|
| 问题 | tensor.mul 传入 tuple 类型 other 未报 TypeError |
| 环境 | 通用 |
| 错误特征 | `Failed: DID NOT RAISE <class 'TypeError'>` |
| 定界过程 | 新增 deprecated 分支 → 放宽了类型约束 → 定界到 API 层 |
| 根因 | 修复 COOTensor 时新增的 list/tuple deprecated 分支意外放宽了 tensor.mul 的类型约束 |
| 修复 | 删除 tensor.mul 中支持 list/tuple 的分支 ([PR#91457](https://gitee.com/mindspore/mindspore/pulls/91457)) |
| 引入类型 | 特性合入引入 (PR#89877) |
| 关键教训 | 为兼容性新增 deprecated 分支时，需检查是否意外放宽了其他 API 的类型约束 |

### Case CS-008: FusedAdamW amsgrad 参数冲突 (#42227)

| 字段 | 内容 |
|------|------|
| 问题 | FusedAdamW 使用 add_param_group 设置 amsgrad=True 时报 TypeError |
| 环境 | 通用 |
| 错误特征 | `TypeError: unsupported operand type(s) for +=: 'NoneType' and 'ParameterTuple'` |
| 定界过程 | 初始化 amsgrad=False → add_param_group amsgrad=True → 内部状态未初始化 |
| 根因 | 初始化时 amsgrad=False 跳过了 max_exp_avg_sq 的创建，后续 add_param_group 设置 amsgrad=True 时访问未初始化的变量 |
| 修复 | 在 add_param_group 中检查并补充初始化 max_exp_avg_sq |
| 引入类型 | 特性合入引入 |
| 关键教训 | 优化器的延迟初始化逻辑需考虑参数组动态添加场景 |

---

## Shape 推导

### Case CS-009: PixelShuffle AbstractProblem (#41973)

| 字段 | 内容 |
|------|------|
| 问题 | mint.nn.PixelShuffle 动态 shape 报 AbstractProblem 错误 |
| 环境 | Ascend, Graph 模式 |
| 错误特征 | `RuntimeError: Invalid abstract;AbstractProblem(Value: DeadNode, ...)` at `control_node_parser.cc:362` |
| 定界过程 | 静态 shape 正常 → 动态 shape 报错 → IR 中发现 DeadNode → 定界到编译器 pass |
| 根因 | 前端脚本变化导致构图变化，产生 DeadNode 节点未被消除，流入后端触发 AbstractProblem |
| 修复 | 新增 switch_simplify pass 消除 DeadNode ([PR#91361](https://gitee.com/mindspore/mindspore/pulls/91361)) |
| 引入类型 | 测试漏测 |
| 关键教训 | **错误信息说"Shape/Abstract"但根因在编译器 pass**。动态 shape 场景需验证 DeadNode 是否被正确消除 |

---

## 编译器/IR

### Case CS-010: Morph *args/**kwargs 编译报错 (#41959)

| 字段 | 内容 |
|------|------|
| 问题 | Morph 内部调用的函数包含 *args, **kwargs 入参时编译报错 |
| 环境 | Graph 模式 |
| 错误特征 | `RuntimeError: Illegal type in the graph: AbstractKeywordArg(...)` at `validator.cc:194` |
| 定界过程 | 仅 Morph 场景出错 → IR 中残留 make_keyword_arg → 定界到编译器 pass 时序 |
| 根因 | `RewriterBeforeOptA` 负责消除 keyword_arg 算子，但 Morph 展开发生在该 pass 之后，导致残留 |
| 修复 | 在 `RewriterAfterOptA` 中补充 keyword_arg 消除逻辑 ([PR#91387](https://gitee.com/mindspore/mindspore/pulls/91387)) |
| 引入类型 | 特性合入引入 (PR#82416) |
| 关键教训 | 新特性的展开时机若晚于清理 pass，需在后续 pass 中补充相同的清理逻辑 |

### Case CS-011: acosh O1 模式反向 dump 报错 (#41967)

| 字段 | 内容 |
|------|------|
| 问题 | acosh 算子 O1 模式反向传播报错 Unknown scalar type 1 |
| 环境 | Ascend, Graph O1 |
| 错误特征 | `RuntimeError: Unknown scalar type 1` at `dump_proto.cc:312` |
| 定界过程 | 仅 O1 出错 → dump IR 路径 → 定界到 IR dump 工具 |
| 根因 | `SetScalarToProto` 未处理 FP16 (scalar type 1) 类型 |
| 修复 | 增加 FP16 和 BF16 类型支持分支 ([PR#91426](https://gitee.com/mindspore/mindspore/pulls/91426)) |
| 引入类型 | 特性合入引入 |
| 关键教训 | IR dump/序列化功能必须覆盖所有支持的数据类型，包括 fp16、bf16 |

---

## Kernel 实现

### Case CS-012: CANN 版本不兼容 — aclrtDevResLimitType (#41948)

| 字段 | 内容 |
|------|------|
| 问题 | 自定义算子在旧 CANN 包执行报错 Error building extension |
| 环境 | Ascend, 旧 CANN 版本 |
| 错误特征 | `RuntimeError: Error building extension 'my_ops'`，链接时符号缺失 |
| 定界过程 | 新 CANN 正常 → 旧 CANN 报错 → 定界到 CANN API 兼容性 |
| 根因 | `aclrtDevResLimitType` 为 CANN 8.3 新增符号，老版本缺失 |
| 修复 | 用编译宏 `#if CANN_VERSION >= CANN_8_3` 隔离 ([PR#91399](https://gitee.com/mindspore/mindspore/pulls/91399)) |
| 引入类型 | CANN 升级 (PR#90666) |
| 关键教训 | 使用新 CANN API 时必须用编译宏保护，确保老版本可编译 |

### Case CS-013: 多线程 core dump — SetMsInternalEnableCustomKernelList (#41935)

| 字段 | 内容 |
|------|------|
| 问题 | O1 自动并行 8p 模式偶现 core dump |
| 环境 | Ascend, O1, 8 卡并行 |
| 错误特征 | `SIGSEGV in std::_Rb_tree_insert_and_rebalance` |
| 定界过程 | 偶现 → 多线程 → 堆栈指向 set 插入 → 定界到线程安全 |
| 根因 | 双重问题：① optional 判断错误导致训练误入推理路径；② set 修改无多线程保护 |
| 修复 | ① 修正 optional 判断；② 使用原子操作保证只初始化一次 ([PR#91480](https://gitee.com/mindspore/mindspore/pulls/91480)) |
| 引入类型 | 特性合入引入 (PR#91141) |
| 关键教训 | 偶现 core dump 重点排查并发访问；optional 需区分"有值"与"值为 true" |

---

## 反向传播

### Case CS-014: expm1/log1p complex 类型求梯度崩溃 (#41954)

| 字段 | 内容 |
|------|------|
| 问题 | ops.expm1/ops.log1p 对 complex 类型 0 维张量求梯度时崩溃 |
| 环境 | 通用 |
| 错误特征 | `RuntimeError: When convert scalar to tensor, the scalar type: Complex128 is invalid` |
| 定界过程 | 仅 complex 类型 → 仅 0 维 → 定界到 ScalarToTensor 工具函数 |
| 根因 | `ScalarToTensor` 不支持 Complex64/Complex128 类型 |
| 修复 | 扩展 ScalarToTensor 支持 complex 类型 ([PR#91404](https://gitee.com/mindspore/mindspore/pulls/91404)) |
| 引入类型 | 特性合入引入 (PR#89877) |
| 关键教训 | 新增 complex 类型支持时，需同步检查底层工具函数是否覆盖该类型 |

---

## 运行时

### Case CS-015: Cell 属性修改 mac arm 报 device address 错误 (#41943)

| 字段 | 内容 |
|------|------|
| 问题 | Cell 中非 Parameter 属性修改在 mac arm 上报 device address 未创建 |
| 环境 | Mac ARM, Graph 模式 |
| 错误特征 | `RuntimeError: Output_idx 0 of node ... output addr is not exist` |
| 定界过程 | 仅 mac arm → runtime 线程数=1 → kernel actor 路径 → 定界到 ref 地址校验 |
| 根因 | mac arm 线程数为 1 退化为 kernel actor 方式，data_prepare_actor 校验 ref 地址时 any 类型输入的地址尚未创建 |
| 修复 | 对 any 类型输入跳过 ref 地址校验 ([PR#91448](https://gitee.com/mindspore/mindspore/pulls/91448)) |
| 引入类型 | 特性合入引入 (PR#90625) |
| 关键教训 | 平台差异（线程数）会触发不同执行路径，any 类型的延迟地址创建与提前校验存在时序冲突 |

### Case CS-016: lazy_inline 导入错误 (#42129)

| 字段 | 内容 |
|------|------|
| 问题 | Pipeline + lazy_inline 场景报 module not callable |
| 环境 | 通用 |
| 错误特征 | `TypeError: 'module' object is not callable` |
| 定界过程 | 直接从错误信息 → 导入路径检查 → 定界到 import 语句 |
| 根因 | 重构后 `common/__init__.py` 不再导出同名函数，`import module as name` 得到模块对象而非函数 |
| 修复 | 改为 `from mindspore import lazy_inline` |
| 引入类型 | 用例未适配 (PR#91253) |
| 关键教训 | 模块重构时需同步更新所有依赖旧路径的导入语句 |

---

## 性能退化

### Case CS-017: GetNext 动态 shape 性能退化 (#41951)

| 字段 | 内容 |
|------|------|
| 问题 | 构造 GetNext 动态 shape 网络多次执行，用例时间从 10 分钟涨至 15 分钟以上 |
| 环境 | Ascend, 动态 shape |
| 错误特征 | 用例超时被杀，无功能错误 |
| 定界过程 | 版本对比 → 动态 shape 独有 → 定界到动态 shape 执行路径 |
| 根因 | 动态 shape 场景下 GetNext 的执行路径存在性能退化 |
| 修复 | 优化动态 shape 执行路径 |
| 引入类型 | 特性合入引入 |
| 关键教训 | 动态 shape 场景的性能需要专项看护，多次执行的累积开销容易被忽略 |

### Case CS-018: Reciprocal complex64 inf 返回 NaN (#42294)

| 字段 | 内容 |
|------|------|
| 问题 | `Tensor.reciprocal` 对非空 shape 的 complex64 inf 输入返回 `nan+nanj`，标量 shape 和 complex128 均正确 |
| 环境 | Ascend, MindSpore 2.9.0 |
| 错误特征 | complex64 `[2]` shape 的 inf 输入 → `nan+nanj`；标量 `[]` → `0+0j`；complex128 `[2]` → `0+0j` |
| 定界过程 | 1) shape 差异暗示不同执行路径 → 标量走 InferValue 常量折叠，非标量走 aclnn kernel 2) 对比 torch_npu 行为完全一致 → 非 MindSpore 问题 3) 对比 PyTorch CPU 基准返回 `0+0j` → aclnn 算子 bug |
| 根因 | `aclnnReciprocal` 对 complex64 类型的 inf 输入处理有缺陷，内部 `conj(x)/|x|²` 计算中 float 精度的 inf²=inf, inf/inf=nan。complex128 (double) 路径实现不同，无此问题 |
| 修复 | CANN 侧修复 `aclnnReciprocal`，MindSpore 侧无需修改（与 torch_npu 行为一致） |
| 引入类型 | CANN aclnn 算子缺陷 |
| 关键教训 | 1) shape 差异（标量 vs 非标量）是执行路径分叉的重要线索 — 标量常量折叠走 C++ 标准库，非标量走 aclnn 2) 三方对比（PyTorch CPU → torch_npu → MindSpore）是定界 aclnn 问题的标准方法 3) complex64 vs complex128 行为差异通常指向底层浮点精度处理 |

### Case CS-019: mint.fix 大数值返回 INT32_MAX (#42295)

| 字段 | 内容 |
|------|------|
| 问题 | `mint.fix(Tensor(1.e+20, dtype=float32))` 在 910A 上返回 `2.14748e+09` (INT32_MAX)，而非预期的 `1e+20` |
| 环境 | Ascend 910A |
| 错误特征 | 返回值 `2147483647` = `INT32_MAX`，说明内部做了 float32→int32→float32 的错误转换，int32 溢出 |
| 定界过程 | 1) `mint.fix` → `mint.trunc` → `Trunc` primitive → `aclnnTrunc`，MindSpore 侧无任何类型转换 2) 910B3 上无法复现，结果正确 3) torch_npu 在 910B3 上结果正确 4) CPU kernel 使用 `std::trunc()` 结果正确 → 定界到 910A 平台的 `aclnnTrunc` 实现 |
| 根因 | `aclnnTrunc` 在 910A 平台上的实现内部将 float 转为 int32 再转回 float，当输入值超出 int32 范围时溢出截断到 `INT32_MAX`。910B 上的实现已无此问题 |
| 修复 | CANN 侧修复 `aclnnTrunc` 910A 实现，MindSpore 侧无需修改 |
| 引入类型 | CANN aclnn 算子缺陷（平台特有） |
| 关键教训 | 1) 返回值为 `2^31-1` 或 `-2^31` 是 float→int32 溢出的标志性特征 2) 同一 aclnn 算子在不同芯片型号（910A vs 910B）上可能有不同实现，需注意平台差异 3) 无法在目标平台复现时，可通过返回值特征（INT32_MAX）直接推断根因 4) `numpy.fix` 用 floor+ceil+select 组合实现可作为 workaround 参考 |

### Case CS-020: repeat_interleave bfloat16 梯度精度偏差 (#1574)

| 字段 | 内容 |
|------|------|
| 问题 | `Tensor.repeat_interleave` bfloat16 梯度 `grad_cmp` 失败，316542/2097152 元素超出容差 (max_diff=0.0625) |
| 环境 | 910B3, CANN 20251121, PyNative |
| 错误特征 | `allclose_nparray` 失败，梯度偏差集中在 bfloat16，正向精度正常 |
| 定界过程 | 1) forward_cmp PASS → 正向无问题 2) grad_cmp FAIL → 反向精度偏差 3) MindSpore 使用 reshape+SumExt 路径 (PR !91183)，PyTorch 基准代码中 `repeats = torch.tensor(self.repeats)` 将 int 转为 tensor，导致 torch 走 scatter_add backward 路径 4) 两条 backward 路径在 bfloat16 下精度不同 → 定界到**测试基准代码** |
| 根因 | 测试框架 `grad_pytorch_impl` 中 `repeats = torch.tensor(self.repeats)` 将 int repeats 包装为 0-d tensor，导致 PyTorch 走 tensor repeats 的 backward 路径 (scatter_add)，而 MindSpore 走 int repeats 的 backward 路径 (reshape+sum)。两条路径在 float32 下结果一致，但在 bfloat16 下 scatter_add 的累加精度与 reshape+sum 不同，产生 max_diff=0.0625 的偏差 |
| 修复 | 测试基准代码中将 `repeats = torch.tensor(self.repeats)` 改为 `repeats = self.repeats`，确保 PyTorch 和 MindSpore 走相同的 backward 路径 |
| 引入类型 | 测试基准代码缺陷 |
| 关键教训 | 1) **`torch.tensor(int)` 与 `int` 在 PyTorch 中走不同的 backward 路径**：tensor repeats 用 scatter_add，int repeats 用 reshape+sum 2) bfloat16 下不同累加路径的精度差异会被放大 3) `grad_cmp` 失败时，不仅要检查 MindSpore 侧，还要检查 PyTorch 基准侧的参数类型是否与 MindSpore 对齐 4) 这是一类**误导性精度问题**：表面是 MindSpore 精度错误，实际是测试基准的 backward 路径不对齐 |

---

## 跨分类案例（误导性定界）

以下案例的初始定界方向与最终根因不一致，具有特殊参考价值：

| Case | 初始定界 | 最终根因 | 误导原因 |
|------|---------|---------|---------|
| CS-001 (#41932) | 精度/数值 | 反向传播 (bprop Select) | 错误信息是 allclose 失败，但根因在反向图结构 |
| CS-009 (#41973) | Shape 推导 | 编译器 pass (DeadNode) | 错误信息是 AbstractProblem，但根因在 pass 缺失 |
| CS-002 (#41934) | 精度/数值 | 基准环境 (TF 版本) | 看似 MS 精度问题，实际是 TF 基准代码的 dtype 处理 |
| CS-013 (#41935) | Kernel 崩溃 | 线程安全 + optional 误用 | 堆栈指向 set 插入，但根因是执行路径判断错误 |
| CS-020 (#1574) | 精度/数值 | 测试基准 backward 路径不对齐 | grad_cmp bfloat16 失败，看似 MindSpore 梯度精度问题，实际是 torch 基准用 `torch.tensor(int)` 走了不同 backward 路径 |
