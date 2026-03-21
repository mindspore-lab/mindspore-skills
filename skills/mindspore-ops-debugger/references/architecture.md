# MindSpore 算子架构与源码导航

> **路径约定**: 本文档中的源码路径相对于 MindSpore 包目录。在工作目录中，实际路径前缀为 `mindspore/mindspore/`。
> 例如 `ops/op_def/yaml/` 的完整路径为 `mindspore/mindspore/ops/op_def/yaml/`，
> `ccsrc/frontend/` 的完整路径为 `mindspore/mindspore/ccsrc/frontend/`。

## 目录

1. [算子整体架构](#1-算子整体架构)
2. [YAML 算子定义](#2-yaml-算子定义)
3. [API 定义层](#3-api-定义层)
4. [Shape/Type 推导 (Infer)](#4-shapetype-推导-infer)
5. [Kernel 实现](#5-kernel-实现)
6. [反向传播 (Bprop)](#6-反向传播-bprop)
7. [Python 接口层](#7-python-接口层)
8. [PyBoost Customize](#8-pyboost-customize)
9. [C++ 算子组合](#9-c-算子组合)
10. [View 算子](#10-view-算子)
11. [源码导航速查表](#11-源码导航速查表)

---

## 1. 算子整体架构

MindSpore 算子从上到下分为以下层次：

```
Python API (tensor_method.py, ops.function, mint)
    │
    ▼
API Definition (ops/api_def/*.yaml)
    │  ─ 绑定 Python 方法到 op_yaml，决定各设备的执行路径
    ▼
Op Definition (ops/op_def/yaml/*.yaml)
    │  ─ 定义参数、返回值、dispatch 策略
    ▼
OpDef (C++) + OpFuncImpl (Infer)
    │  ─ Shape/Type 推导
    ▼
Kernel Dispatch
    ├── CPU:    ops/kernel/cpu/    (NativeCpuKernelMod, nnacl, eigen, mkldnn)
    ├── GPU:    ops/kernel/gpu/    (CUDA kernels, cuDNN, cuBLAS)
    └── Ascend: ops/kernel/ascend/ (ACLNN, AICore, AICPU, GE Adapter)
```

### 关键 C++ 数据结构

`OpDef` 定义在 `mindspore/core/include/ops/op_def.h`：

```cpp
struct OpDef {
  std::string name_;
  std::vector<OpInputArg> args_;
  std::vector<OpOutputArg> returns_;
  std::vector<Signature> signatures_;
  std::unordered_map<std::string, size_t> indexes_;
  OpFuncImpl &func_impl_;
  bool enable_dispatch_;
  bool is_view_;
  bool is_graph_view_;
};
```

---

## 2. YAML 算子定义

### 路径

- 主定义: `mindspore/ops/op_def/yaml/`
- 废弃方法: `mindspore/ops/op_def/yaml/deprecated/`
- 仅推导: `mindspore/ops/op_def/yaml/infer/`
- 函数式: `mindspore/ops/op_def/func_op/`
- 核心定义: `mindspore/core/ops/ops_def/`

### YAML 字段说明

| 字段 | 说明 |
|------|------|
| `args` | 输入参数列表 |
| `args.dtype` | 参数类型: `tensor`, `int`, `bool`, `number`, `TypeId`, `str` 等 |
| `args.default` | 默认值 |
| `args.type_cast` | 类型提升: `number` 表示标量自动转 tensor |
| `args.arg_handler` | 参数处理器: 如 `dtype_to_type_id` |
| `args.prim_init` | 是否在 Primitive 初始化时设置为属性 |
| `args_signature.dtype_group` | 类型提升分组 |
| `returns` | 输出参数列表 |
| `returns.inplace` | inplace 操作的输入索引 |
| `dispatch.enable` | 是否启用 PyBoost dispatch |
| `dispatch.Ascend/CPU/GPU` | 设备特化: `None` 表示不支持，自定义名称表示 Customize |
| `function.disable` | 禁止生成 functional 接口 |
| `class.disable` | 禁止生成 Primitive 类 |
| `class.name` | 自定义 Primitive 类名 |
| `view` | 是否为 View 算子 |
| `bprop_expander` | 是否使用 C++ bprop，`False` 表示各子算子独立 autograd |

### 典型示例

**简单算子** (sigmoid):
```yaml
sigmoid:
    args:
        input:
            dtype: tensor
    returns:
        output:
            dtype: tensor
    dispatch:
        enable: True
```

**带 dispatch 和默认值** (addmm):
```yaml
addmm:
  args:
    input: { dtype: tensor }
    mat1:  { dtype: tensor }
    mat2:  { dtype: tensor }
    beta:  { dtype: number, default: 1 }
    alpha: { dtype: number, default: 1 }
  returns:
    output: { dtype: tensor }
  function:
    disable: True
  dispatch:
    enable: True
    Ascend: AddmmAscend    # Ascend 使用 Customize
    GPU: None              # GPU 不支持 PyBoost
```

**带类型提升** (add):
```yaml
add:
  args:
    input: { dtype: tensor, type_cast: number }
    other: { dtype: tensor, type_cast: number }
  args_signature:
    dtype_group: (input, other)
  returns:
    output: { dtype: tensor }
  dispatch:
    enable: True
    Ascend: AddAscend
```

**多输出** (topk_ext):
```yaml
topk_ext:
  args:
    input: { dtype: tensor }
    k:       { dtype: int }
    dim:     { dtype: int, default: -1 }
    largest: { dtype: bool, default: True }
    sorted:  { dtype: bool, default: True }
  returns:
    values:  { dtype: tensor }
    indices: { dtype: tensor }
  class:
    name: TopkExt
  dispatch:
    enable: True
    GPU: None
```

### 自动生成路径

YAML 定义会自动生成以下文件：
- Python: `mindspore/python/mindspore/ops/auto_generate/`
- C++: `mindspore/ops/op_def/auto_generate/`, `mindspore/ops/include/primitive/auto_generate/`
- PyBoost: `mindspore/pyboost/auto_generate/`, `mindspore/ops/kernel/{ascend,gpu,cpu}/pyboost_impl/auto_generate/`

---

## 3. API 定义层

### 路径

- API 定义: `mindspore/ops/api_def/*.yaml`
- 函数文档: `mindspore/ops/api_def/function_doc/`
- 方法文档: `mindspore/ops/api_def/method_doc/`

### 字段说明

| 字段 | 说明 |
|------|------|
| `op_yaml` | 关联的 op 定义 YAML 文件 |
| `py_method` | Python 回调函数名 (定义在 tensor_method.py 等) |
| `kwonlyargs` | 仅限关键字参数列表 |
| `Ascend/CPU/GPU` | 设备执行路径: `pyboost` 或 `py_method` |
| `interface` | 接口类型: `tensor`, `function`, 或 `tensor, function` |
| `disable_scalar_tensor` | 禁止标量自动转 tensor 的参数 |

### 示例

**全设备 PyBoost** (sigmoid):
```yaml
sigmoid:
  op_yaml: sigmoid_op.yaml
  py_method: tensor_sigmoid
  Ascend: pyboost
  CPU: pyboost
  GPU: pyboost
  interface: tensor
```

**混合路径 + 多入口** (add):
```yaml
add:
  - op_yaml: add_scalar_op.yaml
    py_method: tensor_add_ext
    kwonlyargs: alpha
    Ascend: pyboost
    CPU: py_method
    GPU: py_method
    disable_scalar_tensor: other
    interface: tensor, function

  - op_yaml: add_ext_op.yaml
    py_method: tensor_add_ext
    kwonlyargs: alpha
    Ascend: pyboost
    CPU: pyboost
    GPU: pyboost
    interface: tensor, function

  - op_yaml: deprecated/add_method.yaml
    py_method: deprecated_tensor_add
    Ascend: py_method
    CPU: py_method
    GPU: py_method
    interface: tensor
```

---

## 4. Shape/Type 推导 (Infer)

### 路径

- 新接口 (GeneralInfer): `mindspore/ops/infer/ops_func_impl/`
- 核心推导: `mindspore/core/ops/ops_func_impl/`
- 前端推导: `mindspore/ops/infer/ops_frontend_func_impl/`
- 梯度推导: `mindspore/ops/infer/grad/`
- 符号推导: `mindspore/ops/infer/symbol_ops_impl/`

### 基类层级

```
OpFuncImpl (基类)
    ├── EltwiseOpFuncImpl    ── 逐元素算子, 输出 shape = 输入 shape
    ├── SigmoidFuncImpl      ── 继承 Eltwise, 覆盖类型推导
    ├── GatherFuncImpl       ── 复杂 shape 推导 (batch_dims)
    ├── ConcatFuncImpl       ── 多输入 + 类型提升
    └── ReshapeFuncImpl      ── 动态 shape + -1 推导
```

### 关键接口

```cpp
class OpFuncImpl {
 public:
  // 新版 GeneralInfer 接口 (推荐)
  virtual ShapeArray InferShape(const PrimitivePtr &primitive,
                                const InferInfoPtrList &input_infos) const;
  virtual std::vector<TypeId> InferType(const PrimitivePtr &primitive,
                                        const InferInfoPtrList &input_infos) const;
  virtual bool GeneralInferRegistered() const { return false; }

  // 旧版接口
  virtual BaseShapePtr InferShape(const PrimitivePtr &primitive,
                                  const std::vector<AbstractBasePtr> &input_args) const;
  virtual TypePtr InferType(const PrimitivePtr &primitive,
                            const std::vector<AbstractBasePtr> &input_args) const;
};
```

### 注册方式

```cpp
// 简单注册 (新接口)
REGISTER_SIMPLE_INFER(kNameSigmoid, SigmoidFuncImpl)

// 完整注册 (旧接口)
REGISTER_PRIMITIVE_OP_INFER_IMPL(Softmax, prim::kPrimSoftmax, SoftmaxInfer, false)
```

### 典型实现

**逐元素 (Eltwise)** — 输出与输入同 shape/type:
```cpp
ShapeArray EltwiseOpFuncImpl::InferShape(const PrimitivePtr &primitive,
                                         const InferInfoPtrList &input_infos) const {
  return {input_infos[kInputIndex0]->GetShape()};
}
```

**类型提升 (Sigmoid)** — int/bool 输入提升为 float32:
```cpp
std::vector<TypeId> SigmoidFuncImpl::InferType(const PrimitivePtr &primitive,
                                               const InferInfoPtrList &input_infos) const {
  const auto &input_type_id = input_infos[kInputIndex0]->GetType();
  static const std::vector<TypeId> int_or_bool = {kNumberTypeUInt8, kNumberTypeInt8, kNumberTypeInt16,
                                                  kNumberTypeInt32, kNumberTypeInt64, kNumberTypeBool};
  bool is_int_or_bool = std::any_of(int_or_bool.begin(), int_or_bool.end(),
                                    [&input_type_id](const TypeId &t) { return input_type_id == t; });
  if (is_int_or_bool) {
    return {kNumberTypeFloat32};
  }
  return {input_type_id};
}
```

### 动态 Shape 处理要点

- 使用 `x->IsDynamic()` 判断，不要用 `IsDynamic(shape)`
- `-1` 表示某维度动态，`-2` 表示 rank 动态
- 输出依赖输入值时 (如 Unique, NonZero)，需标记为 compute-dependent

---

## 5. Kernel 实现

### 5.1 CPU Kernel

**路径**:
- Native: `mindspore/ops/kernel/cpu/native/`
- NNACL: `mindspore/ops/kernel/cpu/nnacl/`
- Eigen/MKL-DNN: `mindspore/ops/kernel/cpu/eigen/`, `mkldnn/`
- PyBoost: `mindspore/ops/kernel/cpu/pyboost/`

**基类**: `NativeCpuKernelMod`
- `Init()` — 一次性初始化
- `Resize()` — shape 变化时调用，计算 workspace
- `Launch()` — 主计算入口

**注册**:
```cpp
// 方式1: 工厂注册
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, Sigmoid,
    []() { return std::make_shared<ArithmeticSelfCpuKernelMod>(kSigmoid); });

// 方式2: 宏简化
ARITHMETIC_SELF_CPU_REGISTER(Sigmoid, kSigmoid);
```

### 5.2 GPU Kernel

**路径**:
- CUDA wrapper: `mindspore/ops/kernel/gpu/cuda/`
- CUDA impl: `mindspore/ops/kernel/gpu/cuda_impl/cuda_ops/`
- cuDNN/cuBLAS: `mindspore/ops/kernel/gpu/cudnn/`, `cublas/`
- PyBoost: `mindspore/ops/kernel/gpu/pyboost/`

**CUDA 实现示例** (Sigmoid):
```cuda
template <typename T>
__global__ void SigmoidKernel(size_t size, const T *input, T *output) {
  for (int pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size;
       pos += blockDim.x * gridDim.x) {
    output[pos] = static_cast<T>(1) / (static_cast<T>(1) + exp(-input[pos]));
  }
}
```

**注册** (逐元素):
```cpp
{"Sigmoid",
 {REGISTER_UNARY_FLOAT_TYPE(ElwiseOpType::kSigmoid),
  REGISTER_UNARY_COMPLEX_TYPE(ElwiseOpType::kSigmoid)}},
MS_ELEWISE_KERNEL_FACTORY_REG_BY_CREATOR(Sigmoid);
```

### 5.3 Ascend Kernel

**路径**:
- ACLNN: `mindspore/ops/kernel/ascend/aclnn/`
  - 自动生成: `pyboost_impl/auto_generate/`
  - Customize: `pyboost_impl/customize/`
  - KernelMod: `kernel_mod_impl/customize/`
- ACL Op: `mindspore/ops/kernel/ascend/aclop/`
- HCCL: `mindspore/ops/kernel/ascend/hccl/`
- AICPU: `mindspore/ops/kernel/ascend/aicpu/`

**ACLNN 调用** (PyBoost Customize):
```cpp
LAUNCH_ACLNN(aclnnSigmoid, device_context, op->stream_id(), input_tensor, outputs[0]);
```

**ACLNN KernelMod 注册**:
```cpp
class InplaceSigmoidAscend : public AclnnKernelMod {
  InplaceSigmoidAscend() : AclnnKernelMod("aclnnInplaceSigmoid") {}
  // ...
};
MS_ACLNN_KERNEL_FACTORY_REG(InplaceSigmoid, InplaceSigmoidAscend);
```

### 5.4 动态 Shape Kernel 要点

- `Resize()` 返回 `KRET_INVALID_SHAPE` 表示 shape 含 `-1`
- `ResetResource()` 在 `Resize` 前调用
- 输出动态时: 设置 `is_need_retrieve_output_shape_ = true`，实现 `SyncData()` 和 `GetOutputs()`

---

## 6. 反向传播 (Bprop)

### 路径

- C++ Expander: `mindspore/ccsrc/frontend/expander/bprop/grad_ops/`
  - `grad_nn_ops.cc` — NN 相关算子
  - `grad_math_ops.cc` — 数学算子
  - `grad_array_ops.cc` — 数组算子
  - `grad_comm_ops.cc` — 通信算子
- Python (实验性): `mindspore/python/mindspore/ops/_grad_experimental/`

### 注册宏

```cpp
#define REG_BPROP_BUILDER(name) ...
#define BODYFUNC(v) [](BpropBuilder * (v)) -> NodePtrList
```

### 输入约定

- `i0, i1, ...` — 前向输入
- `iN` — 前向输出 (N = 输入数量)
- `iN+1` — 上游梯度 dout

### 典型模式

**基本反向** (Sigmoid):
```cpp
REG_BPROP_BUILDER("Sigmoid").SetUnusedInputs({i0}).SetBody(BODYFUNC(ib) {
  const auto &out = ib->GetInput(i1);     // 前向输出
  const auto &dout = ib->GetInput(i2);    // 上游梯度
  auto dx = ib->SigmoidGrad(out, dout);
  return {dx};
});
```

**释放无用张量**:
```cpp
REG_BPROP_BUILDER("Conv2D").FreeUselessValues(FreeTensorsOfMul).SetBody(BODYFUNC(ib) { ... });
REG_BPROP_BUILDER("TopkExt").FreeUselessValues_IO({i0, i3, i4}, {i0}).SetBody(BODYFUNC(ib) { ... });
```

**不可导输入返回零**:
```cpp
return {ib->OutZeros(n), ib->OutZeros(m), ib->OutZeros(t)};
```

### 关键注意事项

1. 不可导输入用 `ib->OutZeros(xxx)`
2. 梯度为真零时用 `ib->ZerosLikeExt()` 而非 `ZerosLike`
3. inplace 算子: 不要把用到的输入加入 `unused_inputs`; 叶子节点先用 `x*1` 或 `x+0`
4. 反向中需要前向输入时用 `CloneInplaceInput()`
5. 选择性梯度: 用 `need_compute_grad_out()` 判断是否需要计算

---

## 7. Python 接口层

### 路径

- Tensor 方法: `mindspore/python/mindspore/ops/tensor_method.py`
- 废弃方法: `mindspore/python/mindspore/ops/deprecated_tensor_method.py`
- 函数接口: `mindspore/python/mindspore/ops/function/` (如 `nn_func.py`, `math_func.py`)
- 函数重载: `mindspore/python/mindspore/ops/functional_overload.py`
- Operations: `mindspore/python/mindspore/ops/operations/`
- Mint 接口: `mindspore/python/mindspore/mint/`
- 自动生成: `mindspore/python/mindspore/ops/auto_generate/`

### 接口类型

| 类型 | 调用方式 | 实现位置 |
|------|---------|---------|
| Tensor 方法 | `x.sigmoid()` | `tensor_method.py` → `auto_generate.sigmoid` |
| Functional | `ops.sigmoid(x)` | `ops/function/` → `auto_generate.sigmoid` |
| Mint | `mint.sigmoid(x)` | `mint/__init__.py` → `functional_overload` |
| Operations | `ops.Sigmoid()(x)` | `ops/operations/nn_ops.py` → Primitive |

---

## 8. PyBoost Customize

当 YAML 中 `dispatch` 指定了设备特化名称时，需要手动实现 Customize。

### Ascend Customize

**路径**: `mindspore/ops/kernel/ascend/aclnn/pyboost_impl/customize/`

**模板**:
```cpp
tensor::TensorPtr OpNameAscendCustomize(const std::shared_ptr<OpRunner> &op,
                                        const TensorPtr &input_tensor, ...) {
  // 1. 推导输出
  OpRunner::InferOpOutput(op, input_tensor, ...);
  // 2. 准备输入/输出
  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), input_tensor, ...);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());
  // 3. 异步执行
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>([op, input_tensor, ...]() {
    auto device_context = op->device_context();
    const auto &outputs = op->outputs();
    PyBoostUtils::MallocOpInputs(device_context, input_tensor, ...);
    PyBoostUtils::MallocOpOutputs(device_context, outputs);
    LAUNCH_ACLNN(aclnnOpName, device_context, op->stream_id(), input_tensor, ..., outputs[0]);
  }));
  return op->output(0);
}
```

### CPU Customize (组合算子)

**路径**: `mindspore/ops/kernel/cpu/pyboost/customize/`

**模板** (如 SiLU = Sigmoid * x):
```cpp
void SiLUCPUCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &x_tensor) {
  const auto &sigmoid = CREATE_PYBOOST_OP(Sigmoid, device::DeviceType::kCPU);
  const auto &mul = CREATE_PYBOOST_OP(Mul, device::DeviceType::kCPU);
  const auto &sigmoid_tensor = sigmoid->Call(x_tensor);
  mul->Call(x_tensor, sigmoid_tensor);
  op->set_outputs(mul->outputs());
}
```

---

## 9. C++ 算子组合

### PyNative 模式

**原理**: 每个子算子独立 autograd，YAML 中设置 `bprop_expander: False`

**实现路径**: `mindspore/ops/kernel/{ascend,cpu,gpu}/pyboost/customize/`

**使用 C++ API**: `#include "kernel/functions/auto_generate/functions.h"`

```cpp
// Dense = transpose(weight) + matmul(input, transposed_weight) + bias
auto transposed = transpose(weight, perm);
auto contig = contiguous(transposed);
auto result = matmul_ext(input, contig);
if (bias != nullptr) {
  result = add(result, bias);
}
```

### 图模式 (Meta DSL)

**注册**: `REGISTER_FUNCTION_OP(CustomOp)`

**实现**:
```cpp
BeginFunction(CustomOp, x, y) {
  auto result = Call(Prim(MatMulExt), x, y);
  Return(result);
} EndFunction(CustomOp)
```

**控制流**: `If`, `For`, `While`, `Scan`, `ForiLoop`

**反向**: `PRIMITIVE_BPROP_REG(CustomOp, CustomOpGrad)`

---

## 10. View 算子

View 算子的输出与输入共享存储（无拷贝）。

### 路径

- 实现: `mindspore/ops/view/`
- 注册: `REG_VIEW_STRIDES_CALC_FUN(XXX, XXXCalc)`

### YAML 标记

```yaml
my_view_op:
  args: ...
  returns: ...
  view: True
```

### 需实现

- `shape` — 输出 shape
- `strides` — 输出 strides
- `offset` — 存储偏移

---

## 11. 源码导航速查表

### 给定算子名 `OpName`，快速定位各层代码

| 层次 | 路径模式 | 示例 |
|------|---------|------|
| YAML 定义 | `ops/op_def/yaml/{op_name}_op.yaml` | `sigmoid_op.yaml` |
| API 定义 | `ops/api_def/{op_name}.yaml` | `sigmoid.yaml` |
| Infer (新) | `ops/infer/ops_func_impl/{op_name}.cc/.h` | `sigmoid.cc` |
| Infer (旧) | `core/ops/ops_func_impl/{op_name}.cc` | — |
| CPU Kernel | `ops/kernel/cpu/native/*_cpu_kernel.cc` 或 `ops/kernel/cpu/pyboost/` | `arithmetic_self_cpu_kernel.cc` |
| GPU Kernel | `ops/kernel/gpu/cuda/*_gpu_kernel.cc` 或 `cuda_impl/` | `elementwise_ops_gpu_kernel.cc` |
| GPU CUDA | `ops/kernel/gpu/cuda_impl/cuda_ops/*_impl.cu` | `sigmoid_impl.cu` |
| Ascend ACLNN | `ops/kernel/ascend/aclnn/pyboost_impl/` | `auto_generate/sigmoid.h` |
| Ascend Customize | `ops/kernel/ascend/aclnn/pyboost_impl/customize/` | `sigmoid_grad.cc` |
| Ascend KernelMod | `ops/kernel/ascend/aclnn/kernel_mod_impl/customize/` | `inplace_sigmoid_aclnn_kernel.cc` |
| Bprop | `ccsrc/frontend/expander/bprop/grad_ops/grad_nn_ops.cc` | `REG_BPROP_BUILDER("Sigmoid")` |
| Python Tensor 方法 | `python/mindspore/ops/tensor_method.py` | `tensor_sigmoid` |
| Python Functional | `python/mindspore/ops/function/nn_func.py` | `sigmoid` |
| Mint | `python/mindspore/mint/__init__.py` | `mint.sigmoid` |
| Operations | `python/mindspore/ops/operations/nn_ops.py` | `Sigmoid` |
| 测试 | `tests/st/ops/` | `test_sigmoid.py` |

### 按文件类型搜索

注意: 工作目录中的实际路径前缀为 `mindspore/mindspore/`。

```bash
# 搜索算子 YAML 定义
rg -l "^op_name:" mindspore/mindspore/ops/op_def/yaml/

# 搜索 Infer 实现
rg -l "class OpNameFuncImpl" mindspore/mindspore/ops/infer/

# 搜索 Kernel 注册
rg "FACTORY_REG.*OpName" mindspore/mindspore/ops/kernel/

# 搜索 Bprop 注册
rg 'REG_BPROP_BUILDER\("OpName"\)' mindspore/mindspore/ccsrc/frontend/expander/

# 搜索 Python API
rg "def tensor_op_name" mindspore/mindspore/python/mindspore/ops/
```
