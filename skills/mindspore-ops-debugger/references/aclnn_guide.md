# ACLNN 算子适配指南

## 目录

1. [ACLNN 适配概览](#1-aclnn-适配概览)
2. [分析阶段](#2-分析阶段)
3. [YAML 配置](#3-yaml-配置)
4. [PyNative 实现 (PyBoost Customize)](#4-pynative-实现-pyboost-customize)
5. [图模式实现 (KBK KernelMod)](#5-图模式实现-kbk-kernelmod)
6. [Infer 实现](#6-infer-实现)
7. [反向实现](#7-反向实现)
8. [Mint 接口](#8-mint-接口)
9. [自验指导](#9-自验指导)
10. [常见问题](#10-常见问题)

---

## 1. ACLNN 适配概览

ACLNN 是 Ascend 设备上调用 CANN 算子库的标准接口。适配流程:

```
分析 (MS 接口 vs PTA 接口 vs aclnn 接口)
    │
    ▼
设计 (确定适配方案: Auto / Mapping / Custom)
    │
    ▼
实现
    ├── YAML 定义 + dispatch 配置
    ├── PyNative: PyBoost Customize
    ├── 图模式 (KBK): AclnnKernelMod
    ├── Infer: GeneralInfer
    └── Bprop: REG_BPROP_BUILDER
    │
    ▼
验证 (精度零偏差 + 显存对比 + UT/ST)
    │
    ▼
交付 (设计文档 + PR + 测试 PR)
```

---

## 2. 分析阶段

### 2.1 对比 MS、PTA、aclnn 接口

| 对比项 | 来源 | 检查点 |
|-------|------|--------|
| **MS 接口** | `ops/op_def/yaml/` | 参数名、类型、默认值 |
| **PTA 接口** | PyTorch + torch_npu | 参数映射、行为差异 |
| **aclnn 接口** | CANN 文档 / cann_ops_adpt | 输入输出、约束条件 |

### 2.2 确定适配方案

| 方案 | 条件 | YAML 配置 | 需要写代码 |
|------|------|-----------|-----------|
| **Auto** | MS API 与 aclnn 完全匹配 | `dispatch.enable: True` | 不需要 |
| **Mapping** | MS API 与 aclnn 名称不同但参数匹配 | 在 `aclnn_config.yaml` 中添加映射 | 不需要 |
| **Custom** | 参数不匹配或需要额外处理 | `dispatch.Ascend: OpNameAscend` | 需要 |

### 2.3 分析要点

- **参数映射**: MS 参数如何映射到 aclnn 参数
- **类型处理**: dtype 是否需要转换 (如 TypeId → aclDataType)
- **内存格式**: 输入输出是否需要 contiguous
- **Compute Dependent**: 输出 shape 是否依赖输入值

---

## 3. YAML 配置

### Auto 模式

```yaml
sigmoid:
    args:
        input: { dtype: tensor }
    returns:
        output: { dtype: tensor }
    dispatch:
        enable: True    # 自动匹配 aclnnSigmoid
```

### Mapping 模式

在 `aclnn_config.yaml` 中添加:
```yaml
ms_op_name:
  aclnn_name: aclnnDifferentName
```

### Custom 模式

```yaml
my_op:
    args:
        input: { dtype: tensor }
        dim: { dtype: int, default: -1 }
    returns:
        output: { dtype: tensor }
    dispatch:
        enable: True
        Ascend: MyOpAscend    # 指向 Customize 实现
```

---

## 4. PyNative 实现 (PyBoost Customize)

### 路径

- 头文件: `mindspore/ops/kernel/ascend/aclnn/pyboost_impl/customize/{op_name}.h`
- 实现: `mindspore/ops/kernel/ascend/aclnn/pyboost_impl/customize/{op_name}.cc`

### 模板

**头文件**:
```cpp
#pragma once
#include "ir/tensor.h"
#include "runtime/pynative/op_runner.h"

namespace mindspore::kernel::pyboost {
tensor::TensorPtr MyOpAscendCustomize(const std::shared_ptr<OpRunner> &op,
                                      const TensorPtr &input_tensor,
                                      const Int64ImmPtr &dim);
}  // namespace mindspore::kernel::pyboost
```

**实现**:
```cpp
#include "kernel/ascend/aclnn/pyboost_impl/customize/my_op.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "kernel/ascend/aclnn/pyboost_impl/aclnn_utils.h"

namespace mindspore::kernel::pyboost {
namespace {

void MyOpAscendCall(const std::shared_ptr<OpRunner> &op,
                    const device::DeviceContext *device_context,
                    const TensorPtr &input_tensor,
                    int64_t dim,
                    const std::vector<tensor::TensorPtr> &outputs) {
  LAUNCH_ACLNN(aclnnMyOp, device_context, op->stream_id(),
               input_tensor, dim, outputs[0]);
}

}  // namespace

tensor::TensorPtr MyOpAscendCustomize(const std::shared_ptr<OpRunner> &op,
                                      const TensorPtr &input_tensor,
                                      const Int64ImmPtr &dim) {
  // 1. 推导输出
  OpRunner::InferOpOutput(op, input_tensor, dim);

  // 2. 获取标量值
  auto dim_value = GetValue<int64_t>(dim);

  // 3. 准备输入/输出
  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), input_tensor);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  // 4. 异步 dispatch
  PyBoostUtils::DispatchRun(
      std::make_shared<runtime::PyBoostDeviceTask>([op, input_tensor, dim_value]() {
        auto device_context = op->device_context();
        const auto &outputs = op->outputs();
        PyBoostUtils::MallocOpInputs(device_context, input_tensor);
        PyBoostUtils::MallocOpOutputs(device_context, outputs);
        MyOpAscendCall(op, device_context, input_tensor, dim_value, outputs);
      }));

  return op->output(0);
}

}  // namespace mindspore::kernel::pyboost
```

### Compute-Dependent 算子

当输出 shape 依赖计算结果时 (如 Unique, NonZero):

```cpp
tensor::TensorPtr UniqueAscendCustomize(const std::shared_ptr<OpRunner> &op,
                                        const TensorPtr &input_tensor) {
  OpRunner::InferOpOutput(op, input_tensor);
  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), input_tensor);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  // 同步版本: 需要等待结果以获取实际输出 shape
  auto device_context = op->device_context();
  const auto &outputs = op->outputs();
  PyBoostUtils::MallocOpInputs(device_context, input_tensor);
  PyBoostUtils::MallocOpOutputs(device_context, outputs);

  LAUNCH_ACLNN(aclnnUnique, device_context, op->stream_id(), input_tensor, outputs[0]);

  // 同步等待并更新输出 shape
  op->stream_id()->synchronize();
  auto real_shape = GetActualShape(outputs[0]);
  outputs[0]->set_shape(real_shape);

  return op->output(0);
}
```

### Inplace 算子

```cpp
tensor::TensorPtr InplaceOpAscendCustomize(const std::shared_ptr<OpRunner> &op,
                                           const TensorPtr &input_tensor) {
  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), input_tensor);
  op->set_outputs({input_tensor});  // 输出复用输入

  PyBoostUtils::DispatchRun(
      std::make_shared<runtime::PyBoostDeviceTask>([op, input_tensor]() {
        auto device_context = op->device_context();
        PyBoostUtils::MallocOpInputs(device_context, input_tensor);
        LAUNCH_ACLNN(aclnnInplaceOp, device_context, op->stream_id(), input_tensor);
      }));

  return op->output(0);
}
```

---

## 5. 图模式实现 (KBK KernelMod)

### 路径

`mindspore/ops/kernel/ascend/aclnn/kernel_mod_impl/customize/`

### 模板

```cpp
#include "kernel/ascend/aclnn/aclnn_kernel_mod.h"

namespace mindspore::kernel {
namespace my_op {

class MyOpAscend : public AclnnKernelMod {
 public:
  MyOpAscend() : AclnnKernelMod("aclnnMyOp") {}

  bool Launch(const std::vector<KernelTensor *> &inputs,
              const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs,
              void *stream_ptr) override {
    // 基本实现: 由基类 AclnnKernelMod 处理
    return AclnnKernelMod::Launch(inputs, workspace, outputs, stream_ptr);
  }

  // 如果需要额外 workspace
  void GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                        const std::vector<KernelTensor *> &outputs) override {
    // 获取 aclnn 所需 workspace 大小
    auto input_tensor = inputs[0];
    auto output_tensor = outputs[0];
    GetWorkspaceForResize(input_tensor, output_tensor);
  }
};

MS_ACLNN_KERNEL_FACTORY_REG(MyOp, MyOpAscend);

}  // namespace my_op
}  // namespace mindspore::kernel
```

### 自定义 Launch

当 aclnn 接口需要特殊参数处理时:

```cpp
bool Launch(const std::vector<KernelTensor *> &inputs,
            const std::vector<KernelTensor *> &workspace,
            const std::vector<KernelTensor *> &outputs,
            void *stream_ptr) override {
  auto input = inputs[0];
  auto dim = GetValue<int64_t>(inputs[1]->GetValue());
  auto output = outputs[0];

  // 自定义 aclnn 调用
  LAUNCH_ACLNN(aclnnMyOp, stream_ptr, input, dim, output);
  return true;
}
```

---

## 6. Infer 实现

### 路径

`mindspore/ops/infer/ops_func_impl/`

### 新接口 (GeneralInfer)

```cpp
// my_op.h
#pragma once
#include "ops/ops_func_impl/op_func_impl.h"

namespace mindspore::ops {

class MyOpFuncImpl : public OpFuncImpl {
 public:
  ShapeArray InferShape(const PrimitivePtr &primitive,
                        const InferInfoPtrList &input_infos) const override;
  std::vector<TypeId> InferType(const PrimitivePtr &primitive,
                                const InferInfoPtrList &input_infos) const override;
  bool GeneralInferRegistered() const override { return true; }
};

}  // namespace mindspore::ops
```

```cpp
// my_op.cc
#include "ops/infer/ops_func_impl/my_op.h"

namespace mindspore::ops {

ShapeArray MyOpFuncImpl::InferShape(const PrimitivePtr &primitive,
                                    const InferInfoPtrList &input_infos) const {
  auto input_shape = input_infos[kInputIndex0]->GetShape();
  auto dim = input_infos[kInputIndex1]->GetIntValue();

  // 计算输出 shape
  ShapeVector output_shape = input_shape;
  if (dim.has_value()) {
    auto axis = dim.value();
    if (axis < 0) axis += static_cast<int64_t>(input_shape.size());
    output_shape.erase(output_shape.begin() + axis);
  } else {
    // 动态 dim
    output_shape = ShapeVector(input_shape.size() - 1, -1);
  }
  return {output_shape};
}

std::vector<TypeId> MyOpFuncImpl::InferType(const PrimitivePtr &primitive,
                                            const InferInfoPtrList &input_infos) const {
  return {input_infos[kInputIndex0]->GetType()};
}

REGISTER_SIMPLE_INFER(kNameMyOp, MyOpFuncImpl)

}  // namespace mindspore::ops
```

### 注意事项

- 使用 `ShapeVector` 和 `TypeId`，不要用 `AbstractBasePtr`
- 用 `x->IsDynamic()` 检测动态 shape
- `-1` 表示动态维度，`-2` 表示动态 rank
- 不在 InferShape 中调用 `AddAttr`

---

## 7. 反向实现

### 路径

`mindspore/ccsrc/frontend/expander/bprop/grad_ops/`

### 选择合适的文件

| 算子类别 | 文件 |
|---------|------|
| NN 算子 | `grad_nn_ops.cc` |
| 数学算子 | `grad_math_ops.cc` |
| 数组算子 | `grad_array_ops.cc` |
| 通信算子 | `grad_comm_ops.cc` |

### 反向模板

```cpp
REG_BPROP_BUILDER("MyOp").SetUnusedInputs({i1}).SetBody(BODYFUNC(ib) {
  auto input = ib->GetInput(i0);   // 前向输入
  auto out = ib->GetInput(i1);     // 前向输出 (标记为 unused)
  auto dout = ib->GetInput(i2);    // 上游梯度

  auto dx = ib->MyOpGrad(input, dout);
  return {dx, ib->OutZeros(dim)};  // dim 是标量参数, 返回零梯度
});
```

### 反向注意事项

1. **非可导参数**: 用 `ib->OutZeros(param)` 返回
2. **真零梯度**: 用 `ib->ZerosLikeExt()` 而非 `ZerosLike`
3. **inplace 前向**: 不要把前向输入加入 `unused_inputs`
4. **前向值保护**: 反向需要前向输入时用 `CloneInplaceInput`
5. **显存优化**: 用 `FreeUselessValues` 释放不需要的前向张量
6. **选择性梯度**: 用 `need_compute_grad_out()` 判断

---

## 8. Mint 接口

### 添加 mint 接口

在 `mindspore/python/mindspore/mint/__init__.py` 中添加:

```python
from mindspore.ops.functional_overload import my_op
```

或在 `functional_overload.py` 中定义重载:

```python
def my_op(input, dim=-1):
    return _my_op_impl(input, dim)
```

---

## 9. 自验指导

### 9.1 精度零偏差

目标: 与 PTA (PyTorch + NPU) 输出二进制一致。

```python
import mindspore as ms
import torch
import torch_npu
import numpy as np

# 固定种子
np.random.seed(42)
input_data = np.random.randn(3, 4).astype(np.float32)

# MindSpore
ms_output = mint.my_op(ms.Tensor(input_data)).asnumpy()
np.save("ms_output.npy", ms_output)

# PTA
torch_output = torch.my_op(torch.tensor(input_data).npu()).cpu().numpy()
np.save("torch_output.npy", torch_output)

# 对比 (md5 应相同)
import hashlib
print(hashlib.md5(open("ms_output.npy", "rb").read()).hexdigest())
print(hashlib.md5(open("torch_output.npy", "rb").read()).hexdigest())
```

### 9.2 显存对比

```python
import mindspore as ms

# MindSpore 显存
ms.set_context(device_target="Ascend")
# ... 运行算子 ...
ms_memory = ms.runtime.max_memory_allocated()

# PTA 显存
import torch_npu
# ... 运行算子 ...
pta_memory = torch_npu.npu.max_memory_allocated()

print(f"MS: {ms_memory / 1024**2:.1f} MB")
print(f"PTA: {pta_memory / 1024**2:.1f} MB")
```

### 9.3 正向 + 反向验证

```python
import mindspore as ms
import torch

# 正向对比
ms_out = mint.my_op(ms.Tensor(input_data))
torch_out = torch.my_op(torch.tensor(input_data))
assert np.allclose(ms_out.asnumpy(), torch_out.numpy(), rtol=1e-4, atol=1e-4)

# 反向对比
ms_grad_fn = ms.grad(mint.my_op)
ms_grad = ms_grad_fn(ms.Tensor(input_data))

torch_input = torch.tensor(input_data, requires_grad=True)
torch_out = torch.my_op(torch_input)
torch_out.sum().backward()
torch_grad = torch_input.grad

assert np.allclose(ms_grad.asnumpy(), torch_grad.numpy(), rtol=1e-4, atol=1e-4)
```

---

## 10. 常见问题

### Q: ACLNN 接口报参数类型错误

检查 MS tensor dtype 是否在 aclnn 支持范围内。部分 aclnn 接口不支持 int8、bool 等类型。

### Q: PyBoost Customize 编译失败

检查头文件路径和 namespace。确认 `LAUNCH_ACLNN` 宏的参数与 aclnn 接口匹配。

### Q: KBK 模式精度与 PyNative 不一致

可能是 KernelMod 和 PyBoost Customize 使用了不同的 aclnn 接口或参数。确认两者调用路径一致。

### Q: 动态 shape 在 KBK 下报错

检查 `Resize()` 是否正确处理 `-1` shape，是否返回 `KRET_INVALID_SHAPE`。

### Q: Inplace 算子显存异常

确认 inplace 算子的 YAML 中设置了 `returns.inplace` 指向正确的输入索引。

### Q: 反向精度差

先用 fp32 测试排除精度问题。检查 bprop 中是否正确处理了所有输入的梯度，特别是 `SetUnusedInputs` 配置。
