# MindSpore 特定模式

本文档总结 MindSpore 算子实现的特定模式和最佳实践。

## 1. 算子定义模式 (YAML)

### 1.1 基本 YAML 结构

```yaml
# mindspore/ops/op_def/yaml/add.yaml
- op: Add
  description: |
    Adds two tensors element-wise.
  inputs:
    - name: x
      description: First input tensor
    - name: y
      description: Second input tensor
  outputs:
    - name: output
      description: Output tensor
  infer:
    - python: mindspore.ops.infer.add_infer
  bprop:
    - python: mindspore.ops.bprop.add_bprop
```

**关键点**:
- `op` 字段是算子名称 (PascalCase)
- `inputs`/`outputs` 定义输入输出
- `infer` 指向 shape/dtype 推导函数
- `bprop` 指向反向传播函数

### 1.2 带属性的算子

```yaml
- op: Conv2D
  description: 2D convolution operator
  inputs:
    - name: x
      description: Input tensor
    - name: weight
      description: Convolution kernel
  attrs:
    - name: stride
      type: int
      default: 1
      description: Stride of the convolution
    - name: padding
      type: int
      default: 0
      description: Padding size
  outputs:
    - name: output
      description: Output tensor
```

**关键点**:
- `attrs` 定义算子属性
- 每个属性有 `type` 和 `default` 值

## 2. Shape/Dtype 推导 (Infer)

### 2.1 基本 Infer 函数

```python
# mindspore/ops/infer/add_infer.py
from mindspore.ops.infer import InferShape, InferDtype

class AddInfer(InferShape, InferDtype):
    def infer_shape(self, x_shape, y_shape):
        """Infer output shape."""
        # Broadcasting
        return self.broadcast_shape(x_shape, y_shape)

    def infer_dtype(self, x_dtype, y_dtype):
        """Infer output dtype."""
        # Type promotion
        return self.promote_dtype(x_dtype, y_dtype)
```

**关键点**:
- 继承 `InferShape` 和 `InferDtype`
- 实现 `infer_shape` 和 `infer_dtype` 方法
- 使用内置工具函数 (如 `broadcast_shape`)

### 2.2 动态 Shape 处理

```python
def infer_shape(self, x_shape):
    """Infer shape with dynamic dimensions."""
    batch, channels, height, width = x_shape

    # 检查动态维度
    if batch == -1:
        out_batch = -1
    else:
        out_batch = batch

    # 计算输出 shape
    out_height = (height - kernel_size) // stride + 1
    out_width = (width - kernel_size) // stride + 1

    return (out_batch, channels, out_height, out_width)
```

**关键点**:
- `-1` 表示动态维度
- 动态维度传播到输出

### 2.3 Reduction 算子 Infer

```python
def infer_shape(self, x_shape, axis, keep_dims):
    """Infer shape for reduction operations."""
    if axis is None:
        # Reduce all dimensions
        if keep_dims:
            return tuple([1] * len(x_shape))
        else:
            return ()  # Scalar

    # Normalize negative axis
    if axis < 0:
        axis = len(x_shape) + axis

    # Build output shape
    if keep_dims:
        output_shape = list(x_shape)
        output_shape[axis] = 1
    else:
        output_shape = [s for i, s in enumerate(x_shape) if i != axis]

    return tuple(output_shape)
```

## 3. 反向传播 (Bprop)

### 3.1 基本 Bprop 函数

```python
# mindspore/ops/bprop/add_bprop.py
from mindspore.ops import operations as P

def add_bprop(x, y, out, dout):
    """Backward propagation for Add."""
    # dL/dx = dL/dout * dout/dx = dout * 1 = dout
    # dL/dy = dL/dout * dout/dy = dout * 1 = dout
    return (dout, dout)
```

**关键点**:
- 参数: 前向输入 + 前向输出 + 输出梯度
- 返回: 每个输入的梯度 (tuple)
- 梯度顺序与输入顺序一致

### 3.2 带 Broadcasting 的 Bprop

```python
def add_bprop(x, y, out, dout):
    """Backward with broadcasting."""
    reduce_sum = P.ReduceSum(keep_dims=False)

    # 计算需要 reduce 的维度
    dx = dout
    dy = dout

    # 如果 x 被 broadcast，需要 reduce
    if x.shape != dout.shape:
        reduce_axes = get_broadcast_axes(x.shape, dout.shape)
        dx = reduce_sum(dout, reduce_axes)
        dx = dx.reshape(x.shape)

    # 如果 y 被 broadcast，需要 reduce
    if y.shape != dout.shape:
        reduce_axes = get_broadcast_axes(y.shape, dout.shape)
        dy = reduce_sum(dout, reduce_axes)
        dy = dy.reshape(y.shape)

    return (dx, dy)
```

**关键点**:
- Broadcasting 的反向需要 reduce
- 使用 `ReduceSum` 聚合梯度

### 3.3 Reduction 算子 Bprop

```python
def reduce_sum_bprop(x, axis, keep_dims, out, dout):
    """Backward for ReduceSum."""
    # 梯度需要 broadcast 回原始 shape
    if not keep_dims and axis is not None:
        # 恢复被 reduce 的维度
        dout = P.ExpandDims()(dout, axis)

    # Broadcast 到输入 shape
    dx = P.BroadcastTo(x.shape)(dout)

    return (dx,)
```

**关键点**:
- Reduction 的反向是 broadcast
- 需要恢复被 reduce 的维度

### 3.4 带选择的 Bprop (如 ReLU)

```python
def relu_bprop(x, out, dout):
    """Backward for ReLU."""
    # ReLU: y = max(0, x)
    # dy/dx = 1 if x > 0 else 0
    dx = dout * (x > 0).astype(dout.dtype)
    return (dx,)
```

**关键点**:
- 使用前向输入 `x` 判断梯度是否传播
- 类型转换确保 dtype 一致

## 4. ACLNN 适配模式

### 4.1 Python 层调用

```python
# mindspore/ops/operations/nn_ops.py
from mindspore.ops.primitive import Primitive

class Add(Primitive):
    """Add primitive."""

    @prim_attr_register
    def __init__(self):
        """Initialize Add."""
        self.init_prim_io_names(inputs=['x', 'y'], outputs=['output'])

    def __call__(self, x, y):
        """Call ACLNN kernel."""
        return self.run_aclnn(x, y)
```

### 4.2 C++ Kernel 注册

```cpp
// mindspore/ccsrc/plugin/device/ascend/kernel/aclnn/add_aclnn_kernel.cc
#include "plugin/device/ascend/kernel/aclnn/aclnn_kernel_mod.h"

class AddAclnnKernelMod : public AclnnKernelMod {
 public:
  bool Launch(const std::vector<KernelTensor *> &inputs,
              const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs,
              void *stream_ptr) override {
    // 获取输入
    auto x = inputs[0];
    auto y = inputs[1];
    auto output = outputs[0];

    // 调用 ACLNN
    auto ret = aclnnAdd(x->device_ptr(), y->device_ptr(), output->device_ptr(), stream_ptr);
    if (ret != ACL_SUCCESS) {
      MS_LOG(ERROR) << "aclnnAdd failed: " << ret;
      return false;
    }

    return true;
  }
};

// 注册 kernel
MS_ACLNN_KERNEL_FACTORY_REG(Add, AddAclnnKernelMod);
```

**关键点**:
- 继承 `AclnnKernelMod`
- 实现 `Launch` 方法
- 使用 `MS_ACLNN_KERNEL_FACTORY_REG` 注册

## 5. PyNative vs Graph 模式

### 5.1 PyNative 模式特点

```python
import mindspore as ms
ms.set_context(mode=ms.PYNATIVE_MODE)

# 动态图模式，逐行执行
x = ms.Tensor([1, 2, 3])
y = x + 1  # 立即执行
print(y)   # 可以直接打印
```

**特点**:
- 动态执行，易于调试
- 支持 Python 控制流
- 性能较低

### 5.2 Graph 模式特点

```python
import mindspore as ms
ms.set_context(mode=ms.GRAPH_MODE)

@ms.jit
def add_func(x, y):
    return x + y

# 静态图模式，编译后执行
result = add_func(x, y)
```

**特点**:
- 静态编译，性能高
- 不支持动态控制流
- 调试困难

### 5.3 兼容两种模式

```python
def my_op(x, y):
    """Compatible with both modes."""
    # 避免使用 Python 控制流
    # 使用 MindSpore 算子实现逻辑
    return ops.where(x > 0, x + y, x - y)
```

**关键点**:
- 使用 MindSpore 算子而非 Python 逻辑
- 避免 `if`/`for` 等控制流

## 6. Dtype 处理

### 6.1 支持的 Dtype

```python
import mindspore as ms

# 常用 dtype
ms.float32  # float32
ms.float16  # float16
ms.int32    # int32
ms.int64    # int64
ms.bool_    # bool
```

### 6.2 Dtype 推导

```python
def infer_dtype(self, x_dtype, y_dtype):
    """Infer output dtype with type promotion."""
    # 类型提升规则
    if x_dtype == ms.float64 or y_dtype == ms.float64:
        return ms.float64
    elif x_dtype == ms.float32 or y_dtype == ms.float32:
        return ms.float32
    elif x_dtype == ms.float16 or y_dtype == ms.float16:
        return ms.float16
    else:
        return x_dtype  # 整数类型
```

### 6.3 Dtype 转换

```python
# 显式转换
x_fp32 = x.astype(ms.float32)

# 在算子内部转换
def forward(self, x):
    if x.dtype != ms.float32:
        x = x.astype(ms.float32)
    # 处理...
    return result
```

## 7. 错误处理

### 7.1 Python 层验证

```python
class Add(Primitive):
    def __call__(self, x, y):
        # 类型检查
        if not isinstance(x, Tensor):
            raise TypeError(f"x must be Tensor, got {type(x)}")

        # Shape 检查
        if x.ndim != y.ndim:
            raise ValueError(f"x and y must have same ndim, got {x.ndim} and {y.ndim}")

        # Dtype 检查
        if x.dtype != y.dtype:
            raise TypeError(f"x and y must have same dtype, got {x.dtype} and {y.dtype}")

        return self.run_aclnn(x, y)
```

### 7.2 C++ 层错误处理

```cpp
bool AddAclnnKernelMod::Launch(...) {
  // 检查输入
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "Invalid inputs or outputs";
    return false;
  }

  // 调用 ACLNN
  auto ret = aclnnAdd(...);
  if (ret != ACL_SUCCESS) {
    MS_LOG(ERROR) << "aclnnAdd failed with error code: " << ret;
    return false;
  }

  return true;
}
```

## 8. 测试模式

### 8.1 基本测试结构

```python
# tests/st/ops/test_add.py
import pytest
import numpy as np
import mindspore as ms
from mindspore import Tensor, ops

class TestAdd:
    def test_add_basic(self):
        """Test basic functionality."""
        x = Tensor([1, 2, 3], ms.float32)
        y = Tensor([4, 5, 6], ms.float32)
        result = ops.add(x, y)

        expected = Tensor([5, 7, 9], ms.float32)
        assert np.allclose(result.asnumpy(), expected.asnumpy())

    def test_add_broadcast(self):
        """Test broadcasting."""
        x = Tensor([[1], [2], [3]], ms.float32)  # (3, 1)
        y = Tensor([4, 5, 6], ms.float32)        # (3,)
        result = ops.add(x, y)

        assert result.shape == (3, 3)
```

### 8.2 梯度测试

```python
def test_add_grad(self):
    """Test gradient computation."""
    from mindspore import grad

    def forward(x, y):
        return ops.add(x, y).sum()

    # 计算梯度
    grad_fn = grad(forward, grad_position=(0, 1))

    x = Tensor([1.0, 2.0, 3.0], ms.float32)
    y = Tensor([4.0, 5.0, 6.0], ms.float32)

    dx, dy = grad_fn(x, y)

    # 梯度应该都是 1
    assert np.allclose(dx.asnumpy(), np.ones(3))
    assert np.allclose(dy.asnumpy(), np.ones(3))
```

### 8.3 边界情况测试

```python
def test_add_edge_cases(self):
    """Test edge cases."""
    # 空张量
    x = Tensor([], ms.float32)
    y = Tensor([], ms.float32)
    result = ops.add(x, y)
    assert result.shape == (0,)

    # 零维张量
    x = Tensor(1.0, ms.float32)
    y = Tensor(2.0, ms.float32)
    result = ops.add(x, y)
    assert result.shape == ()
    assert result.asnumpy() == 3.0
```

## 9. 性能优化

### 9.1 使用 JIT 编译

```python
@ms.jit
def optimized_func(x, y):
    """JIT compiled function."""
    return x + y

# 首次调用会编译
result = optimized_func(x, y)  # 慢
# 后续调用直接使用编译结果
result = optimized_func(x, y)  # 快
```

### 9.2 避免频繁 CPU-NPU 传输

```python
# ❌ 错误: 频繁传输
for i in range(100):
    x_npu = Tensor(data[i]).to("Ascend")
    result = ops.add(x_npu, y_npu)
    result_cpu = result.asnumpy()  # 每次都传回 CPU

# ✅ 正确: 批量处理
x_npu = Tensor(data).to("Ascend")  # 一次传输
results = []
for i in range(100):
    result = ops.add(x_npu[i], y_npu)
    results.append(result)
results_cpu = ops.stack(results).asnumpy()  # 一次传回
```

### 9.3 使用 Inplace 操作

```python
# 非 inplace
x = x + 1  # 分配新内存

# Inplace (如果支持)
ops.assign_add(x, 1)  # 原地修改
```

## 10. 调试技巧

### 10.1 打印中间结果

```python
import mindspore as ms
ms.set_context(mode=ms.PYNATIVE_MODE)  # 切换到动态图

def debug_func(x, y):
    result = x + y
    print(f"x: {x}")
    print(f"y: {y}")
    print(f"result: {result}")
    return result
```

### 10.2 检查 NaN/Inf

```python
def check_nan_inf(tensor, name="tensor"):
    """Check for NaN/Inf in tensor."""
    if ops.isnan(tensor).any():
        raise ValueError(f"{name} contains NaN")
    if ops.isinf(tensor).any():
        raise ValueError(f"{name} contains Inf")
```

### 10.3 使用 Profiler

```python
from mindspore import Profiler

# 启动 profiler
profiler = Profiler(output_path='./profiler_data')

# 运行代码
result = model(input)

# 停止并分析
profiler.analyse()
```

## 11. 常见问题

### 11.1 Shape 推导错误

```
ValueError: For 'Add', the shape of 'x' and 'y' must be broadcastable
```

**解决**: 检查 infer_shape 函数，确保正确处理 broadcasting

### 11.2 Dtype 不匹配

```
TypeError: For 'Add', the dtype of 'x' and 'y' must be the same
```

**解决**: 在 infer_dtype 中添加类型提升逻辑

### 11.3 梯度为零

```
Gradient is zero for all inputs
```

**解决**: 检查 bprop 函数，确保梯度正确传播

### 11.4 Graph 模式编译失败

```
RuntimeError: The 'if' statement is not supported in graph mode
```

**解决**: 使用 MindSpore 算子替代 Python 控制流

## 12. 源码路径参考

```
mindspore/
├── ops/
│   ├── op_def/yaml/          # 算子 YAML 定义
│   ├── infer/                # Shape/Dtype 推导
│   ├── bprop/                # 反向传播
│   └── operations/           # Python 算子类
├── ccsrc/
│   ├── plugin/device/ascend/kernel/aclnn/  # ACLNN kernel
│   └── frontend/expander/bprop/            # Bprop 实现
└── python/mindspore/ops/     # Python API
```

## 参考资料

- MindSpore 算子开发指南
- ACLNN 算子适配文档
- MindSpore API 文档
- 算子测试框架 2.0 文档
