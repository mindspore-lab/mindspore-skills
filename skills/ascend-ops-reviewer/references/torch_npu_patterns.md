# torch_npu 特定模式

本文档总结 torch_npu 算子实现的特定模式和最佳实践。

## 1. 算子注册模式

### 1.1 TORCH_LIBRARY_IMPL 注册

```cpp
// torch_npu/csrc/aten/ops/AddKernel.cpp
TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("add.Tensor", TORCH_FN(add_npu));
  m.impl("add_.Tensor", TORCH_FN(add_npu_));  // inplace variant
}
```

**关键点**:
- `PrivateUse1` 是 NPU 设备的标识
- 函数名需要匹配 PyTorch 标准 (如 `add.Tensor`)
- Inplace 变体使用 `_` 后缀

### 1.2 算子实现签名

```cpp
at::Tensor add_npu(
    const at::Tensor& self,
    const at::Tensor& other,
    const at::Scalar& alpha) {
  // 实现
}
```

**关键点**:
- 参数类型和顺序必须与 PyTorch 标准一致
- 使用 `const at::Tensor&` 避免拷贝
- 返回新张量 (非 inplace)

## 2. ACLNN 算子调用模式

### 2.1 基本调用流程

```cpp
#include "torch_npu/csrc/aten/ops/aclnn/AclnnUtils.h"

at::Tensor add_npu(const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha) {
  // 1. 输入检查
  TORCH_CHECK(self.device().is_privateuseone(), "Expected NPU tensor");

  // 2. 输出张量分配
  at::Tensor result = at::empty_like(self);

  // 3. 调用 ACLNN
  EXEC_NPU_CMD(aclnnAdd, self, other, alpha, result);

  return result;
}
```

**关键点**:
- 使用 `EXEC_NPU_CMD` 宏调用 ACLNN 算子
- 输出张量需要预先分配
- 错误检查使用 `TORCH_CHECK`

### 2.2 Inplace 变体

```cpp
at::Tensor& add_npu_(at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha) {
  // Inplace 操作直接修改 self
  EXEC_NPU_CMD(aclnnInplaceAdd, self, other, alpha);
  return self;
}
```

**关键点**:
- 返回类型是 `at::Tensor&` (引用)
- 直接修改输入张量
- 函数名以 `_` 结尾

### 2.3 多输出算子

```cpp
std::tuple<at::Tensor, at::Tensor> topk_npu(
    const at::Tensor& self,
    int64_t k,
    int64_t dim,
    bool largest,
    bool sorted) {

  at::Tensor values = at::empty({...}, self.options());
  at::Tensor indices = at::empty({...}, self.options().dtype(at::kLong));

  EXEC_NPU_CMD(aclnnTopK, self, k, dim, largest, sorted, values, indices);

  return std::make_tuple(values, indices);
}
```

**关键点**:
- 返回 `std::tuple`
- 每个输出都需要预分配
- indices 通常是 `int64` 类型

## 3. 设备内存管理

### 3.1 张量分配

```cpp
// 分配与输入相同属性的张量
at::Tensor output = at::empty_like(input);

// 分配指定 shape 和 dtype
at::Tensor output = at::empty(
    {batch, channels, height, width},
    input.options().dtype(at::kFloat)
);

// 分配在 NPU 设备上
at::Tensor output = at::empty(
    {n},
    at::TensorOptions().device(at::kPrivateUse1).dtype(at::kFloat)
);
```

### 3.2 内存拷贝

```cpp
// CPU -> NPU
at::Tensor npu_tensor = cpu_tensor.to(at::kPrivateUse1);

// NPU -> CPU
at::Tensor cpu_tensor = npu_tensor.to(at::kCPU);

// NPU -> NPU (同设备拷贝)
at::Tensor copy = npu_tensor.clone();
```

### 3.3 临时 Buffer 管理

```cpp
// 使用 RAII 管理临时内存
{
  at::Tensor temp = at::empty({size}, input.options());
  // 使用 temp
  // 离开作用域自动释放
}
```

## 4. Stream 管理

### 4.1 获取当前 Stream

```cpp
#include "torch_npu/csrc/core/NPUStream.h"

c10_npu::NPUStream stream = c10_npu::getCurrentNPUStream();
```

### 4.2 异步执行

```cpp
// ACLNN 算子默认是异步的
EXEC_NPU_CMD(aclnnAdd, self, other, alpha, result);
// 此时 kernel 可能还在执行

// 需要同步时
c10_npu::npuSynchronizeDevice();
```

### 4.3 多 Stream 并发

```cpp
c10_npu::NPUStream stream1 = c10_npu::getStreamFromPool();
c10_npu::NPUStream stream2 = c10_npu::getStreamFromPool();

{
  c10_npu::NPUStreamGuard guard(stream1);
  // 在 stream1 上执行
  EXEC_NPU_CMD(aclnnOp1, ...);
}

{
  c10_npu::NPUStreamGuard guard(stream2);
  // 在 stream2 上执行
  EXEC_NPU_CMD(aclnnOp2, ...);
}
```

## 5. 错误处理

### 5.1 输入验证

```cpp
at::Tensor add_npu(const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha) {
  // 设备检查
  TORCH_CHECK(self.device().is_privateuseone(), "Expected NPU tensor, got ", self.device());
  TORCH_CHECK(other.device().is_privateuseone(), "Expected NPU tensor, got ", other.device());

  // Shape 检查
  TORCH_CHECK(self.dim() == other.dim(), "Tensors must have same number of dimensions");

  // Dtype 检查
  TORCH_CHECK(self.scalar_type() == other.scalar_type(), "Tensors must have same dtype");

  // 实现...
}
```

### 5.2 ACLNN 错误处理

```cpp
// EXEC_NPU_CMD 会自动检查返回值
// 如果失败会抛出异常，包含详细错误信息
EXEC_NPU_CMD(aclnnAdd, self, other, alpha, result);
```

## 6. Dtype 处理

### 6.1 支持的 Dtype

torch_npu 常见支持的 dtype:
- `at::kFloat` (float32)
- `at::kHalf` (float16)
- `at::kBFloat16` (bfloat16)
- `at::kInt` (int32)
- `at::kLong` (int64)
- `at::kBool`

### 6.2 Dtype 转换

```cpp
// 检查并转换 dtype
at::Tensor input_fp32 = input;
if (input.scalar_type() != at::kFloat) {
  input_fp32 = input.to(at::kFloat);
}

// 处理...

// 转换回原始 dtype
if (result.scalar_type() != input.scalar_type()) {
  result = result.to(input.scalar_type());
}
```

### 6.3 混合精度

```cpp
// 自动类型提升
at::ScalarType result_dtype = at::promoteTypes(
    self.scalar_type(),
    other.scalar_type()
);

at::Tensor result = at::empty(output_shape, self.options().dtype(result_dtype));
```

## 7. Shape 推导

### 7.1 Broadcasting

```cpp
#include "torch_npu/csrc/aten/ops/OpInterface.h"

// 使用 PyTorch 的 broadcasting 工具
at::IntArrayRef self_shape = self.sizes();
at::IntArrayRef other_shape = other.sizes();

// 推导输出 shape
std::vector<int64_t> output_shape = at::infer_size(self_shape, other_shape);
```

### 7.2 Reduction

```cpp
// Reduction 操作的 shape 推导
std::vector<int64_t> output_shape;
for (int64_t i = 0; i < input.dim(); i++) {
  if (i == dim) {
    if (keepdim) {
      output_shape.push_back(1);
    }
    // else: 不添加这个维度
  } else {
    output_shape.push_back(input.size(i));
  }
}
```

## 8. 常见 API

### 8.1 张量属性查询

```cpp
// Shape
at::IntArrayRef shape = tensor.sizes();
int64_t dim = tensor.dim();
int64_t size_at_dim = tensor.size(0);

// Dtype
at::ScalarType dtype = tensor.scalar_type();

// Device
c10::Device device = tensor.device();
bool is_npu = device.is_privateuseone();

// 连续性
bool is_contiguous = tensor.is_contiguous();

// 元素数量
int64_t numel = tensor.numel();
```

### 8.2 张量操作

```cpp
// Reshape (view)
at::Tensor reshaped = tensor.reshape({-1, channels});

// Transpose
at::Tensor transposed = tensor.transpose(0, 1);

// Slice
at::Tensor sliced = tensor.slice(0, start, end);

// Squeeze/Unsqueeze
at::Tensor squeezed = tensor.squeeze(dim);
at::Tensor unsqueezed = tensor.unsqueeze(dim);
```

## 9. 测试模式

### 9.1 单元测试结构

```python
# test/test_ops/test_add.py
import torch
import torch_npu

class TestAdd(unittest.TestCase):
    def test_add_basic(self):
        # CPU 参考实现
        cpu_a = torch.randn(2, 3)
        cpu_b = torch.randn(2, 3)
        cpu_result = cpu_a + cpu_b

        # NPU 实现
        npu_a = cpu_a.npu()
        npu_b = cpu_b.npu()
        npu_result = npu_a + npu_b

        # 对比
        self.assertTrue(torch.allclose(
            cpu_result,
            npu_result.cpu(),
            atol=1e-5,
            rtol=1e-5
        ))
```

### 9.2 边界情况测试

```python
def test_add_edge_cases(self):
    # 空张量
    self.assertEqual((torch.tensor([]).npu() + torch.tensor([]).npu()).shape, torch.Size([0]))

    # 零维张量
    scalar = torch.tensor(1.0).npu()
    result = scalar + scalar
    self.assertEqual(result.item(), 2.0)

    # Broadcasting
    a = torch.randn(3, 1).npu()
    b = torch.randn(1, 4).npu()
    result = a + b
    self.assertEqual(result.shape, torch.Size([3, 4]))
```

## 10. 性能优化技巧

### 10.1 避免不必要的同步

```cpp
// ❌ 错误: 频繁同步
for (int i = 0; i < n; i++) {
  EXEC_NPU_CMD(aclnnOp, ...);
  c10_npu::npuSynchronizeDevice();  // 每次都同步!
}

// ✅ 正确: 批量执行后同步
for (int i = 0; i < n; i++) {
  EXEC_NPU_CMD(aclnnOp, ...);
}
c10_npu::npuSynchronizeDevice();  // 只同步一次
```

### 10.2 使用 Contiguous 张量

```cpp
// 确保输入是连续的，提高性能
at::Tensor input_contiguous = input.contiguous();
EXEC_NPU_CMD(aclnnOp, input_contiguous, ...);
```

### 10.3 复用输出 Buffer

```cpp
// Inplace 操作避免分配
at::Tensor& result = self;  // 复用输入
EXEC_NPU_CMD(aclnnInplaceOp, result, ...);
```

## 11. 调试技巧

### 11.1 打印调试信息

```cpp
#include <iostream>

std::cout << "Tensor shape: " << tensor.sizes() << std::endl;
std::cout << "Tensor dtype: " << tensor.scalar_type() << std::endl;
std::cout << "Tensor device: " << tensor.device() << std::endl;
```

### 11.2 检查 NaN/Inf

```cpp
bool has_nan = at::isnan(tensor).any().item<bool>();
bool has_inf = at::isinf(tensor).any().item<bool>();

TORCH_CHECK(!has_nan, "Tensor contains NaN");
TORCH_CHECK(!has_inf, "Tensor contains Inf");
```

### 11.3 性能分析

```python
import torch
import torch_npu

# 使用 profiler
with torch.autograd.profiler.profile(use_npu=True) as prof:
    result = model(input.npu())

print(prof.key_averages().table(sort_by="npu_time_total"))
```

## 12. 常见问题

### 12.1 设备不匹配

```
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cpu and npu:0!
```

**解决**: 确保所有输入都在 NPU 上

### 12.2 ACLNN 算子不存在

```
RuntimeError: ACLNN operator not found: aclnnXxx
```

**解决**: 检查 CANN 版本，该算子可能需要更高版本

### 12.3 Shape 不匹配

```
RuntimeError: The size of tensor a (3) must match the size of tensor b (4) at non-singleton dimension 1
```

**解决**: 检查 broadcasting 规则，或显式 reshape

## 参考资料

- torch_npu 官方文档
- ACLNN 算子手册
- PyTorch C++ API 文档
- CANN 开发指南
