# torch_npu 常见修复模式

## 目录

1. [ACLNN 算子适配修复](#1-aclnn-算子适配修复)
2. [Format 处理修复](#2-format-处理修复)
3. [dtype 转换修复](#3-dtype-转换修复)
4. [算子注册修复](#4-算子注册修复)
5. [反向传播修复](#5-反向传播修复)
6. [运行时修复](#6-运行时修复)

---

## 1. ACLNN 算子适配修复

### 1.1 添加 DO_COMPATIBILITY fallback

**问题**: 新 ACLNN 接口在旧 CANN 版本上不存在，导致 `undefined symbol`。

**修复**: 添加 DO_COMPATIBILITY 宏，回退到 aclops 实现。

```cpp
// op_plugin/ops/opapi/XxxKernelNpu.cpp

// Before: no fallback
at::Tensor xxx_npu(const at::Tensor& self) {
    auto result = npu_preparation::apply_tensor(self);
    EXEC_NPU_CMD(aclnnXxx, self, result);
    return result;
}

// After: add DO_COMPATIBILITY fallback
at::Tensor xxx_npu(const at::Tensor& self) {
    DO_COMPATIBILITY(aclnnXxx, acl_op::xxx_npu(self));

    auto result = npu_preparation::apply_tensor(self);
    EXEC_NPU_CMD(aclnnXxx, self, result);
    return result;
}
```

### 1.2 修复 ACLNN 参数不匹配

**问题**: aclnn 接口签名变更，参数数量或类型不匹配。

**修复**: 对齐 CANN 头文件中的函数签名。

```cpp
// Before: old signature
EXEC_NPU_CMD(aclnnXxx, self, other, result);

// After: new signature adds alpha parameter
EXEC_NPU_CMD(aclnnXxx, self, other, alpha, result);
```

### 1.3 添加 Workspace 处理

**问题**: ACLNN 算子需要额外 workspace 但未分配。

**修复**: 使用 EXEC_NPU_CMD 宏（自动处理 workspace），或手动分配。

```cpp
// EXEC_NPU_CMD automatically handles workspace via GetWorkspaceSize + Execute
// If manual control is needed:
uint64_t ws_size = 0;
aclnnXxxGetWorkspaceSize(self_desc, result_desc, &ws_size, &executor);
void* ws_addr = nullptr;
if (ws_size > 0) {
    auto ws_tensor = at::empty({static_cast<int64_t>(ws_size)}, self.options().dtype(at::kByte));
    ws_addr = ws_tensor.data_ptr();
}
aclnnXxx(ws_addr, ws_size, executor, stream);
```

---

## 2. Format 处理修复

### 2.1 强制基础格式输入

**问题**: 算子不支持 NZ/FRACTAL_Z 等私有格式输入。

**修复**: 在算子入口处转换为基础格式。

```cpp
// Before: directly use input (may be in NZ format)
at::Tensor xxx_npu(const at::Tensor& self) {
    auto result = npu_preparation::apply_tensor(self);
    EXEC_NPU_CMD(aclnnXxx, self, result);
    return result;
}

// After: ensure base format input
at::Tensor xxx_npu(const at::Tensor& self) {
    DO_COMPATIBILITY(aclnnXxx, acl_op::xxx_npu(self));

    at::Tensor self_cp = self;
    if (!at_npu::native::FormatHelper::IsBaseFormatType(self)) {
        self_cp = at_npu::native::custom_ops::npu_format_cast(self, ACL_FORMAT_ND);
    }
    auto result = npu_preparation::apply_tensor_without_format(self_cp);
    EXEC_NPU_CMD(aclnnXxx, self_cp, result);
    return result;
}
```

### 2.2 使用 apply_tensor_without_format

**问题**: `apply_tensor` 会继承输入的 format，导致输出 format 不正确。

**修复**: 使用 `apply_tensor_without_format` 创建基础格式输出。

```cpp
// Before: inherits input format (may cause issues)
auto result = npu_preparation::apply_tensor(self);

// After: always use base format for output
auto result = npu_preparation::apply_tensor_without_format(output_size, self.options());
```

### 2.3 修复非连续张量

**问题**: 非连续张量传入 ACLNN 导致计算错误。

**修复**: 在算子入口处确保张量连续。

```cpp
at::Tensor xxx_npu(const at::Tensor& self) {
    at::Tensor self_cp = self.is_contiguous() ? self : self.contiguous();
    // ... proceed with self_cp
}
```

---

## 3. dtype 转换修复

### 3.1 添加 dtype 提升

**问题**: NPU fp16 精度不足导致计算溢出。

**修复**: 将输入提升到 float32 计算后再转回。

```cpp
at::Tensor xxx_npu(const at::Tensor& self) {
    DO_COMPATIBILITY(aclnnXxx, acl_op::xxx_npu(self));

    // Promote to float32 for precision
    bool need_cast = (self.scalar_type() == at::kHalf || self.scalar_type() == at::kBFloat16);
    at::Tensor self_cp = need_cast ? self.to(at::kFloat) : self;

    auto result = npu_preparation::apply_tensor_without_format(self_cp);
    EXEC_NPU_CMD(aclnnXxx, self_cp, result);

    // Cast back to original dtype
    return need_cast ? result.to(self.scalar_type()) : result;
}
```

### 3.2 修复 dtype 推导

**问题**: 输出 dtype 推导错误，使用了错误的 dtype。

**修复**: 显式指定输出 dtype。

```cpp
// Before: wrong dtype inference
auto result = npu_preparation::apply_tensor(self);

// After: explicit dtype
auto result = npu_preparation::apply_tensor_without_format(
    output_size, self.options().dtype(at::kFloat));
```

### 3.3 处理 bool 类型

**问题**: NPU 不支持 bool 类型的某些运算。

**修复**: 将 bool 转为 int 计算后转回。

```cpp
at::Tensor xxx_npu(const at::Tensor& self) {
    if (self.scalar_type() == at::kBool) {
        auto self_int = self.to(at::kInt);
        auto result = xxx_npu(self_int);
        return result.to(at::kBool);
    }
    // ... normal path
}
```

---

## 4. 算子注册修复

### 4.1 添加算子注册

**问题**: 算子未注册到 PrivateUse1 dispatch key。

**修复**: 在 TORCH_LIBRARY_IMPL 中添加注册。

```cpp
// In the appropriate registration file
TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("xxx", TORCH_FN(op_plugin::xxx_npu));
    m.impl("xxx.out", TORCH_FN(op_plugin::xxx_out_npu));
}
```

### 4.2 添加 Autograd 注册

**问题**: 算子有 forward 实现但缺少 autograd 注册，导致 backward 失败。

**修复**: 注册到 AutogradPrivateUse1。

```cpp
TORCH_LIBRARY_IMPL(aten, AutogradPrivateUse1, m) {
    m.impl("xxx", TORCH_FN(op_plugin::xxx_autograd));
}
```

### 4.3 修复 DO_COMPATIBILITY 注册

**问题**: DO_COMPATIBILITY 回退到 aclops 但 aclops 命名空间中没有对应实现。

**修复**: 确保 aclops 中有对应实现，或提供内联 fallback。

```cpp
// Option 1: ensure aclops implementation exists
DO_COMPATIBILITY(aclnnXxx, acl_op::xxx_npu(self, other));

// Option 2: inline fallback when no aclops implementation
DO_COMPATIBILITY(aclnnXxx, [&]() {
    // Fallback implementation using OpCommand
    at_npu::native::OpCommand cmd;
    cmd.Name("Xxx").Input(self).Input(other).Output(result).Run();
    return result;
}());
```

---

## 5. 反向传播修复

### 5.1 修复 backward 函数

**问题**: backward 实现中 dtype 或 shape 处理错误。

**修复**: 确保 grad 的 dtype 和 shape 与 forward 输出一致。

```cpp
at::Tensor xxx_backward_npu(const at::Tensor& grad_output, const at::Tensor& self) {
    DO_COMPATIBILITY(aclnnXxxBackward, acl_op::xxx_backward_npu(grad_output, self));

    // Ensure grad has correct dtype
    at::Tensor grad = grad_output.scalar_type() == self.scalar_type()
        ? grad_output
        : grad_output.to(self.scalar_type());

    auto grad_input = npu_preparation::apply_tensor_without_format(self);
    EXEC_NPU_CMD(aclnnXxxBackward, grad, self, grad_input);
    return grad_input;
}
```

### 5.2 添加缺失的 backward

**问题**: 算子只有 forward 没有 backward，导致 autograd 失败。

**修复**: 实现 backward 并注册到 AutogradPrivateUse1。

```cpp
// 1. Implement backward in opapi/
at::Tensor xxx_backward_npu(const at::Tensor& grad, const at::Tensor& self) {
    DO_COMPATIBILITY(aclnnXxxBackward, acl_op::xxx_backward_npu(grad, self));
    auto grad_input = npu_preparation::apply_tensor_without_format(self);
    EXEC_NPU_CMD(aclnnXxxBackward, grad, self, grad_input);
    return grad_input;
}

// 2. Create autograd function
class XxxFunction : public torch::autograd::Function<XxxFunction> {
    static at::Tensor forward(torch::autograd::AutogradContext* ctx,
                              const at::Tensor& self) {
        ctx->save_for_backward({self});
        return op_plugin::xxx_npu(self);
    }
    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::variable_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        return {op_plugin::xxx_backward_npu(grad_outputs[0], saved[0])};
    }
};
```

---

## 6. 运行时修复

### 6.1 添加 Stream 同步

**问题**: 异步执行导致数据竞争或结果不正确。

**修复**: 在关键点添加 stream 同步。

```cpp
// Add synchronization after critical operations
c10_npu::NPUStream stream = c10_npu::getCurrentNPUStream();
aclrtSynchronizeStream(stream);
```

### 6.2 修复内存分配

**问题**: 输出 tensor 大小计算错误导致内存越界。

**修复**: 修正 output size 计算。

```cpp
// Before: wrong output size
auto output_size = {self.size(0), self.size(1)};

// After: correct output size calculation
auto output_size = op_infer::broadcast_ops_npu_output_size(self, other);
auto result = npu_preparation::apply_tensor_without_format(output_size, self.options());
```

### 6.3 修复设备检查

**问题**: 输入张量在不同设备上导致运行时错误。

**修复**: 添加设备一致性检查。

```cpp
at::Tensor xxx_npu(const at::Tensor& self, const at::Tensor& other) {
    TORCH_CHECK(self.device() == other.device(),
        "Expected all tensors to be on the same device, but found at least two devices, ",
        self.device(), " and ", other.device());
    // ...
}
```
