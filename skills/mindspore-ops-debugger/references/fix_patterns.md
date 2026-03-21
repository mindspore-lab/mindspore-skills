# MindSpore 算子常见修复模式

## 目录

1. [精度修复模式](#1-精度修复模式)
2. [Shape 推导修复模式](#2-shape-推导修复模式)
3. [API 签名修复模式](#3-api-签名修复模式)
4. [Kernel 修复模式](#4-kernel-修复模式)
5. [Bprop 修复模式](#5-bprop-修复模式)
6. [编译器/IR 修复模式](#6-编译器ir-修复模式)
7. [运行时修复模式](#7-运行时修复模式)
8. [测试用例修复模式](#8-测试用例修复模式)

---

## 1. 精度修复模式

### 1.1 修复基准比对 dtype 不一致

**问题**: 基准框架 (TF/Torch) 版本升级导致 dtype 自动提升行为变化。

**关联 Issue**: #41934 (nn.Adam), Case CS-002

**修复**: 在基准比对代码中显式指定 dtype。

```python
# 修复前: TF 自动提升 dtype
class TestAdam(OpsFactory):
    def forward_tensorflow_impl(self, ...):
        dense = tf.keras.layers.Dense(units)  # TF 2.18 自动提升为 float32

# 修复后: 显式传入 dtype
class TestAdam(OpsFactory):
    def forward_tensorflow_impl(self, ...):
        dense = tf.keras.layers.Dense(units, dtype=self.dtype)  # 保持 fp16
```

### 1.2 消除反向图中的 Select

**问题**: backward 中 Select 操作在 GE 上导致内存异常。

**关联 Issue**: #41932 (ops.pow), Case CS-001

**修复**: 重写反向逻辑避免使用 Select。

```cpp
// 修复前: 反向使用 Select
auto dx = ib->Select(condition, grad_x, zeros);

// 修复后: 使用 Mul + Cast 替代 Select
auto mask = ib->Cast(condition, input_dtype);
auto dx = ib->Mul(grad_x, mask);
```

### 1.3 切换到 mint 接口

**问题**: 旧模型使用 ops 接口导致精度或随机数不一致。

**关联 Issue**: #42126 (mindone I2VGenXLUNet), Case CS-005

**修复**: 将 `mindspore.ops` 调用替换为 `mindspore.mint`。

```python
# 修复前
import mindspore.ops as ops
output = ops.sigmoid(x)

# 修复后
import mindspore.mint as mint
output = mint.sigmoid(x)
```

### 1.4 CANN 变更后重新基线

**问题**: CANN 升级后算子计算结果变化，但结果仍在合理范围内。

**关联 Issue**: #41977 (DeepSeek loss), Case CS-005

**修复**: 更新基线数据 (baseline)，调整 tolerance。

```python
# 调整容差
np.testing.assert_allclose(ms_output, ref_output, rtol=1e-3, atol=1e-3)
# 改为
np.testing.assert_allclose(ms_output, ref_output, rtol=5e-3, atol=5e-3)
```

### 1.5 aclnn 特殊值缺陷 — 定界到 CANN 侧

**问题**: aclnn 算子对 inf/nan 等特殊值处理不正确，仅特定 dtype 触发（如 complex64 有问题但 complex128 正常）。

**关联 Issue**: #42294 (Reciprocal), Case CS-018

**定界方法**: 三方对比验证。

```python
# 1. PyTorch CPU (基准)
x = torch.tensor([np.inf+0j, np.inf+0j], dtype=torch.complex64)
torch.reciprocal(x)  # tensor([0.+0.j, 0.+0.j]) ← 正确

# 2. torch_npu (aclnn)
x_npu = x.npu()
torch.reciprocal(x_npu)  # tensor([nan+nanj, nan+nanj]) ← 错误

# 3. MindSpore (aclnn)
x_ms = ms.Tensor([np.inf+0j, np.inf+0j], dtype=ms.complex64)
x_ms.reciprocal()  # [nan+nanj, nan+nanj] ← 与 torch_npu 一致

# 结论: MindSpore == torch_npu != PyTorch CPU → aclnn 算子 bug
```

**处理**: 提交 CANN 问题单，MindSpore 侧无需修改。

---

## 2. Shape 推导修复模式

### 2.1 修复广播规则

**问题**: InferShape 中广播条件检查不充分。

**修复**: 补充广播兼容性检查。

```cpp
// 修复前: 简单检查
if (condition_shape != input_shape) {
  MS_EXCEPTION(ValueError) << "shapes can not broadcast";
}

// 修复后: 正确的广播检查
for (size_t i = 0; i < max_rank; ++i) {
  auto cond_dim = i < condition_rank ? condition_shape[condition_rank - 1 - i] : 1;
  auto input_dim = i < input_rank ? input_shape[input_rank - 1 - i] : 1;
  if (cond_dim != input_dim && cond_dim != 1 && input_dim != 1 &&
      cond_dim != -1 && input_dim != -1) {
    MS_EXCEPTION(ValueError) << "shapes can not broadcast";
  }
}
```

### 2.2 修复动态 shape 推导

**问题**: 静态 shape 正确但动态 shape 报错。

**修复**: 在 InferShape 中正确处理 -1 维度。

```cpp
// 修复前: 未考虑动态 shape
auto dim = input_shape[axis];
MS_CHECK_VALUE(dim > 0, "dimension must be positive");

// 修复后: 跳过动态维度检查
auto dim = input_shape[axis];
if (!IsDynamic(dim)) {
  MS_CHECK_VALUE(dim > 0, "dimension must be positive");
}
```

### 2.3 添加编译 Pass 清理无效节点

**问题**: IR 中残留 DeadNode 导致后端解析失败。

**关联 Issue**: #41973 (PixelShuffle), Case CS-009

**修复**: 添加 switch_simplify 等优化 pass。

```cpp
// 在 pipeline 中添加 pass
void AddOptimizePasses(OptPassGroupMap *passes) {
  // ... existing passes ...
  passes->emplace_back("switch_simplify", std::make_shared<SwitchSimplify>());
}
```

---

## 3. API 签名修复模式

### 3.1 修复参数数量不匹配

**问题**: Python 接口参数数量与 YAML 定义不一致。

**修复**: 统一 YAML 定义和 Python 实现。

```yaml
# 修复 YAML: 补充缺失的参数
op_name:
  args:
    input: { dtype: tensor }
    param1: { dtype: int, default: 0 }     # 新增
    param2: { dtype: bool, default: True }  # 新增
  returns:
    output: { dtype: tensor }
```

### 3.2 修复参数校验

**问题**: 缺少参数类型或取值范围校验。

**修复**: 添加校验逻辑。

```python
# 修复: 在 Python 层添加校验
def op_function(input, alpha=1, amsgrad=False):
    if not isinstance(alpha, (int, float)):
        raise TypeError(f"alpha must be a number, got {type(alpha)}")
    if alpha < 0:
        raise ValueError(f"alpha must be non-negative, got {alpha}")
```

### 3.3 修复错误类型一致性

**问题**: 不同 API 抛出不一致的异常类型。

**修复**: 统一异常类型。

```cpp
// 修复前: 使用 value_error
MS_EXCEPTION(ValueError) << "group not initialized";

// 修复后: 统一使用 runtime_error
MS_EXCEPTION(RuntimeError) << "group not initialized";
```

### 3.4 防止自动类型转换

**问题**: PyNative 下 list 被自动转为 tuple。

**关联 Issue**: #41971 (scatternd), Case CS-006

**修复**: 在 ConvertSequence 中区分 list 和 tuple。

```cpp
// 修复: 保持原始类型
if (py::isinstance<py::list>(input)) {
  // 不自动转为 tuple，保持 list
  return ConvertToList(input);
}
```

---

## 4. Kernel 修复模式

### 4.1 CANN 版本兼容

**问题**: 代码使用了新 CANN 版本的 API，老版本编译失败。

**关联 Issue**: #41948 (自定义算子), Case CS-012

**修复**: 用宏隔离版本差异。

```cpp
// 修复: 条件编译
#if CANN_VERSION >= CANN_8_3
  aclrtDevResLimitType limit_type = ...;
  aclrtSetDevResLimit(limit_type, value);
#else
  // 老版本的替代实现
  SetLegacyLimit(value);
#endif
```

### 4.2 修复线程安全

**问题**: 多线程访问共享数据结构导致 crash。

**关联 Issue**: #41935 (自动并行 core dump), Case CS-013

**修复**: 加锁或使用原子操作。

```cpp
// 修复前: 无锁访问
static std::set<std::string> kernel_list;
kernel_list.insert(name);

// 修复后: 加锁
static std::mutex kernel_list_mutex;
static std::set<std::string> kernel_list;
{
  std::lock_guard<std::mutex> lock(kernel_list_mutex);
  kernel_list.insert(name);
}
```

### 4.3 修复 optional 使用

**问题**: 未检查 optional 有值就访问。

**修复**: 添加 has_value 检查。

```cpp
// 修复前
auto value = opt_value.value();

// 修复后
if (opt_value.has_value()) {
  auto value = opt_value.value();
} else {
  // 处理无值情况
  MS_LOG(WARNING) << "optional value not set";
}
```

### 4.4 修复设备地址创建

**问题**: 运行时在地址创建前就检查 ref 地址。

**修复**: 调整检查顺序或跳过特殊情况。

```cpp
// 修复: any-type 输入跳过 ref 地址检查
if (input_type == kTypeAny) {
  continue;  // 跳过 ref 地址检查
}
CheckRefAddress(input);
```

### 4.5 动态 shape kernel 修复

**问题**: Resize 中未处理 -1 shape。

**修复**: 正确返回 KRET_INVALID_SHAPE。

```cpp
int MyKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                         const std::vector<KernelTensor *> &outputs) {
  auto ret = KernelMod::Resize(inputs, outputs);
  if (ret != KRET_OK) return ret;

  auto shape = inputs[0]->GetShapeVector();
  if (std::any_of(shape.begin(), shape.end(),
                  [](int64_t s) { return s < 0; })) {
    return KRET_INVALID_SHAPE;
  }
  // ... 正常处理
  return KRET_OK;
}
```

---

## 5. Bprop 修复模式

### 5.1 添加缺失的 bprop

**问题**: 新算子没有注册 bprop。

**修复**: 添加 REG_BPROP_BUILDER。

```cpp
REG_BPROP_BUILDER("NewOp").SetUnusedInputs({i2}).SetBody(BODYFUNC(ib) {
  auto input = ib->GetInput(i0);
  auto out = ib->GetInput(i1);
  auto dout = ib->GetInput(i3);
  auto dx = ib->Mul(dout, ib->SomeGradFunc(input, out));
  return {dx};
});
```

### 5.2 修复 SetUnusedInputs

**问题**: 错误地把反向需要的输入标记为 unused。

**修复**: 确认反向图中实际使用的输入。

```cpp
// 修复前: 错误标记
REG_BPROP_BUILDER("Op").SetUnusedInputs({i0, i1}).SetBody(BODYFUNC(ib) {
  auto input = ib->GetInput(i0);  // 实际用到了 i0！
  // ...
});

// 修复后: 不标记 i0
REG_BPROP_BUILDER("Op").SetUnusedInputs({i1}).SetBody(BODYFUNC(ib) {
  auto input = ib->GetInput(i0);
  // ...
});
```

### 5.3 不可导输入返回零

**问题**: 非 tensor 输入需要返回合适的零梯度。

**修复**: 使用 OutZeros。

```cpp
// 对于非 tensor 参数 (如 int, bool, dtype)
return {dx, ib->OutZeros(n), ib->OutZeros(m), ib->OutZeros(dtype_param)};
```

### 5.4 Inplace 算子 bprop

**问题**: inplace 操作修改了 input，但 bprop 需要原始 input。

**修复**: 使用 CloneInplaceInput。

```cpp
REG_BPROP_BUILDER("InplaceOp").SetBody(BODYFUNC(ib) {
  // 使用 clone 获取 inplace 前的值
  auto original_input = ib->CloneInplaceInput(i0);
  auto dout = ib->GetInput(i2);
  auto dx = ib->SomeGrad(original_input, dout);
  return {dx};
});
```

### 5.5 选择性梯度计算

**问题**: 对不需要梯度的输入也做了完整计算。

**修复**: 用 need_compute_grad_out 判断。

```cpp
REG_BPROP_BUILDER("Op").SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(i0);
  auto y = ib->GetInput(i1);
  auto dout = ib->GetInput(i3);

  auto dx = x->need_compute_grad_out()
      ? ib->ComputeGradX(x, y, dout)
      : ib->OutZeros(x);
  auto dy = y->need_compute_grad_out()
      ? ib->ComputeGradY(x, y, dout)
      : ib->OutZeros(y);
  return {dx, dy};
});
```

---

## 6. 编译器/IR 修复模式

### 6.1 添加 IR 节点的 Bprop 支持

**问题**: 编译器引入新的 IR 节点但没有对应 bprop。

**修复**: 为新 IR 节点注册 bprop。

### 6.2 添加优化 Pass

**问题**: 特定 IR 模式未被优化 pass 覆盖。

**修复**: 在 pipeline 中添加新 pass 或扩展现有 pass。

### 6.3 修复控制流解析

**问题**: 控制流节点解析逻辑不完整。

**修复**: 在 `control_node_parser.cc` 中补充处理逻辑。

---

## 7. 运行时修复模式

### 7.1 修复导入错误

**问题**: 导入了 module 而非 callable。

**关联 Issue**: #42129 (lazy_inline), Case CS-016

**修复**: 修正 import 语句。

```python
# 修复前
from mindspore import initializer  # 导入了 module

# 修复后
from mindspore.common.initializer import initializer  # 导入了 callable
```

### 7.2 修复设备特定路径

**问题**: 平台差异导致走入不同的执行路径。

**修复**: 添加平台检测和条件处理。

---

## 8. 测试用例修复模式

### 8.1 移除不支持的测试配置

**问题**: 测试用例使用了后端不支持的 dtype 或配置。

**修复**: 在 OpInfo 中正确标记支持的 dtype。

```python
# 修复: 移除不支持的 float64 scalar
OpInfo(
    name='mint.op',
    dtypes_ascend=[ms.float16, ms.float32],  # 移除 float64
    # ...
)
```

### 8.2 添加 skip 条件

**问题**: 特定环境下测试预期失败。

```python
@pytest.mark.skipif(device_target == 'Ascend' and dtype == 'float16',
                    reason="fp16 precision insufficient on Ascend")
def test_op_precision():
    ...
```

### 8.3 调整 tolerance

```python
# 根据 dtype 设置不同 tolerance
tolerance = {
    ms.float16: (1e-2, 1e-2),
    ms.float32: (1e-4, 1e-4),
    ms.float64: (1e-6, 1e-6),
}
rtol, atol = tolerance[dtype]
```
