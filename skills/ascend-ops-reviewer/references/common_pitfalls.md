# 算子实现常见陷阱

本文档总结算子实现中的常见陷阱和错误模式，帮助快速识别潜在问题。

## 1. Shape 推导陷阱

### 1.1 忘记处理 Broadcasting

**问题**: 二元算子未考虑输入 shape 不同的情况

```python
# ❌ 错误: 假设两个输入 shape 相同
def add_infer_shape(x_shape, y_shape):
    return x_shape  # 错误!

# ✅ 正确: 应用 broadcasting 规则
def add_infer_shape(x_shape, y_shape):
    return broadcast_shape(x_shape, y_shape)
```

**检测**: 搜索二元算子的 infer 函数，检查是否调用 broadcast 相关函数

### 1.2 动态 Shape 未考虑

**问题**: 只处理静态 shape，动态 shape (-1) 时出错

```python
# ❌ 错误: 直接计算 shape
output_size = input_shape[0] * input_shape[1]  # -1 * 10 = -10 错误!

# ✅ 正确: 检查动态维度
if input_shape[0] == -1:
    output_size = -1
else:
    output_size = input_shape[0] * input_shape[1]
```

**检测**: 搜索 shape 计算代码，检查是否有 -1 判断

### 1.3 Reduction 维度错误

**问题**: reduce 操作未正确处理 keepdim 参数

```python
# ❌ 错误: 忘记 keepdim
def sum_infer_shape(input_shape, axis):
    return [s for i, s in enumerate(input_shape) if i != axis]

# ✅ 正确: 考虑 keepdim
def sum_infer_shape(input_shape, axis, keepdim=False):
    if keepdim:
        return [s if i != axis else 1 for i, s in enumerate(input_shape)]
    else:
        return [s for i, s in enumerate(input_shape) if i != axis]
```

**检测**: 搜索 reduce 相关算子，检查 keepdim 处理

### 1.4 负索引未处理

**问题**: axis=-1 等负索引未转换为正索引

```python
# ❌ 错误: 直接使用负索引
output_shape[axis] = 1  # axis=-1 时错误!

# ✅ 正确: 转换负索引
if axis < 0:
    axis = len(input_shape) + axis
output_shape[axis] = 1
```

## 2. Dtype 陷阱

### 2.1 整数除法截断

**问题**: Python 2 风格整数除法导致精度损失

```python
# ❌ 错误: 整数除法
result = a / 2  # Python 2 中 5/2=2

# ✅ 正确: 浮点除法或显式转换
result = a / 2.0
# 或
result = a // 2  # 明确使用整除
```

**检测**: 搜索除法操作，检查操作数类型

### 2.2 Float16 精度损失

**问题**: Float16 范围小 (±65504)，容易溢出

```python
# ❌ 错误: 直接计算可能溢出
result = x * y  # x, y 是 float16，结果可能 > 65504

# ✅ 正确: 提升到 float32 计算
result = x.astype(float32) * y.astype(float32)
result = result.astype(float16)
```

**检测**: 搜索 float16 相关代码，检查是否有精度保护

### 2.3 隐式类型转换

**问题**: 不同 dtype 运算导致意外类型提升

```python
# ❌ 错误: 未明确类型
result = int32_tensor + float32_tensor  # 结果是什么类型?

# ✅ 正确: 显式转换
result = int32_tensor.astype(float32) + float32_tensor
```

**检测**: 搜索混合 dtype 运算，检查类型提升规则

### 2.4 整数溢出

**问题**: 整数运算结果超出类型范围

```cpp
// ❌ 错误: int32 乘法可能溢出
int32_t result = a * b;  // a=100000, b=100000 溢出!

// ✅ 正确: 提升到 int64
int64_t result = static_cast<int64_t>(a) * b;
```

**检测**: 搜索整数乘法/加法，检查是否有溢出保护

## 3. 内存安全陷阱

### 3.1 裸指针未检查

**问题**: 指针解引用前未检查 nullptr

```cpp
// ❌ 错误: 直接解引用
auto data = tensor->data_ptr();
data[0] = 1.0;  // data 可能是 nullptr!

// ✅ 正确: 检查指针
auto data = tensor->data_ptr();
if (data == nullptr) {
    throw std::runtime_error("Null pointer");
}
data[0] = 1.0;
```

**检测**: 搜索 `->` 和 `*ptr`，检查前面是否有 nullptr 检查

### 3.2 数组越界

**问题**: 索引访问未检查边界

```cpp
// ❌ 错误: 未检查索引
int value = array[index];  // index 可能 >= size!

// ✅ 正确: 边界检查
if (index < 0 || index >= size) {
    throw std::out_of_range("Index out of bounds");
}
int value = array[index];
```

**检测**: 搜索 `[index]`，检查是否有边界检查

### 3.3 资源未释放

**问题**: malloc/new 后未 free/delete

```cpp
// ❌ 错误: 内存泄漏
float* buffer = new float[size];
// ... 使用 buffer
return;  // 忘记 delete[]!

// ✅ 正确: RAII 或显式释放
std::unique_ptr<float[]> buffer(new float[size]);
// 或
float* buffer = new float[size];
// ... 使用 buffer
delete[] buffer;
```

**检测**: 搜索 `new`/`malloc`，检查是否有对应的 `delete`/`free`

### 3.4 Use-after-free

**问题**: 访问已释放的内存

```cpp
// ❌ 错误: 使用已释放内存
delete[] buffer;
buffer[0] = 1.0;  // 错误!

// ✅ 正确: 释放后置空
delete[] buffer;
buffer = nullptr;
```

**检测**: 搜索 `delete`/`free`，检查后续是否还有访问

## 4. 数值稳定性陷阱

### 4.1 Log(0) / Div(0)

**问题**: 对零或负数取对数，除零

```python
# ❌ 错误: 可能 log(0)
result = np.log(x)  # x=0 时 -inf!

# ✅ 正确: 添加 epsilon
result = np.log(x + 1e-8)
# 或检查
if x <= 0:
    raise ValueError("x must be positive")
```

**检测**: 搜索 `log`/`/`，检查是否有零值保护

### 4.2 Exp 溢出

**问题**: exp(x) 在 x 很大时溢出

```python
# ❌ 错误: exp 溢出
result = np.exp(x)  # x=1000 时溢出!

# ✅ 正确: Clip 输入
result = np.exp(np.clip(x, -100, 100))
```

**检测**: 搜索 `exp`，检查输入范围

### 4.3 累加精度损失

**问题**: 大量浮点数累加导致精度损失

```python
# ❌ 错误: 直接累加
sum = 0.0
for x in large_array:
    sum += x  # 精度损失!

# ✅ 正确: 使用 Kahan 求和或库函数
sum = np.sum(large_array)  # NumPy 有优化
```

**检测**: 搜索循环中的累加操作

### 4.4 Sqrt 负数

**问题**: 对负数开方

```python
# ❌ 错误: sqrt 负数
result = np.sqrt(x)  # x<0 时 NaN!

# ✅ 正确: 检查或 clip
result = np.sqrt(np.maximum(x, 0))
```

**检测**: 搜索 `sqrt`，检查输入范围

## 5. 梯度计算陷阱

### 5.1 忘记处理 Inplace 操作

**问题**: inplace 操作破坏梯度计算

```python
# ❌ 错误: inplace 修改输入
def forward(x):
    x += 1  # inplace!
    return x

def backward(grad):
    return grad  # 错误! x 已被修改
```

**检测**: 搜索 `+=`/`*=` 等 inplace 操作

### 5.2 梯度链断裂

**问题**: 中间变量未保存导致梯度无法回传

```python
# ❌ 错误: 未保存中间变量
def forward(x):
    y = x * 2
    z = y + 1
    return z  # 未保存 y!

def backward(grad_z):
    # 无法计算 grad_y!
    pass
```

**检测**: 检查 backward 函数是否使用了 forward 中的中间变量

### 5.3 特殊值梯度未定义

**问题**: 某些点梯度不存在或不连续

```python
# ❌ 错误: ReLU(0) 梯度未定义
def relu_backward(x, grad):
    return grad * (x > 0)  # x=0 时?

# ✅ 正确: 明确定义
def relu_backward(x, grad):
    return grad * (x > 0).astype(float)  # x=0 时梯度为 0
```

**检测**: 搜索分段函数，检查边界点梯度

### 5.4 高阶梯度未考虑

**问题**: 只实现一阶梯度，二阶梯度错误

```python
# ❌ 错误: backward 不可微
def backward(grad):
    return grad * (x > 0)  # 不可微!

# ✅ 正确: 使用可微操作
def backward(grad):
    return grad * sigmoid(x)  # 可微
```

**检测**: 检查 backward 函数是否使用了不可微操作

## 6. API 兼容性陷阱

### 6.1 参数顺序不一致

**问题**: 参数顺序与 PyTorch/NumPy 不同

```python
# ❌ 错误: 参数顺序错误
def conv2d(input, kernel, stride, padding):  # PyTorch 是 (input, weight, bias, stride, padding)
    pass
```

**检测**: 对比官方文档，检查参数顺序

### 6.2 默认值不一致

**问题**: 默认参数值与标准不同

```python
# ❌ 错误: 默认值不同
def relu(x, inplace=True):  # PyTorch 默认是 False!
    pass
```

**检测**: 对比官方文档，检查默认值

### 6.3 错误类型不一致

**问题**: 抛出的异常类型与标准不同

```python
# ❌ 错误: 异常类型错误
if x < 0:
    raise Exception("x must be positive")  # 应该是 ValueError!

# ✅ 正确
if x < 0:
    raise ValueError("x must be positive")
```

**检测**: 检查异常类型是否符合 Python 标准

## 7. 测试陷阱

### 7.1 只测试正常情况

**问题**: 未测试边界和异常情况

```python
# ❌ 错误: 只测试正常输入
def test_add():
    assert add(1, 2) == 3  # 空张量? 零维? 大张量?
```

**检测**: 检查测试是否覆盖边界情况

### 7.2 精度阈值过大

**问题**: allclose 的 atol/rtol 设置过大

```python
# ❌ 错误: 阈值太大
assert np.allclose(result, expected, atol=1e-1)  # 10% 误差!

# ✅ 正确: 合理阈值
assert np.allclose(result, expected, atol=1e-5, rtol=1e-5)
```

**检测**: 搜索 `allclose`，检查阈值设置

### 7.3 未测试梯度

**问题**: 只测试前向，未测试反向

```python
# ❌ 错误: 缺少梯度测试
def test_relu():
    output = relu(input)
    assert output.shape == input.shape  # 未测试梯度!
```

**检测**: 检查是否有 gradcheck 或梯度对比测试

## 8. 性能陷阱

### 8.1 不必要的拷贝

**问题**: 可以用 view 的地方用了 copy

```python
# ❌ 错误: 不必要的拷贝
result = x.copy()
result = result.reshape(new_shape)

# ✅ 正确: 使用 view
result = x.reshape(new_shape)  # 不拷贝
```

**检测**: 搜索 `copy`/`clone`，检查是否必要

### 8.2 循环中的重复计算

**问题**: 循环不变量在循环内重复计算

```python
# ❌ 错误: 重复计算
for i in range(n):
    scale = compute_scale(x)  # 每次都计算!
    result[i] = data[i] * scale

# ✅ 正确: 提取到循环外
scale = compute_scale(x)
for i in range(n):
    result[i] = data[i] * scale
```

**检测**: 检查循环内是否有不依赖循环变量的计算

### 8.3 小块频繁分配

**问题**: 循环中频繁分配小内存

```cpp
// ❌ 错误: 频繁分配
for (int i = 0; i < n; i++) {
    float* temp = new float[10];  // 每次都分配!
    // ...
    delete[] temp;
}

// ✅ 正确: 复用 buffer
float* temp = new float[10];
for (int i = 0; i < n; i++) {
    // 复用 temp
}
delete[] temp;
```

**检测**: 检查循环内是否有内存分配

## 快速检测清单

检视代码时，按以下顺序快速扫描:

1. **搜索关键词**: `TODO`, `FIXME`, `XXX`, `HACK`
2. **指针操作**: `*`, `->`, `[]` → 检查空指针和边界
3. **数学函数**: `log`, `exp`, `sqrt`, `/` → 检查特殊值
4. **类型转换**: `cast`, `astype`, `to` → 检查精度损失
5. **内存操作**: `new`, `malloc`, `delete`, `free` → 检查泄漏
6. **循环**: `for`, `while` → 检查边界和性能
7. **异常**: `throw`, `raise` → 检查类型和信息
8. **测试**: 检查是否覆盖边界、异常、梯度

## 参考资料

- PyTorch 算子实现规范
- MindSpore 算子开发指南
- NumPy Broadcasting 规则
- CUDA 编程最佳实践
