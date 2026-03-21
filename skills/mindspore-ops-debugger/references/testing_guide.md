# MindSpore 算子测试指南

## 目录

1. [测试体系概览](#1-测试体系概览)
2. [测试框架 2.0 (OpsFactory)](#2-测试框架-20-opsfactory)
3. [OpInfo 数据定义](#3-opinfo-数据定义)
4. [OpsFactory 使用](#4-opsfactory-使用)
5. [输入生成](#5-输入生成)
6. [编写新算子测试](#6-编写新算子测试)
7. [测试命名规范](#7-测试命名规范)
8. [MindSporeTest 结构](#8-mindsporetest-结构)
9. [调试测试用例](#9-调试测试用例)

---

## 1. 测试体系概览

MindSpore 算子测试分为两个层面:

| 层面 | 位置 | 框架 | 用途 |
|------|------|------|------|
| **源码内测试** | `mindspore/tests/st/ops/` | OpsFactory (测试框架 2.0) | 算子正确性、精度对比 |
| **独立测试套** | `MindSporeTest/` | pytest + OpsFactory | CI/CD 回归测试 |

### 测试级别

| 级别 | 说明 | 用途 |
|------|------|------|
| level0 | 冒烟测试 | 基本功能验证 |
| level1 | 核心功能 | 主要场景覆盖 |
| level2 | 扩展测试 | 边界条件、异常输入 |
| level3 | 全量测试 | 完整 dtype/shape 覆盖 |
| level4 | 性能测试 | 性能基线对比 |

---

## 2. 测试框架 2.0 (OpsFactory)

### 核心组件

```
OpInfo          ─ 算子元信息 (名称、引用实现、dtype 支持、输入生成)
    │
    ▼
OpsFactory      ─ 基类，提供 forward/grad 对比方法
    ├── UnaryOpsFactory     ─ 一元算子
    ├── BinaryOpsFactory    ─ 二元算子
    ├── ReductionOpsFactory ─ 规约算子
    └── CustomOpsFactory    ─ 自定义算子
    │
    ▼
OpSampleInput   ─ 输入容器，支持 astorch()/asnumpy() 转换
```

### 关键路径

- OpInfo 定义: `tests/st/ops/share/_op_info/op_database.py`
- OpsFactory 基类: `tests/st/ops/share/_internal/meta.py`
- 输入辅助: `tests/st/ops/share/_internal/sample_inputs.py`
- 测试文件: `tests/st/ops/test_*.py`

---

## 3. OpInfo 数据定义

### OpInfo 字段

```python
@dataclass
class OpInfo:
    name: str                          # 算子名, 如 'mint.add'
    op: callable                       # MindSpore 算子函数
    ref: callable                      # 参考实现 (通常是 torch.xxx)
    customize_op_func: callable = None # 自定义 MindSpore 算子
    op_func_without_kwargs: callable = None  # 无关键字参数版本 (用于 grad)

    # dtype 支持
    dtypes_ascend: list = None         # Ascend 支持的 dtype
    dtypes_cpu: list = None            # CPU 支持的 dtype
    dtypes_gpu: list = None            # GPU 支持的 dtype

    # 输入生成函数
    op_basic_reference_inputs_func: callable = None   # 基本输入
    op_extra_reference_inputs_func: callable = None   # 扩展输入
    op_dynamic_inputs_func: callable = None           # 动态 shape 输入
    op_error_inputs_func: callable = None             # 错误输入 (异常测试)

    # 梯度配置
    grad_position: tuple = (0,)        # 需要求梯度的输入位置
```

### 子类

```python
class BinaryOpInfo(OpInfo):
    """二元算子, 默认 grad_position=(0, 1)"""
    pass

class UnaryOpInfo(OpInfo):
    """一元算子, 默认 grad_position=(0,)"""
    pass

class ReductionOpInfo(OpInfo):
    """规约算子"""
    pass
```

### 示例: mint.add

```python
'mint.add': BinaryOpInfo(
    name='mint.add',
    op=mint.add,
    op_func_without_kwargs=add_ext_func_grad_without_kwargs,
    ref=torch.add,
    dtypes_ascend=[ms.float16, ms.float32, ms.bfloat16, ms.int32, ms.int64,
                   ms.complex64, ms.complex128, ms.bool_],
    dtypes_cpu=[ms.float16, ms.float32, ms.float64, ms.int32, ms.int64,
                ms.complex64, ms.complex128, ms.bool_],
    dtypes_gpu=[ms.float16, ms.float32, ms.float64, ms.int32, ms.int64,
                ms.complex64, ms.complex128, ms.bool_, ms.bfloat16],
    op_basic_reference_inputs_func=basic_sample_inputs_add_sub_ext,
    op_dynamic_inputs_func=dynamic_sample_inputs_add_sub_ext,
    op_error_inputs_func=error_inputs_add_sub_ext_func,
)
```

---

## 4. OpsFactory 使用

### 基本测试模式

```python
import pytest
from tests.st.ops.share._internal.meta import get_op_info, BinaryOpsFactory

# 获取所有二元算子 OpInfo
binary_op_db = [...]

@pytest.mark.parametrize("mode", ['pynative', 'kbk'])
@pytest.mark.parametrize("op_info", binary_op_db)
def test_binary_op_forward(mode, op_info):
    fact = BinaryOpsFactory(op_info=get_op_info(op_info))
    fact.set_context_mode(mode=mode)
    fact.test_op_reference()
```

### OpsFactory 核心方法

| 方法 | 说明 |
|------|------|
| `set_context_mode(mode)` | 设置执行模式: 'pynative' / 'kbk' |
| `test_op_reference()` | 对比 forward 结果与 ref 实现 |
| `forward_mindspore_impl(inputs)` | MindSpore 前向执行 |
| `grad_mindspore_impl(inputs)` | MindSpore 梯度执行 |
| `compare_with_torch(ms_out, torch_out)` | 精度对比 |
| `assert_equal(ms_out, ref_out, rtol, atol)` | 断言相等 |

### 对比流程

```python
# OpsFactory 内部逻辑简化
def test_op_reference(self):
    for sample_input in self.get_sample_inputs():
        # MindSpore 执行
        ms_out = self.forward_mindspore_impl(sample_input)
        # Reference (torch) 执行
        ref_out = self.ref(*sample_input.astorch())
        # 对比
        self.assert_equal(ms_out, ref_out)
```

---

## 5. 输入生成

### OpSampleInput

```python
class OpSampleInput:
    """算子输入容器"""
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def astorch(self):
        """转为 PyTorch tensor"""
        return [to_torch(a) for a in self.args], {k: to_torch(v) for k, v in self.kwargs.items()}

    def asnumpy(self):
        """转为 numpy array"""
        return [to_numpy(a) for a in self.args], {k: to_numpy(v) for k, v in self.kwargs.items()}
```

### 输入生成函数示例

```python
def basic_sample_inputs_add_sub_ext(op_info, dtype, device):
    """生成 add/sub 的基本测试输入"""
    yield OpSampleInput(
        make_tensor((3, 4), dtype=dtype, device=device),
        make_tensor((3, 4), dtype=dtype, device=device),
    )
    # 广播
    yield OpSampleInput(
        make_tensor((3, 4), dtype=dtype, device=device),
        make_tensor((4,), dtype=dtype, device=device),
    )
    # 标量
    yield OpSampleInput(
        make_tensor((3, 4), dtype=dtype, device=device),
        2.0,
    )

def dynamic_sample_inputs_add_sub_ext(op_info, dtype, device):
    """动态 shape 输入"""
    yield OpSampleInput(
        make_tensor((3, None), dtype=dtype, device=device),
        make_tensor((3, None), dtype=dtype, device=device),
    )

def error_inputs_add_sub_ext_func(op_info, dtype, device):
    """异常输入 (应该报错)"""
    yield OpSampleInput(
        make_tensor((3,), dtype=dtype, device=device),
        make_tensor((4,), dtype=dtype, device=device),  # shape 不兼容
    ), ValueError
```

### make_tensor 辅助

```python
from tests.st.ops.share._internal.sample_inputs import make_tensor

# 创建指定 shape/dtype 的随机 tensor
tensor = make_tensor((3, 4), dtype=ms.float32, device='Ascend')
```

### wrap_sample_inputs

```python
from tests.st.ops.share._internal.sample_inputs import wrap_sample_inputs

# 为特定 sample 添加 skip 或 loss 覆盖
wrapped = wrap_sample_inputs(
    basic_samples,
    skips=[
        SampleSkip(index=2, reason="fp16 precision too low"),
    ],
    loss_overrides=[
        SampleLossOverride(index=0, rtol=1e-2, atol=1e-2),
    ],
)
```

---

## 6. 编写新算子测试

### 步骤

1. **添加 OpInfo** 到 `op_database.py`
2. **实现输入生成函数**
3. **添加测试用例**
4. **运行并验证**

### 完整示例: 新增 my_op 测试

#### Step 1: 定义 OpInfo

```python
# 在 op_database.py 中添加
'mint.my_op': UnaryOpInfo(
    name='mint.my_op',
    op=mint.my_op,
    ref=torch.my_op,
    dtypes_ascend=[ms.float16, ms.float32],
    dtypes_cpu=[ms.float16, ms.float32, ms.float64],
    dtypes_gpu=[ms.float16, ms.float32, ms.float64],
    op_basic_reference_inputs_func=basic_sample_inputs_my_op,
    op_dynamic_inputs_func=dynamic_sample_inputs_my_op,
    op_error_inputs_func=error_inputs_my_op,
    grad_position=(0,),
)
```

#### Step 2: 输入生成

```python
def basic_sample_inputs_my_op(op_info, dtype, device):
    # 基本形状
    yield OpSampleInput(make_tensor((3, 4), dtype=dtype, device=device))
    # 空 tensor
    yield OpSampleInput(make_tensor((0,), dtype=dtype, device=device))
    # 标量
    yield OpSampleInput(make_tensor((), dtype=dtype, device=device))
    # 高维
    yield OpSampleInput(make_tensor((2, 3, 4, 5), dtype=dtype, device=device))

def dynamic_sample_inputs_my_op(op_info, dtype, device):
    yield OpSampleInput(make_tensor((None, 4), dtype=dtype, device=device))

def error_inputs_my_op(op_info, dtype, device):
    # 空输入应该报错
    yield OpSampleInput(None), TypeError
```

#### Step 3: 测试用例

```python
import pytest
from tests.st.ops.share._internal.meta import get_op_info, UnaryOpsFactory

@pytest.mark.parametrize("mode", ['pynative', 'kbk'])
def test_my_op_forward(mode):
    fact = UnaryOpsFactory(op_info=get_op_info('mint.my_op'))
    fact.set_context_mode(mode=mode)
    fact.test_op_reference()

@pytest.mark.parametrize("mode", ['pynative', 'kbk'])
def test_my_op_grad(mode):
    fact = UnaryOpsFactory(op_info=get_op_info('mint.my_op'))
    fact.set_context_mode(mode=mode)
    fact.test_op_grad()
```

#### Step 4: 运行

```bash
# 单个测试
pytest tests/st/ops/test_my_op.py -v -s

# 指定模式
pytest tests/st/ops/test_my_op.py -v -k "pynative"

# 指定超时
pytest tests/st/ops/test_my_op.py -v --timeout=900
```

---

## 7. 测试命名规范

### MindSporeTest 命名约定

| 前缀 | 含义 | 示例 |
|------|------|------|
| `test_f_` | functional 接口 | `test_f_abs_float32` |
| `test_n_` | nn 接口 | `test_n_relu_forward` |
| `test_p_` | primitive 接口 | `test_p_add_broadcast` |
| `test_t_` | tensor 方法 | `test_t_sigmoid_fp16` |

### 用例结构

```python
class TestMyOp:
    """
    Test: my_op 算子测试
    Precondition: MindSpore installed with Ascend support
    Backend: Ascend, CPU
    Level: level1
    """

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    def test_f_my_op_float32(self):
        """
        Feature: my_op operator
        Description: Test my_op with float32 input
        Expectation: Output matches PyTorch reference
        """
        ...
```

---

## 8. MindSporeTest 结构

```
MindSporeTest/
├── README.md           # 测试规范
├── share/              # 共享工具
│   ├── _op_info/       # OpInfo 定义
│   ├── _internal/      # 内部工具 (meta.py, sample_inputs.py)
│   └── ops/            # 算子测试公共方法
├── operations/         # 算子功能测试
├── dynamic_shape/      # 动态 shape 测试
├── compiler/           # 编译器测试
├── parallel/           # 并行测试
├── bert/               # BERT 模型测试
├── resnet50/           # ResNet50 模型测试
└── ...
```

---

## 9. arg_mark 装饰器规范

### 强制要求：mem_peak

`arg_mark` 在以下条件**同时满足**时，强制要求提供 `mem_peak` 参数，否则 pytest 收集阶段直接抛出 `ValueError: wrong mem_peak value`：

- `card_mark='onecard'`
- `plat_marks` 包含 `'platform_ascend'` 或 `'platform_ascend910b'`
- `level_mark` 为 `'level0'` 或 `'level1'`

```python
# 错误写法 — 会在 collect 阶段报错
@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'],
          level_mark='level0', card_mark='onecard', essential_mark='essential')

# 正确写法 — 必须加 mem_peak
@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'],
          level_mark='level0', card_mark='onecard', essential_mark='essential',
          mem_peak=1024.0)
```

`mem_peak` 单位为 MB，表示该用例预期的峰值显存占用。对于小型算子测试，`1024.0` 是合理的默认值。

### 不需要 mem_peak 的情况

- `plat_marks` 不含 `ascend`（纯 GPU/CPU 用例）
- `level_mark` 为 `'level2'` 及以上
- `card_mark` 为 `'allcards'`

### PR 审查检查点

审查包含新测试用例的 PR 时，检查所有 `platform_ascend + level0/level1 + onecard` 组合的 `arg_mark` 是否都有 `mem_peak`：

```bash
# 找出缺少 mem_peak 的 arg_mark（粗略检查）
grep -n "platform_ascend" tests/st/ops/test_func_xxx.py | grep "level0\|level1"
```

---

## 10. 调试测试用例

### 常用 pytest 参数

```bash
# 显示详细输出
pytest -v -s

# 只运行匹配的用例
pytest -k "sigmoid and float32"

# 只运行失败的用例
pytest --lf

# 设置超时
pytest --timeout=300

# 只运行标记的用例
pytest -m "level1"
```

### 调试精度问题

```python
import numpy as np
import mindspore as ms

# 手动对比
ms_out = my_op(ms.Tensor(input_data)).asnumpy()
ref_out = torch.my_op(torch.tensor(input_data)).numpy()

# 详细差异分析
diff = np.abs(ms_out - ref_out)
print(f"Max absolute diff: {diff.max()}")
print(f"Mean absolute diff: {diff.mean()}")
print(f"Max relative diff: {(diff / (np.abs(ref_out) + 1e-8)).max()}")
print(f"Mismatch count: {(diff > 1e-4).sum()} / {diff.size}")
```

### 动态 Shape 测试

```python
import mindspore as ms

# 设置动态 shape 输入
net = MyNet()
dyn_input = ms.Tensor(shape=[None, 4], dtype=ms.float32)
net.set_inputs(dyn_input)

# 用不同 shape 运行
out1 = net(ms.Tensor(np.random.randn(3, 4).astype(np.float32)))
out2 = net(ms.Tensor(np.random.randn(5, 4).astype(np.float32)))
```

---

## 10. 常见陷阱

### PyTorch 基准参数类型影响 backward 路径

编写 `grad_pytorch_impl` 时，确保传给 PyTorch 算子的参数类型与 MindSpore 侧一致。PyTorch 对不同参数类型可能走完全不同的 backward 路径，在低精度 dtype (bfloat16/fp16) 下产生不可忽略的精度差异。

典型案例：`repeat_interleave` 的 `repeats` 参数

```python
# 错误: torch.tensor(int) 创建 0-d tensor，PyTorch 走 scatter_add backward
repeats = torch.tensor(self.repeats)  # self.repeats = 6

# 正确: 直接传 int，PyTorch 走 reshape+sum backward（与 MindSpore 一致）
repeats = self.repeats
```

`torch.tensor(6)` 和 `6` 在 forward 结果完全一致，但 backward 路径不同：
- `int` repeats → reshape + sum（与 MindSpore 的 SumExt 路径对齐）
- `tensor` repeats → scatter_add（bfloat16 下累加精度不同）

参考: CS-020 (#1574), M-014
