# MindSpore CPU Operator Test Generator Skill

This skill provides comprehensive guidance and templates for generating test cases for MindSpore CPU operators based on the "CPU算子转测checklist.md" and existing high-quality test files like `test_abs.py`.

## When to Use This Skill

Use this skill when you need to:
- Write new test files for MindSpore CPU operators
- Validate existing test files against the checklist
- Improve test coverage for operators
- Ensure compliance with MindSpore testing standards

## How to Use This Skill

When working on a new operator test file, invoke this skill to get:
1. A comprehensive test template
2. Checklist item mapping
3. Helper function patterns
4. Best practices and common pitfalls

## Test Case Template

### 1. Standard Imports and Setup

```python
# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Operator_name op test case - comprehensive test suite"""
# pylint: disable=unused-variable
import pytest
import numpy as np
import torch
import mindspore as ms
from mindspore import mint, ops, jit
from tests.utils.mark_utils import arg_mark
from tests.utils.tools import allclose_nparray, needs_backward
```

> **Note**: Update the copyright year to the current year (2026).

### 2. Helper Functions

```python
def generate_random_input(shape, dtype):
    """Generate random input data for testing."""
    if np.issubdtype(dtype, np.integer):
        return np.random.randint(-10, 10, shape).astype(dtype)
    if np.issubdtype(dtype, np.complexfloating):
        return (np.random.uniform(-10, 10, shape) + 1j *
                np.random.uniform(-10, 10, shape)).astype(dtype)
    return np.random.uniform(-1, 1, shape).astype(dtype)


def generate_special_input(shape, dtype, special_type):
    """Generate input with special values."""
    if special_type == "inf":
        x = np.ones(shape, dtype=dtype)
        x[0] = np.inf
        x[-1] = -np.inf
        return x
    if special_type == "nan":
        x = np.ones(shape, dtype=dtype)
        x[0] = np.nan
        return x
    if special_type == "zero":
        return np.zeros(shape, dtype=dtype)
    if special_type == "large":
        return np.random.uniform(1e6, 1e9, shape).astype(dtype)
    if special_type == "small":
        return np.random.uniform(-1e-9, 1e-9, shape).astype(dtype)
    return generate_random_input(shape, dtype)


def generate_ones_grad(shape, dtype):
    """Generate gradient of ones."""
    return np.ones(shape).astype(dtype)


def operator_name_forward_func(input, target=None, **kwargs):
    """Forward function for mint.operator_name."""
    if target is None:
        return mint.operator_name(input, **kwargs)
    return mint.operator_name(input, target, **kwargs)


def operator_name_backward_func(input, target=None, **kwargs):
    """Backward function for mint.operator_name."""
    def forward(input):
        if target is None:
            return mint.operator_name(input, **kwargs)
        return mint.operator_name(input, target, **kwargs)
    return ops.grad(forward, (0,))(input)


# KBK mode wrappers
KBK_OPERATOR_NAME_FORWARD = jit(operator_name_forward_func, backend="ms_backend", jit_level="O0")
KBK_OPERATOR_NAME_BACKWARD = jit(operator_name_backward_func, backend="ms_backend", jit_level="O0")
```

### 3. Wrapper Functions

```python
def operator_name_forward(mode, input, target=None, **kwargs):
    """Operator forward wrapper."""
    if mode == 'pynative':
        return operator_name_forward_func(input, target, **kwargs)
    return KBK_OPERATOR_NAME_FORWARD(input, target, **kwargs)


def operator_name_backward(mode, input, target=None, **kwargs):
    """Operator backward wrapper."""
    if mode == 'pynative':
        return operator_name_backward_func(input, target, **kwargs)
    return KBK_OPERATOR_NAME_BACKWARD(input, target, **kwargs)
```

## Checklist to Test Case Mapping

| Checklist Requirement | Test Function | Parameters | Notes |
|---------------------|---------------|------------|-------|
| 默认参数场景 | `test_operator_name_std` | mode: ['pynative', 'KBK'] | Basic functionality |
| 空 Tensor | `test_operator_name_empty_tensor` | mode: ['pynative', 'KBK'] | Shapes: (0,), (2, 0), (0, 3, 4) |
| inf 和 nan | `test_operator_name_special_values` | mode: ['pynative', 'KBK'], special_type: ['inf', 'nan', 'zero'] | Special values handling |
| 数据类型覆盖 | `test_operator_name_dtype_coverage` | mode: ['pynative'], dtype: [float32, float64] | Support dtypes |
| 输入维度 0D-8D | `test_operator_name_dimensions` | mode: ['pynative', 'KBK'], shape: [(), (1,), (2,3), ...] | 0D to 8D coverage |
| 动态 shape/rank | `test_operator_name_dynamic_shape` | mode: ['pynative', 'KBK'] | Dynamic dimensions |
| 非连续输入 | `test_operator_name_non_contiguous` | mode: ['pynative', 'KBK'] | Transposed tensors |
| bf16 | `test_operator_name_bf16` | mode: ['pynative', 'KBK'] | bfloat16 support |
| 正向精度验证 | `test_operator_name_precision` | mode: ['pynative', 'KBK'] | High precision testing |
| 反向支持 | `test_operator_name_backward_*` | Various | Gradient computation |
| Functional 用例 | `test_operator_name_functional_interface` | mode: ['pynative', 'KBK'] | Direct API calls |
| JIT 模式 | `test_operator_name_jit_mode` | mode: ['pynative', 'KBK'] | JIT compilation |
| 大尺寸张量 | `test_operator_name_large_tensors` | mode: ['pynative', 'KBK'] | Large tensor handling |
| 0 偏差 | `test_operator_name_zero_bias` | mode: ['pynative', 'KBK'] | Zero deviation from PyTorch |
| vmap | `test_operator_name_vmap` | mode: ['pynative'], batch_size: [8, 16, 32, 64, 128] | Batch processing |
| 广播 | `test_operator_name_broadcasting` | mode: ['pynative', 'KBK'] | Broadcasting support |
| 单算子实现 | `test_operator_name_backward_single_operator` | mode: ['pynative', 'KBK'] | Verify single operator |

## Test Function Templates

### Template 1: Standard Test

```python
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0',
          card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_operator_name_std(mode):
    """
    Feature: Standard forward and backward features.
    Description: Test standard functionality of operator_name.
    Expectation: Expect correct result consistent with PyTorch.
    """
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")

    input_data = generate_random_input((4, 4), np.float32)
    target = generate_random_input((4, 4), np.float32) if has_target else None

    # PyTorch reference
    input_pt = torch.tensor(input_data)
    target_pt = torch.tensor(target) if target is not None else None
    expect = generate_expect_forward_output(input_pt, target_pt)

    # MindSpore output
    output = operator_name_forward(mode, ms.Tensor(input_data),
                                   ms.Tensor(target) if target is not None else None)

    allclose_nparray(expect.detach().numpy(), output.asnumpy(), equal_nan=True)
```

### Template 2: Empty Tensor Test

```python
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0',
          card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_operator_name_empty_tensor(mode):
    """
    Feature: Empty tensor handling for operator_name.
    Description: Test operator_name with empty tensor input.
    Expectation: Results match PyTorch implementation.
    """
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")

    shapes = [(0,), (2, 0), (0, 3, 4)]
    for shape in shapes:
        input_data = generate_random_input(shape, np.float32)
        target = generate_random_input(shape, np.float32)

        input_pt = torch.tensor(input_data)
        target_pt = torch.tensor(target)
        expect = generate_expect_forward_output(input_pt, target_pt)

        output = operator_name_forward(mode, ms.Tensor(input_data), ms.Tensor(target))

        assert output.shape == expect.shape
        allclose_nparray(expect.detach().numpy(), output.asnumpy(), equal_nan=True)
```

### Template 3: Special Values Test

```python
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0',
          card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
@pytest.mark.parametrize('special_type', ['inf', 'nan', 'zero'])
def test_operator_name_special_values(mode, special_type):
    """
    Feature: Special value handling for operator_name.
    Description: Test operator_name with inf, nan, and zero values.
    Expectation: Results match PyTorch implementation.
    """
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")

    shape = (4, 4)
    input_data = generate_special_input(shape, np.float32, special_type)
    target = generate_random_input(shape, np.float32)

    input_pt = torch.tensor(input_data)
    target_pt = torch.tensor(target)
    expect = generate_expect_forward_output(input_pt, target_pt)

    output = operator_name_forward(mode, ms.Tensor(input_data), ms.Tensor(target))
    allclose_nparray(expect.detach().numpy(), output.asnumpy(), equal_nan=True)
```

### Template 4: Dimension Coverage Test

```python
@arg_mark(plat_marks=['cpu_linux'], level_mark='level1',
          card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
@pytest.mark.parametrize('shape', [
    (), (1,), (2, 3), (2, 3, 4), (2, 2, 3, 4), (2, 2, 2, 3, 4)
])
def test_operator_name_dimensions(mode, shape):
    """
    Feature: Dimension coverage for operator_name.
    Description: Test operator_name with 0D to 5D inputs.
    Expectation: Results match PyTorch implementation.
    """
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")

    input_data = generate_random_input(shape, np.float32)
    target = generate_random_input(shape, np.float32)

    input_pt = torch.tensor(input_data)
    target_pt = torch.tensor(target)
    expect = generate_expect_forward_output(input_pt, target_pt)

    output = operator_name_forward(mode, ms.Tensor(input_data), ms.Tensor(target))
    allclose_nparray(expect.detach().numpy(), output.asnumpy(), equal_nan=True)
```

### Template 5: Dynamic Shape Test

```python
@arg_mark(plat_marks=['cpu_linux'], level_mark='level1',
          card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_operator_name_dynamic_shape(mode):
    """
    Feature: Dynamic shape support for operator_name.
    Description: Test operator_name with dynamic batch and dimensions.
    Expectation: Results match PyTorch implementation.
    """
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")

    for _ in range(5):
        batch_size = np.random.randint(2, 10)
        feat_size = np.random.randint(2, 10)
        shape = (batch_size, feat_size)

        input_data = generate_random_input(shape, np.float32)
        target = generate_random_input(shape, np.float32)

        input_pt = torch.tensor(input_data)
        target_pt = torch.tensor(target)
        expect = generate_expect_forward_output(input_pt, target_pt)

        output = operator_name_forward(mode, ms.Tensor(input_data), ms.Tensor(target))
        allclose_nparray(expect.detach().numpy(), output.asnumpy(), equal_nan=True)
        assert output.shape == expect.shape
```

### Template 6: Non-Contiguous Input Test

```python
@arg_mark(plat_marks=['cpu_linux'], level_mark='level1',
          card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_operator_name_non_contiguous(mode):
    """
    Feature: Non-contiguous input support for operator_name.
    Description: Test operator_name with non-contiguous tensor input.
    Expectation: Results match PyTorch implementation.
    """
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")

    shape = (4, 4, 4)
    input_data = generate_random_input(shape, np.float32)
    target = generate_random_input(shape, np.float32)

    input_pt = torch.tensor(input_data).transpose(0, 1)
    target_pt = torch.tensor(target).transpose(0, 1)
    assert not input_pt.is_contiguous()

    input_ms = mint.transpose(ms.Tensor(input_data), 0, 1)
    target_ms = mint.transpose(ms.Tensor(target), 0, 1)
    assert not input_ms.is_contiguous()

    expect = generate_expect_forward_output(input_pt, target_pt)
    output = operator_name_forward(mode, input_ms, target_ms)
    allclose_nparray(expect.detach().numpy(), output.asnumpy(), equal_nan=True)
```

### Template 7: BF16 Test

```python
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0',
          card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_operator_name_bf16(mode):
    """
    Feature: bf16 dtype support for operator_name.
    Description: Test operator_name with bfloat16 dtype.
    Expectation: Results match PyTorch implementation with appropriate precision.
    """
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")

    input_data = generate_random_input((4, 4), np.float32)
    target = generate_random_input((4, 4), np.float32)

    ms_bf16 = ms.bfloat16
    torch_bf16 = torch.bfloat16

    ms_input = ms.Tensor(input_data, dtype=ms_bf16)
    ms_target = ms.Tensor(target, dtype=ms_bf16)
    pt_input = torch.tensor(input_data, dtype=torch_bf16, requires_grad=True)
    pt_target = torch.tensor(target, dtype=torch_bf16)

    expect = generate_expect_forward_output(pt_input, pt_target)
    output = operator_name_forward(mode, ms_input, ms_target)

    expect_float = expect.to(torch.float32)
    ms_output_float = output.astype(ms.float32)
    allclose_nparray(expect_float.detach().numpy(), ms_output_float.asnumpy(),
                      atol=1e-2, equal_nan=True)
```

### Template 8: Vmap Test

```python
@arg_mark(plat_marks=['cpu_linux'], level_mark='level1',
          card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative'])
@pytest.mark.parametrize('batch_size', [8, 16, 32, 64, 128])
def test_operator_name_vmap(mode, batch_size):
    """
    Feature: vmap support for operator_name.
    Description: Test operator_name with vmap for batch processing.
    Expectation: Results match PyTorch implementation.
    """
    from mindspore import vmap

    ms.context.set_context(mode=ms.PYNATIVE_MODE)

    input_data = generate_random_input((batch_size, 4, 4), np.float32)
    target = generate_random_input((batch_size, 4, 4), np.float32)

    def batched_forward(input_batch, target_batch):
        return mint.operator_name(input_batch, target_batch)

    vmap_forward = vmap(batched_forward, in_axes=0)

    input_pt = torch.tensor(input_data)
    target_pt = torch.tensor(target)
    expect = generate_expect_forward_output(input_pt, target_pt)

    output = vmap_forward(ms.Tensor(input_data), ms.Tensor(target))
    allclose_nparray(expect.detach().numpy(), output.asnumpy(), equal_nan=True)
```

## Decision Tree for Test Scenarios

```
Operator has target/input?
├── YES: Include target parameter in tests
│   └── Both inputs must have same shape? (e.g., loss functions)
│       ├── YES: Skip broadcasting tests
│       └── NO: Include broadcasting tests
│
└── NO: Single input operator (e.g., activation functions)
    └── Include broadcasting tests if applicable

Operator supports backward?
├── YES: Include backward tests
│   └── Single operator implementation?
│       ├── YES: Include backward_single_operator test
│       └── NO: Document in comments
│
└── NO: Skip backward tests

Dtype support?
├── Float only: Test float32, float64, bf16
├── Integer support: Test int8, int16, int32, int64, uint8
└── Complex support: Test complex64, complex128

Special considerations?
├── Element-wise: Skip vmap (redundant)
├── Loss function: Skip vmap (not typical)
└── Not applicable: Document reasons in skip decorators
```

## Common Pitfalls and Solutions

### 1. Segmentation Faults
**Problem**: Dynamic shape tests cause segfaults
**Solution**: Skip or use concrete shapes with random variation

### 2. Framework Bugs
**Problem**: Some reduction modes don't work (e.g., smooth_l1_loss 'none', 'sum')
**Solution**: Skip affected tests and document in comments

### 3. Broadcasting Not Supported
**Problem**: Operator requires same shape inputs
**Solution**: Skip broadcasting tests with clear reason

### 4. Type Conversion Errors
**Problem**: float atol vs int atol in allclose_nparray
**Solution**: Use `atol=1e-4` (float) or cast to int

### 5. NaN Handling
**Problem**: Different NaN handling between PyTorch and MindSpore
**Solution**: Use `equal_nan=True` in allclose_nparray

## Checklist Validation Command

```bash
# Run tests for a specific operator
pytest tests/st/mint/test_operator_name.py -v --tb=short

# Run all tests
pytest tests/st/mint/ -v --tb=short

# Check coverage
pytest tests/st/mint/test_operator_name.py --collect-only
```

## File Structure

```
mindspore_op_plugin/tests/st/mint/
├── test_operator_name.py      # Main test file
├── test_abs.py                # Reference for element-wise operators
├── test_smooth_l1_loss.py     # Reference for loss functions
└── test_nn_conv2d.py          # Reference for nn operators
```

## Best Practices

1. **Update copyright year** - Use current year (2026) in copyright header
2. **Follow test_abs.py patterns** - It's the gold standard
3. **Use parametrized tests** - Cover multiple cases efficiently
3. **Include both pynative and KBK modes** - Test both execution modes
4. **Add proper docstrings** - Document feature, description, expectation
5. **Use appropriate tolerances** - bf16 needs larger atol (1e-2)
6. **Skip rather than fail** - Mark inapplicable tests with reasons
7. **Document framework bugs** - Comment on known issues
8. **Test edge cases** - Empty tensors, special values, large tensors
