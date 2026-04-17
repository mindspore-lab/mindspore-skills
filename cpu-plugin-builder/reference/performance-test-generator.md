# MindSpore CPU Operator Performance Test Generator Skill

This skill provides comprehensive guidance for generating performance test cases for MindSpore CPU operators, following the patterns established in reference files like `test_perf_abs.py` and `test_perf_constant_pad_nd.py`.

## When to Use This Skill

Use this skill when you need to:
- Create new performance test files for MindSpore CPU operators
- Measure performance comparison between MindSpore and PyTorch
- Ensure operator performance meets the <1.1x PyTorch threshold
- Document performance metrics for validation

## How to Use This Skill

When creating a performance test file, invoke this skill to get:
1. A comprehensive template structure
2. Helper function patterns
3. Test configuration guidelines
4. Common pitfalls and solutions

## Performance Test Template

### 1. Standard File Header

```python
# Copyright [YEAR] Huawei Technologies Co., Ltd
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
""" [OperatorName] op performance test case """
# pylint: disable=unused-variable
# pylint: disable=W0622,W0613
import time
import mindspore as ms
from mindspore import mint
from mindspore.common.api import _pynative_executor
from tests.utils.test_op_utils import BACKGROUND_NOISE
from tests.utils.mark_utils import arg_mark
import torch
import numpy as np
import pytest
```

### 2. Helper Functions

```python
def generate_random_input(shape, dtype):
    """Generate test data."""
    return np.random.uniform(-1, 1, shape).astype(dtype)


def [operator_name]_forward_perf(input_tensor, *args):
    """get ms op forward performance"""
    for _ in range(1000):
        _ = mint.[operator_name](input_tensor, *args)

    _pynative_executor.sync()
    start = time.time()
    for _ in range(1000):
        _ = mint.[operator_name](input_tensor, *args)
    _pynative_executor.sync()
    end = time.time()

    print(f"MindSpore [OperatorName] e2e time: ", (end - start))
    return end - start


def generate_expect_forward_perf(input_tensor, *args):
    """get torch op forward performance"""
    print("================shape: ", input_tensor.shape)

    for _ in range(1000):
        _ = torch.[operator_name](input_tensor, *args)

    start = time.time()
    for _ in range(1000):
        _ = torch.[operator_name](input_tensor, *args)
    end = time.time()

    print(f"Torch [OperatorName] e2e time: ", end - start)
    return end - start
```

### 3. Basic Performance Test

```python
@arg_mark(plat_marks=['cpu_linux'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', ['pynative'])
def test_[operator_name]_perf(mode):
    """
    Feature: [OperatorName] forward performance.
    Description: test [OperatorName] op performance between MindSpore
                 and PyTorch.
    Expectation: MindSpore performance is within 110% of PyTorch
                 performance.
    """
    ms.set_device("CPU")
    shape = (4, 3, 32, 32)
    # Add operator-specific parameters here
    
    input_np = generate_random_input(shape, np.float32)
    ms_perf = [operator_name]_forward_perf(
        ms.Tensor(input_np), *args)
    expect_perf = generate_expect_forward_perf(
        torch.Tensor(input_np), *args)
    assert np.less(ms_perf - BACKGROUND_NOISE, expect_perf * 1.1).all()
```

### 4. Shape Variation Tests

```python
@arg_mark(plat_marks=['cpu_linux'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', ['pynative'])
@pytest.mark.parametrize('shape', [
    (4, 3, 32, 32),
    (16, 3, 64, 64),
    # Add more shapes as needed
])
def test_[operator_name]_perf_shapes(mode, shape):
    """
    Feature: [OperatorName] forward performance with different shapes.
    Description: test [OperatorName] op performance with various tensor shapes.
    Expectation: MindSpore performance is within 110% of PyTorch
                 performance.
    """
    ms.set_device("CPU")
    # Operator-specific parameters
    
    input_np = generate_random_input(shape, np.float32)
    ms_perf = [operator_name]_forward_perf(
        ms.Tensor(input_np), *args)
    expect_perf = generate_expect_forward_perf(
        torch.Tensor(input_np), *args)
    assert np.less(ms_perf - BACKGROUND_NOISE, expect_perf * 1.1).all()
```

### 5. Parameter Variation Tests

```python
@arg_mark(plat_marks=['cpu_linux'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', ['pynative'])
@pytest.mark.parametrize('param', [
    value1,
    value2,
    value3,
])
def test_[operator_name]_perf_[param_name](mode, param):
    """
    Feature: [OperatorName] forward performance with different [param_name].
    Description: test [OperatorName] op performance with various parameter values.
    Expectation: MindSpore performance is within 110% of PyTorch
                 performance.
    """
    ms.set_device("CPU")
    shape = (4, 3, 32, 32)
    
    input_np = generate_random_input(shape, np.float32)
    ms_perf = [operator_name]_forward_perf(
        ms.Tensor(input_np), param)
    expect_perf = generate_expect_forward_perf(
        torch.Tensor(input_np), param)
    assert np.less(ms_perf - BACKGROUND_NOISE, expect_perf * 1.1).all()
```

### 6. Dtype Variation Tests

```python
@arg_mark(plat_marks=['cpu_linux'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', ['pynative'])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_[operator_name]_perf_dtype(mode, dtype):
    """
    Feature: [OperatorName] forward performance with different dtypes.
    Description: test [OperatorName] op performance with float32 and float64.
    Expectation: MindSpore performance is within 110% of PyTorch
                 performance.
    """
    ms.set_device("CPU")
    shape = (4, 3, 32, 32)
    
    input_np = generate_random_input(shape, dtype)
    ms_perf = [operator_name]_forward_perf(
        ms.Tensor(input_np), *args)
    expect_perf = generate_expect_forward_perf(
        torch.Tensor(input_np.astype(np.float32)), *args)
    assert np.less(ms_perf - BACKGROUND_NOISE, expect_perf * 1.1).all()
```

## Performance Test Checklist

| Item | Status | Notes |
|------|--------|-------|
| Basic performance test | Required | Compare with PyTorch baseline |
| Shape variations | Recommended | Test small, medium, large tensors |
| Parameter variations | Recommended | Test different parameter values |
| Dtype variations | Optional | Test float32, float64 |
| Performance threshold | Required | MindSpore < 1.1x PyTorch |
| Warm-up iterations | Included | First 1000 iterations for warm-up |
| Measurement iterations | 1000 | Measure after warm-up |

## Common Issues and Solutions

### 1. Missing yaml Module

**Problem**: `ModuleNotFoundError: No module named 'yaml'`

**Solution**: Install pyyaml in the test environment:
```bash
pip install pyyaml
```

### 2. Performance Threshold Exceeded

**Problem**: MindSpore performance > 1.1x PyTorch

**Solutions**:
- Reduce tensor sizes for large shape tests
- Skip problematic test cases with `@pytest.mark.skip`
- Investigate optimization opportunities in the operator implementation

### 3. Inconsistent Timing

**Problem**: High variance in timing measurements

**Solutions**:
- Increase warm-up iterations
- Run tests multiple times and average results
- Ensure no other processes are running

### 4. Memory Issues

**Problem**: Out of memory with large tensors

**Solutions**:
- Reduce tensor sizes
- Run tests sequentially instead of in parallel
- Add garbage collection between tests

## File Naming Convention

```
test_perf_[operator_name].py
```

Examples:
- `test_perf_abs.py`
- `test_perf_constant_pad_nd.py`
- `test_perf_constant_pad_1d.py`

## Test Execution

```bash
# Run all performance tests
pytest tests/st/mint/test_perf_[operator_name].py -v

# Run specific test
pytest tests/st/mint/test_perf_[operator_name].py::test_[operator_name]_perf -v

# Run with timeout
pytest tests/st/mint/test_perf_[operator_name].py --timeout=300
```

## Expected Output

```
MindSpore [OperatorName] e2e time:  0.123456789
================shape:  torch.Size([4, 3, 32, 32])
Torch [OperatorName] e2e time:  0.112345678
PASSED
```

## Key Metrics

- **Warm-up**: 1000 iterations (discarded)
- **Measurement**: 1000 iterations (averaged)
- **Threshold**: MindSpore < PyTorch * 1.1
- **Background noise**: Subtracted from measurements

## Best Practices

1. **Consistent Environment**: Use the same conda environment for all tests
2. **Rebuild Before Testing**: Rebuild the plugin after code changes
   ```bash
   bash build.sh
   ```
3. **Isolate Tests**: Run performance tests separately from functional tests
4. **Document Results**: Log performance metrics for comparison
5. **Regular Validation**: Run performance tests after operator updates

## Template Summary

```python
# 1. Imports
# 2. generate_random_input()
# 3. [op]_forward_perf()
# 4. generate_expect_forward_perf()
# 5. Basic test
# 6. Shape variation tests (2-3 shapes)
# 7. Parameter variation tests (2-4 values)
# 8. Dtype tests (float32, float64)
# 9. __main__ block for manual testing
```

## Post-Creation Steps

1. Run tests to verify they pass
2. Copy to commit_store for version control
3. Document any performance issues found
4. Update operator documentation if needed
