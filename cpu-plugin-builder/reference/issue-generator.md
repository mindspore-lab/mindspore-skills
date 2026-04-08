# MindSpore OP Plugin Issue Generator Skill

This skill generates GitHub/GitCode issue templates for MindSpore CPU operator适配 (adapter/implementation) tasks.

## When to Use

Use this skill when you need to create an issue template for MindSpore OP Plugin CPU operators that includes:
- Operator background and description
- Design rationale
- Related modules and operators
- Test design and plans
- Performance benchmarks
- Reference implementations

## Workflow

### Step 1: Gather Information

Before generating the issue, collect the following information:

1. **Test Files**
   - Location: `/Users/litingyu/lty_work/mindspore_op_plugin/tests/st/mint/`
   - Look for: `test_*.py` (functional tests) and `test_perf_*.py` (performance tests)
   - Read the test file to understand:
     - Operator interface (`mint.nn.functional.xxx` or `mint.nn.xxx`)
     - Test scenarios covered
     - Parameters and their values
     - Input shapes and dtypes
     - PyTorch reference implementation

2. **Kernel Files**
   - Location: `/Users/litingyu/lty_work/mindspore_op_plugin/op_plugin/ops/kernel/`
   - Look for: `xxx.cc` (forward) and `xxx_grad.cc` (backward)
   - If not found, search for pattern: `xxx_ext.cc` (extended implementations)
   - Identify the ATen kernel being called (e.g., `at::xxx_out`)

3. **Operator Details**
   - PyTorch reference: `torch.nn.functional.xxx` or `torch.nn.xxx`
   - Formula/algorithm
   - Key parameters
   - Input/output shapes
   - Related operators

### Step 2: Analyze Test File

Read the test file and extract:

```python
# Key patterns to identify:

# 1. Operator interface
def xxx_forward_func(x, param1, param2):
    return mint.nn.functional.xxx(x, param1=param1, param2=param2)

# 2. PyTorch reference
def generate_expect_forward_output(x, param1, param2):
    return torch.nn.functional.xxx(x, param1=param1, param2=param2)

# 3. Test scenarios
- Standard forward/backward
- Different dtypes: float32, float64
- Different parameters
- Different input shapes
- Edge values
- Backward propagation
```

### Step 3: Generate Issue Template

Use the following structure for the issue template:

## Issue Template Structure

```markdown
# Issue Title
[Feature]: [OP Plugin] [OPS] [CPU] mint.nn.XXX算子适配

# Issue Content Sections

## 1. Background (背景描述)
- Operator description
- PyTorch reference implementation
- Key features/characteristics
- Formula/algorithm

## 2. Design Rationale (设计思路)
- Forward implementation
- Backward implementation
- Implementation approach
- Current status (kernel files, test files)

## 3. Related Modules (与其他模块的相关性描述)
- Functional interface
- Module interface
- MindSpore execution modes (PyNative, KBK)
- Dynamic shape support
- Related operators

## 4. Test Design (测试设计与测试计划)
### Functional Tests
| Scenario | Description | Acceptance Criteria |
|----------|-------------|-------------------|
| ... | ... | ... |

### Performance Tests
| Scenario | Description | Acceptance Criteria |
|----------|-------------|-------------------|
| ... | ... | ... |

## 5. Other Information (其他信息)
- Reference implementation
- Operator type
- Test markers
- Files involved

## 6. Algorithm Details (算法详解)
- Formula
- Parameters
- Output shape calculation
- Advantages

## 7. Application Scenarios (应用场景)
- Common use cases
- Industries/domains
```

## Template Sections Explained

### Section 1: Background
```markdown
###

` 🚀 背景描述torch.nn.functional.xxx` 是 PyTorch 中的 XXX 算子，用于...

**参考实现**：PyTorch `torch.nn.functional.xxx(input, params)`

**关键特性**：
- Feature 1
- Feature 2
- Feature 3
```

### Section 2: Design
```markdown
### 设计思路

- **前向**：调用 ATen `at::xxx_out` 实现 XXX 功能
- **反向**：调用 ATen `at::xxx_backward_out` 实现梯度回传
- **实现方式**：通过 OP Plugin 内核...

**现有实现状态**：
- `op_plugin/ops/kernel/xxx.cc`：前向 kernel
- `op_plugin/ops/kernel/xxx_grad.cc`：反向 kernel
- `tests/st/mint/test_xxx.py`：功能测试
- `tests/st/mint/test_perf_xxx.py`：性能测试
```

### Section 3: Test Design

Use tables to document tests:

**Functional Tests Table:**
| Scenario |
|----------|------------ | Description | Acceptance-|------------|
| 标准前反向 | Default parameters | PyTorch 一致 |
| dtype 覆盖 | float32/float64致 |
| 多维度覆盖 | PyTorch 一 | Various shapes | Output shape/numerical correct |
| 参数变化 | Different params | PyTorch 一致 |
| 反向传播 | Gradient check | Gradient matches PyTorch |

**Performance Tests Table:**
| Scenario | Description | Acceptance |
|----------|-------------|------------|
| 前向性能（小） | Small tensor | (ms_time - BACKGROUND_NOISE) ≤ torch_time * 1.1 |
| 前向性能（中） | Medium tensor | (ms_time - BACKGROUND_NOISE) ≤ torch_time * 1.1 |
| 前向性能（大） | Large tensor | (ms_time - BACKGROUND_NOISE) ≤ torch_time * 1.1 |

### Section 4: Related Operators

```markdown
### 与相关算子的关系

| 算子 | 关系 |
|------|------|
| Operator A | Relationship description |
| Operator B | Relationship description |
```

### Section 5: Algorithm Details

```markdown
### 算法详解

**XXX 公式**：
```
formula here
```

**参数说明**：
- Param 1: Description
- Param 2: Description

**输出尺寸计算**：
```
Output shape formula
```
```

## Example: Complete Issue Template

See `/Users/litingyu/lty_work/issue_template_*.md` for complete examples:
- `issue_template_softplus.md`
- `issue_template_threshold.md`
- `issue_template_softshrink.md`
- `issue_template_smooth_l1_loss.md`
- `issue_template_group_norm.md`
- `issue_template_replication_pad_3d.md`
- `issue_template_upsamplingnearest2d.md`
- `issue_template_upsamplingbilinear2d.md`
- `issue_template_zeropad1d.md`
- `issue_template_constantpad.md`
- `issue_template_constant_pad_1d.md`
- `issue_template_constant_pad_2d.md`
- `issue_template_constant_pad_3d.md`

## Key Information to Extract from Test Files

### From Test File Header
- Test file location
- Test functions covered
- Execution modes (pynative, KBK)
- Dtypes tested

### From Helper Functions
```python
def generate_random_input(shape, dtype):
    """Generate test data."""
    return np.random.uniform(-10, 10, shape).astype(dtype)

def generate_expect_forward_output(x, param1, param2):
    """PyTorch reference output."""
    return torch.nn.functional.xxx(x, param1=param1, param2=param2)

def xxx_forward_func(x, param1, param2):
    """MindSpore forward function."""
    return mint.nn.functional.xxx(x, param1=param1, param2=param2)
```

### From Test Cases
```python
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
@pytest.mark.parametrize('shape', [...])
@pytest.mark.parametrize('dtype', [...])
@pytest.mark.parametrize('param', [...])
def test_xxx_xxx(mode, shape, dtype, param):
    """Test description."""
    # Implementation
```

## Kernel File Analysis

### Forward Kernel Pattern
```cpp
extern "C" int XxxExt(int nparam, void **params, int *ndims, int64_t **shapes,
                       const char **dtypes, void *stream, void *extra) {
  auto tensors = ConvertToATenTensors(nparam, params, ndims, shapes, dtypes, extra, c10::kCPU);
  auto at_input = tensors[0];
  auto at_output = tensors[nparam - 1];
  
  // Get parameters
  auto param = input_utils.GetScalarInput(1);  // or GetIntInput, GetFloatInput
  
  // Call ATen kernel
  at::xxx_out(at_output, at_input, param);
  
  return 0;
}
```

### Backward Kernel Pattern
```cpp
extern "C" int XxxGradExt(int nparam, void **params, int *ndims, int64_t **shapes,
                          const char **dtypes, void *stream, void *extra) {
  auto tensors = ConvertToATenTensors(nparam, params, ndims, shapes, dtypes, extra, c10::kCPU);
  auto at_grad_output = tensors[0];
  auto at_input = tensors[1];
  auto at_grad_input = tensors[nparam - 1];
  
  // Call ATen backward kernel
  at::xxx_backward_out(at_grad_input, at_grad_output, at_input, ...);
  
  return 0;
}
```

## Acceptance Criteria

### Functional Tests
- Results match PyTorch implementation within tolerance
- Forward output numerical accuracy (typically atol=1e-4)
- Backward gradient numerical accuracy
- All execution modes work (PyNative, KBK)
- All dtypes supported

### Performance Tests
```python
# Standard acceptance criterion
(ms_time - BACKGROUND_NOISE) <= torch_time * 1.1

# Where BACKGROUND_NOISE is typically 0.005-0.01 seconds
```

## Output

Generate the issue template in `/Users/litingyu/lty_work/issue_template_xxx.md` with:

1. Issue title: `[Feature]: [OP Plugin] [OPS] [CPU] mint.nn.XXX算子适配`
2. Complete markdown content following the template structure
3. Creation instructions for GitCode/GitHub

## Commands

### Create Issue Template
```bash
# Read test files
cat /Users/litingyu/lty_work/mindspore_op_plugin/tests/st/mint/test_xxx.py

# Read kernel files
cat /Users/litingyu/lty_work/mindspore_op_plugin/op_plugin/ops/kernel/xxx.cc
cat /Users/litingyu/lty_work/mindspore_op_plugin/op_plugin/ops/kernel/xxx_grad.cc

# Create issue template
write /Users/litingyu/lty_work/issue_template_xxx.md "<template_content>"
```

### GitCode Issue Creation
1. Visit: https://gitcode.net/{owner}/{repo}/issues
2. Click "New Issue"
3. Title: `[Feature]: [OP Plugin] [OPS] [CPU] mint.nn.XXX算子适配`
4. Body: Copy template content
5. Create Issue

## Quality Checklist

- [ ] Operator name and PyTorch reference are correct
- [ ] Formula/algorithm is accurately described
- [ ] All test scenarios are documented
- [ ] Kernel files are identified
- [ ] Related operators are listed
- [ ] Application scenarios are meaningful
- [ ] Acceptance criteria are realistic
- [ ] File paths are accurate
- [ ] Execution modes are covered

## Examples

### Softplus Issue Template
```markdown
### 🚀 背景描述

`torch.nn.functional.softplus` 是 PyTorch 中的 Softplus 激活函数...

**参考实现**：PyTorch `torch.nn.functional.softplus(input, beta=1.0, threshold=20.0)`

### 设计思路

- **前向**：调用 ATen `at::softplus_out` 实现 Softplus 激活
- **反向**：调用 ATen `at::softplus_backward_out` 实现梯度回传

### 测试设计

| 场景 | 说明 | 验收 |
|------|------|------|
| 标准前反向 | 默认 beta=1.0, threshold=20.0 | 与 PyTorch 一致 |
| 不同 beta | 0.5, 1.0, 2.0, 5.0 | 与 PyTorch 一致 |
```
