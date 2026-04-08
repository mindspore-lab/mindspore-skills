# Skill: Determining and Implementing Independent Gradient Operators in op_plugin

## Overview

This skill guides you through the process of determining whether an independent gradient (Grad/GradExt) operator needs to be implemented in the op_plugin, and how to implement it if required.

## Step 1: Identify the Operator's Gradient Type

MindSpore operators can have two types of backward implementations:

### Type 1: Composed Gradient
The gradient is composed of basic operators (e.g., `Sin` uses `Cos` and `Mul`). **No independent Grad operator needed.**

### Type 2: Independent Grad Operator
The bprop explicitly calls an independent `XXXGrad` or `XXXGradExt` primitive. **Independent Grad operator must be implemented.**

## Step 2: Determine Which Type Your Operator Uses

### Method 1: Check if Grad Primitive Exists

```bash
# Check if Grad primitive exists in MindSpore
python -c "from mindspore.ops import functional as F; print(hasattr(F._grad_ops, 'YourOpGrad'))"
```

### Method 2: Search in MindSpore Source Code

```bash
# Search for bprop registration
grep -r "REG_BPROP_BUILDER.*YourOp\|@bprop_getters.register.*YourOp" \
  <mindspore-source>/mindspore/ccsrc/frontend/expander/grad/ \
  <mindspore-source>/mindspore/python/mindspore/ops/_grad_experimental/ \
  --include="*.py" --include="*.cc"

# Search for Grad primitive definition
grep -r "class YourOpGrad\|YourOpGrad.*Primitive" \
  <mindspore-source>/mindspore/ops/auto_generate/gen_ops_prim.py
```

### Method 3: Check op_plugin Kernel Directory

```bash
# If Grad kernel exists, it must be implemented
ls op_plugin/ops/kernel/ | grep -i "yourop.*grad"
```

## Step 3: Analyze the Gradient Requirements

### If Composed Gradient (No Independent Grad)
- Check which basic operators are used in the composition
- Ensure all dependency operators are implemented in op_plugin
- Examples:
  - `Abs`: uses `Sign` and `Mul`
  - `Acos`: uses `Neg`, `Rsqrt`, `Sub`, `Square`

### If Independent Grad Operator Required
- You MUST implement the Grad operator in op_plugin
- Use the corresponding ATen backward function

## Step 4: Implement the Independent Gradient Operator

### Case Study: GeluGradExt

#### 4.1 Analyze the MindSpore Definition

```python
# From mindspore/ops/auto_generate/gen_ops_prim.py
class GeluGradExt(Primitive):
    r"""
    Computes gradients for Gelu operation.
    """
    @prim_arg_register
    def __init__(self):
        pass

    def __call__(self, x, dout):
        return super().__call__(x, dout)
```

#### 4.2 Identify Parameters

For `GeluGradExt`:
- **Input 0**: `grad` - upstream gradient (dy)
- **Input 1**: `input` - original input to gelu
- **Input 2**: `approximate` - approximation mode enum ("none" or "tanh")
- **Output**: `dinput` - gradient with respect to input

#### 4.3 Find Corresponding ATen Function

Check `third_party/libtorch/include/ATen/` for ATen operators:
- `at::gelu_backward` - ATen's gelu backward function
- `at::gelu_backward_out(output, grad, input, approximate)` - out variant

#### 4.4 Implement the Gradient Operator

Create file: `op_plugin/ops/kernel/gelu_grad_ext.cc`

```cpp
/**
 * Copyright 2025 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <torch/extension.h>

#include "utils/op_utils.h"

namespace op_plugin {
namespace aten_op {
extern "C" int GeluGradExt(int nparam, void **params, int *ndims, int64_t **shapes,
                           const char **dtypes, void *stream, void *extra) {
  // Step 1: Extract non-tensor parameters from extra
  // For GeluGradExt: parameter[2] = approximate (int64_t enum)
  KernelInputInfo &input_info = *static_cast<KernelInputInfo *>(extra);
  KernelInputUtils input_utils(input_info);
  int64_t approximate_enum = input_utils.GetIntInput(2);
  c10::string_view approximate = (approximate_enum == 1) ? "tanh" : "none";

  // Step 2: Convert MindSpore tensors to ATen tensors
  // Order: [grad, input, output]
  auto tensors = ConvertToATenTensors(nparam, params, ndims, shapes, dtypes, extra, c10::kCPU);
  auto grad = tensors[0];           // upstream gradient (dy)
  auto input = tensors[1];          // original input to gelu
  auto dinput = tensors[nparam - 1]; // output gradient (dL/dx)

  // Step 3: Call ATen backward operator
  // Use _out variant to write directly to output tensor
  at::gelu_backward_out(dinput, grad.contiguous(), input.contiguous(), approximate);

  return 0;  // Return 0 for success
}
}  // namespace aten_op
}  // namespace op_plugin
```

## Parameter Mapping Reference

### Common Parameter Indices

| Index | Meaning | Description |
|-------|---------|-------------|
| `tensors[0]` | First input | Usually the first input tensor |
| `tensors[1]` | Second input | Usually the second input tensor or input |
| `tensors[nparam-1]` | Output | Always the last parameter |
| `input_utils.GetIntInput(idx)` | Non-tensor int | Integer parameter at index |
| `input_utils.GetScalarInput(idx)` | Non-tensor scalar | Scalar parameter at index |

### Input/Output Mapping Examples

**Binary operators (e.g., AddGrad):**
```cpp
auto grad = tensors[0];      // upstream gradient
auto other = tensors[1];     // other operand
auto dinput = tensors[2];    // gradient w.r.t. input
```

**Unary operators (e.g., GeluGradExt):**
```cpp
auto grad = tensors[0];      // upstream gradient
auto input = tensors[1];     // original input
auto dinput = tensors[2];    // gradient w.r.t. input
```

**With non-tensor parameters:**
```cpp
// Get non-tensor int parameter
int64_t param_value = input_utils.GetIntInput(parameter_index);

// Get non-tensor scalar parameter
at::Scalar scalar_param = input_utils.GetScalarInput(scalar_index);
```

## Step 5: Build and Verify

### Build the Plugin

```bash
# Windows
build.bat

# Linux
bash build.sh
```

### Check Registration

Look for this line in build output:
```
-- Found operator: GeluGradExt in gelu_grad_ext.cc
```

### Test the Operator

```python
import mindspore as ms
from mindspore import mint

ms.set_context(mode=ms.PYNATIVE_MODE)

# Test forward
x = ms.Tensor([[1.0, 2.0], [3.0, 4.0]])
y = mint.gelu(x)
print("Forward:", y)

# Test backward
x.requires_grad = True
grad = ms.ops.grad(lambda x: mint.gelu(x).sum(), (0,))(x)
print("Gradient:", grad)
```

## Quick Reference: Decision Tree

```
Is there a dedicated Grad primitive in MindSpore?
├── No → Check if basic operators are used (composed gradient)
│         └── Ensure all dependency operators are implemented
└── Yes → Must implement the Grad operator in op_plugin
          ├── Find ATen backward function
          ├── Map parameters correctly
          └── Implement in <op>_grad.cc or <op>_grad_ext.cc
```

## Common ATen Backward Functions

| MindSpore Operator | ATen Backward Function |
|---------------------|------------------------|
| Gelu | `at::gelu_backward_out` |
| Sigmoid | `at::sigmoid_backward_out` |
| Relu | `at::relu_backward_out` |
| Softplus | `at::softplus_backward_out` |
| Add | N/A (composed: Mul + Add) |
| Abs | N/A (composed: Sign) |

## Troubleshooting

### "The kernel XXXGrad unregistered"
**Cause**: Grad operator not implemented in op_plugin
**Solution**: Implement the Grad operator following the template above

### Parameter Index Mismatch
**Cause**: Incorrect parameter ordering
**Solution**: Verify parameter order from MindSpore primitive definition

### Non-tensor Parameter Issues
**Cause**: Incorrect extraction of non-tensor parameters
**Solution**: Use `input_utils.GetIntInput()` or `input_utils.GetScalarInput()` with correct index

## Examples Reference

### Simple Gradient Implementation (Unary)

```cpp
// sigmoid_grad.cc
extern "C" int SigmoidGradExt(int nparam, void **params, int *ndims, int64_t **shapes,
                              const char **dtypes, void *stream, void *extra) {
  auto tensors = ConvertToATenTensors(nparam, params, ndims, shapes, dtypes, extra, c10::kCPU);
  auto grad = tensors[0];
  auto output = tensors[1];
  auto dinput = tensors[nparam - 1];

  at::sigmoid_backward_out(dinput, grad, output);
  return 0;
}
```

### With Approximation Parameter

```cpp
// gelu_grad_ext.cc (from example above)
int64_t approximate_enum = input_utils.GetIntInput(2);
c10::string_view approximate = (approximate_enum == 1) ? "tanh" : "none";
at::gelu_backward_out(dinput, grad.contiguous(), input.contiguous(), approximate);
```

## Summary

1. **Check** if Grad primitive exists in MindSpore
2. **If yes**: Implement independent Grad operator in op_plugin
3. **If no**: Verify dependency operators are implemented (composed gradient)
4. **Map parameters** correctly: [inputs..., output] order
5. **Use ATen _out variants** for direct output writing
6. **Extract non-tensor parameters** using `input_utils`
7. **Build and test** to verify registration
