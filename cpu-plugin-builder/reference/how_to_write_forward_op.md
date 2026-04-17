
### HOW TO WRITE KERNEL FILE
When you know the op name, create the operator file at `mindspore_op_plugin/op_plugin/ops/kernel/<op_name>.cc`:

#### Template for unary operators (1 input, 1 output):
`OpName` -> MindSpore Primitive OP name to use
`op_name_out` -> Aten op name to use

```cpp

#include <string.h>
#include <torch/extension.h>
#include <iostream>

#include "utils/op_utils.h"

namespace op_plugin {
namespace aten_op {
extern "C" int OpName(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes, void *stream,
                      void *extra) {
  // Parameter list: [input, output]
  auto tensors = ConvertToATenTensors(nparam, params, ndims, shapes, dtypes, extra, c10::kCPU);
  auto at_input = tensors[0];   // input tensor
  auto at_output = tensors[1];  // output tensor

  // Call ATen interface: output = op_name(input)
  at::op_name_out(at_output, at_input);

  return 0;
}
}  // namespace aten_op
}  // namespace op_plugin
```

#### Template for binary operators (2 inputs, 1 output):

```cpp
#include <string.h>
#include <torch/extension.h>
#include <iostream>

#include "utils/op_utils.h"

namespace op_plugin {
namespace aten_op {
extern "C" int OpName(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes, void *stream,
                      void *extra) {
  // Parameter list: [input, other, output]
  auto tensors = ConvertToATenTensors(nparam, params, ndims, shapes, dtypes, extra, c10::kCPU);

  auto at_input = tensors[0];   // first input tensor
  auto at_other = tensors[1];   // second input tensor
  auto at_output = tensors[2];  // output tensor

  // Call ATen interface: output = op_name(input, other)
  at::op_name_out(at_output, at_input, at_other);

  return 0;
}
}  // namespace aten_op
}  // namespace op_plugin
```

#### Template with scalar parameters:

```cpp
#include <string.h>
#include <torch/extension.h>
#include <iostream>

#include "utils/op_utils.h"

namespace op_plugin {
namespace aten_op {
extern "C" int OpName(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes, void *stream,
                      void *extra) {
  // Parameter list: [input, scalar_param, output]
  auto tensors = ConvertToATenTensors(nparam, params, ndims, shapes, dtypes, extra, c10::kCPU);
  auto at_input = tensors[0];   // input tensor
  auto at_output = tensors[2];  // output tensor

  // Extract scalar parameter
  KernelInputInfo &input_info = *static_cast<KernelInputInfo *>(extra);
  KernelInputUtils input_utils(input_info);
  constexpr size_t scalar_idx = 1;
  double scalar_value = input_utils.GetFloatInput(scalar_idx);

  // Call ATen interface: output = op_name(input, scalar)
  at::op_name_out(at_output, at_input, scalar_value);

  return 0;
}
}  // namespace aten_op
}  // namespace op_plugin
```
**Important conventions:**
- Use `#include <torch/extension.h>` (includes all ATen ops) instead of specific headers like `<ATen/ops/xxx.h>`
- Use `auto` (not `auto &`) for tensor variables
- Use `at_` prefix for ATen tensor variables (e.g., `at_input`, `at_output`)
- Use hardcoded index for output tensor (e.g., `tensors[1]` for unary, `tensors[2]` for binary)
- Add comments explaining parameter list and key operations
- Add blank line before `at::xxx_out()` call and after it

### CHECK IMPL. AGAIN

#### Handling Optional Parameters

When ATen operators have optional parameters (like `c10::optional<int64_t>`), you must check the op_def YAML to find the sentinel value used for "None".

**Step 1: Check the op_def YAML**

Location: `mindspore/mindspore/ops/op_def/yaml/<op_name>_op.yaml`

Example from `cross_op.yaml`:
```yaml
dim:
  dtype: int
  default: -65530    # Sentinel value for None
  prim_init: True
```

**Step 2: Convert sentinel value to c10::optional**

```cpp
// Wrong - passes sentinel value directly to ATen
int64_t dim = input_utils.GetIntInput(2);
at::cross_out(at_output, at_input, at_other, dim);  // -65530 is invalid!

// Correct - convert sentinel to c10::nullopt
auto dim = input_utils.GetIntInput(2);
c10::optional<int64_t> dim_opt = (dim == -65530) ? c10::nullopt : c10::optional<int64_t>(dim);
at::cross_out(at_output, at_input, at_other, dim_opt);
```

**Common sentinel values:**

| Type | Sentinel Value | Meaning |
|------|----------------|---------|
| `int` | `-65530` | None |
| `float` | Check YAML | None |

**Alternative pattern using IsNoneInput:**

```cpp
c10::optional<int64_t> dim = c10::nullopt;
if (!input_utils.IsNoneInput(2)) {
  dim = input_utils.GetIntInput(2);
}
```