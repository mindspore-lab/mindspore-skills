
### HOW TO WRITE BACKWARD OP

When you know the forward op name, use api-helper skill to get backward op needed for forward op.

#### Case 1: Standalone Grad Operator:
 - If the backward uses `Emit("XXXGrad", ...)`, it is dedicated grad operator.
 - write xxx_grad.cc in `op_plugin/ops/kernel/xxx_grad.cc`
 - e.g. for gelu_ext_grad.cc

```
#include <torch/extension.h>

#include "utils/op_utils.h"

namespace op_plugin {
namespace aten_op {
extern "C" int GeluGradExt(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes, void *stream,
                           void *extra) {
  KernelInputInfo &input_info = *static_cast<KernelInputInfo *>(extra);
  KernelInputUtils input_utils(input_info);
  int64_t approximate_enum = input_utils.GetIntInput(2);
  c10::string_view approximate = (approximate_enum == 1) ? "tanh" : "none";

  auto tensors = ConvertToATenTensors(nparam, params, ndims, shapes, dtypes, extra, c10::kCPU);
  auto grad = tensors[0];
  auto input = tensors[1];
  auto dinput = tensors[nparam - 1];

  at::gelu_backward_out(dinput, grad.contiguous(), input.contiguous(), approximate);
  return 0;
}
}  // namespace aten_op
}  // namespace op_plugin
```

#### Case 2: Multiple Primitive Operators

 - For `AcosExt` backward which uses `ib->Neg(dout) * ib->Rsqrt(ib->Sub(..., ib->Square(x)))`:
 - backward op is Neg,Rsqrt,Square, so you should write three neg.cc, rsqrt.cc, square.cc in `op_plugin/ops/kernel/xxx_grad.cc`
 - e.g. for neg.cc

```
#include <string.h>
#include <torch/extension.h>
#include <iostream>

#include "utils/op_utils.h"

namespace op_plugin {
namespace aten_op {
extern "C" int Neg(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes, void *stream,
                   void *extra) {
  // Parameter list: [input, output]
  auto tensors = ConvertToATenTensors(nparam, params, ndims, shapes, dtypes, extra, c10::kCPU);
  auto at_input = tensors[0];
  auto at_output = tensors[nparam - 1];

  // Call ATen interface: output = -input
  at::neg_out(at_output, at_input);

  return 0;
}
}  // namespace aten_op
}  // namespace op_plugin

```

#### Case 3: BinopGradCommon Pattern

`BinopGradCommon` is used by binary ops that support broadcasting (e.g. `SubExt`,
`MulExt`, `DivExt`). It handles reducing the gradient back to the original tensor
shape after the forward broadcast.

**What it expands to:**
```
dx = SumExt(dout, reduce_axes) -> Reshape(dx, x.shape)
dy = SumExt(dout, reduce_axes) -> Reshape(dy, y.shape)
```

Concretely, the backward needs two primitives:
- `SumExt` (or `ReduceSum` in older graphs) — reduces along broadcast dims
- `Reshape` — restores the original input shape

**Required kernel files:**
- `op_plugin/ops/kernel/sum_ext.cc`
- `op_plugin/ops/kernel/reshape.cc`

**Before writing:** check whether these files already exist:
```bash
ls op_plugin/ops/kernel/sum_ext.cc op_plugin/ops/kernel/reshape.cc 2>/dev/null
```
If both exist, no action needed — the `find_bprop.sh` script also flags this.

If `sum_ext.cc` is missing, implement `SumExt` using `at::sum_out` with the
provided axes. If `reshape.cc` is missing, implement `Reshape` using
`at::reshape` (note: Reshape takes the target shape as an integer list input,
not a tensor attribute).

### NOTES:
 - Always run `find_bprop.sh` first; it auto-detects BinopGradCommon and lists
   the required files.
 - Check `op_plugin/ops/kernel/` before writing any backward kernel — if it
   already exists, skip it and note "existed" in the report.