
### HOW TO FIND ATEN INTERFACE
When you know the MindSpore op YAML, pick the best matching ATen operator.

#### Checklist for selecting ATen operator:
`ms_op` -> MindSpore op name in YAML
`aten_op` -> candidate ATen operator name

- Read `mindspore/mindspore/ops/op_def/yaml/<op_name>_op.yaml` for args, returns, attrs, defaults, and training/inference behavior.
- Search ATen schemas in `pytorch/aten/src/ATen/native/native_functions.yaml` and generated headers in `pytorch/build/aten/src/ATen/ops/*_ops.h` or `pytorch/aten/src/ATen/`.
- Prefer same-name operators and `_out` variants to write directly into output tensors.
- If outputs mismatch, choose the closest semantic operator and safely ignore extra outputs if needed.
- Validate semantics by checking native implementation in `pytorch/aten/src/ATen/native/*.cpp`.

**Important conventions:**
- Use `pytorch/` paths for ATen sources and generated headers.
- Only use low-level ops (e.g., `native_batch_norm_out`) when a higher-level wrapper does not match semantics.
- If no `_out` exists, use `at::xxx()` then `copy_()` into output.

### CHECK IMPL. AGAIN

#### If paths or generated files are missing
- Download PyTorch 2.1 source into your workspace (e.g. `pytorch/`).
- Generate ATen headers: `python -m torchgen.gen -s aten/src/ATen -d build/aten/src/ATen` (run in the PyTorch repo root).

#### BatchNorm example (batch_norm_ext)
`batch_norm_ext` -> MindSpore op
`_batch_norm_impl_index` -> selected ATen op

- MS YAML: `mindspore/mindspore/ops/op_def/yaml/batch_norm_ext_op.yaml` returns `output`, `saved_mean`, `saved_variance`.
- ATen candidate `batch_norm` returns only 1 tensor, mismatch.
- ATen candidate `native_batch_norm_out` is lower-level and lacks wrapper behavior.
- Selected `_batch_norm_impl_index` returns `(out, save_mean, save_invstd, reserve, impl_index)` and matches semantics.
- Ignore the last two return values when mapping to MS outputs.
- Implementation reference: `mindspore_op_plugin/op_plugin/ops/kernel/batch_norm_ext.cc`.

#### Convolution example (conv2d_ext)
`conv2d_ext` -> MindSpore op
`convolution_out` -> selected ATen op

- MS YAML: `mindspore/mindspore/ops/op_def/yaml/conv2d_ext_op.yaml` returns `output`.
- ATen `conv2d` has no `_out` variant, cannot write directly to output tensor.
- Selected `convolution_out` supports bias/stride/padding/dilation/groups and writes into `out`.
- For non-transposed conv2d, use `transposed=false` and `output_padding={0, 0}`.
- Implementation reference: `mindspore_op_plugin/op_plugin/ops/kernel/conv2d_ext.cc`.

#### Call chain (PyTorch forward)
- `torch.nn.functional.batch_norm` -> `torch.batch_norm`
- `at::batch_norm` -> `at::_ops::batch_norm::call` (generated)
- Dispatcher -> `at::native::batch_norm` in `pytorch/aten/src/ATen/native/Normalization.cpp`
- `at::_batch_norm_impl_index` -> backend `cudnn_batch_norm` / `miopen_batch_norm` / `native_batch_norm`

#### Reference files
- `pytorch/aten/src/ATen/native/native_functions.yaml`
- `pytorch/build/aten/src/ATen/Functions.h`
- `pytorch/build/aten/src/ATen/ops/*_ops.h`
- `pytorch/aten/src/ATen/`
