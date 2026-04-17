### HOW TO FIND ATEN INFERFACE 

Search for the corresponding ATen operator:

1. **Location**: `./third_party/libtorch/include/ATen/`
2. **Prefer `_out` variants**: Use `at::xxx_out()` when available to write directly to output tensor
3. **Fallback to copy**: If no `_out` variant exists, use `at::xxx()` then `copy_()` to output

Reference files:
- `aten/src/ATen/native/native_functions.yaml` for operator definitions
- `aten/src/ATen/templates/RedispatchFunctions.h` for `_out` variants
