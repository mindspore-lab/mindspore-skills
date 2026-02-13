### Understand Mindspore API Call Chains

This guide helps you locate and identify API/operator call chains in the MindSpore codebase.

### Instructions

#### Step 1A: Identify mint API Forward Call Chain
- mint import entry -> `mindspore/python/mindspore/mint/__init__.py`
- API definition -> `mindspore/ops/api_def/`
- Op definition -> `mindspore/ops/op_def/yaml/`

```python
# Example: mint.linspace
from mindspore.ops.function.math_func import linspace_ext as linspace
```

This tells you:
- **Module path**: `mindspore.ops.function.math_func`
- **Internal function name**: `linspace_ext`
- **Public API name**: `linspace`
- **Operator name**: `LinSpaceExt` (CamelCase of `linspace_ext`)

##### mint Case 1: Overloaded Operators
For example, `mint.max` has multiple implementations in `mindspore/ops/api_def/max.yaml`, and is imported from functional overload:

```python
from mindspore.ops.functional_overload import max
```

Then map each overload item to its `op_yaml`:
- `max_op.yaml`
- `max_dim_op.yaml`
- `maximum_op.yaml`

##### mint Case 2: Inplace Operators
`Tensor.sub_` (inplace subtraction) usually maps to inplace op files such as:
- `inplace_sub_ext.cc`

The `_` suffix in Python often indicates inplace behavior.

##### mint Case 3: Wrapper/Module APIs
Some APIs are wrapper/module APIs and are neither overload nor inplace.

Example: `mint.nn.AdaptiveMaxPool1d`
- `mint.nn.layer.pooling.AdaptiveMaxPool1d.construct`
- `mint.nn.functional.adaptive_max_pool1d`
- `ops.auto_generate.gen_ops_prim.adaptive_max_pool1d_op(...)`
- primitive from YAML/class name: `AdaptiveMaxPool1D`

---

#### Step 1B: Identify Tensor API Forward Call Chain
For `Tensor.xxx` APIs, resolve in this order.

1. Locate Tensor method entry:
- `mindspore/python/mindspore/common/tensor.py`
- `mindspore/python/mindspore/ops/tensor_method.py` (for `py_method` route)

2. Classify the route:
- **Python registry route**:
  - `Tensor.xxx` in `common/tensor.py` calls `tensor_operator_registry.get("xxx")`
  - lookup binding in `mindspore/python/mindspore/ops/functional.py` via `setattr(tensor_operator_registry, "xxx", ...)`
  - then continue to bound function/primitive
- **py_method route**:
  - check `mindspore/ops/api_def/<op>.yaml` for `py_method: tensor_xxx`
  - resolve `tensor_xxx` in `mindspore/python/mindspore/ops/tensor_method.py`
- **C++ TensorPy route**:
  - method is implemented in C++ (TensorPy method descriptor)
  - locate implementation in `mindspore/ccsrc/pynative/utils/pyboost/custom/tensor.cc`
  - common pattern: `Tensor::xxx -> mindspore::kernel::pyboost::xxx`

3. Resolve primitive name:
- `mindspore/python/mindspore/ops/auto_generate/gen_ops_prim.py`
- `mindspore/ops/op_def/yaml/*_op.yaml`

##### Tensor route example: `Tensor.repeat`
- C++ method: `Tensor::repeat` in `mindspore/ccsrc/pynative/utils/pyboost/custom/tensor.cc`
- forward dispatch: `mindspore::kernel::pyboost::repeat(...)`
- primitive: `Repeat` in `gen_ops_prim.py`

---

#### Step 2: Identify Backward Operator Name
After Step 1 (mint or Tensor), once you get the **correct primitive/operator name**:

1. Search `mindspore/ccsrc/frontend/expander/grad/` for that exact name.
2. Find `REG_BPROP_BUILDER("<CorrectOpName>")` registration.
3. Parse the full BODYFUNC and list each operator call:
- Find `ib->OperatorName(...)`
- List all operators (not only `Emit("XXXGrad", ...)`)

##### Backward Case 1: Standalone Grad Operator
- backward uses `Emit("XXXGrad", ...)`, which is a dedicated grad operator.

##### Backward Case 2: Multiple Primitive Operators
- Example: `AcosExt` backward uses `Neg`, `Rsqrt`, `Sub`, `Square`, and mul expression.

##### Backward Case 3: Mixed Composite Backward Chain
Some bprop bodies mix helper primitives and emitted grad ops.

Example `AdaptiveMaxPool1D` backward chain:
- `ExpandDims`
- `Emit("AdaptiveMaxPool2DGrad", ...)`
- `Reshape`
- `OutZeros(output_size)`

Scope rules:
- kernel-required primitives: `ExpandDims`, `Reshape`, `Squeeze`, `Transpose`, `Cast`, etc.
- graph/meta helpers (no kernel file): `OutZeros`, `ShapeCalc`, `TupleGetItem`, `EmitValue`

### Key Search Locations

| Purpose                           | Path                                                           |
| --------------------------------- | -------------------------------------------------------------- |
| mint imports                      | `mindspore/python/mindspore/mint/__init__.py`                  |
| Tensor methods                    | `mindspore/python/mindspore/common/tensor.py`                  |
| Tensor py_method implementations  | `mindspore/python/mindspore/ops/tensor_method.py`              |
| Tensor registry bindings          | `mindspore/python/mindspore/ops/functional.py`                 |
| Tensor C++ pyboost custom methods | `mindspore/ccsrc/pynative/utils/pyboost/custom/tensor.cc`      |
| Function implementations          | `mindspore/python/mindspore/ops/function/*.py`                 |
| Primitive definitions             | `mindspore/python/mindspore/ops/auto_generate/gen_ops_prim.py` |
| Op YAML definitions               | `mindspore/ops/op_def/yaml/*_op.yaml`                          |
| Gradient definitions              | `mindspore/ccsrc/frontend/expander/grad/`                      |

### Examples

**Example A: mint API (`mint.acos`)**
```bash
# mint import
rg -n "acos_ext as acos" mindspore/python/mindspore/mint/__init__.py

# backward registration
rg -n "REG_BPROP_BUILDER\(\"AcosExt\"\)" mindspore/ccsrc/frontend/expander/grad
```

**Example B: Tensor API (`Tensor.repeat`)**
```bash
# Tensor py_method (if any)
rg -n "def tensor_repeat" mindspore/python/mindspore/ops/tensor_method.py

# Tensor C++ method path
rg -n "Tensor::repeat" mindspore/ccsrc/pynative/utils/pyboost/custom/tensor.cc

# primitive
rg -n "class Repeat\(|repeat_op=" mindspore/python/mindspore/ops/auto_generate/gen_ops_prim.py
```
