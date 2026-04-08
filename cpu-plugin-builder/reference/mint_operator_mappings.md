### MINT OPERATOR NAME MAPPINGS

**CRITICAL**: When answering questions about `mint.*` operators, ALWAYS check this mapping first to avoid confusion between operator variants (e.g., `ACos` vs `AcosExt`).

## Why This Matters

MindSpore has **multiple implementations** of trigonometric and math functions:
- **Legacy operators**: `ACos`, `Asin`, `Atan` (older implementation)
- **Extended operators**: `AcosExt`, `AsinExt`, `AtanExt` (newer, used by mint)

**ALWAYS verify which one `mint.*` actually uses** before searching for backward implementations!

## How to Check

**Primary Source**: `mindspore/python/mindspore/mint/__init__.py`

Look for the import statement:
```python
from mindspore.ops.function.math_func import acos_ext as acos
```

This tells you:
- `mint.acos` → imports `acos_ext` → uses `AcosExt` operator (NOT `ACos`!)

## Common Mappings

| mint API | Import Name | Operator Name | ⚠️ NOT This |
|----------|-------------|---------------|-------------|
| `mint.acos` | `acos_ext` | `AcosExt` | ❌ `ACos` |
| `mint.asin` | `asin_ext` | `AsinExt` | ❌ `Asin` |
| `mint.atan` | `atan_ext` | `AtanExt` | ❌ `Atan` |
| `mint.acosh` | `acosh_ext` | `AcoshExt` | ❌ `Acosh` |
| `mint.asinh` | `asinh_ext` | `AsinhExt` | ❌ `Asinh` |
| `mint.atanh` | `atanh` | `Atanh` | ✓ (no Ext) |
| `mint.sin` | `sin` | `Sin` | ✓ (no Ext) |
| `mint.cos` | `cos` | `Cos` | ✓ (no Ext) |
| `mint.tan` | `tan` | `Tan` | ✓ (no Ext) |
| `mint.add` | `add` | `AddExt` | (overloaded) |
| `mint.sub` | `sub` | `SubExt` | (overloaded) |
| `mint.mul` | `mul` | `Mul` | (overloaded) |
| `mint.pow` | `pow_ext` | `PowExt` | ❌ `Pow` |

## Operator Naming Patterns

### Pattern 1: Direct Import (No Ext)
```python
from mindspore.ops.functional import sin
```
→ `mint.sin` uses `Sin` operator

### Pattern 2: _ext Import
```python
from mindspore.ops.function.math_func import acos_ext as acos
```
→ `mint.acos` uses `AcosExt` operator (NOT `ACos`!)

### Pattern 3: Overloaded Operators
```python
from mindspore.ops.functional_overload import max
```
→ May map to multiple operators (e.g., `MaxExt`, `MaxDimExt`, `MaximumExt`)
→ Check `mindspore/ops/api_def/max` for variants

## Backward Operator Differences

### Example: ACos vs AcosExt

**Legacy `ACos` backward**:
```cpp
REG_BPROP_BUILDER("ACos").SetUnusedInputs({i1}).SetBody(BODYFUNC(ib) {
  auto dx = ib->Emit("ACosGrad", {x, dout});  // Uses dedicated ACosGrad operator
  return {dx};
});
```

**Modern `AcosExt` backward**:
```cpp
REG_BPROP_BUILDER("AcosExt").SetUnusedInputs({i1}).SetBody(BODYFUNC(ib) {
  // Inline computation with elementary ops
  dx = ib->Neg(dout) * ib->Rsqrt(ib->Sub(ib->Tensor(1, ib->GetDtype(x)), ib->Square(x)));
  return {dx};
});
```

**Key Difference**:
- `ACos` uses a **dedicated backward operator** `ACosGrad`
- `AcosExt` uses **inline elementary operations** (`Neg`, `Mul`, `Rsqrt`, `Sub`, `Square`)

## Verification Workflow

When asked "What is the backward operator of mint.X":

1. ✅ **Check mint/__init__.py** to find the import
   ```bash
   grep "as X$" mindspore/python/mindspore/mint/__init__.py
   ```

2. ✅ **Search for the CORRECT operator name**
   ```bash
   grep "REG_BPROP_BUILDER(\"XxxExt\")" mindspore/ccsrc/frontend/expander/grad/grad_math_ops.cc
   ```

3. ❌ **DO NOT assume** it's the obvious name
   - ❌ `mint.acos` → `ACos` (WRONG!)
   - ✅ `mint.acos` → `AcosExt` (CORRECT!)

## Quick Reference Commands

```bash
# Find mint.acos mapping
grep "as acos$" mindspore/python/mindspore/mint/__init__.py

# Find AcosExt backward
grep -A 10 "REG_BPROP_BUILDER(\"AcosExt\")" mindspore/ccsrc/frontend/expander/grad/grad_math_ops.cc

# List all _ext imports in mint
grep "_ext as" mindspore/python/mindspore/mint/__init__.py
```

## When in Doubt

**ALWAYS read the import line in mint/__init__.py FIRST** before searching for backward operators!
