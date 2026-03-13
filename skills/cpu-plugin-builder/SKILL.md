---
name: cpu-plugin-builder
description: >
  Build MindSpore CPU operator kernels in mindspore_op_plugin. Covers the full
  workflow: primitive discovery (yaml → CamelCase name), ATen interface mapping,
  forward and backward kernel authoring (.cc files under op_plugin/ops/kernel/),
  bprop chain tracing, and build/test. Use for mint.*, Tensor.*, and
  nn.functional.* ops. Not for GPU/CUDA, Ascend/ACLNN, or Python-level ops.
---

# CPU Plugin Builder

Develop CPU operator kernels for MindSpore's `mindspore_op_plugin`, which wraps
ATen (libtorch) functions. Each kernel is a `.cc` file under
`op_plugin/ops/kernel/` exposing an `extern "C" int <PrimitiveName>(...)` entry
point.

## Prerequisites

Before starting, confirm:
- `MINDSPORE_ROOT` — absolute path to the MindSpore source tree
- `PLUGIN_KERNEL_DIR` — absolute path to `op_plugin/ops/kernel/`
- The `api_name` you are implementing (e.g. `sub`, `mul`, `log`, `amax`)

---

## Step 1: Resolve API → Primitive Names

**Goal**: discover every overload (yaml file) for the target API and which
primitives already have kernels.

Run:
```bash
bash skills/cpu-plugin-builder/scripts/find_ops.sh \
  <api_name> <MINDSPORE_ROOT> <PLUGIN_KERNEL_DIR>
```

**CHECKPOINT 1** — state: (a) total primitives found, (b) which are `missing`.
Example: `SubExt missing, SubScalar missing (2 total)`.

---

## Step 2: Find the ATen Interface

For each primitive with `status=missing`, find the corresponding ATen op.

Read: `./reference/how_to_find_aten_interface.md`

**CHECKPOINT 2** — one line per missing primitive: `SubExt → at::sub_out`.

---

## Step 3: Trace the Backward Chain

For each primitive you need to implement, run:
```bash
bash skills/cpu-plugin-builder/scripts/find_bprop.sh \
  <PrimitiveName> <MINDSPORE_ROOT>
```

Read the printed bprop body carefully. Identify every primitive op used in it.
If `BinopGradCommon` is detected, the script will note the required kernels
(`sum_ext.cc`, `reshape.cc`).

Then re-run `find_ops.sh` for each backward primitive to check kernel status.

Read: `./reference/how_to_write_backward_op.md` (especially the BinopGradCommon
section if that pattern was detected).

**CHECKPOINT 3** — for each backward primitive: name, file, action (write/skip).
Example: `SumExt → sum_ext.cc (exists, skip)`, `Reshape → reshape.cc (missing, write)`.

---

## Step 4: Write Forward Kernel(s)

Read: `./reference/how_to_write_forward_op.md`

For each missing forward primitive, write `<snake_case_name>.cc` in
`op_plugin/ops/kernel/`.

**CHECKPOINT 4** — list files written: `sub_ext.cc` (written), etc.

---

## Step 5: Write Backward Kernel(s)

Read: `./reference/how_to_write_backward_op.md`

For each missing backward primitive identified in Checkpoint 3, write the
corresponding `.cc` file. Skip any that already `exist` per the script output.

**CHECKPOINT 5** — list files written or skipped: `reshape.cc` (written), `sum_ext.cc` (existed, skipped).

---

## Step 6: Build and Test

```bash
cd mindspore_op_plugin
bash build.sh
source env.source
pytest tests/st/mint/test_<api_name>.py
```

**CHECKPOINT 6** — state test result: `5 passed` or the first failing test name + error type. Fix failures before writing the report.

---

## Step 7: Report

Produce a structured summary:

```
## CPU Plugin Build Report: mint.<api_name>

### Forward kernels
| PrimitiveName | file           | action  |
|---------------|----------------|---------|
| SubExt        | sub_ext.cc     | written |

### Backward kernels
| PrimitiveName | file           | action  |
|---------------|----------------|---------|
| SumExt        | sum_ext.cc     | existed |
| Reshape       | reshape.cc     | written |

### Test result
pytest tests/st/mint/test_sub.py — 5 passed in 3.21s
```
