---
name: cpu-plugin-builder
description: Build MindSpore CPU operators by adapting ATen (libtorch) operators via mindspore_op_plugin. Use when implementing ops in op_plugin/ops/kernel/, writing kernel .cc files
---

# CPU Plugin Builder

This skill helps you develop CPU operators for MindSpore's op_plugin that call ATen (libtorch) operators.

## When to Use

Use this skill when:
- Implementing CPU operators for mindspore_op_plugin
- Writing forward and backward (gradient) operators kernel `.cc` files under `op_plugin/ops/kernel/`

## Instructions

### Step 1: Load api-helper skill to find op name.
Find the MindSpore forward/backward primitive op names in mindspore/ (regardless of
mint/Tensor). 

### Step 1.5: Verify primitive existence before writing forward kernel
- Check whether a real forward primitive operator exists in MindSpore for this API (for example in `ops/api_def`, `ops/op_def/yaml`, generated prims).
- If no forward primitive/YAML exists and the API is implemented as a Python composite, **do not create a new forward plugin kernel** for a non-existent primitive.
- In this composite case, verify that all primitive operators used by the composite are already implemented in `mindspore_op_plugin/op_plugin/ops/kernel/`.
- If any required composite primitive is missing, implement only those missing primitive kernels.

### Step 2: Find corresponding torch ATen Interface
Find the Aten interface in pytorch/ .
must read mindspore-skills/sklls/cpu-plugin-builder/reference/how_to_find_aten_interface.md

### Step 3: Write the Forward Operator kernel file
Implement in mindspore_op_plugin/op_plugin/ops/kernel/.
must match those primitive names and Aten interfaces found in step 1, step 1.5, and step 2.
must read mindspore-skills/sklls/cpu-plugin-builder/reference/how_to_write_forward_op.md

### Step 4: Write the Backward Operator kernel file
Check all Backward Operator found in step 1, if not impl in mindspore_op_plugin/op_plugin/ops/kernel/. impl them.
must match those primitive names and Aten interfaces found in step 1 and 2.
must read mindspore-skills/sklls/cpu-plugin-builder/reference/how_to_write_backward_op.md

### Step 5: Write the functional test 
Implement in mindspore_op_plugin/tests/st/mint/test_{API_name}.py
must read mindspore-skills/skills/cpu-plugin-builder/reference/how_to_write_the _functional_test

### Step 6: Build and run test
cd `mindspore_op_plugin`
build with `bash build.sh`
get env ready : `source env.source`
run test : `python tests/run_tests.py --type functional --op op_name`

### Step 7: Review code
must read mindspore-skills/skills/cpu-plugin-builder/reference/how_to_review_code

### Step 8: Write Report of Each Step
report contains: forward opname(list out kernel file name), backward op name(list out kernel file name), test result
