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

**Action**: Load api-helper skill to understand the operator API.

**Output**: Status table showing completion.

---

### Step 2: Find corresponding torch ATen Interface

**Must Read**: `./reference/how_to_find_aten_interface.md`

**Action**: 
- Search ATen headers in `third_party/libtorch/include/ATen/ops/`
- Confirm the ATen function signature
- Check if `_out` variant exists

**Output**: Status table with ATen interface details.

---

### Step 3: Write the Forward Operator

**Must Read**: `./reference/how_to_write_forward_op.md`

**Action**:
- Create kernel file in `op_plugin/ops/kernel/{op_name}_ext.cc`
- Follow naming convention: Function name must be `{OpName}Ext`
- Use ATen interface (prefer `_out` variant if available)

**Output**: Status table with kernel file name.

---

### Step 4: Write the Backward Operator (if needed)

**Must Read**: `./reference/implementing-gradient-operators.md`

**Action**: Create backward kernel if operator requires gradient computation.

**Output**: Status table (can be skipped if no backward needed).

---

### Step 5: Write Test Files

**Must Read**: 
- `./reference/test-generator.md` 
- `./reference/performance-test-generator.md`

**Action**:
1. Create `tests/st/mint/test_{op_name}.py` - Functional tests
2. Create `tests/st/mint/test_perf_{op_name}.py` - Performance tests

**Validation Checklist** (MUST complete all):
- [ ] Test file follows naming convention
- [ ] Uses `@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')`
- [ ] Parametrizes mode: `@pytest.mark.parametrize('mode', ['pynative', 'KBK'])`
- [ ] rtol=0, atol=0 in allclose_nparray
- [ ] Cover standard functionality
- [ ] Cover 0D input (if applicable)
- [ ] Cover multi-dimensions (0D-6D or more)
- [ ] Cover non-contiguous inputs (use `mint.transpose`)
- [ ] Cover boundary values
- [ ] Cover vmap (level_mark='level1') with batch sizes [8, 16, 32, 64]
- [ ] Cover different data types (if applicable)
- [ ] Do NOT test unsupported scenarios (dynamic shape, auto num_classes=-1, etc.)

**Output**: Status table showing test coverage.

---

### Step 6: Build and Run Tests

**Environment**: `my_x64_env`

**Commands**:
```bash
conda activate my_x64_env
cd mindspore_op_plugin
bash build.sh
source env.source
pytest tests/st/mint/test_{op_name}.py -v
```

**Validation Checklist**:
- [ ] Build succeeds without errors
- [ ] All tests pass (show count: X passed)
- [ ] No import errors
- [ ] No runtime errors

**Output**: Status table with build and test results.

---

### Step 7: Pylint Check

**Command**:
```bash
python -m pylint tests/st/mint/test_{op_name}.py --disable=all --enable=E,W
```

**Validation Checklist**:
- [ ] Score >= 9.0/10 (preferably 10/10)
- [ ] No errors (E)
- [ ] No critical warnings (W)

**Output**: Status table with pylint score.

---

### Step 8: Code Review

**Action**: Load `op-plugin-code-reviewer` skill and run checks.

**Validation Checklist**:
- [ ] Coding rules (copyright 2026, no Chinese comments, no try/except skip)
- [ ] ATen interface correct (prefer `_out` if available)
- [ ] No unnecessary validations
- [ ] No unnecessary type casts
- [ ] Test baseline requirements met
- [ ] Test coverage complete per test-generator.md
- [ ] Performance tests adequate

**Output**: Code review report with PASS/FAIL status.

---

### Step 9: Generate Status Report

**MUST DO**: Only proceed after Steps 5-8 ALL PASS.

**Report Format**:
```
==============================================================
{Operator Name} CPU Plugin Implementation Report
==============================================================

1. FORWARD OPERATOR
   - Kernel File: {file_name}
   - ATen Interface: {aten_function}
   - Status: ✅ Implemented

2. BACKWARD OPERATOR
   - Status: {✅ Implemented / ⏭️ Not Needed}

3. TEST COVERAGE
   - Functional Tests: {X} test cases
   - Performance Tests: {Y} scenarios
   - All Tests: ✅ PASSED

4. CODE QUALITY
   - Pylint Score: {X.XX}/10
   - Code Review: ✅ PASSED

5. VALIDATION SUMMARY
   - Build: ✅ Success
   - Tests: ✅ {X}/{X} Passed
   - Pylint: ✅ Passed
   - Code Review: ✅ Passed

==============================================================
```

---

### Step 10: Write Issue

**Must Read**: `./reference/issue-generator.md`

**Action**: Create detailed issue document.

**Include**:
- Background description
- Design思路
- Test coverage summary (reference Step 9 report)
- Known limitations
- Files involved

**Output**: Issue markdown file.

---

### Step 11: Create Commit and Push

**Commands**:
```bash
cd commit_store/mindspore_op_plugin
git checkout master
git branch {operator_name}
git checkout {operator_name}
git add op_plugin/ops/kernel/{op_name}_ext.cc
git add tests/st/mint/test_{op_name}.py
git add tests/st/mint/test_perf_{op_name}.py
git commit -m "feat: add {op_name} operator support

- Add {OpName}Ext forward kernel implementation
- Add functional tests ({X} test cases)
- Add performance tests

Test results: {X} passed"
git push -u origin {operator_name}
```

**Output**: Branch pushed to remote with commit hash.

---

## Workflow Summary Table

| Step | Task | Validation | Output |
|------|------|------------|--------|
| 1 | Load api-helper | - | Status table |
| 2 | Find ATen interface | Read how_to_find_aten_interface.md | Status table |
| 3 | Write forward op | Read how_to_write_forward_op.md | Status table |
| 4 | Write backward op | Read implementing-gradient-operators.md | Status table (optional) |
| 5 | Write tests | Read test-generator.md, performance-test-generator.md | Test coverage checklist |
| 6 | Build & test | Use my_x64_env | Build + test results |
| 7 | Pylint | pylint score >= 9.0 | Pylint report |
| 8 | Code review | Load op-plugin-code-reviewer | Review report |
| 9 | Status report | All previous steps PASS | Implementation report |
| 10 | Write issue | Read issue-generator.md | Issue document |
| 11 | Commit & push | Git operations | Remote branch |

**CRITICAL**: Do NOT proceed to next step if current step has failures.
**CRITICAL**: Always print status table after completing each step.
