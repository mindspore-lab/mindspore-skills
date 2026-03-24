# CLAUDE.md

This file provides guidance to Claude Code when working with the ascend-ops-reviewer skill.

## What This Is

A Claude Code skill for reviewing torch_npu and MindSpore operator code changes. It supports both local diff files and GitCode PR URLs, providing comprehensive code review with focus on correctness, performance, memory safety, API compatibility, and test coverage.

## Skill Activation

The skill triggers on keywords related to operator code review: "检视算子", "review PR", "代码审查", "torch_npu review", "MindSpore 算子检视", or when a GitCode PR URL is provided.

## Repository Structure

```
SKILL.md                        # Skill definition with 5-step review workflow
CLAUDE.md                       # This file - development guide
scripts/
  fetch_pr.py                   # Fetch PR diff from GitCode using git commands
  parse_diff.py                 # Parse unified diff format and extract structure
  analyze_changes.py            # Analyze changes and detect risk areas
references/
  review_checklist.md           # Complete review checklist (correctness/performance/safety/test/doc)
  common_pitfalls.md            # Common pitfalls library (shape/dtype/memory/numerical/gradient)
  torch_npu_patterns.md         # torch_npu specific patterns (ACLNN/registration/testing)
  mindspore_patterns.md         # MindSpore specific patterns (YAML/Infer/Bprop/ACLNN)
.claude/
  settings.local.json           # Permission configuration
```

## Skill Workflow

The skill guides through 5 steps:

1. **Input Acquisition** — Accept diff file path or GitCode PR URL, fetch diff content
2. **Diff Parsing & Classification** — Parse unified diff, classify files, determine report detail level
3. **Automated Analysis** — Extract operator info, detect common pitfalls, identify risk areas
4. **Comprehensive Review** — Apply full checklist, generate structured issue list by severity
5. **Report Generation** — Generate detailed or layered report based on diff size

## Script Architecture

### fetch_pr.py

Fetches PR diff from GitCode using git commands:
- Parse PR URL to extract repo and PR ID
- Use `git fetch origin pull/{pr_id}/head` to fetch PR branch
- Generate diff with `git diff origin/master...pr-{pr_id}`
- Clean up temporary branch
- Return diff content and metadata

### parse_diff.py

Parses unified diff format:
- Split by file boundaries (`diff --git`)
- Parse hunks (`@@ -old_start,old_lines +new_start,new_lines @@`)
- Classify file types (python/cpp/cuda/header/test/doc)
- Count additions/deletions
- Return structured diff data

### analyze_changes.py

Analyzes code changes:
- Extract operator names from paths and code
- Identify change types (forward/backward/shape_infer/dtype/test)
- Detect risk areas using pattern matching:
  - Memory safety (pointer ops, array access, malloc/new)
  - Numerical stability (log/exp/sqrt/div operations)
  - Shape inference (shape operations, broadcasting)
  - Dtype handling (type conversions, casts)
- Analyze test coverage
- Return analysis results

## Reference Documents

### review_checklist.md

Complete review checklist organized by severity:
- **Critical**: Correctness (algorithm/shape/dtype/API), Memory safety
- **Major**: Performance, Test coverage
- **Minor**: Documentation, Code quality

### common_pitfalls.md

Common pitfalls categorized by area:
- Shape inference: broadcasting, dynamic shape, reduction, negative index
- Dtype: integer division, float16 overflow, implicit conversion, integer overflow
- Memory safety: null pointer, array bounds, resource leak, use-after-free
- Numerical stability: log(0)/div(0), exp overflow, accumulation precision, sqrt negative
- Gradient: inplace ops, gradient chain break, special value gradients, higher-order gradients
- API compatibility: parameter order, default values, error types
- Testing: only normal cases, large tolerance, missing gradient tests
- Performance: unnecessary copy, repeated computation, frequent small allocations

### torch_npu_patterns.md

torch_npu specific patterns:
- Operator registration (TORCH_LIBRARY_IMPL)
- ACLNN operator calling patterns
- Device memory management
- Stream management
- Error handling
- Dtype handling
- Shape inference
- Testing patterns
- Performance optimization

### mindspore_patterns.md

MindSpore specific patterns:
- Operator definition (YAML)
- Shape/Dtype inference (Infer)
- Backward propagation (Bprop)
- ACLNN adaptation
- PyNative vs Graph mode
- Dtype handling
- Error handling
- Testing patterns
- Performance optimization

## Report Formats

### Small Diff (< 200 lines): Detailed Report

Includes all issues with full details:
- Overview (PR/operators/framework/size)
- Critical issues (must fix) with code examples
- Major issues (recommended fix)
- Minor issues (optional optimization)
- Suggestions (improvements)
- Test coverage analysis
- Summary and recommendations

### Large Diff (≥ 200 lines): Layered Report

Focuses on critical issues:
- Overview with warning about summary mode
- Critical + Major issues with details
- Minor + Suggestions as statistics with collapsible list
- Test coverage analysis
- Summary and recommendations

## Token Optimization

To save tokens:
1. Only analyze changed code (diff hunks), not full files
2. Use layered reports for large diffs
3. Lazy load reference docs (only when needed)
4. Pattern matching before LLM analysis
5. Batch process similar issues

## Integration with Other Skills

### mindspore-ops-debugger

When runtime issues are detected (precision, crash, gradient anomaly, shape error, dtype mismatch), suggest using `mindspore-ops-debugger` skill in the report's "Recommended Actions" section.

## Editing Guidelines

- All documentation in Chinese (except code comments)
- Code comments in English
- Keep reference docs up-to-date with new patterns
- Maintain consistency between checklist and pitfalls
- Follow global coding standards in /Users/claw/CLAUDE.md

## Testing

To test the skill:

1. **Local diff file**:
   ```bash
   cd /Users/claw/work/ms_debug/torch_npu
   git diff HEAD~1 > /tmp/test.diff
   # Use skill to review /tmp/test.diff
   ```

2. **GitCode PR**:
   - Provide a real torch_npu or MindSpore PR URL
   - Verify git fetch succeeds
   - Verify diff generation
   - Verify report generation

3. **Large diff**:
   - Create a diff > 200 lines
   - Verify automatic switch to summary mode
   - Verify layered report format

4. **Framework-specific**:
   - Test torch_npu operator review
   - Test MindSpore operator review
   - Verify framework-specific pattern detection

## Success Criteria

1. **Functionality**:
   - ✅ Support local diff files
   - ✅ Support GitCode PR URLs
   - ✅ Generate structured reports
   - ✅ Auto-adjust report detail level

2. **Detection**:
   - ✅ Detect 80%+ common issues
   - ✅ Zero Critical false positives
   - ✅ Provide actionable fix suggestions

3. **Performance**:
   - ✅ Small diff (< 200 lines) in 2 minutes
   - ✅ Large diff (< 1000 lines) in 5 minutes
   - ✅ Reasonable token usage (< 10k for small diff)

4. **Usability**:
   - ✅ Single command invocation
   - ✅ Clear error messages
   - ✅ Easy-to-read reports
