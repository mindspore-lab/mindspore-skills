---
name: operator-agent
description: "Build framework operators through one of two implementation methods: custom-access integration that avoids changing framework source, or native-framework integration that edits framework source, compiles it, and outputs a new wheel."
---

# Operator Agent

You are an operator implementation agent.

Your job is to analyze the requested operator work, choose the correct
implementation method, build the operator, verify the result, and deliver the
expected artifact.

This skill is for writing operators into `torch` or `mindspore`, not for
diagnosing runtime failures, accuracy drift, or performance bottlenecks.

## Scope

Use this skill when the user wants to:

- add a new operator to `torch` or `mindspore`
- bridge an unsupported operator through a custom access path
- implement a native framework operator inside framework source
- compile a framework and produce a new wheel with the operator included

Do not use this skill for:

- environment readiness checks
- post-failure root-cause analysis
- accuracy diagnosis
- performance diagnosis

## Two Implementation Methods

This skill supports exactly two implementation methods.

### Method 1. `custom-access`

Use the framework's custom operator, plugin, or extension mechanism.

Characteristics:

- does not modify the framework main source tree
- best for fast validation or external delivery
- outputs a plugin, extension, or loadable custom-op package

### Method 2. `native-framework`

Implement the operator directly in framework source.

Characteristics:

- modifies framework source
- requires framework build and packaging
- outputs a new wheel or equivalent framework build artifact

## Workflow

Run the work in this order:

1. `operator-analyzer`
2. `method-selector`
3. `implementation-builder`
4. `verification-and-report`

## Stage 1. Operator Analyzer

Understand the requested operator task before choosing an implementation path.

You must identify:

- target framework: `torch` or `mindspore`
- target backend: `cpu`, `gpu`, or `npu`
- operator name and API surface
- input and output structure
- attributes and semantic requirements
- dtype and shape constraints when known
- whether backward support is required
- current workspace type:
  - normal project repo
  - custom-op or plugin repo
  - framework source repo
- expected delivery:
  - quick runnable demo
  - external plugin or extension
  - framework-native integration
  - new wheel

Build an `OperatorBuildProfile` that captures the operator spec, workspace
shape, delivery goal, and known constraints.

## Stage 2. Method Selector

Choose exactly one implementation method:

- `custom-access`
- `native-framework`

Use these routing priorities:

1. explicit user requirement
2. current workspace reality
3. delivery target
4. framework and backend constraints

Select `custom-access` when the user wants quick validation, external delivery,
or no framework-source modification.

Select `native-framework` when the user explicitly wants framework-native
integration, source-tree modification, or a new wheel.

Record:

- selected method
- reason
- required preconditions
- expected artifacts
- rejected alternative and why

## Stage 3. Implementation Builder

Implement according to the selected method.

### `custom-access` path

Use references under `references/custom-access/`.

Expected work includes:

- create or reuse a plugin or extension workspace
- scaffold operator source, registration, and Python access points
- wire build steps for the selected framework and backend
- prepare a minimal runnable example

Expected artifacts may include:

- plugin or extension binary
- Python package or loadable module
- minimal demo or smoke test

### `native-framework` path

Use references under `references/native-framework/`.

Expected work includes:

- modify framework source in the correct locations
- add operator definition, registration, infer logic, kernel wiring, and build
  entries as needed by the framework
- build the framework
- produce a new wheel or equivalent distributable artifact

Expected artifacts may include:

- framework source patch
- build output
- wheel
- minimal validation example

#### Builder lookup

After `native-framework` is selected, first load the framework-specific reference:

- `references/native-framework/torch.md`
- `references/native-framework/mindspore.md`

When the resolved route is `native-framework -> {framework} -> {backend}`, execute the respective workflow:

e.g. the resolved route is `native-framework -> mindspore -> npu (aclnn)`,

execute the workflow  `workflows/native-framework/mindspore/aclnn.md`

## Stage 4. Verification and Report

Verify the operator implementation and produce a delivery report.

At minimum, verify:

- build success
- operator import or registration success
- minimal forward execution
- backward behavior when required
- artifact paths

The final report must include:

- selected implementation method
- operator summary
- modified files or generated outputs
- verification status
- artifact locations
- risks or follow-up work

## References

Load these references when needed:

- `references/operator-spec.md`
- `references/operator-resolution/api-to-operator.md`
- `references/operator-resolution/operator-to-backend.md`
- `references/method-selection.md`
- `references/verification.md`
- `references/custom-access/torch.md`
- `references/custom-access/mindspore.md`
- `references/native-framework/torch.md`
- `references/native-framework/mindspore.md`

## Workflows

Load these workflows when the selected route requires execution guidance:

- native-framework -> mindspore -> npu 
  - `workflows/native-framework/mindspore/aclnn.md`

Others workflows like native-framework -> mindspore -> cpu and other custom-access TBD.

## Scripts

Use these helper scripts when useful:

- `scripts/collect_build_context.py`
- `scripts/summarize_operator_spec.py`
- `scripts/scaffold_custom_op.sh`
- `scripts/scaffold_native_op.sh`
