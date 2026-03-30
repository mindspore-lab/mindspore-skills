# MindSpore Native-Framework

Use this reference when the selected method is `native-framework` and the
target framework is `mindspore`.

Focus on:

- source-tree integration points
- operator definition, registration, and build updates
- framework rebuild and wheel output


## When To Read

Read this reference after:

- `references/operator-spec.md`
- `references/method-selection.md`

Use it to decide how the resolved branch should be handled inside the MindSpore framework source tree.


## Native Implementation Lane Entry

When the resolved branch clearly maps to the NPU/ACLNN backend lane, continue with:
- `workflows/native-framework/mindspore/aclnn.md`


## MindSpore Native Integration Surfaces

Typical source areas include:

- `mindspore/ops/op_def/yaml/`
- `mindspore/ops/api_def/`
- `mindspore/ops/op_def/yaml/doc/`
- infer implementation under MindSpore ops sources
- PyBoost / KBK ACLNN sources when the selected lane requires them
- Python export locations when the resolved interface is public

Use repository state as the final source of truth when workflow text and source layout differ.

## MindSpore Reference

This hlep understanding mindspore call chain, especally help api and operator resolution.

- `references/operator-resolution/api-to-operator.md`
- `references/operator-resolution/operator-to-backend.md`

`api-to-operator.md` resolve:

- `mint.*`, `Tensor.*`, wrapper, alias, and overload identity
- the correct active operator branch
- the final Primitive/operator name

use `operator-to-backend.md` after the operator branch is fixed to determine:

- whether backend support is visible from static source
- whether the current operator branch uses `auto_generate` or `customize` when ACLNN evidence is present

