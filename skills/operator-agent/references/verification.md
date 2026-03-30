# Verification

This file owns final verification for `operator-agent`.

Use it after the selected workflow has finished code changes. Do not use this
file for API resolution, backend selection, or builder-step routing.

## Minimum Verification Set

Always verify:

- resolved API/operator identity still matches the modified branch
- build success or generation success for the touched path
- operator registration or import success
- minimal forward execution
- backward execution when required
- artifact locations and reuse instructions

## Identity Sanity Checks

Before reporting success, confirm:

- the public API symbol still maps to the resolved operator branch
- alias or `_ext` cases were not collapsed to the wrong Primitive name
- backward lookup, when discussed, uses the same resolved operator identity

If the request started from `mint.*`, `Tensor.*`, or a wrapper API, verify the
answer against the real export/import path instead of guessing from the public
name alone.

## Implementation Checks

Confirm the implementation state required by the selected lane:

- YAML / dispatch changes are present when the branch definition changed
- generated files are refreshed when generation was required
- infer changes exist when shape/type behavior was part of the task
- runtime implementation evidence exists for the selected backend lane
- Python export changes exist when `export_required = true`

For ACLNN-oriented work, also check:

- the branch-local ACLNN mode matches the intended path (`auto_generate` or
  `customize`)
- expected PyBoost / KBK evidence exists for the chosen path

## Runtime Smoke Checks

At minimum, record whether you validated:

- import or registration path
- one representative forward case
- one representative backward case when required

When the operator semantics make it relevant, also note whether these were
covered or intentionally skipped:

- dynamic shape or dynamic rank
- scalar / tuple / `None`-like inputs
- empty tensor behavior
- special values such as `inf` / `nan`

## Documentation And Optional Artifacts

Documentation and scratch templates are not mandatory for every execution path,
but if they were touched or required by the task, report them explicitly.

Examples:

- public API export/doc updates
- local PTA review notes
- local ACLNN call-chain notes

## Report Output

Report:

- modified files
- generated artifacts or rebuild outputs
- what was verified directly
- what was not verified
- known risks or blockers
