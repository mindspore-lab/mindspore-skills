# Workflow: MindSpore Native ACLNN Lane

## Goal

Execute the ACLNN implementation lane after the control plane has already
resolved:

- the correct API/operator branch
- the backend gap
- the implementation method: `native-framework`
- the framework: `mindspore`
- the backend lane: `npu`/`aclnn`

This file is intentionally thin. It orchestrates the ACLNN lane but does not
own API resolution, method selection, or final verification.

## Preconditions

Before entering this workflow, the following must already be fixed:

- `resolved_api`
- `resolved_operator`
- `selected_method = native-framework`
- `selected_framework = mindspore`
- `selected_backend_lane = npu or aclnn`

These decisions belong to:

- `references/operator-spec.md`
- `references/method-selection.md`
- `references/operator-resolution/api-to-operator.md`
- `references/operator-resolution/operator-to-backend.md`
- `references/native-framework/mindspore.md`

## Inputs

At minimum:

- `resolved_api`
- `resolved_operator`
- `workspace_root`
- `backward_required` when already known
- `existing_coverage` when already known

## Outputs

This lane must produce:

- ACLNN precheck result
- forward implementation result
- backward implementation result when required
- modified file list
- generated artifact list
- verification handoff data

## Execution Order

1. Run `./aclnn/00-pre-checks.md`
2. Run the ACLNN implementation flow in order:
   - `./aclnn/01-yaml-definition.md`
   - `./aclnn/02-code-generation.md`
   - `./aclnn/03-general-infer.md`
   - `./aclnn/04-pyboost.md`
   - `./aclnn/05-kbk.md`
3. If backward work is required, run `./aclnn/06-bprop.md`
4. When export/doc/test work is required, continue with:
   - `./aclnn/07-export.md`
   - `./aclnn/08-testing.md`
   - `./aclnn/09-docs.md`
5. Hand off to `references/verification.md`

## Notes

- This lane owns ACLNN-specific execution only.
- It must not redo routing decisions.
- If a future MindSpore native lane is added, it should become a sibling lane
  under `workflows/native-framework/mindspore/`.
