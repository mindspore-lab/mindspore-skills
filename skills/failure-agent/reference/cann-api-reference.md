# CANN API Reference

Use this file as a lightweight reminder for Ascend operator/API-side diagnosis.

Default structured inputs live in:

- `reference/index/cann_error_index.yaml`
- `reference/index/cann_aclnn_api_index.yaml`

Optional maintenance-side raw docs may also exist when explicitly generated:

- `reference/index/aclError.md`
- `reference/index/aclnnApiError.md`
- `reference/index/aclnn_api_compact.md`

## When to use

Use this reference when the failure clearly sits in:

- ACLNN parameter validation
- ACLNN runtime/internal errors
- CANN operator constraints
- operator adaptation or backend path selection

## ACLNN Two-Phase Pattern

Many operator paths follow:

1. `aclnnXxxGetWorkspaceSize`
2. `aclnnXxx`

If the first phase fails, suspect parameter, shape, dtype, or kernel-package
issues before blaming execution.

## Useful Error Families

- `161xxx`: parameter validation
- `361xxx`: runtime error
- `561xxx`: internal, infer shape, tiling, kernel package, or OPP path issues

High-value mappings:

- `561003`: kernel not found
- `561107`: OPP path not configured
- `561112`: operator kernel package not found

## Practical Checks

- confirm `ASCEND_OPP_PATH` is valid
- confirm current CANN version matches the expected operator path
- distinguish compile-time TBE failures from runtime ACLNN failures
- keep operator shape and dtype constraints explicit when reproducing

## Related References

- use [pta-diagnosis](pta-diagnosis.md) for quick code routing
- use [backend-diagnosis](backend-diagnosis.md) for broader layer triage
- use [mindspore-dianosis](mindspore-dianosis.md) if source-level investigation is needed

Prefer the YAML indexes first. Only read the optional markdown sources when the
structured entries are insufficient for the current diagnosis.
