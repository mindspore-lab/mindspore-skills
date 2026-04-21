# HF Transformers Guardrails

## Scope and precedence
- This file contains route-specific rules for the `hf-transformers` route.
- If any rule conflicts with `SKILL.md`, follow this file.

## Guardrails
- Avoid custom compatibility wrappers unless required.
- Use diff-based insertion when updating auto maps.
- Keep changes minimal and aligned with existing MindOne patterns.
- `register_buffer` is supported in MindSpore; do not remove it as part of device-handling cleanup.
- Do not add `model.device`, `.device` compatibility properties, or other
  PyTorch-style device shims; remove device-only usage from migrated code,
  tests, and examples instead.
- Standalone HF-style model repositories are an exception to the default rule
  against migrating configuration and tokenization files when local
  `trust_remote_code=False` loading requires those components.
- Do not change processor `tokenizer_class`, `attributes`,
  `image_processor_class`, or `video_processor_class` by default; do it only
  when target Auto capability is missing and record the reason.
- Do not load `AutoProcessor` inside model `__init__` to infer preprocessing
  defaults; prefer config values or local processor defaults.
- Preserve the route's `mindspore.mint` and `mindspore.mint.nn` conversions by
  default after auto-convert.
- Do not batch-convert auto-generated `mint.*` and `mint.nn.*` usages back to
  `mindspore.nn.*`, `ops.*`, or legacy MindSpore APIs as a generic cleanup
  step.
- If a specific operator must move away from `mint`, document the concrete
  reason such as missing support, target-repo convention, or verified semantic
  mismatch.
- Model coding standards:
  - Import MindSpore as `import mindspore` (avoid `import mindspore as ms`).
  - Use `from mindspore import nn` and define modules as `nn.Cell`.
  - `nn.Cell` guidance applies to module base classes and structure; it is not
    a blanket instruction to rewrite layer implementations away from `mint.nn`.

## Response expectations
- List reference files consulted.
- Summarize edits and note any risks or TODOs.
- Suggest next tests when appropriate.
