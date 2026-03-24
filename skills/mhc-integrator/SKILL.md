---
name: mhc-integrator
description: Add manifold-constrained hyper-connections (mHC) to decoder-only or causal LLMs in PyTorch or Hugging Face Transformers. Use when Codex needs to port the mHC residual-stream design from a notebook, paper, or existing implementation into a decoder stack by extending config classes, adding per-block mHC modules, expanding and reducing residual streams, wiring initialization, exposing load or train flags, or adding tests for output shape and doubly-stochastic residual mappings.
---

# mHC Integrator

Use this skill to retrofit mHC into an existing decoder block without changing the model's public hidden size.
Treat mHC as a residual-stream wrapper around attention and MLP, not as a new attention mechanism.

## Core workflow

1. Inspect the target model's config class, decoder block, model forward path, training entrypoints, and any code-generation constraints.
2. Add config fields for `use_mhc`, `mhc_expansion_rate`, `mhc_sinkhorn_iterations`, `mhc_gating_factor_init`, and `mhc_output_reduce`, plus argument validation.
3. Add an mHC module that consumes `[batch, seq, streams, hidden]`, flattens to `[batch, seq, streams * hidden]`, and emits `pre`, `post`, and `residual` mappings.
4. Keep the non-mHC decoder path behaviorally identical; gate the new path behind `config.use_mhc`.
5. Expand streams once after embeddings and reduce them once before final norm or task heads.
6. Wire custom initialization so mHC starts close to stable identity mixing instead of arbitrary branch interference.
7. Expose the flag at load and training callsites and add focused tests.

## Integration rules

- Do not change the attention module's public input or output hidden size. mHC reduces `[B, S, N, D]` to `[B, S, D]` before attention or MLP and restores the residual stream afterward.
- Build masks, cache metadata, and RoPE inputs from the base `[B, S, D]` embeddings, not from the expanded residual tensor.
- Reduce streams before the final model norm and LM head so downstream heads and losses stay unchanged.
- Keep mapping logits in `float32` for Sinkhorn stability even when the model runs in `bf16` or `fp16`.
- Do not modify `modular_xx.py`. Do not regenerate `modeling_xx.py`. Directly modify `modeling_xx.py` at the real decoder implementation site unless the user explicitly asks for a regeneration workflow.
- Prefer `mhc_output_reduce="mean"` when retrofitting pretrained checkpoints. Use `"sum"` only when intentionally matching a reference implementation that sums branches.

## Implementation notes

- The notebook prototype expands embeddings with `unsqueeze(...).repeat(...)` and merges them back with `sum(dim=2)`.
- The provided Qwen3 port keeps the prototype's mapping structure but uses a simpler `RMSNorm -> Linear` path instead of the fused RMSNorm trick from the notebook. Revisit the fused version only if profiling shows the mapping path is a bottleneck.
- Initialize `alpha` near zero so pretrained residual dynamics are not immediately destabilized.
- Bias the residual mapping toward an identity matrix at initialization so each stream mostly preserves itself before training.
- When `mhc_expansion_rate == 1`, special-case formulas that would otherwise call `log(0)`.

## References

Open only what you need:

- `references/implementation-pattern.md`: generic porting recipe, shapes, pseudocode, and initialization rules.
- `references/qwen3-case-study.md`: mapping from the provided notebook, Qwen3 files, tests, and training script.
- `references/validation-checklist.md`: minimum and extended validation matrix plus failure symptoms.
