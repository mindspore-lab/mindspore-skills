# MindSpore Mint API Methodology

## Error Priority Rules

1. Prefer `unknown` over a guessed `yes` or `no`.
2. For `func_op`, GRAPH `unknown` does not mean unsupported. It means the GRAPH path should be understood through expansion.
3. `multi_overload_op` must not be interpreted with a single-primitive mental model.
4. For `high_level_module`, inherited primitive/support facts from `construct` must be read as inherited facts, not direct class facts.
5. `runtime_utility` APIs should not continue operator mapping; treat primitive mapping as `not_applicable`.
6. `dispatch.enable=True` is only an adapter-layer clue, not a final support conclusion.

## How To Read The Index

1. Read `semantic_kind` first.
2. Then read `trust_level` to know whether the fact is direct, inherited, scenario-dependent, or expansion-based.
3. Then read `primitive` or `possible_primitives`.
4. Then read `pynative_support` and `graph_kbk_o0_support`.
5. Then read `aclnn`.
6. If `llm_warning` is not empty, follow the warning before using the summary as a hard conclusion.

## General Workflow

1. Start from the public API and resolve re-exports, `from ... import *`, and aggregator modules. For class symbols in aggregator modules, do not stop at the first star import; collect candidates and choose the best same-name definition.
2. For class APIs, analyze `construct` first. Allow one layer of `self.helper(...)`. Do not recurse through nested class APIs indefinitely.
3. Besides direct `return mint.nn.functional.xxx(...)`, also support callable members bound in `__init__`, for example `self.xxx = ops.auto_generate.SomeOp(...)` and then `self.xxx(...)` inside `construct`.
4. Primitive resolution priority: `api_def -> op_def -> class.name`, then `python symbol -> ops.auto_generate -> op_def`, then small manual mappings for generated names that do not match op_def names.
5. `dispatch.enable=True` means an adapter layer is generated: pyboost for PYNATIVE and KBK for GRAPH `jit_level='O0'`.
6. `dispatch.{platform}=None` means that platform adapter is not generated and should prefer `no` even if later name-based kernel matches exist.
7. PYNATIVE support: Ascend uses pyboost customize, `LAUNCH_ACLNN(...)`, or downstream Ascend pyboost chains; CPU/GPU use kernel registration evidence.
8. GRAPH KBK O0 support: Ascend uses KBK aclnn kernel-mod or register evidence; CPU/GPU use kernel factory or fallback evidence.
9. Ascend KBK registration must include both `MS_ACLNN_KERNEL_FACTORY_REG(...)` and `MS_ACLNN_COMMON_KERNEL_FACTORY_REG(...)`.
10. `aclnn.interfaces` means direct primitive aclnn integration; `aclnn.effective_interfaces` means the final aclnn interfaces actually reached by the execution chain, and it can be multi-valued.
11. If CPU/GPU registry and fallback have both been checked and neither exists, emit `no` instead of leaving `unknown`.
12. For `func_op`, PYNATIVE still follows dispatch/runtime evidence; GRAPH should be modeled through `meta_dsl/func_op` expansion rather than direct kernel registration.
13. `yes/no/unknown` rule: closed evidence gives `yes`; in-scope negative evidence gives `no`; only unresolved static analysis gives `unknown`.
14. For simple wrappers, helper/setup calls such as seed generation must be kept as `prelude_calls` and must not replace the returned operator call as the terminal symbol.
15. Module-level primitive-instance bindings such as `randint_like_ = RandIntLike()` should be resolved as primitive-instance terminals when a wrapper returns `randint_like_(...)`.

## Examples

### mindspore.mint.AdaptiveAvgPool1d
- API: `mindspore.mint.AdaptiveAvgPool1d`
- Primitive: `AdaptiveAvgPool1D`
- Execution chain: `construct -> mint.nn.functional.adaptive_avg_pool1d -> ops.auto_generate.adaptive_avg_pool1d`
- PYNATIVE: Ascend=`yes`, CPU=`no`, GPU=`no`
- GRAPH KBK O0: Ascend=`yes`, CPU=`no`, GPU=`no`
- Effective aclnn interface: `aclnnAdaptiveAvgPool2d`
- Conclusion: this class API reaches `aclnnAdaptiveAvgPool2d` through a customized Ascend path.

### mindspore.mint.xlogy
- API: `mindspore.mint.xlogy`
- Primitive: `Xlogy` plus scalar overload primitive(s)
- PYNATIVE: Ascend/CPU/GPU all close with runtime evidence
- GRAPH KBK O0: Ascend/CPU/GPU all close with runtime evidence
- aclnn: direct aclnn path exists
- Conclusion: all three backends are supported in both modes.

### mindspore.mint.sum
- API: `mindspore.mint.sum`
- Primitive: `SumExt`
- CPU/GPU kernel mapping: `SumExt -> ReduceSum`
- PYNATIVE: Ascend/CPU/GPU=`yes`
- GRAPH KBK O0: Ascend/CPU/GPU=`yes`; CPU/GPU can also be supported by fallback
- Conclusion: a small manual primitive-to-kernel-name map is necessary for some cases.

### mindspore.mint.add
- API: `mindspore.mint.add`
- implementation_type: `multi_overload_op`
- Primitive: `AddScalar`, `AddExt`
- PYNATIVE: Ascend/CPU/GPU=`yes`
- GRAPH KBK O0: Ascend/CPU/GPU=`yes`
- fallback: `AddExt` can be supported in GRAPH CPU/GPU through `REG_FALLBACK_BUILDER("AddExt")`
- Conclusion: this is a multi-overload API; do not read it as a single-primitive operator.

### mindspore.mint.AdaptiveAvgPool3d
- API: `mindspore.mint.AdaptiveAvgPool3d`
- Primitive: `AdaptiveAvgPool3DExt`
- Execution chain: `construct -> mint.nn.functional.adaptive_avg_pool3d -> ops.auto_generate.adaptive_avg_pool3d_ext`
- PYNATIVE: Ascend=`yes`, CPU=`yes`, GPU=`no`
- GRAPH KBK O0: Ascend=`yes`, CPU=`yes`, GPU=`no`
- Effective aclnn interfaces: `aclnnMean`, `aclnnAdaptiveAvgPool3d`
- Conclusion: `dispatch.GPU=None` directly negates GPU; the Ascend customize path reaches two aclnn interfaces.

### mindspore.mint.BCEWithLogitsLoss
- API: `mindspore.mint.BCEWithLogitsLoss`
- Primitive: `BCEWithLogitsLoss`
- Execution chain: `construct -> self.bce_with_logits -> ops.auto_generate.BCEWithLogitsLoss`
- PYNATIVE: Ascend/CPU/GPU=`yes`
- GRAPH KBK O0: Ascend may still be `unknown`; CPU/GPU are already closed
- Effective aclnn interface: `aclnnBinaryCrossEntropyWithLogits`
- Conclusion: class analysis must also recognize callable members bound in `__init__`, not only direct calls inside `construct`.

### mindspore.mint.minimum
- API: `mindspore.mint.minimum`
- Primitive: `Minimum`
- PYNATIVE: Ascend/CPU/GPU=`yes`
- GRAPH KBK O0: Ascend/CPU/GPU=`yes`
- Ascend KBK evidence: `MS_ACLNN_COMMON_KERNEL_FACTORY_REG(Minimum, aclnnMinimum, 3)`
- Conclusion: auto-generated `MS_ACLNN_COMMON_KERNEL_FACTORY_REG(...)` must be treated as valid Ascend KBK evidence.

### mindspore.mint.CosineEmbeddingLoss
- API: `mindspore.mint.CosineEmbeddingLoss`
- Primitive: `CosineEmbeddingLoss`
- Execution chain: `construct -> mint.nn.functional.cosine_embedding_loss -> ops.auto_generate.cosine_embedding_loss`
- func_op: `cosine_embedding_loss_op.yaml` has `bprop_expander: False`; GRAPH expansion is implemented in `meta_dsl/func_op/cosine_embedding_loss.cc`
- func_op_expands_to: `Mul`, `SumExt`, `Sqrt`, `Div`, `ClampMin`, `MeanExt`, ...
- Conclusion: in GRAPH mode, this API should be understood through func_op expansion rather than direct-kernel registration.

### mindspore.mint.randint_like
- API: `mindspore.mint.randint_like`
- Primitive: `RandIntLike`
- Execution chain: `mint.randint_like -> ops.function.random_func.randint_like_ext -> randint_like_`
- Prelude calls: `default_generator._step`
- Terminal kind: `primitive_instance`
- PYNATIVE: Ascend=`yes`, CPU=`no`, GPU=`no`
- GRAPH KBK O0: Ascend=`yes`, CPU=`no`, GPU=`no`
- Effective aclnn interface: `aclnnInplaceRandom`
- Conclusion: for simple wrappers, helper calls must not replace the returned primitive-instance terminal.

### mindspore.mint.distributed.get_rank
- API: `mindspore.mint.distributed.get_rank`
- semantic_kind: `runtime_utility`
- Primitive mapping: not applicable
- grad/aclnn: `not_applicable`
- Conclusion: this is a runtime utility API and should not be forced into operator mapping.

