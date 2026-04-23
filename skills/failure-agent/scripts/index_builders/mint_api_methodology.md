# MindSpore Mint API Index Builder Methodology

This methodology applies to `generate_mindspore_failure_index.py` rule
maintenance, index field semantics, builder regression tests, and alignment of
MindSpore mint API index generation policy.

## Error Priority Rules

1. Prefer `unknown` over a guessed `yes` or `no`.
2. For `func_op`, backend support must be judged through expansion. A backend is `yes` only when every expanded primitive closes on that backend.
3. `multi_overload_op` must not be interpreted with a single-primitive mental model or a sibling-overload support union.
4. For `high_level_module`, inherited primitive/support facts from `construct` must be read as inherited facts, not direct class facts.
5. `runtime_utility` APIs should not continue operator mapping; treat primitive mapping as `not_applicable`.
6. `dispatch.enable=True` is only an adapter-layer clue, not a final support conclusion.

## How To Read The Index

1. Read `call_chain_kind` and `resolution_kind` first to know which resolver produced the primitive/support facts.
2. Then read `implementation_type` for the API shape.
3. Then read `trust_level` as a reliability tier: `certain`, `strong`, `conditional`, `weak`, or `not_applicable`.
4. Then read `primitive` or `possible_primitives`.
5. Then read `pynative_support` and `graph_kbk_o0_support`.
6. Then read `fact_origin` to see whether the fact came from direct resolution, construct inheritance, alias inheritance, overload dispatch, or func expansion.
7. Then read `support_reason_kind` as the single explanation field for why support is modeled that way. `func_dispatch` means GRAPH KBK O0 was closed from func_op `dispatch` plus `meta_dsl`, not from extracted expansion primitives.
8. Then read `aclnn`.
9. Then read `path_hints` to jump directly to the key source files instead of grepping the repo again.
10. Read `summary` only as a compact restatement of structured fields; when it conflicts with structured facts, trust the structured facts.

## Path Hints

- `path_hints` is grouped for LLM retrieval, not a full file inventory.
- `api_def_paths` stores the `api_def/*.yaml` files actually used for this API.
- `dispatch_paths` stores branch selection and overload routing files such as `functional_overload.py`, `api_def/*.yaml`, `functional_map.cc`, and `pyboost_overload_functions.cc`.
- `implementation_paths` stores the real Python or C++ implementation entry files, for example `gen_ops_def.py`, `tensor_method.py`, wrapper files, or composite wrapper files. Files already present in `dispatch_paths` are excluded to avoid duplication.
- `op_def_paths` stores the real `op_yaml` or `func_op yaml` files that close the primitive facts.
- `kernel_paths` are split by exec mode and backend, so `pynative.ascend` and `graph_kbk_o0.cpu` can point to different backend implementation files. ACLNN customize, pyboost, and kernel-mod files are carried inside the relevant `kernel_paths.*.ascend` buckets instead of a separate `aclnn_paths` list.
- `infer_paths` records real InferShape / InferType implementation files, for example `ops/infer/ops_func_impl/*.cc` or `ops/infer/symbol_ops_impl/*.cc`.
- Gradient file locations are **not** in `path_hints`. Use the top-level `grad.impl[]` field instead, which carries path + anchor alongside semantic labels (primitive, kind, scope_kind).

## Field Responsibility

Each piece of information belongs to exactly one field. `path_hints` is the **pure locator layer** (path + anchor only), while top-level semantic fields carry analysis conclusions.

| Information | Authoritative field | Notes |
|---|---|---|
| Gradient file locations | `grad.impl[]` | Includes primitive, kind, scope_kind alongside path+anchor |
| ACLNN interface names | `aclnn.interfaces` / `aclnn.effective_interfaces` | Symbolic names, no file paths |
| ACLNN kernel file locations | `kernel_paths.*.ascend` | File path + anchor pointing to actual registration macros |
| Backend support conclusions | `support_matrix` / `pynative_support` / `graph_kbk_o0_support` | Final yes/no/unknown verdicts |
| Compact text overview | `summary` | Denormalized; when conflicts arise, trust structured fields |

`aclnn` and `kernel_paths.*.ascend` are **complementary**: `aclnn` answers "which aclnn interfaces does this API use" (name level), while `kernel_paths.*.ascend` answers "which file and line to look at" (file level). They are not redundant.

## General Workflow

1. Start from the public API and resolve re-exports, `from ... import *`, and aggregator modules. For class symbols in aggregator modules, do not stop at the first star import; collect candidates and choose the best same-name definition.
2. For class APIs, analyze `construct` first. Allow one layer of `self.helper(...)`. Do not recurse through nested class APIs indefinitely.
3. Besides direct `return mint.nn.functional.xxx(...)`, also support callable members bound in `__init__`, for example `self.xxx = ops.auto_generate.SomeOp(...)` and then `self.xxx(...)` inside `construct`.
4. Primitive resolution priority: real terminal/effective call chain first, then symbol-to-op_def resolution, then small manual mappings for generated names that do not match op_def names. Do not fall back to canonical api_def guesses when the real terminal symbol is unresolved.
5. `dispatch.enable=True` means an adapter layer is generated: pyboost for PYNATIVE and KBK for GRAPH `jit_level='O0'`.
6. `dispatch.{platform}=None` means that platform adapter is not generated and should prefer `no` even if later name-based kernel matches exist.
7. Resolve the real execution chain from `terminal_calls`, `composed_of`, and construct mapping before deciding backend support.
8. `api_def` overload entries are ordered. Overload branch matching follows the YAML order; this is not an unordered union.
9. `interface:function` and `interface:tensor` only mean that a branch participates in functional or tensor overload generation. They do not by themselves imply backend support.
10. PYNATIVE support: Ascend uses pyboost customize, `LAUNCH_ACLNN(...)`, or downstream Ascend pyboost chains; CPU/GPU use kernel registration evidence for normal ops. For `functional_overload`, also read branch-level `api_def` backend declarations.
11. Single-primitive view ops are a special PYNATIVE case: do not require exact-name aclnn/kernel factory closure. If `pyboost_api.cc -> pyboost_core.cc -> kernel::pyboost::<view_op>() -> *_view_impl` closes statically, treat that as positive PYNATIVE evidence for CPU/GPU/Ascend. View composite functions such as `FlattenExt -> flatten_ext_impl -> reshape_impl` may also close through this path.
12. GRAPH KBK O0 support: Ascend uses KBK aclnn kernel-mod or register evidence; CPU/GPU use kernel factory or fallback evidence for normal ops. For `functional_overload`, also read branch-level `api_def` backend declarations and Python fallback branches.
13. Single-primitive `graph_view` ops are a special GRAPH case: if the op yaml marks `graph_view: True` and a matching `MS_HOST_REG_KERNEL(<Primitive>, ...)` exists under `mindspore/ops/kernel/host/view/kernel_mod_impl`, treat that as positive GRAPH Ascend evidence.
14. Base view primitives such as `Reshape` and `Squeeze` can close GRAPH Ascend through `RT_KERNEL + IsNopNode + nop_op_to_memcpy_/MemoryCopyAsync`; this is not ACLNN or ACLOP evidence. Composite view APIs such as `FlattenExt` may close GRAPH Ascend only when their fallback builder explicitly reaches an inner primitive that has this RT/NOP closure.
15. Ascend KBK registration must include both `MS_ACLNN_KERNEL_FACTORY_REG(...)` and `MS_ACLNN_COMMON_KERNEL_FACTORY_REG(...)`.
16. `aclnn.interfaces` means direct primitive or branch-level declared aclnn integration; `aclnn.effective_interfaces` means the final aclnn interfaces actually reached by the execution chain, and it can be multi-valued. When one source file contains multiple sibling primitives or kernel classes, effective interfaces must be extracted from the current primitive/class scope only; do not borrow sibling ACLNN calls from the same file or legacy primitive files.
17. If CPU/GPU registry and fallback have both been checked and neither exists, emit `no` instead of leaving `unknown`.
18. Do not use sibling overload hits or implicit `Ext -> base primitive` fallback as positive CPU/GPU evidence. Only explicit manual maps may cross primitive names.
19. `yes/no/unknown` rule: closed evidence gives `yes`; in-scope negative evidence gives `no`; only unresolved static analysis gives `unknown`.
20. For simple wrappers, helper/setup calls such as seed generation must be kept as `prelude_calls` and must not replace the returned operator call as the terminal symbol.
20a. Prelude helper primitives must not block backend support when a wrapper has a proven single terminal primitive. Example: `adaptive_avg_pool2d_ext` may call `shape_(input)` only to normalize `output_size=None`; the support primitive remains `AdaptiveAvgPool2DExt`, not `AdaptiveAvgPool2DExt + Shape`.
20b. Pure Python prelude such as `validator.check_value_type`, `isinstance`, `tuple(...)`, or `None` normalization is not an operator chain and must not turn a closed terminal primitive into `unresolved_composite_chain`.
21. Module-level primitive-instance bindings such as `randint_like_ = RandIntLike()` should be resolved as primitive-instance terminals when a wrapper returns `randint_like_(...)`.
22. For generated bindings that return `*_impl(...)`, resolve the terminal through `pyboost_inner_prim.py` explicit instance bindings such as `stack_ext_impl = _PyboostStackExtPrim()` and then recover the primitive from the bound class base `StackExtPrim_ -> StackExt`.
23. For `functional_overload` APIs, use `api_def` to enumerate overload branches, `functional_map.cc` to recover GRAPH branch primitives, and `pyboost_overload_functions.cc` to recover PYNATIVE branch primitives.
24. For `functional_overload` branch backends, `pyboost` = direct backend support. `py_method` = Python fallback and must be resolved through the real Python implementation.
25. For fixed C++ instance bridges under `functional_overload`, close only through the proven fixed mapping. Example: `einsum_ext -> functional_overload.einsum -> _einsum_instance` resolves through `functional_map.cc` to `EinsumExt` and then through `einsum_ext_op.yaml` dispatch for Ascend; do not borrow legacy `Einsum` primitive kernels or bprop facts.
25a. Other fixed functional-overload bridges must also use exact mapped primitives only. Example: `nn.functional.conv2d -> functional_overload.conv2d -> functional_map.cc` resolves to `Conv2DExt` / `Conv2DPadding`; do not borrow legacy `Conv2D` kernel evidence for CPU/GPU.
26. `py_method` is not automatically `yes`: executable fallback chain -> `yes`; raise/error stub -> `no`; unresolved dynamic Python chain -> `unknown`.
27. For `functional_overload` in GRAPH mode, if a `deprecated/*.yaml` branch exists, prefer the deprecated branch before non-deprecated branches when modeling graph-side matching.
28. `functional_overload` APIs should prefer `possible_primitives` plus overload-branch support targets over a guessed single terminal primitive, and aggregate backend support branch by branch instead of using a sibling primitive union.
29. `functional_overload` ACLNN facts must also be aggregated branch by branch. Do not leave API-level `aclnn` at totally empty or fully unknown when branch-level ACLNN interfaces are already proven.
30. When some overload branches have proven ACLNN interfaces and others remain unresolved, keep the proven `interfaces` / `effective_interfaces` but set `aclnn.mode = unknown`.
31. `grad` must follow the same real forward scope as primitive/support facts. Do not use sibling overloads, canonical metadata primitives, or helper-only symbols as grad facts.
32. `functional_overload` grad is not automatically `unknown`: if all non-deprecated `interface:function` branches collapse to the same stable overload family and every branch has the same closed grad conclusion, the API-level grad may be promoted to `explicit_bprop` or `autodiff`.
33. If `functional_overload` branches still represent genuinely different function signatures or placeholder/scenario-dependent dispatch paths, keep API-level grad as `unknown` even when some branch primitives have bprop builders.
34. Python `Cell.bprop` is a valid grad fact source when the real forward chain enters that Cell, for example `SyncBatchNormInner.bprop`.
35. `grad.backward_primitives` records backward-path primitive/operator dependencies and is not a duplicate of forward `primitive`.
36. For single-primitive `real_terminal` APIs whose resolved `op_yaml` has no `dispatch`, support may still close directly: Ascend can use `REG_ADPT_DESC(...)` adapter registrations under `op_adapter/op_declare`, and CPU/GPU can use exact kernel factory registrations.
37. The no-`dispatch` direct-support rule is currently limited to simple single-primitive direct terminals. Do not apply it to multi-primitive, scenario-dependent, overload, or composite multi-hop APIs without separately closing those chains.
38. Module-level aliases to generated functions, such as `tensor_gt = greater`, should resolve through the imported generated function and then to the terminal primitive.
39. `_get_cache_prim(P.Xxx)` and `_get_cache_prim(Xxx)` with a statically known primitive class are equivalent to constructing and calling that primitive. Dynamic `_get_cache_prim(expr)` must remain unresolved.
40. Pyboost inner primitive classes such as `_PyboostSearchSortedPrim` can be resolved through their generated base class (`SearchSortedPrim_ -> SearchSorted`) when the inheritance chain is static.
41. A class whose `construct` is a pure `return input` pass-through should be modeled as Python pass-through support rather than as an `Identity` primitive. Do not add `identity_op.yaml` or borrow `Identity` kernel evidence for this case.

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
- call_chain_kind: `functional_overload`
- Primitive: keep `primitive=[]`; use `possible_primitives = [Xlogy, XLogYScalarSelf, XLogYScalarOther]`
- PYNATIVE / GRAPH: aggregate branch support per overload instead of forcing one primitive
- Grad: all non-deprecated function-overload branches close to explicit bprop, so API-level grad can be promoted to `explicit_bprop`
- Conclusion: overload-dispatched support remains branch-based, but grad can still collapse when branch conclusions fully agree.

### mindspore.mint.sum
- API: `mindspore.mint.sum`
- Primitive: `SumExt`
- CPU/GPU kernel mapping: `SumExt -> ReduceSum`
- PYNATIVE: Ascend/CPU/GPU=`yes`
- GRAPH KBK O0: Ascend/CPU/GPU=`yes`; CPU/GPU can also be supported by fallback
- Conclusion: a small manual primitive-to-kernel-name map is necessary for some cases.

### mindspore.mint.add
- API: `mindspore.mint.add`
- call_chain_kind: `functional_overload`
- Primitive: keep `primitive=[]`; use `possible_primitives = [AddScalar, AddExt]`
- Support: aggregate per-overload branch instead of treating the wrapper itself as a direct terminal primitive
- Conclusion: do not read this API as a single-primitive operator.

### mindspore.mint.sub
- API: `mindspore.mint.sub`
- call_chain_kind: `functional_overload`
- Primitive: keep `primitive=[]`; use `possible_primitives = [SubScalar, SubExt]`
- Support: aggregate per-overload branch instead of treating the wrapper itself as a direct terminal primitive
- Effective aclnn interfaces: `aclnnSub`, `aclnnSubs`
- Grad: all non-deprecated function-overload branches close to explicit bprop, so API-level grad can be promoted to `explicit_bprop`
- Conclusion: overload wrappers must aggregate ACLNN branch by branch, and may also collapse grad when branch conclusions fully agree.

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
- implementation_type: `runtime_utility`
- Primitive mapping: not applicable
- grad/aclnn: `not_applicable`
- Conclusion: this is a runtime utility API and should not be forced into operator mapping.

### View-related flags
- `view_op`: only mark APIs that uniquely close to a single view primitive and are not overload/composite/scenario-dependent wrappers.
- `has_view_op`: mark APIs whose real in-scope primitive set contains at least one view primitive only when the API itself is not already `view_op`.
- Conclusion: `view_op` and `has_view_op` are mutually exclusive; composite APIs such as `mindspore.mint.diff` may carry `has_view_op`, while pure view APIs such as `mindspore.mint.narrow` only carry `view_op`.
