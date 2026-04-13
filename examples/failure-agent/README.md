# Failure-Agent Examples

This example is for users who want two things quickly:

1. the key full picture of failures when running a script or model
2. the failure items that `failure-agent` has already shown it can diagnose and route

## 1. Coverage Map

Use these as the main buckets. This keeps the current failure classification
shape, but makes the map and maturity explicit.

| Failure class | Item | What users usually see | What `failure-agent` checks next | Status |
| --- | --- | --- | --- | --- |
| startup failure | environment or library startup failure | import error, missing libs, bad environment, invalid backend/device setup, version mismatch, missing runtime variables such as CANN env vars | environment/runtime mismatch, library incompatibility, config incompatibility | planned |
| compile or graph-build failure | MindSpore matmul dtype mismatch | model parameters and input tensors use incompatible dtypes, such as `float32` weights with default NumPy `float64` input | print the input and parameter dtypes, cast them to the same supported dtype, and rerun the same operator | validated |
| compile or graph-build failure | attention value-state transpose/layout bug | query/key/value tensors do not share the same layout, so attention matmul or later reshape steps fail or produce invalid output shapes | print the q/k/v tensor shapes, align the transpose order across them, and rerun the minimal attention block | validated |
| runtime execution failure | MindSpore backward API misuse | the code assumes PyTorch-style `.backward()` semantics, but MindSpore requires the gradient path to be built with `ops.GradOperation` or the proper training wrapper | replace the backward call with the MindSpore gradient API and verify gradients on a minimal forward pass | validated |
| runtime execution failure | embedding index cast to float | token ids are converted to floating point before `nn.Embedding`, but embeddings expect integer indices such as `LongTensor` | keep `input_ids` as integer indices, remove the float cast, and rerun the embedding path | validated |
| checkpoint load or save failure | checkpoint mismatch or resume failure | checkpoint load mismatch, missing keys, unexpected keys, corrupt file, save failure, or broken resume behavior | checkpoint/resume issue, model asset issue | planned |
| evaluation failure | eval-path failure | training runs, but eval path fails, metric path fails, eval inputs are incompatible, or model assets are wrong on the eval path | dataset/input issue, config incompatibility, model asset issue | planned |
| oom | memory exhaustion | out-of-memory, allocation failure, device memory exhaustion, fragmentation, or oversized batch/model input | model/input size pressure, config incompatibility, backend/runtime issue | planned |
| timeout or hang | dead wait or distributed stall | no progress, dead wait, blocked collective, or one-rank upstream failure masked as timeout | communication/timeout issue, environment/runtime mismatch, upstream rank failure | planned |
| backend or communication failure | backend runtime or collective failure | HCCL/NCCL/CANN/ACLNN/backend error codes, low-level runtime faults, unsupported backend path, or backend operator failure | backend/runtime issue, communication/timeout issue, library incompatibility | planned |

## 2. Validated Coverage

These rows are directly evidenced by the current worked example and concrete local
failure items.

| Covered class | Covered item | Evidence form | Example / Demo | Result |
| --- | --- | --- | --- | --- |
| compile or graph-build failure | MindSpore matmul dtype mismatch | concrete local failure item | local dtype-mismatch example in this example set | the agent can classify the dtype mismatch, point to the failing operator boundary, and validate the fix by rerunning with aligned dtypes |
| compile or graph-build failure | attention value-state transpose/layout bug | worked Qwen3 diagnosis evidence | broken Qwen3 model script diagnosis | the agent can tie the failure to the mismatched q/k/v layout and route the next fix to a minimal attention-block check |
| runtime execution failure | MindSpore backward API misuse | concrete local failure item | local backward-misuse example in this example set | the agent can identify the PyTorch-style backward assumption and redirect the fix toward the MindSpore gradient path |
| runtime execution failure | embedding index cast to float | worked Qwen3 diagnosis evidence | broken Qwen3 model script diagnosis | the agent can identify the float-cast embedding misuse, rank it as the top root-cause candidate, and validate the fix on the embedding path |

Every validated example above maps back to at least one primary row in the
Coverage Map.

## 3. Worked Example

### Problem

A broken Qwen3 model script fails because the model code contains invalid tensor
typing and attention-layout logic.

- **Mode:** `fix`
- **Example prompt:** `/fix "according to the error, fix the issue"`

### Map Position

- Failure class: runtime execution failure
- Item: embedding index cast to float

Secondary mapped row:
- Failure class: compile or graph-build failure
- Item: attention value-state transpose/layout bug

### Observed Evidence

- failing artifact is a broken Qwen3 model script
- the embedding path calls `nn.Embedding` with `input_ids.float()`
- the attention path uses a different transpose order for value than for query
  and key
- these patterns can cause an early runtime failure or a later downstream
  training failure

### What `failure-agent` Does

- inspect the failing model code and visible failure evidence
- map the issue to the right failure class
- rank the likely root causes
- suggest low-risk checks before bigger changes
- propose one concrete fix before applying it

### Ranked Root-Cause Candidates

| Rank | Candidate | Failure class | Why it is likely | First check |
| --- | --- | --- | --- | --- |
| 1 | embedding index cast to float | `runtime execution failure` | the model calls `nn.Embedding` with `input_ids.float()`, but embedding indices should stay integer typed | check the embedding input dtype and remove the float cast |
| 2 | attention value-state transpose/layout bug | `compile or graph-build failure` | query and key use one layout while value uses a different transpose order, which can break attention matmul or reshape logic | print q/k/v shapes and align the transpose order |
| 3 | secondary training failure caused by an earlier forward bug | `runtime execution failure` | a wrong dtype or tensor layout in the forward path can cause a later training failure, even if the later error is only a downstream symptom | re-run the minimal forward path first and identify the earliest failing operator |

### Outcome

`failure-agent` can diagnose the likely root cause, propose a targeted fix, and
verify the failing path after the fix is applied.

## 4. Current Boundary

### Currently Strong Coverage

- concrete routing for local dtype/layout/runtime misuse examples
- worked-example diagnosis that traces back to explicit map rows
- ranked root-cause output tied to visible evidence
- low-risk validation moves before larger changes

### Not Yet Fully Covered

- startup failures are present as taxonomy coverage, but not yet shown through a
  worked example in this doc
- checkpoint load/save failures are not yet backed by a worked example here
- evaluation failures are not yet backed by a worked example here
- oom, timeout/hang, and backend/communication failures are represented in the
  map, but not yet validated by a concrete example in this doc

### Handoff / Boundary Notes

The map preserves the current failure classification direction from the merged
doc and reference taxonomy. The validated rows are limited to concrete items
that are directly evidenced by the current local example material or the worked
Qwen3 diagnosis. The broader failure classes stay visible in the Coverage Map,
but remain planned until this example set includes direct evidence for them.

---

## Reference Sources

- `skills/failure-agent/reference/failure-taxonomy.md`
- `skills/failure-agent/reference/failure-showcase.md`
- `skills/failure-agent/reference/root-cause-validation.md`
