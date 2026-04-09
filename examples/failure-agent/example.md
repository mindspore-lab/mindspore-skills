# Failure-Agent Example

This example is for users who want two things quickly:

1. the key full picture of failures when running a script or model
2. the failure items that `failure-agent` has proved useful at diagnosing and fixing

## 1. What `failure-agent` does

`failure-agent` is for post-failure diagnosis.

It helps you:

- identify the failure stage
- classify the failure type
- rank likely root causes from evidence
- suggest the next low-risk checks or fix moves

## 2. Full Picture Of Failures

Use these as the main buckets.

| Failure class | What users usually see | What `failure-agent` validates next |
| --- | --- | --- |
| Startup failure | import error, missing libs, bad environment, invalid backend/device setup, version mismatch, missing runtime variables such as CANN env vars | environment/runtime mismatch, library incompatibility, config incompatibility |
| Compile or graph-build failure | graph compile failure, infer failure, abstract/type/shape errors, unsupported graph constructs, operator shape or layout mismatch | config incompatibility, dataset/input issue, operator-related issue |
| Runtime execution failure | crash or runtime exception after launch, dtype mismatch, bad embedding index dtype, backward API misuse, invalid tensor layout during execution | backend/runtime issue, operator-related issue, config incompatibility |
| Checkpoint load or save failure | checkpoint load mismatch, missing keys, unexpected keys, corrupt file, save failure, resume failure | checkpoint/resume issue, model asset issue |
| Evaluation failure | training runs, but eval path fails, metric path failure, incompatible eval inputs, bad model assets on the eval path | dataset/input issue, config incompatibility, model asset issue |
| OOM | out-of-memory, allocation failure, device memory exhaustion, fragmentation, oversized batch size or model input | model/input size pressure, config incompatibility, backend/runtime issue |
| Timeout or hang | no progress, dead wait, distributed stall, blocked collective, one-rank upstream failure masked as timeout | communication/timeout issue, environment/runtime mismatch, upstream rank failure |
| Backend or communication failure | HCCL/NCCL/CANN/ACLNN/backend error codes, low-level runtime faults, unsupported backend path, backend operator failure | backend/runtime issue, communication/timeout issue, library incompatibility |

## 3. Failure Items Proven Useful In Practice

These are concrete failure items from local examples that `failure-agent` can classify and route well.

| Failure class | Failure item | What it usually means | First fix / validation move |
| --- | --- | --- | --- |
| compile or graph-build failure | MindSpore matmul dtype mismatch | Model parameters and input tensors use incompatible dtypes, such as `float32` weights with default NumPy `float64` input | Print the input and parameter dtypes, cast them to the same supported dtype, and rerun the same operator |
| compile or graph-build failure | Attention value-state transpose/layout bug | Query/key/value tensors do not share the same layout, so attention matmul or later reshape steps fail or produce invalid output shapes | Print the q/k/v tensor shapes, align the transpose order across them, and rerun the minimal attention block |
| runtime execution failure | MindSpore backward API misuse | The code assumes PyTorch-style `.backward()` semantics, but MindSpore requires the gradient path to be built with `ops.GradOperation` or the proper training wrapper | Replace the backward call with the MindSpore gradient API and verify gradients on a minimal forward pass |
| runtime execution failure | Embedding index cast to float | Token ids are converted to floating point before `nn.Embedding`, but embeddings expect integer indices such as `LongTensor` | Keep `input_ids` as integer indices, remove the float cast, and rerun the embedding path |

## 4. Worked By `failure-agent`

### Example: Broken Qwen3 Model Script Diagnosis

**Problem**

A broken Qwen3 model script fails because the model code contains invalid tensor typing and attention-layout logic.

- **Mode:** `fix`
- **Example prompt:** `/fix "according to the error, fix the issue"`

**Observed evidence**

- failing artifact is a broken Qwen3 model script
- the embedding path calls `nn.Embedding` with `input_ids.float()`
- the attention path uses a different transpose order for value than for query and key
- these patterns can cause an early runtime failure or a later downstream training failure

**What `failure-agent` does on this case**

- inspect the failing model code and visible failure evidence
- map the issue to the right failure class
- rank the likely root causes
- suggest low-risk checks before bigger changes
- propose one concrete fix before applying it

**Ranked root-cause candidates**

| Rank | Candidate | Failure class | Why it is likely | First check |
| --- | --- | --- | --- | --- |
| 1 | Embedding index cast to float | `runtime execution failure` | The model calls `nn.Embedding` with `input_ids.float()`, but embedding indices should stay integer typed | Check the embedding input dtype and remove the float cast |
| 2 | Attention value-state transpose/layout bug | `compile or graph-build failure` | Query and key use one layout while value uses a different transpose order, which can break attention matmul or reshape logic | Print q/k/v shapes and align the transpose order |
| 3 | Secondary training failure caused by an earlier forward bug | `runtime execution failure` | A wrong dtype or tensor layout in the forward path can cause a later training failure, even if the later error is only a downstream symptom | Re-run the minimal forward path first and identify the earliest failing operator |

**Result**

`failure-agent` can diagnose the likely root cause, propose a targeted fix, and verify the failing path after the fix is applied.

For deeper details, see:

- `skills/failure-agent/reference/failure-taxonomy.md`
- `skills/failure-agent/reference/failure-showcase.md`
- `skills/failure-agent/reference/root-cause-validation.md`
