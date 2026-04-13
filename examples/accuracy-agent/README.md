# Accuracy Agent — Examples

## 1. Coverage Map

The current v1 example coverage is centered on accuracy problems that appear
after the workload already runs successfully. The map below describes what kinds
of mismatches the doc currently covers, what users usually see first, and what
`accuracy-agent` checks next.

| Class | Item | Typical symptom | What the agent checks next | Status |
| --- | --- | --- | --- | --- |
| configuration or input alignment | step1 loss mismatch | loss diverges at the first meaningful step even though the run completes | re-check aligned weights, inputs, preprocessing, tokenizer, masks, and labels before narrowing to modules | validated |
| module or operator semantics | forward output mismatch | final output or an early module output differs between baseline and target on the same input | use structured tensor comparison to find the first stable mismatch, then verify module inputs, parameters, dtype, API parameters, and device placement | validated |
| backward or update consistency | later divergence after a matching start | step1 matches, but gradients, updates, or later losses drift | compare gradients, one-step updates, loss scale, grad clipping, and distributed reduction behavior | planned |
| numerical stability | non-fatal NaN or Inf | the job finishes or remains comparable, but invalid values appear during the run | find the first invalid step or module, then inspect overflow, ranges, and unstable precision paths | planned |
| evaluation or metric consistency | eval-only or cross-platform regression | training/inference finishes, but final outputs or metrics differ across platforms or versions | compare golden outputs first, then metric/postprocessing definitions and the earliest internal mismatch that affects them | planned |
| no trusted baseline | cannot name a clean reference side | the user sees a possible accuracy problem but has no baseline they trust | reduce to a minimal golden case and build the smallest meaningful confidence signal before making root-cause claims | planned |

## 2. Validated Coverage

These rows are already demonstrated by the merged example material and the
current supported demo path.

| Covered class | Covered item | Evidence form | Example / Demo | Result |
| --- | --- | --- | --- | --- |
| configuration or input alignment | step1 loss mismatch | supported comparison workflow | `/diagnose step1 loss mismatch between torch_npu baseline and mindspore target, check train_log.txt` | the doc already shows this as a supported diagnose path |
| module or operator semantics | forward output mismatch | worked cross-framework example | zero-deviation torch_npu vs MindSpore inference alignment demo | the doc shows layer-by-layer comparison leading to API/default alignment or `mindspore.mint` replacement |

Every validated example above maps back to at least one primary row in the
Coverage Map.

## 3. Worked Example

### Problem

Torch NPU and MindSpore inference scripts for the same model both run on Ascend,
but the outputs do not align. The expected result is zero-deviation or
machine-epsilon-level alignment.

### Map Position

- Class: module or operator semantics
- Item: forward output mismatch

### Observed Evidence

- the baseline is `torch_npu` on Ascend and the target is MindSpore on Ascend
- the run succeeds on both sides, so this is not a failure/readiness problem
- the first user-visible symptom is an output mismatch rather than a crash
- likely causes include API default mismatch, legacy operator path mismatch, or
  init-time device inconsistency

### What the Agent Does

- keep the baseline fixed and treat it as the source of truth
- use layer-by-layer structured tensor comparison to locate the first stable
  divergence point
- verify module inputs before narrowing inside a mismatching module
- if the mismatch stabilizes at one operator, align API parameters first and
  prefer a direct `mindspore.mint` replacement before considering deeper rewrite

### Outcome

The agent narrows the drift to the first meaningful mismatch and applies a small
alignment fix on the MindSpore side until outputs match the baseline.

### Demo

<img src="../../docs/assets/accuracy_agent.gif" width="720" />

User prompt (CN):

> 执行 run_llm_infer.sh，运行同一个模型的 torch_npu 和 mindspore 版本推理脚本，检查输出结果是否有精度误差。
> - 预期精度应该是绝对的0偏差对齐，不允许有微小误差。
> - /accuracy-agent 定位并修复这个问题。

User prompt (EN):

> Run run_llm_infer.sh to execute torch_npu and mindspore inference scripts for
> the same model and check whether the outputs have precision errors.
> - Expected precision should be absolute zero-deviation alignment, no minor
>   errors allowed.
> - /accuracy-agent locate and fix this issue.

## 4. Current Boundary

### Currently Strong Coverage

- single-device single-card Ascend
- `torch_npu` vs MindSpore on Ascend
- inference / eager-mode alignment
- zero-deviation or machine-epsilon-level output comparison
- diagnosis paths centered on configuration/input alignment, forward mismatch,
  and output-level regression

### Not Yet Fully Covered

- training-time backward or optimizer mismatch flows
- graph mode vs eager mode comparison as a validated example region
- multi-card and multi-node accuracy analysis
- mixed precision (AMP) and quantization-heavy accuracy cases
- same-framework version/config regression examples with validated demos

### Handoff / Boundary Notes

Support dimensions such as topology, framework pairing, task type, distributed
framework, and precision context are boundary/support context for this doc.
They are not the Coverage Map itself.

The current supported region is:
- single-card Ascend
- `torch_npu` vs MindSpore
- eager-mode inference

Other scenarios remain planned expansion areas rather than validated map rows.

---

## How to Use

### Prerequisites

You need a baseline script (accuracy reference) and a target script (with
precision issues), in an environment where both can run and reproduce the
problem. No strict workspace layout is required.

### Modes and Commands

The accuracy-agent supports two modes:

- **diagnose** — diagnosis only, no code changes
- **fix** — diagnose first, then propose, confirm, and apply a fix

Use `/diagnose` or `/fix` in any supported CLI environment (mindspore-cli,
Claude Code, OpenCode, Gemini CLI, Codex, etc). See the main
[README](../../README.md) for installation instructions.

Describe the accuracy problem and specify which script is the baseline and
which is the target. Point to relevant logs or output files if available.

The agent produces a diagnosis report with root-cause analysis. In fix mode, it
also proposes and applies a concrete fix after user confirmation.

### Example Prompts

```text
/fix run run_llm_infer.sh, torch_npu and mindspore inference outputs have precision errors, expected zero-deviation alignment
```

```text
/diagnose step1 loss mismatch between torch_npu baseline and mindspore target, check train_log.txt
```
