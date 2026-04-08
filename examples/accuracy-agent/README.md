# Accuracy Agent — Examples

## Scenario Panorama

The accuracy-agent is designed to diagnose and fix accuracy problems across a
range of model training and inference scenarios on Ascend NPU. The tables below
show the full landscape of planned scenarios and their current support status.

### Overview

| Dimension | Supported | Planned |
| --- | --- | --- |
| Hardware topology | Single-card Ascend | Multi-card, multi-node |
| Framework pairing | torch_npu vs MindSpore | GPU PyTorch vs Ascend, same-framework diffs |
| Task type | Inference (eager) | Training, graph mode |
| Distributed framework | — | DDP, Hyper-Parallel, MindSpeed, etc. |
| Precision | FP32 / FP16 / BF16 | Mixed precision (AMP), quantization |

The target hardware is always Ascend NPU. The baseline may run on GPU or Ascend
depending on the framework pairing.

### By Hardware Topology

| Topology | Status |
| --- | --- |
| Single-device single-card on Ascend | Supported |
| Single-machine multi-card on Ascend | Planned |
| Multi-machine multi-card on Ascend | Planned |

### By Framework Pairing

**Primary focus:**

| Framework Pairing | Status |
| --- | --- |
| torch_npu vs MindSpore (both on Ascend) | Supported |
| PyTorch (GPU) vs MindSpore (Ascend) | Planned |
| PyTorch (GPU) vs torch_npu (Ascend) | Planned |

**Secondary (same-framework comparisons):**

| Framework Pairing | Status |
| --- | --- |
| MindSpore vs MindSpore (version or config diff) | Planned |
| torch_npu vs torch_npu (cross-version) | Planned |

### By Task Type

| Task Type | Status |
| --- | --- |
| Inference (eager mode) | Supported |
| Training (forward + backward + optimizer) | Planned |
| Graph mode vs eager mode | Planned |

### By Distributed Framework

| Distributed Framework | Status |
| --- | --- |
| Native distributed (DDP / parallel strategies) | Planned |
| Hyper-Parallel, MindSpeed, etc. | Planned |

### By Precision Context

FP32, FP16, and BF16 dtypes are supported. Mixed precision (AMP) scenarios and
quantization (INT8 / INT4) are planned for future support.

### Additional Scenarios (Future)

- Checkpoint conversion / migration accuracy
- Long-term training stability / convergence drift
- Operator fusion accuracy impacts
- Cross-framework-version accuracy regression

---

## Currently Supported Scenario

**Single-device single-card Ascend, torch_npu vs MindSpore, eager-mode
inference, zero-deviation alignment.**

The torch_npu model script serves as the accuracy baseline. The goal is to
locate and fix accuracy issues in the MindSpore model script until outputs match
with zero or machine-epsilon-level deviation.

### What It Can Diagnose

- Cross-framework API specification or default parameter misalignment
- Kernel path differences between legacy MindSpore operators and torch_npu
  operators
- Computation device inconsistency during model initialization

Examples: LayerNorm eps default differs between torch and MindSpore; RoPE
`inv_freq` computed on CPU vs NPU due to different default device policies.

### Diagnosis, Fix, and Expected Outcome

The agent uses layer-by-layer structured tensor comparison to locate the first
divergence point, then narrows down to the responsible module or operator.

Fixes include aligning API parameters, replacing legacy `mindspore.nn` /
`mindspore.ops` operators with `mindspore.mint` equivalents, and aligning
init-computed state across frameworks.

The expected outcome is zero-deviation or machine-epsilon-level alignment
between torch_npu and MindSpore outputs.

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

Result: accuracy-agent compares results, traces the numerical drift to its
source, and applies the fix automatically.

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
