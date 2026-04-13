# Performance-Agent Examples

The examples here are built around real runnable scripts. The current canonical
demo script is
[examples/performance-agent/scripts/train-qwen3-npu.py](scripts/train-qwen3-npu.py).

## 1. Coverage Map

The current v1 example coverage is centered on performance bottlenecks that
appear after the workload already runs successfully. The map below describes the
dominant bottleneck classes the examples currently cover, what signals usually
show up first, and how `performance-agent` analyzes them.

| Bottleneck class | Typical signal | How the agent analyzes it | Expected next move | Status |
| --- | --- | --- | --- | --- |
| host launch or idle gap | existing metrics or trace evidence show low utilization and visible gaps between useful compute slices | inspect existing metrics first, then separate host-side gaps from input stalls and look for launch density issues | propose one host-overhead reduction and compare idle gap or step time after rerun | validated |
| dominant profiler-visible bottleneck | profiler summaries expose step, communication, memory, input, trace, or hotspot imbalance | use structured helper outputs to rank the dominant bottleneck before proposing any change | choose one dominant bottleneck and keep optimization scoped to that class | validated |
| input pipeline stall | device waits before compute begins or input stages dominate the pre-compute gap | compare pre-compute idle time, input stage time, and utilization before changing kernels | keep the first optimization in the input path and validate with shorter idle time | planned |
| graph build or recompilation | compile sections dominate latency or recompilation repeats | separate first-run compile cost from steady-state latency and compare compile count/time | recommend one graph-stabilization change and rerun | planned |
| communication bottleneck | collective time or step tail grows under distributed scale | inspect collective count, collective time, and overlap near backward/update | recommend one communication-focused change first | planned |
| memory pressure | peak memory or memory-heavy operators cap batch size and slow the run | compare peak memory, top memory-heavy stage, and any secondary slowdown signals | recommend one memory-focused change and validate peak memory movement | planned |

## 2. Validated Coverage

These rows are already demonstrated by the merged example material.

| Covered class | Evidence form | Example / Demo | Result |
| --- | --- | --- | --- |
| host launch or idle gap | runnable script scan with existing metrics | `demos/qwen3-script-scan.md` | the agent scans a runnable script, identifies one obvious bottleneck, proposes one repair, and compares before/after results |
| dominant profiler-visible bottleneck | explicit profiler-backed workflow | `demos/qwen3-profiler-workflow.md` | the agent collects profiler data, ranks the dominant bottleneck from structured evidence, proposes one optimization, and compares the result |

Every validated example above maps back to at least one primary row in the
Coverage Map.

## 3. Worked Example

### Problem

A Qwen3 training script already runs, but the user reports that it is too slow
and wants one dominant bottleneck identified before any optimization is applied.

### Map Position

- Bottleneck class: dominant profiler-visible bottleneck
- Typical signal: profiler summaries expose one dominant imbalance across step,
  communication, memory, input, trace, or hotspot views

### Observed Evidence

- the script already runs successfully, so this is not a failure/readiness case
- the user explicitly asks for profiler collection before optimization
- structured summaries are available or can be collected for step breakdown,
  communication, memory pressure, input pipeline, trace gaps, and operator
  hotspots

### What the Agent Does

- confirm that the workload already runs
- collect profiler data only because the user explicitly requested it
- build structured summaries instead of guessing from one raw signal
- rank the dominant bottleneck from profiler evidence
- propose one single-point optimization and wait for confirmation before making
  changes

### Outcome

The agent turns the profiler outputs into one dominant bottleneck judgment,
chooses one targeted optimization, and validates the result with a before/after
comparison.

## 4. Current Boundary

### Currently Strong Coverage

- runnable script scan with existing metrics
- explicit profiler-backed diagnosis when the user asks for collection
- single dominant bottleneck selection before optimization
- one-change-at-a-time fix mode with before/after comparison
- examples grounded in `train-qwen3-npu.py`

### Not Yet Fully Covered

- dedicated validated examples for memory-dominant cases
- dedicated validated examples for dataloader stalls
- dedicated validated examples for communication-heavy distributed runs
- dedicated validated examples for graph compile/recompilation bottlenecks

### Handoff / Boundary Notes

The demos remain important evidence, but they are subordinate to the bottleneck
map in this doc. They show how validated regions of the Coverage Map behave in
practice; they are not the top-level structure.

The current validated region is strongest for:
- script scan with existing metrics
- explicit profiler-driven dominant bottleneck selection
- single-point optimization with before/after comparison

---

## Supporting Demos

- `demos/qwen3-script-scan.md`
  Uses `examples/performance-agent/scripts/train-qwen3-npu.py` with an
  existing runnable baseline, then scans the script and metrics to compare the
  impact of a repair.
- `demos/qwen3-profiler-workflow.md`
  Uses `examples/performance-agent/scripts/train-qwen3-npu.py` when the user
  explicitly asks for profiler collection, then follows the full inject,
  collect, analyze, repair, and compare workflow.

## Planned Additions

- `memory-dominant real case`
- `dataloader stall real case`
- `communication-heavy distributed real case`

## Reference Sources

- The formal examples now use the real runtime behavior and real metrics of
  `examples/performance-agent/scripts/train-qwen3-npu.py` as the baseline
