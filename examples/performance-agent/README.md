# Performance-Agent Examples

The examples here are built around real runnable scripts. The current canonical
demo script is
[examples/performance-agent/scripts/train-qwen3-npu.py](scripts/train-qwen3-npu.py).

## Panorama

| Scenario | What `performance-agent` should do | Current status | Example |
| --- | --- | --- | --- |
| Script scan plus obvious bottleneck repair | Scan a runnable script and existing metrics, locate an obvious bottleneck, repair it, and compare the result | Ready to demo | `demos/qwen3-script-scan.md` |
| Explicit profiler collection | When the user explicitly asks for profiler collection, inject profiler code, collect traces, analyze the bottleneck, repair it, and compare the result | Ready to demo | `demos/qwen3-profiler-workflow.md` |
| Multi-dimensional structured diagnosis | Produce ranked bottleneck conclusions from step, communication, memory, input, trace, and hotspot evidence | Supported | Covered inside the profiler demo |
| Single-point optimization in fix mode | Diagnose first, propose one optimization, wait for confirmation, then apply and verify it | Supported | Covered inside both demos |
| Specialized cases for memory-heavy, dataloader stall, or graph compile issues | Cover more bottleneck classes with real cases | Planned | Not completed |

## Completed Demos

- `demos/qwen3-script-scan.md`
  Uses `examples/performance-agent/scripts/train-qwen3-npu.py` with an
  existing runnable baseline, then scans the script and metrics to compare the
  impact of a repair.
- `demos/qwen3-profiler-workflow.md`
  Uses `examples/performance-agent/scripts/train-qwen3-npu.py` when the user
  explicitly asks for profiler
  collection, then follows the full inject, collect, analyze, repair, and
  compare workflow.

## Planned Additions

- `memory-dominant real case`
- `dataloader stall real case`
- `communication-heavy distributed real case`

## Reference Sources

- The formal examples now use the real runtime behavior and real metrics of
  `examples/performance-agent/scripts/train-qwen3-npu.py` as the baseline
