# Demo: Qwen3 Script Scan And Compare

## Goal

Demonstrate the first real `performance-agent` path:

- the script already runs
- the agent scans the script and existing metrics
- the agent identifies an obvious bottleneck
- after user confirmation, the agent makes one minimal repair
- the script is rerun and the before/after result is compared

Current fixed script:

- [examples/performance-agent/scripts/train-qwen3-npu.py](../scripts/train-qwen3-npu.py)

## Prerequisites

- `examples/performance-agent/scripts/train-qwen3-npu.py` already runs
  successfully in the target environment
- the script has completed at least one successful run
- `qwen3-finetuned/metrics.json` already exists, or the user has at least one
  preserved run log
- the user already has a working CANN / NPU / PTA runtime environment

## Recommended Prompt

```text
/fix Analyze the performance bottleneck of C:\workspace\mindspore-skills\examples\performance-agent\scripts\train-qwen3-npu.py. Prefer the existing run results and metrics.json. If you can identify one obvious bottleneck, propose a single optimization first, wait for my confirmation, then apply it and compare the before/after gain.
```

For a diagnose-only version, use:

```text
/diagnose Optimize Qwen3 performance, analyze the bottleneck, and compare the optimization gain.
```

## Expected Agent Actions

1. Confirm that the script already runs, so a crash is not misrouted as a
   performance problem.
2. Scan `examples/performance-agent/scripts/train-qwen3-npu.py`.
3. Read existing high-signal artifacts such as
   `qwen3-finetuned/metrics.json`, logs, or equivalent outputs.
4. Present one dominant bottleneck instead of many speculative changes.
5. Explain the expected gain and the minimal change scope.
6. Wait for user confirmation.
7. Modify the script or a narrow configuration area.
8. Rerun the script.
9. Compare the before/after metrics.

## Expected Artifacts

- a performance diagnosis conclusion
- one clearly identified dominant bottleneck
- the modified `examples/performance-agent/scripts/train-qwen3-npu.py` or
  another minimal related change
- a before/after comparison that includes at least one core metric, such as:
  - `train_steps_per_second`
  - step time
  - another training speed metric already recorded by the script

## Success Criteria

This demo is successful when:

- the agent does not guess multiple bottlenecks when evidence is weak
- the agent drives only one main optimization point
- the rerun produces a before/after comparison
- if the gain is not obvious, the agent reports that honestly instead of
  forcing a success claim

## Boundary Notes

- This demo does not require profiler collection.
- If the existing metrics are not strong enough for a reliable diagnosis, the
  agent should explicitly recommend switching to the profiler demo instead of
  continuing to guess.
