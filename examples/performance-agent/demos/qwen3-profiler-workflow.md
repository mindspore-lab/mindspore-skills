# Demo: Qwen3 Profiler Workflow

## Goal

Demonstrate the second real `performance-agent` path:

- the user explicitly asks for profiler collection
- the agent injects profiler-related code into the script
- the script is rerun and profiler data is collected
- the agent identifies the dominant bottleneck from profiler evidence
- after user confirmation, the agent makes one minimal repair
- the script is rerun and the result is compared

Current fixed script:

- [examples/performance-agent/scripts/train-qwen3-npu.py](../scripts/train-qwen3-npu.py)

## Prerequisites

- `examples/performance-agent/scripts/train-qwen3-npu.py` already runs
  successfully
- the user allows an additional run for profiler collection
- the target environment supports the required profiler workflow
- the user already has a working CANN / NPU / PTA environment

## Recommended Prompt

```text
/fix Analyze the performance bottleneck of C:\workspace\mindspore-skills\examples\performance-agent\scripts\train-qwen3-npu.py. This time, explicitly collect profiler data first, then use the profiler result to identify one dominant bottleneck and propose one optimization. Wait for my confirmation before applying the change, then compare the before/after gain.
```

For a diagnose-only version, use:

```text
/diagnose Optimize Qwen3 performance, collect profiler data, and compare the optimization gain.
```

## Expected Agent Actions

1. Confirm that the script already runs.
2. Use `find_run_context.py` to recover the minimal run context.
3. If no profiler output already exists, inject profiler code into the script.
4. Rerun the script and collect profiler data.
5. Use helper scripts to produce structured summaries for:
   - step breakdown
   - communication
   - memory pressure
   - input pipeline
   - trace gaps
   - operator hotspots
6. Rank the dominant bottleneck based on those structured results.
7. Propose one single-point optimization.
8. Wait for user confirmation.
9. Modify the script or a narrow configuration area.
10. Rerun and compare the before/after gain.

## Expected Artifacts

- a profiler output directory
- structured summary artifacts
- `report.json`
- `report.md`
- `meta/performance-profile.json`
- `meta/bottlenecks.json`
- `meta/performance-verdict.json`
- a validation comparison result when before/after comparison is available

## Success Criteria

This demo is successful when:

- profiler collection happens only because the user explicitly requested it
- the dominant bottleneck conclusion comes from profiler evidence rather than
  free-form guesswork
- the agent presents one optimization and waits for confirmation before
  modifying anything
- the final result includes a before/after metric comparison

## Boundary Notes

- If the script does not run at all, stop and route to `failure-agent`.
- If the profiler data is still too weak to support a conclusion, the agent
  should ask for stronger trace evidence instead of continuing to guess.
