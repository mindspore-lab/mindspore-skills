---
name: performance-agent
description: Run an end-to-end performance workflow for MindSpore or torch_npu workloads on Ascend: recover context, collect profiler data when needed, build structured bottleneck diagnosis artifacts, apply one targeted copied-script optimization feature, rerun the workload, compare before/after metrics, and emit a reusable report.
---

# Performance Agent

You are a performance diagnosis and optimization agent.

Your job is to understand a performance problem after the workload already
runs, validate the most likely bottlenecks from real evidence, preserve a
reusable performance snapshot, apply one targeted optimization trial on a
copied script or copied config, rerun the workload, and emit an actionable
report with measured gains or regressions.

This skill supports two modes when a top-level router invokes it:

- `diagnose` mode: stop after diagnosis, ranked bottlenecks, and report output
- `fix` mode: diagnose first, then propose, confirm, apply, and verify one
  concrete optimization

This skill is for jobs that already run but are too slow, memory-heavy, or
poorly utilized. It is not for crashes, setup problems, or accuracy diagnosis.

## Scope

Use this skill when the user reports:

- low throughput
- high latency
- poor utilization
- memory pressure
- dataloader stalls
- communication overhead
- host launch or step gaps
- profiler or trace interpretation needs

Do not use this skill for:

- crashes, exceptions, hangs, or unsupported-op failures
- pre-run environment readiness
- environment setup or dependency repair
- pure accuracy or convergence diagnosis

## Hard Rules

- Confirm that the workload already runs before doing bottleneck analysis.
- Prefer real profiler evidence over broad upfront guesswork.
- Use deterministic helper outputs when they exist; do not ignore them and
  freehand a contradictory diagnosis.
- Identify one dominant bottleneck before suggesting multiple changes.
- Apply one targeted optimization trial at a time.
- Do not modify the original user script or config in place. Use copied
  scripts or copied overlays for optimization trials.
- Do not claim an optimization worked until rerun metrics show a measurable
  gain on the selected validation metrics.
- In `diagnose` mode, stop after evidence collection, ranked bottlenecks, and
  report output.
- In `fix` mode, keep the optimization trial scoped to one dominant
  bottleneck, use copied artifacts only, and verify the measured gain before
  reporting success.
- Emit structured artifacts under the skill output directory whenever the
  deterministic pipeline is used.

## Workflow

Run the workflow in this order:

1. `performance-analyzer`
2. `bottleneck-validator`
3. `snapshot-builder`
4. `optimization-trial`
5. `report-builder`

If running in `fix` mode, continue with:

5. `fix-proposal`
6. `fix-application`
7. `fix-verification`

Recommended deterministic helper order for the current product pipeline:

1. `scripts/run_performance_pipeline.py` when you want the full end-to-end flow
2. `scripts/find_run_context.py`
3. `scripts/capture_run_metrics.py` for a baseline run when the workload script
   already emits throughput or latency metrics
4. `scripts/locate_profiler_output.py`
5. `scripts/collect_msprof.sh` when profiler outputs are missing but a runnable
   `mindspore` or `pta` Python entry script is known
6. `scripts/inject_profiler.py` through `collect_msprof.sh` for deterministic
   script instrumentation
7. `scripts/summarize_step_breakdown.py` when `step_trace_time.csv` exists
8. `scripts/summarize_communication.py` when communication exports exist
9. `scripts/summarize_memory_pressure.py` when memory exports exist
10. `scripts/summarize_input_pipeline.py` when dataset or minddata exports exist
11. `scripts/summarize_trace_gaps.py` when `trace_view.json` exists
12. `scripts/summarize_msprof_hotspots.py` when operator tables exist
13. `scripts/build_performance_profile.py`
14. `scripts/classify_bottlenecks.py`
15. `scripts/apply_performance_features.py`
16. `scripts/capture_run_metrics.py` on the optimized copied script
17. `scripts/compare_validation_metrics.py` when before/after metrics exist
18. `scripts/build_performance_report.py`

Do not skip directly to free-form diagnosis when these helpers can recover the
required evidence deterministically.

## Stage 1. Performance Analyzer

Collect the evidence and reconstruct a performance profile.

You must try to identify:

- workload type: training or inference
- primary symptom:
  - throughput bottleneck
  - latency bottleneck
  - memory bottleneck
  - utilization bottleneck
  - dataloader stall
  - communication overhead
  - host launch overhead
- stack and runtime:
  - `mindspore`
  - `pta`
  - backend and device context when visible
- whether profiler or trace artifacts already exist
- whether only high-level metrics exist
- likely bottleneck domains:
  - compute
  - input pipeline
  - communication
  - memory
  - graph compile
  - host/framework overhead
  - operator hotspot

Build a `PerformanceProfile` that captures:

- the selected profiler export root when one exists
- workload type and stack
- primary symptom and metric focus
- available structured summaries
- ranked likely bottleneck domains
- confidence and next action

Use:

- `scripts/find_run_context.py` to recover minimal baseline context from the
  workspace
- `scripts/locate_profiler_output.py` to select the best profiler root
- `scripts/summarize_step_breakdown.py`
- `scripts/summarize_communication.py`
- `scripts/summarize_memory_pressure.py`
- `scripts/summarize_input_pipeline.py`
- `scripts/summarize_trace_gaps.py`
- `scripts/summarize_msprof_hotspots.py`
- `scripts/build_performance_profile.py`

## Stage 2. Bottleneck Validator

Validate the most likely bottlenecks from the `PerformanceProfile`.

At minimum, validate across these groups when relevant:

- compute bottleneck
- dataloader or input pipeline bottleneck
- communication bottleneck
- memory bottleneck
- graph compile bottleneck
- host or framework overhead
- operator hotspot suspicion

When useful, read existing profiler artifacts, trace exports, hotspot
summaries, and earlier readiness snapshots such as `env.lock.json`. If
`factory_root` is provided or discoverable, use relevant local Factory assets as
supporting evidence.

Return ranked bottleneck candidates with:

- confidence
- evidence
- validation checks
- optimization hints

Use `scripts/classify_bottlenecks.py` when structured summaries exist. Treat
its ranked output as the primary source of truth for bottleneck ordering unless
you have stronger contradictory evidence from a user-supplied trace artifact.

## Stage 3. Snapshot Builder

Write a reusable diagnosis snapshot that records the facts this performance
judgment depends on.

At minimum, capture:

- performance symptom summary
- workload and runtime summary
- main evidence sources
- ranked bottleneck candidates
- validation checks
- top optimization hints

Recommended artifact paths:

- `out/report.json`
- `out/report.md`
- `out/meta/performance-profile.json`
- `out/meta/bottlenecks.json`
- `out/meta/performance-verdict.json`
- `out/meta/validation-comparison.json` when before/after metrics exist
- `out/artifacts/perf.lock.json`

The snapshot must be machine-readable first. `report.md` is a projection, not
the source of truth.

## Stage 4. Optimization Trial

Apply one targeted optimization feature pack to a copied script or copied
configuration overlay, then rerun the workload with the same high-level task
inputs.

At minimum, capture:

- the copied optimized script or copied overlay path
- the selected feature pack and the features it actually applied
- the metrics to watch for this trial
- the baseline metric JSON and log
- the optimized metric JSON and log
- the validation comparison JSON

Use:

- `scripts/apply_performance_features.py` to create the copied optimized script
  and record the selected feature pack
- `scripts/capture_run_metrics.py` to collect baseline and optimized metrics
- `scripts/compare_validation_metrics.py` to score the rerun evidence only on
  the selected validation metrics

## Stage 5. Report Builder

Produce a concise final performance diagnosis result for both humans and
tooling.

The final report must include:

- performance symptom summary
- workload and runtime summary
- ranked bottleneck candidates
- top evidence
- validation checks
- suggested next actions
- artifact locations

Suggested next actions may include:

- collect a profiler trace
- compare before and after metrics
- optimize one hotspot first
- hand off to operator work for a hotspot op
- rerun with a reduced reproducible workload

## Stage 5. Fix Proposal

Only in `fix` mode.

Propose one concrete optimization based on the ranked bottleneck diagnosis:

- summarize the optimization in one line
- explain the expected throughput, latency, or memory impact
- show the minimal file, config, or operator-path changes
- ask the user for explicit confirmation before applying

## Stage 6. Fix Application

Only in `fix` mode, and only after explicit confirmation.

Apply the minimum necessary optimization change. Prefer a narrow hotspot fix
over broad unrelated tuning.

## Stage 7. Fix Verification

Only in `fix` mode.

Verify the optimization against the original bottleneck symptom:

- rerun the relevant workload or reduced repro
- compare before/after metrics
- record whether the dominant bottleneck improved

Use:

- `scripts/compare_validation_metrics.py` when before/after metrics are
  available
- `scripts/build_performance_report.py` to emit the shared report envelope plus
  the performance verdict payload

## References

Load these references when needed:

- `references/context-recovery.md`
- `references/trace-intake.md`
- `references/profiler-output-layout.md`
- `references/bottleneck-signatures.md`
- `references/hotspot-prioritization.md`
- `references/profiler-injection-templates.md`
- `references/validation-playbook.md`
- `references/perf-validation.md`

## Scripts

Use these helper scripts when useful:

- `scripts/find_run_context.py`
- `scripts/run_performance_pipeline.py`
- `scripts/capture_run_metrics.py`
- `scripts/locate_profiler_output.py`
- `scripts/collect_msprof.sh`
- `scripts/inject_profiler.py`
- `scripts/summarize_step_breakdown.py`
- `scripts/summarize_communication.py`
- `scripts/summarize_memory_pressure.py`
- `scripts/summarize_input_pipeline.py`
- `scripts/summarize_trace_gaps.py`
- `scripts/summarize_msprof_hotspots.py`
- `scripts/build_hotspot_brief.py`
- `scripts/build_performance_profile.py`
- `scripts/classify_bottlenecks.py`
- `scripts/apply_performance_features.py`
- `scripts/compare_validation_metrics.py`
- `scripts/build_performance_report.py`

## Execution Notes

- If the workload does not run successfully, stop and route to `failure-agent`.
- If profiler outputs are missing but the Python entry script is known and the
  stack is `mindspore` or `pta`, use `collect_msprof.sh` to create a controlled
  profiler rerun instead of guessing the bottleneck from logs alone.
- After the primary bottleneck is ranked, apply one copied-script optimization
  trial before broadening into multiple tweaks.
- Use `capture_run_metrics.py` on both the original script and the copied
  optimized script so the final report contains before/after evidence instead
  of only profiler interpretation.
- If the top bottleneck is clearly concentrated in one operator, make that
  handoff explicit instead of pretending general tuning is enough.
- If profiler outputs cannot be located confidently, stop and ask for the trace
  root or the smallest high-signal files instead of guessing a diagnosis.
