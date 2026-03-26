---
description: Run an end-to-end performance workflow for MindSpore or torch_npu workloads on Ascend: collect profiler data when needed, build structured diagnosis artifacts, apply one copied optimization feature, rerun, and validate the measured gain
---

# Performance Agent

Direct specialist entry for performance bottlenecks in workloads that already
run successfully but are too slow, memory-heavy, or poorly utilized across
MindSpore and torch_npu.

For most users, prefer:

- `/diagnose <problem>` to auto-route into `performance-agent` in diagnose mode
- `/fix <problem>` to auto-route into `performance-agent` in fix mode

Use `/performance-agent` only when you already know the problem is a
performance case and want to force the specialist directly.

Load the `performance-agent` skill and follow its workflow in either:

- `diagnose` mode for evidence, root cause, and report only
- `fix` mode for diagnose, one copied optimization trial, rerun, and verify

The product pipeline prefers structured profiler summaries and reusable report
artifacts over free-form diagnosis when the required evidence is available. In
`fix` mode it also applies one copied optimization trial, reruns the workload,
and validates the measured gain instead of stopping at diagnosis only.

## Typical Inputs

- runtime context and symptom description
- profiler trace root or exported profiler directory if available
- throughput, latency, memory, utilization, or communication symptoms
- earlier readiness snapshot if available
- optional before/after metric JSON for validation comparison
- optional output directory for structured artifacts such as `report.json`,
  `report.md`, `meta/performance-profile.json`, and
  `meta/performance-verdict.json`
