---
description: Preflight a local single-machine workspace to decide whether it is ready to train or run inference now, what is missing before execution, and whether safe user-space readiness fixes should run
---

# Readiness

Use this as the top-level readiness entrypoint before running ad-hoc
environment probes.

Always load the `readiness-agent` skill first instead of starting with manual
shell checks.

Do not begin by probing helper CLI usage with guessed flags such as
`--verbose` or incomplete argument lists. Invoke the top-level readiness
pipeline once for the real workspace and present only the final structured
result to the user.

Load the `readiness-agent` skill and follow its readiness certification
workflow for:

- selected Python resolution
- execution target discovery
- dependency closure and compatibility validation
- blocker classification
- optional safe user-space remediation and revalidation
- readiness report build

Use `/readiness` when the user is asking:

- can this repo train
- can this repo run inference
- check my environment before running
- validate my workspace
- run a preflight
- verify config, model, dataset, checkpoint, environment, or framework compatibility
- prefer a specific `cann_path` when the machine has a custom CANN install
- download missing model or dataset assets from Hugging Face during readiness fix
- prefer `pta` or `mindspore` for the readiness check

If the workload already runs and the user is reporting a crash, wrong result,
or performance issue, do not stay here. Redirect to:

- `/diagnose`
- `/fix`
