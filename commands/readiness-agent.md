---
description: Direct specialist entry for checking whether a local single-machine workspace is ready to train or run inference now, what is missing before execution, and whether safe user-space readiness fixes should run
---

# Readiness Agent

Direct specialist entry for local single-machine readiness certification before
execution.

For most users, prefer:

- `/readiness <workspace or problem>`

Use `/readiness-agent` only when you already know you want to force the
readiness specialist directly.

Load the `readiness-agent` skill and follow its readiness certification
workflow:

1. selected Python resolution
2. execution target discovery
3. dependency closure and compatibility validation
4. blocker classification
5. optional safe user-space remediation and revalidation
6. readiness report build

Do not run exploratory `--help` or guessed-flag helper invocations before the
real readiness pass. Call the top-level readiness pipeline for the actual
workspace and report the structured verdict instead of raw CLI parser output.

## Typical Inputs

- code folder or working directory
- framework hint such as `pta` or `mindspore` if the user already has a
  concrete preference
- optional `cann_path` when CANN is installed in a custom location
- Hugging Face model or dataset repo IDs when the workspace should materialize
  missing local assets from remote sources
- intended target or explicit entry script if already known
- training or inference config, model, dataset, or checkpoint paths if already
  known
- selected Python or selected environment root if already known
- explicit minimal smoke command if the user already knows a safe one
