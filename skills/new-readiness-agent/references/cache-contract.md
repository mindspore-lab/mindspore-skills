# New-Readiness-Agent Cache Contract

Run-scoped artifacts live under:

- `runs/<run_id>/out/`

Workspace latest cache lives under:

- `runs/latest/new-readiness-agent/`

Downstream agents should prefer:

- `runs/latest/new-readiness-agent/workspace-readiness.lock.json`

If the user explicitly provides a run-scoped artifact path, downstream agents
may read:

- `runs/<run_id>/out/artifacts/workspace-readiness.lock.json`

`workspace-readiness.lock.json` must remain the stable downstream contract for:

- current phase and whether confirmation is still pending
- final selected target, launcher, framework, and runtime environment
- required packages
- missing items
- warnings
- confirmation metadata
- evidence summary
- update timestamp
