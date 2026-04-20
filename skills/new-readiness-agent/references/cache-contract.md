# New-Readiness-Agent Cache Contract

Run-scoped artifacts live under:

- `readiness-output/attempts/<attempt_id>/current/`
- `readiness-output/attempts/<attempt_id>/final/`

Run-scoped writing is phase-sensitive:

- `NEEDS_CONFIRMATION` runs only persist the lightweight continuation state:
  - `meta/readiness-verdict.json`
  - `artifacts/workspace-readiness.lock.json`
  - `artifacts/confirmation-step.json`
- validated runs additionally persist the full diagnostic bundle:
  - `report.json`
  - `report.md`
  - `logs/run.log`
  - `meta/env.json`
  - `meta/inputs.json`

Workspace latest cache lives under:

- `readiness-output/latest/new-readiness-agent/`

Downstream agents should prefer:

- `readiness-output/latest/new-readiness-agent/workspace-readiness.lock.json`

If the user explicitly provides a run-scoped artifact path, downstream agents
may read:

- `readiness-output/attempts/<attempt_id>/current/artifacts/workspace-readiness.lock.json`
- `readiness-output/attempts/<attempt_id>/final/artifacts/workspace-readiness.lock.json`

The attempt directory groups one end-to-end readiness certification flow. A
stepwise rerun with `--confirm field=value` should reuse the active
`<attempt_id>` instead of creating a brand-new sibling directory for every
confirmation step.

`workspace-readiness.lock.json` must remain the stable downstream contract for:

- current phase and whether confirmation is still pending
- the current per-field confirmation step and remaining confirmation queue
- the portable structured-question projection for the current step when present
- final selected target, launcher, framework, and runtime environment
- final selected assets for config, model, dataset, and checkpoint
- selected asset source types such as `local_path`, `hf_cache`, `hf_hub`,
  `script_managed_remote`, and `inline_config`
- selected asset locators such as paths, repo IDs, cache paths, and split names
- required packages
- missing items
- warnings
- confirmation metadata
- HF cache layout and remote-asset evidence summary when present
- evidence summary
- update timestamp
