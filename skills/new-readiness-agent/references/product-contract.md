# New-Readiness-Agent Product Contract

`new-readiness-agent` has one user-visible responsibility:

- certify whether the selected local single-machine workspace can start the
  intended training or inference workflow now

This skill is strictly read-only. It does not fix the environment, install
packages, or launch the real workload.

The user-visible result must contain:

- `phase`
- `status`
- `confirmation_required`
- `pending_confirmation_fields`
- `current_confirmation`
- `can_run`
- `target`
- `summary`
- `missing_items`
- `warnings`
- `next_action`

The internal verdict must preserve:

- detected candidates for target, launcher, framework, and environment
- detected asset candidates for config, model, dataset, and checkpoint
- selected final values and their sources
- selected asset source types and locators
- whether the run is still awaiting user confirmation or already validated
- the current per-field confirmation step and its numbered options
- the current step's portable structured-question projection for host-native
  question tools
- near-launch validation checks
- framework compatibility details, including installed versions, local-table
  status, and recommended package specs when available
- environment candidates and ranking
- HF cache layout and remote-asset evidence when detected
- readiness lock and confirmation step references
- workspace latest cache references
