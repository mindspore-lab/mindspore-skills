# New-Readiness-Agent Decision Rules

## Workspace Boundary

- only use the selected workspace as the certification boundary
- do not borrow scripts, configs, or assets from sibling repos or bundled
  examples
- use current environment variables only for runtime evidence such as Ascend or
  CANN

## Target and Framework

- prefer explicit `target`, `framework_hint`, and `launcher_hint`
- otherwise infer from launch commands, launch scripts, configs, dependencies,
  and entry scripts
- downgrade confidence when evidence conflicts

## Environment Selection

- keep `control Python` separate from the runtime environment
- prefer:
  1. launch-command environment
  2. current active virtual environment
  3. workspace-local virtual environment
  4. system Python
- if the launch command conflicts with the active environment, prefer the
  launch-command environment
- always preserve the full candidate list for user confirmation and downstream
  tooling

## Ready Threshold

- `READY` requires a selected runtime environment and passing near-launch
  validation
- `WARN` is allowed when no hard blocker is proven but confidence gaps remain
- `BLOCKED` is required when the selected runtime path or required assets are
  missing or near-launch validation fails

