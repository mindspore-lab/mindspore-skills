# Readiness-Agent Examples

The examples here describe only real, deployable readiness workflows. They do
not use fake environments to demonstrate `READY`.

## Panorama

| Scenario | What `readiness-agent` should do | Current status | Example |
| --- | --- | --- | --- |
| Training preparation in a new workspace | Create a workspace-local environment, install the framework and dependencies, add a training script, download the model and dataset, and complete readiness certification | Ready to demo | `demos/qwen3-0.6b-pta-new-workspace.md` |
| MindSpore training preparation in a new workspace | Similar to PTA, but the target framework is MindSpore | Planned | Not completed |
| Inference preparation in a new workspace | Prepare scripts, model assets, dependencies, and smoke checks for an inference target | Planned | Not completed |
| Missing-item analysis in an existing workspace | Check whether scripts, model assets, datasets, checkpoints, and Python environments are missing | Supported | Covered inside the completed demo |
| Automatic remediation in fix mode | Install `uv`, create `.venv`, install packages, scaffold example scripts, and download declared assets | Supported | Covered inside the completed demo |
| System-level driver or CANN installation | Outside the skill boundary | Explicitly excluded | Not applicable |

## Completed Demo

- `demos/qwen3-0.6b-pta-new-workspace.md`
  Demonstrates how `readiness-agent` can help in a brand-new workspace, after
  the user has already installed the NPU driver and CANN, by preparing the PTA
  virtual environment, training script, model, and dataset, then pushing the
  workspace to `READY`.

## Planned Additions

- `Qwen3 0.6B + MindSpore + new workspace`
- `Qwen3 0.6B inference + new workspace`
- `remote asset download failure / cache behavior`

## Reference Sources

- [skills/readiness-agent/templates/qwen3_0_6b_training_example.py](../../skills/readiness-agent/templates/qwen3_0_6b_training_example.py)
  can serve as a recipe reference when the agent needs to generate a training
  script in fix mode
