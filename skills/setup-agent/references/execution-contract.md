# Setup-Agent Execution Contract

This reference defines how `setup-agent` should behave at runtime from the
user's perspective. Treat it as the UI and reporting contract.

## Streaming Console Output

During execution, print each step as it starts and when it finishes. Use short,
status-first lines so the user can follow progress in real time.

Preferred pattern:

```text
setup-agent : checking os...
setup-agent : os passed: Ubuntu 22.04 aarch64
setup-agent : checking work dir...
setup-agent : work dir passed: /path/to/current/workdir
setup-agent : checking npu visibility...
setup-agent : npu visibility failed: `npu-smi` not available
setup-agent : checking cann...
setup-agent : cann failed: toolkit version file missing
```

Output rules:
- emit a `checking ...` line before every major step
- emit a `passed`, `failed`, `warn`, or `skip` line after each step
- include the concrete reason in the same line
- keep the stream chronological
- if the workflow stops early, print the stop reason immediately
- if no NPU is detected, print that later driver and CANN checks were skipped

Major steps that must stream:
- os
- npu visibility
- driver
- cann
- Ascend env sourcing
- work dir
- uv
- uv environment selection
- local model directories
- model selection
- hugging face download
- MindSpore
- torch
- torch_npu
- runtime dependencies
- training scripts
- checkpoint files
- final summary

When training scripts or checkpoint files are found, print the resolved file
paths in the stream output.

Preferred pattern:

```text
setup-agent : training scripts passed: ./train.py, ./scripts/finetune.py
setup-agent : checkpoint files passed: ./weights/model.safetensors
```

## Report Artifacts

Every run must produce standard outputs under `runs/<run_id>/out/`:

- `report.json`
- `report.md`
- `logs/run.log`
- `logs/verify.log`
- `artifacts/README.md`
- `meta/env.json`
- `meta/inputs.json`

Required report content:
- OS information
- current work dir
- NPU visibility and `npu-smi` result
- driver, firmware, and CANN state
- `set_env.sh` sourcing result
- `uv` availability, direct shell resolution status, any PATH update action, selected environment, and Python details from inside `uv`
  - direct shell resolution status
  - PATH update action
- MindSpore results
- `torch` / `torch_npu` results
- runtime dependency and install results
  - `transformers`
  - `tokenizers`
  - `datasets`
  - `accelerate`
  - `safetensors`
  - `diffusers`
- work dir artifact results
  - local model directory findings
  - candidate model directory list
  - selected model path
  - selected model source (`local` or `huggingface`)
  - training scripts
  - checkpoint files
  - matched training script paths
  - matched checkpoint paths
- smoke test results
- manual system-layer remediation steps if needed
- the `https://www.hiascend.com/cann/download` link when Ascend driver,
  framework, or toolkit is missing
- generic Hugging Face download guidance when training scripts or checkpoint
  files are missing from the current work dir
- download/auth failure reason when a Hugging Face model cannot be fetched

Use only these status values:
- `PASS`
- `FAIL`
- `WARN`
- `SKIP`
- `INFO`

## Final Mailbox Summary

At the end of the run, print a mailbox-style final summary to the console even
if the run fails early.

The final summary must include:
- current work dir
- which components are already installed
- which components are missing or not yet installed
- which checks were skipped
- selected model path when present
- whether the selected model was reused locally or downloaded
- matched training script paths when present
- matched checkpoint paths when present
- the failure reason if the run failed
- the next manual action or next automated step
- the `https://www.hiascend.com/cann/download` link when the Ascend environment
  is incomplete
- Hugging Face guidance when the current work dir is missing scripts or
  checkpoint files

Preferred summary shape:

```text
setup-agent : final summary
workdir:
- /path/to/current/workdir
installed:
- uv 0.10.9
- python 3.10.12
selected_model:
- ./models/qwen-7b
model_source:
- local
training_scripts:
- ./train.py
checkpoint_files:
- ./weights/model.safetensors
missing:
- Ascend driver
- CANN toolkit
- training script
- checkpoint file
skipped:
- MindSpore
- torch_npu
failure:
- local machine is not an Ascend runtime environment
next:
- install driver and CANN on a Linux Ascend host, then rerun setup-agent
- download missing scripts or checkpoints from Hugging Face into the current work dir
```
