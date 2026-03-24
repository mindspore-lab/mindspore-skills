# Setup-Agent Execution Contract

This reference defines how `setup-agent` should behave at runtime from the
user's perspective. Treat it as the UI and reporting contract.

## Streaming Console Output

During execution, print each step as it starts and when it finishes. Use short,
status-first lines so the user can follow progress in real time.

Preferred pattern:

```text
setup-agent : checking os...
setup-agent : os PASS: Ubuntu 22.04 aarch64
setup-agent : checking work dir...
setup-agent : work dir PASS: /path/to/current/workdir
setup-agent : checking npu visibility...
setup-agent : npu visibility FAIL: `npu-smi` not available
setup-agent : checking cann...
setup-agent : cann FAIL: toolkit version file missing
```

Output rules:
- emit a `checking ...` line before every major step
- emit a `PASS`, `FAIL`, `WARN`, or `SKIP` line after each step
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
- final mailbox summary

When training scripts or checkpoint files are found, print the resolved file
paths in the stream output.

Preferred pattern:

```text
setup-agent : training scripts PASS: ./train.py, ./scripts/finetune.py
setup-agent : checkpoint files PASS: ./weights/model.safetensors
```

## Console Contract

Do not write `.md`, `.json`, or other result artifacts under `runs/` during
`setup-agent` execution. The run result is the streamed console output plus the
final boxed mailbox summary.

The console output must still cover:
- OS information
- current work dir
- NPU visibility and `npu-smi` result
- driver, firmware, and CANN state
- detected CANN version used for framework compatibility resolution
- `set_env.sh` sourcing result
- `uv` availability, direct shell resolution status, any PATH update action, selected environment root, selected interpreter path, and Python details from inside `uv`
  - direct shell resolution status
  - PATH update action
  - selected environment root
  - selected interpreter path
- MindSpore results
- `torch` / `torch_npu` results
- framework compatibility reasoning
  - installed framework version(s)
  - recommended compatible version(s)
  - mismatch cause such as Python range, framework version, or PTA build suffix
  - whether a replacement was offered and whether the user confirmed it
- runtime dependency and install results
  - direct `uv pip install --python <selected_python_path> ...` remediation inside the selected `uv` environment
  - `transformers`
  - `tokenizers`
  - `datasets`
  - `accelerate`
  - `safetensors`
  - `diffusers`
  - Python packages installed to recover a failed framework import or smoke test
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
- final mailbox summary that reflects any successful installs or repairs
- manual system-layer remediation steps if needed
- framework package installs or replacements performed inside the selected `uv`
  environment
- the `https://www.hiascend.com/cann/download` link when Ascend driver or
  toolkit is missing
- generic Hugging Face download guidance when training scripts or checkpoint
  files are missing from the current work dir
- China mirror fallback guidance using `HF_ENDPOINT=https://hf-mirror.com`
  when the Hugging Face download fails due to network reachability problems
- download/auth failure reason when a Hugging Face model cannot be fetched

Use only these status values:
- `PASS`
- `FAIL`
- `WARN`
- `SKIP`
- `INFO`

## Final Mailbox Summary

At the end of the run, print a final boxed mailbox summary to the console even
if the run fails early.

The final mailbox summary must:
- use an ASCII box
- keep labels aligned
- keep the field order fixed
- use the title `setup-agent : Success` or `setup-agent : Fail`
- keep every field on one line
- use `none` for missing values

The final mailbox summary must include these fields in this exact order:
- `workdir`
- `device`
- `uv`
- `framework`
- `model_deps`
- `model`
- `script`
- `ckpt`
- `fixed`
- `failed`
- `suggestion`

Do not require intermediate `run.log` or `verify.log` files during environment
checking. If a component is installed or repaired mid-run, reflect that only in
the final mailbox summary.

Formatting rules:
- left label width is fixed at 10 characters before `:`
- values should be shortened instead of wrapping
- do not print `PASS`
- for healthy components, show version, source, or path directly
- `fixed` is one sentence summarizing what the agent repaired
- `failed` is one sentence summarizing the final failure reason, or `none`
- `suggestion` is one sentence telling the user how to repair what the agent
  could not fix, or `none`

Preferred example:

```text
+----------------------------------------------------------------------------------+
| setup-agent : Success                                                            |
+----------------------------------------------------------------------------------+
| workdir    : /path/to/current/workdir                                            |
| device     : driver=24.1.rc2 | CANN=8.0.RC3                                      |
| uv         : 0.6.14 | env=./.venv | py=3.10.14                                   |
| framework  : mindspore=2.4.1 | torch=2.3.1 | torch_npu=2.3.1                     |
| model_deps : transformers=4.44.2 | tokenizers=0.19.1                             |
| model      : local | ./models/qwen2-7b                                           |
| script     : ./train.py                                                          |
| ckpt       : ./models/qwen2-7b/model.safetensors                                 |
| fixed      : installed uv and updated PATH                                       |
| failed     : none                                                                |
| suggestion : none                                                                |
+----------------------------------------------------------------------------------+
```
