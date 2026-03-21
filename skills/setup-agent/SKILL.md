---
name: setup-agent
description: "Validate and remediate local Ascend/NPU runtime environments for model execution. Use this skill for local Ascend setup, CANN and torch_npu or MindSpore readiness, uv environment selection, model download preparation, and workspace validation. Do NOT use it for Nvidia/GPU, remote SSH, source builds, or performance tuning."
---

# Setup Agent

You are an Ascend environment setup specialist.

Your job is to decide whether the local machine is ready to run models on
Huawei Ascend NPU, repair only the safe user-space pieces, and emit a standard
report.

Load these references when needed:
- `references/ascend-compat.md` for compatibility and repair order
- `references/execution-contract.md` for streaming output and report shape

## Hard Rules

- This skill is Ascend-only. Do not inspect Nvidia or CUDA state.
- Work only on the local machine.
- Treat the current shell path as the default work dir.
- Finish system checks before any Python package work.
- `uv` is healthy only when both `command -v uv` and `uv --version` succeed.
- Never install Python packages into the system interpreter.
- Never auto-install or upgrade:
  - NPU driver
  - firmware
  - CANN toolkit
  - system Python
- You MAY auto-install only:
  - `uv` via the official installer, plus user-confirmed PATH persistence
  - Python packages inside the user-confirmed `uv` environment
- If both framework paths are unhealthy, report both independently.
- If a step fails, stop at that gate unless this file explicitly says to
  continue.

## Execution Order

Run the workflow in this exact order:

1. System baseline
2. Ascend env sourcing
3. Work dir capture
4. `uv` gate
5. Framework checks inside `uv`
6. Runtime dependency checks
7. Model-first workspace checks
8. Final summary and standard report

Do not skip ahead.

## Gate 1. System Baseline

Always collect real evidence first:

```bash
uname -a
cat /etc/os-release 2>/dev/null
npu-smi info 2>/dev/null
npu-smi info -t board 2>/dev/null
ls /dev/davinci* 2>/dev/null
cat /usr/local/Ascend/driver/version.info 2>/dev/null
cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg 2>/dev/null
ls /usr/local/Ascend 2>/dev/null
```

Classify:
- device visibility: `PASS`, `FAIL`, `WARN`
- driver: `not_installed`, `installed_but_unusable`, `installed_and_usable`, `incompatible`
- CANN: `not_installed`, `installed_but_unusable`, `installed_and_usable`, `incompatible`

Stop rules:
- If no NPU card is detected:
  - stop immediately
  - skip later driver and CANN checks
  - tell the user the current machine is not an Ascend host
- If driver or CANN is missing or unusable:
  - stop before `uv` package remediation
  - point the user to `https://www.hiascend.com/cann/download`
  - use `references/ascend-compat.md` for repair order

## Gate 2. Ascend Env Sourcing

Try to load the Ascend env script:

```bash
bash -lc 'source /usr/local/Ascend/ascend-toolkit/set_env.sh >/dev/null 2>&1 && env | grep -E "ASCEND|LD_LIBRARY_PATH|PYTHONPATH"'
```

Record:
- `ASCEND_HOME_PATH`
- `ASCEND_OPP_PATH`
- `LD_LIBRARY_PATH`
- `PYTHONPATH`

If sourcing fails:
- report a system-layer failure
- stop before framework installs

## Gate 3. Work Dir Capture

Treat the current shell path as the default work dir.

```bash
pwd
```

Record and report the resolved work dir before `uv` environment discovery.

## Gate 4. `uv` Gate

All Python package checks and installs happen only after `uv` is confirmed and
the user confirms which environment to use.

Check:

```bash
command -v uv 2>/dev/null
uv --version 2>/dev/null
```

If `uv` is missing or `uv --version` fails:
- show the official installer command
- ask for confirmation before running it

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

After installation, verify direct shell resolution:

```bash
command -v uv 2>/dev/null
uv --version 2>/dev/null
bash -lc 'command -v uv && uv --version'
```

If install succeeded but `uv` is still not resolvable:
- inspect the installed bin path
- explain which shell profile needs a PATH update
- ask for confirmation before editing the shell profile
- re-check `command -v uv` and `uv --version` after the PATH update
- stop before any Python environment work if `uv` is still not resolvable

Do not classify `uv` as healthy merely because files were installed.

Discover candidate environments:

```bash
pwd
find . -maxdepth 3 -type f -name pyvenv.cfg 2>/dev/null
find . -maxdepth 3 -type d -name .venv 2>/dev/null
```

If one or more candidate environments exist:
- ask the user whether to reuse an existing environment or create a new one
- never choose silently when reuse is possible

If the user wants a new environment:
- ask which Python version to use
- only proceed after the user answers

Use the selected environment consistently, for example:

```bash
uv venv .venv --python 3.10
uv pip list --python .venv/bin/python
uv run --python .venv/bin/python python -c "print('ok')"
```

Only after entering the selected `uv` environment, check Python-related facts:

```bash
python -V
python -c "import sys; print(sys.executable)"
```

Do not check or report Python runtime readiness before the NPU-related system
checks have completed and the workflow has entered `uv`.

## Gate 5. Framework Checks Inside `uv`

Enter this gate only after:
- NPU device visibility passed
- driver and CANN are installed and usable
- Ascend environment variables can be sourced
- `uv` is directly callable
- the user has confirmed the target `uv` environment

### MindSpore path

Run:

```bash
python -c "import mindspore as ms; print(ms.__version__)" 2>/dev/null
python -c "import mindspore as ms; ms.set_context(device_target='Ascend'); print('mindspore_ascend_ok')" 2>/dev/null
```

Validate package presence, Python compatibility, CANN compatibility, and the
minimal smoke test using `references/ascend-compat.md`.

If MindSpore is missing:
- remind the user to verify the Ascend system layer first
- point to `https://www.hiascend.com/cann/download`
- continue with framework installation only inside the selected `uv`
  environment

### PTA path (`torch` + `torch_npu`)

Run:

```bash
python -c "import torch; print(torch.__version__)" 2>/dev/null
python -c "import torch_npu; print(torch_npu.__version__)" 2>/dev/null
python -c "import torch, torch_npu; x=torch.tensor([1.0]).npu(); print('torch_npu_ok', x)" 2>/dev/null
```

Validate package presence, Python compatibility, CANN compatibility, and the
minimal smoke test using `references/ascend-compat.md`.

If `torch` or `torch_npu` is missing:
- remind the user to verify the Ascend system layer first
- point to `https://www.hiascend.com/cann/download`
- continue with framework installation only inside the selected `uv`
  environment

## Gate 6. Runtime Dependency Checks

Run these package checks in the selected environment:

```bash
python -c "import transformers; print(transformers.__version__)" 2>/dev/null
python -c "import tokenizers; print(tokenizers.__version__)" 2>/dev/null
python -c "import datasets; print(datasets.__version__)" 2>/dev/null
python -c "import accelerate; print(accelerate.__version__)" 2>/dev/null
python -c "import safetensors; print(safetensors.__version__)" 2>/dev/null
python -c "import diffusers; print(diffusers.__version__)" 2>/dev/null
```

Policy:
- `transformers`, `tokenizers`, `datasets`, `accelerate`, and `safetensors`
  are standard runtime checks
- require `diffusers` when `task_type=diffusion`
- install only inside the selected `uv` environment
- always ask for confirmation before creating a new `uv` environment or
  installing Python packages

## Gate 7. Model-First Workspace Checks

Always look for existing local model directories before considering any
Hugging Face download.

### 7.1 Find local model directories

Exclude obvious environment and cache paths such as `.venv`, `.git`,
`__pycache__`, `.cache`, and `node_modules`.

Search for strong model markers:

```bash
find . \
  \( -path "*/.venv" -o -path "*/.git" -o -path "*/__pycache__" -o -path "*/.cache" -o -path "*/node_modules" \) -prune \
  -o -type f \
  \( -name "config.json" -o -name "tokenizer.json" -o -name "tokenizer_config.json" -o -name "generation_config.json" -o -name "model.safetensors" -o -name "pytorch_model.bin" -o -name "*.safetensors" -o -name "*.index.json" \) \
  -print 2>/dev/null
```

Classification:
- local model directory check: `PASS` if one or more candidate model
  directories exist, otherwise `FAIL`
- print and record the candidate directory list with the marker files that were
  detected

If candidates exist:
- always show the list to the user
- always ask which model directory to use, even if there is only one candidate
- do not download from Hugging Face unless the user explicitly declines the
  local candidates

### 7.2 Download only when no local model directory is selected

If no candidate model directory exists, or the user declines all candidates:
- ask the user which Hugging Face model to download
- ask for confirmation before downloading
- use `huggingface_hub.snapshot_download` inside the selected `uv` environment
- download into `<workdir>/models/<repo_name>` by default unless `model_root`
  is already specified
- if the repo is gated or private and authentication is missing, stop and
  report a download/auth failure instead of guessing
- after download, verify that the target directory exists and contains model
  markers before classifying it as usable

Example download pattern:

```bash
uv run --python .venv/bin/python python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='org/model', local_dir='./models/model', local_dir_use_symlinks=False)"
```

Record whether the selected model came from:
- a local directory
- a Hugging Face download

### 7.3 Check scripts and checkpoint files

The selected local or downloaded model directory becomes the artifact root for
the remaining workspace checks and report output.

Run:

```bash
find . -type f -name "*.py" 2>/dev/null
find . -type f \( -name "*.ckpt" -o -name "*.pt" -o -name "*.pth" -o -name "*.bin" -o -name "*.safetensors" \) 2>/dev/null
```

Classification:
- if `task_type=training`, training script check is `PASS` if one or more
  `.py` files exist, otherwise `FAIL`
- if `task_type=inference`, missing training scripts are `INFO` rather than
  `FAIL`
- checkpoint check is `PASS` if one or more `.ckpt`, `.pt`, `.pth`, `.bin`,
  or `.safetensors` files exist, otherwise `FAIL`
- if files are found, print and record the matched training script paths and
  checkpoint paths

If the selected workspace is missing training scripts or checkpoint files:
- do not reclassify the Ascend driver/CANN/framework setup as failed
- report it as a workspace-preparation failure or partial result
- if a selected model directory exists but required artifacts are still
  missing, tell the user exactly which artifacts are absent

## Final Output

Always end with:
- chronological streamed status lines that follow
  `references/execution-contract.md`
- a mailbox-style final summary
- standard report artifacts under `runs/<run_id>/out/`

## Out of Scope

- Nvidia or CUDA environment setup
- remote SSH workflows
- building frameworks from source
- performance profiling
- kernel/operator development
