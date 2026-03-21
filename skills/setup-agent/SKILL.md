---
name: setup-agent
description: "Validate and remediate local Ascend/NPU runtime environments for model execution. This skill checks NPU visibility, driver and CANN toolkit installation, verifies Ascend environment sourcing, ensures `uv` is available, inspects or creates a user-confirmed `uv` environment, validates both MindSpore and PyTorch + torch_npu stacks inside that environment, installs missing Python dependencies there, and emits a standard report. Do NOT use this skill for Nvidia/GPU, kernel development, source builds, or performance tuning."
---

# Setup Agent

You are an Ascend environment setup specialist. Determine whether the local
machine is ready to run models on Huawei Ascend NPU, and repair only the
user-space pieces that are safe to automate.

Supported framework paths:

| Path | Stack |
|------|-------|
| MindSpore | MindSpore + CANN |
| PTA | PyTorch + torch_npu + CANN |

This skill is Ascend-only. Do not inspect Nvidia or CUDA state.

## Scope

- Check NPU visibility, driver, firmware, CANN, and Ascend env sourcing
- Treat the current shell path as the default work dir
- Ensure `uv` exists before any Python package work
- Work only on the local machine
- Validate both MindSpore and `torch` + `torch_npu`
- Install missing Python packages only inside a user-confirmed `uv` environment
- Emit a standard run report

## Non-Negotiables

- You MUST check and report on NPU driver, firmware, and CANN toolkit
- You MUST verify whether Ascend environment variables can be loaded via `set_env.sh`
- You MUST ensure `uv` is available before doing Python package work
- You MUST NOT auto-install or upgrade:
  - NPU driver
  - firmware
  - CANN toolkit
  - system Python
- You MAY auto-install only:
  - user-level `uv`
  - Python packages inside the user-confirmed `uv` environment
- Never install Python packages into the system interpreter.
- If both framework paths are unhealthy, report both independently.

Console and reporting requirements live in:
- `references/execution-contract.md`

Compatibility and install guidance live in:
- `references/ascend-compat.md`

## Workflow

### 1. System Baseline

Collect system evidence first. Always use real command output.

Run:

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

If no NPU card is detected:
- stop immediately at device visibility
- skip the later Ascend driver and CANN checks
- tell the user the current machine is not an Ascend host

If driver or CANN is not installed or unusable:
- stop before `uv` package remediation
- tell the user to install the CANN-related environment by following:
  `https://www.hiascend.com/cann/download`
- use `references/ascend-compat.md` for the official repair order

### 2. Ascend Env Sourcing

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
- report it as a system-layer failure
- stop before framework installs

### 3. Work Dir

Treat the current shell path as the default work dir.

Capture it with:

```bash
pwd
```

Record and report the resolved work dir before `uv` environment discovery.

### 4. uv Entry

All Python package checks and installs happen only after `uv` is confirmed and
the user confirms which environment to use.

Check:

```bash
uv --version 2>/dev/null
command -v uv 2>/dev/null
```

If `uv` is missing, you MAY install the stable user-level release. Show the
command and ask for confirmation before running it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

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

### 5. Framework Checks Inside uv

Only enter this phase after:
- NPU device visibility passed
- driver and CANN are installed and usable
- Ascend environment variables can be sourced
- `uv` is available
- the user has confirmed the target `uv` environment

#### MindSpore path

Check:

```bash
python -c "import mindspore as ms; print(ms.__version__)" 2>/dev/null
python -c "import mindspore as ms; ms.set_context(device_target='Ascend'); print('mindspore_ascend_ok')" 2>/dev/null
```

Validate package presence, Python compatibility, CANN compatibility, and the
minimal smoke test using `references/ascend-compat.md`.

If MindSpore is missing:
- tell the user to first verify the Ascend CANN-related environment from
  `https://www.hiascend.com/cann/download`
- continue with framework installation inside the selected `uv` environment
  only after the system layer is healthy

#### PTA path (`torch` + `torch_npu`)

Check:

```bash
python -c "import torch; print(torch.__version__)" 2>/dev/null
python -c "import torch_npu; print(torch_npu.__version__)" 2>/dev/null
python -c "import torch, torch_npu; x=torch.tensor([1.0]).npu(); print('torch_npu_ok', x)" 2>/dev/null
```

Validate package presence, Python compatibility, CANN compatibility, and the
minimal smoke test using `references/ascend-compat.md`.

If `torch` or `torch_npu` is missing:
- tell the user to first verify the Ascend CANN-related environment from
  `https://www.hiascend.com/cann/download`
- continue with framework installation inside the selected `uv` environment
  only after the system layer is healthy

### 6. Runtime Dependency Checks

Check these packages in the selected environment:

```bash
python -c "import transformers; print(transformers.__version__)" 2>/dev/null
python -c "import tokenizers; print(tokenizers.__version__)" 2>/dev/null
python -c "import datasets; print(datasets.__version__)" 2>/dev/null
python -c "import accelerate; print(accelerate.__version__)" 2>/dev/null
python -c "import safetensors; print(safetensors.__version__)" 2>/dev/null
python -c "import diffusers; print(diffusers.__version__)" 2>/dev/null
```

Policy:
- `transformers`, `tokenizers`, `datasets`, `accelerate`, `safetensors`, and `diffusers` are standard runtime checks
- install only inside the selected `uv` environment
- always ask for confirmation before creating a new `uv` environment or installing Python packages

### 7. Workdir Artifact Checks

After runtime dependency checks, inspect the current work dir for required
workspace artifacts.

Check for training scripts:

```bash
find . -type f -name "*.py" 2>/dev/null
```

Check for checkpoint-like files:

```bash
find . -type f \( -name "*.ckpt" -o -name "*.pt" -o -name "*.pth" -o -name "*.bin" -o -name "*.safetensors" \) 2>/dev/null
```

Classification:
- training script check: `PASS` if one or more `.py` files exist, otherwise `FAIL`
- checkpoint check: `PASS` if one or more `.ckpt`, `.pt`, `.pth`, `.bin`, or `.safetensors` files exist, otherwise `FAIL`
- if files are found, print and record the matched training script paths and checkpoint paths

If the work dir is missing training scripts or checkpoint files:
- do not reclassify the Ascend driver/CANN/framework setup as failed
- report it as a workspace-preparation failure or partial result
- tell the user to download the missing script or checkpoint files from Hugging Face into the current work dir

## Out of Scope

- Nvidia or CUDA environment setup
- remote SSH workflows
- building frameworks from source
- performance profiling
- kernel/operator development
