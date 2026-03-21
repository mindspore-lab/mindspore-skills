# Ascend Runtime Compatibility and Setup Guidance

This reference is the lookup source for `setup-agent`. Use it after collecting
real system evidence. Do not guess compatibility.

## Quick Use

Use this file in this order:

1. Check `Driver / Firmware / CANN Matrix`
2. If the system layer is healthy, choose the framework table:
   - `MindSpore on Ascend`
   - `PyTorch + torch_npu on Ascend`
3. If a required system component is missing, go directly to
   `Official Installation Guides` and `Repair Policy`

Decision rule:
- no NPU card detected: stop immediately and skip later driver/CANN checks
- missing driver or missing CANN: stop at the system layer
- `set_env.sh` cannot be sourced: stop at the system layer
- system layer healthy: continue to `uv` and Python package checks
- unknown version tuple: mark `WARN`, do not silently treat as supported

## Driver / Firmware / CANN Matrix

Use this table for bottom-up validation. Driver and firmware are the baseline;
CANN must match them before any framework path is considered.

| CANN | Min Driver | Min Firmware | Supported Chips | Typical Use |
|------|------------|--------------|-----------------|-------------|
| 8.1.RC1 | 24.1.rc3 | 7.5.0.1.129 | Ascend 910B/C | Newer 910B/910C deployments |
| 8.0.RC3 | 24.1.rc2 | 7.3.0.1.100 | Ascend 910B/C | Common production baseline |
| 8.0.RC2 | 24.1.rc1 | 7.1.0.6.220 | Ascend 910B | Transitional release |
| 8.0.RC1 | 23.0.6 | 7.1.0.5.220 | Ascend 910B | Older 910B stacks |
| 7.3.0 | 23.0.5 | 7.1.0.3.220 | Ascend 910A/B | Legacy supported path |
| 7.1.0 | 23.0.3 | 7.1.0.1.220 | Ascend 910A/B | Legacy supported path |

Interpretation:
- `PASS`: driver and firmware meet or exceed the minimum for the detected CANN
- `FAIL`: driver or firmware is below the required minimum
- `WARN`: one of the detected versions is not in the table

Stop conditions:
- no Ascend NPU card is present on the machine
- `npu-smi info` fails and no usable backup evidence exists
- driver version file is missing
- CANN toolkit version file is missing
- `set_env.sh` is missing or cannot be sourced

## MindSpore on Ascend

Use this section only after the system layer is healthy.

| MindSpore | Recommended CANN | Minimum CANN | Python | Typical Use |
|-----------|------------------|--------------|--------|-------------|
| 2.5.0 | 8.1.RC1 | 8.0.RC3 | 3.8-3.11 | Current recommended stable line |
| 2.4.1 | 8.0.RC3 | 8.0.RC2 | 3.8-3.11 | Common production baseline |
| 2.4.0 | 8.0.RC2 | 8.0.RC1 | 3.8-3.11 | Transitional release |
| 2.3.1 | 8.0.RC1 | 7.3.0 | 3.8-3.10 | Legacy support |
| 2.3.0 | 7.3.0 | 7.1.0 | 3.8-3.10 | Legacy support |

Validation checklist:
- import succeeds inside the selected `uv` environment
- Python version is within the supported range
- CANN version satisfies at least the minimum
- `ms.set_context(device_target='Ascend')` succeeds after sourcing Ascend env

Decision rule:
- exact or clearly in-range tuple: `PASS`
- version present but below minimum CANN or Python range: `FAIL`
- version not listed: `WARN`

Official verification:
- https://www.mindspore.cn/install

## PyTorch + torch_npu on Ascend

Use this section only after the system layer is healthy.

| torch | torch_npu | Recommended CANN | Python | Typical Use |
|-------|-----------|------------------|--------|-------------|
| 2.4.x | 2.4.x | 8.1.RC1 | 3.9-3.11 | Newer Ascend stacks |
| 2.3.x | 2.3.x | 8.0.RC3 | 3.9-3.11 | Common production baseline |
| 2.1.x | 2.1.x | 8.0.RC1 | 3.8-3.10 | Older but still common |
| 2.0.x | 2.0.x | 7.3.0 | 3.8-3.10 | Legacy support |

Validation checklist:
- `torch` import succeeds
- `torch_npu` import succeeds
- `torch` and `torch_npu` major/minor versions align
- Python version is within the supported range
- a minimal NPU tensor smoke test succeeds after sourcing Ascend env

Decision rule:
- aligned major/minor and compatible Python/CANN tuple: `PASS`
- `torch` and `torch_npu` major/minor mismatch: `FAIL`
- version known but below required CANN/Python range: `FAIL`
- exact tuple not listed: `WARN`

The setup-agent should tell the user to verify against the PTA release notes if
the exact tuple is not listed.

## Detection Hints

Use these hints when classifying evidence gathered by the skill.

### System-layer healthy

Typical signals:
- `npu-smi info` lists one or more NPUs
- `/usr/local/Ascend/driver/version.info` exists
- `/usr/local/Ascend/ascend-toolkit/latest/version.cfg` exists
- `source /usr/local/Ascend/ascend-toolkit/set_env.sh` succeeds

### Installed but unusable

Typical signals:
- version files exist, but `npu-smi info` fails
- version files exist, but sourcing `set_env.sh` fails
- environment variables remain incomplete after sourcing

### Python-layer ready

Typical signals:
- selected `uv` environment is known
- framework import succeeds
- smoke test succeeds on Ascend

## Official Installation Guides

Use these links when the setup-agent detects missing or unusable system-layer
components:

- Ascend CANN download portal: https://www.hiascend.com/cann/download
- MindSpore install guide: https://www.mindspore.cn/install
- Ascend CANN community downloads: https://www.hiascend.com/software/cann/community
- Ascend documentation portal: https://www.hiascend.com/document
- `uv` install guide: https://docs.astral.sh/uv/getting-started/installation/

Recommended manual repair order:

1. Install or repair the Ascend driver package that matches the target chip
2. Install the matching CANN toolkit release
3. Source `/usr/local/Ascend/ascend-toolkit/set_env.sh`
4. Re-run `npu-smi info`
5. Return to `uv` and Python package checks only after the system layer is healthy

If the Ascend driver, framework, or toolkit is missing, the setup-agent should
explicitly remind the user to start from:
- https://www.hiascend.com/cann/download

## Repair Policy

Allowed automation:
- install user-level `uv`
- create or reuse a user-confirmed `uv` environment
- install missing Python packages inside that environment

Forbidden automation:
- auto-install driver
- auto-install firmware
- auto-install CANN toolkit
- install Python packages into the system interpreter

Stop and hand back to the user when:
- no NPU card is detected
- `npu-smi info` fails
- driver or CANN is missing
- `set_env.sh` cannot be sourced
- the user has not confirmed which `uv` environment to use
