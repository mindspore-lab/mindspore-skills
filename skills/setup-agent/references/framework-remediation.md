# Framework Remediation Workflow

Load this reference during Gate 5 and Gate 6.

Use it when:
- a framework package is missing
- an installed framework is incompatible with the detected CANN version
- a framework import or smoke test fails because a Python package may be missing
- runtime dependencies must be checked or installed inside the selected `uv`
  environment

## Scope

This reference covers only the Python-layer remediation workflow after the
system layer is healthy.

Do not use this file to repair:
- NPU driver
- firmware
- CANN toolkit
- system Python

Use `references/ascend-compat.md` for compatibility tables and source
precedence. Use `references/execution-contract.md` for streamed output and
final reporting.

Treat the detected CANN version as the primary selector for framework
validation and remediation.

## Framework Resolution Order

For each framework path, use this order:

1. Detect the current CANN version from the system-layer evidence
2. Detect the interpreter version from `selected_python_path`
3. Resolve compatible framework candidates from
   `references/ascend-compat.md`
4. For MindSpore only, if the local table does not classify the tuple, look up
   the detected release on the official `https://www.mindspore.cn/versions`
   page and treat that result as user-confirmed reference data rather than
   silent auto-remediation input
5. For PTA only, if the local table does not classify the tuple, prefer the
   bundled helper:

```bash
uv run --python <selected_python_path> python <setup_agent_skill_root>/scripts/pta_compat_lookup.py --cann 8.1.RC1 --torch 2.4.0 --torch-npu 2.4.0.post4 --python 3.10 --remote-fallback
```

Resolve `<setup_agent_skill_root>` from the installed `setup-agent` skill
directory before invoking the helper. Do not resolve the helper path relative
to the user work dir.

6. Compare the installed framework version against the compatible candidate set
7. Run the framework smoke test only after compatibility classification

For each framework path, use this remediation order:

1. Resolve the compatible target version from the detected CANN version and the
   current Python version
2. If the framework is missing, install that resolved target version inside the
   selected `uv` environment
3. If the framework import fails because a Python package is missing, install
   the missing Python package inside the selected `uv` environment and retry
4. If the framework is installed but incompatible, ask for confirmation before
   replacing it with the resolved compatible version
5. Re-run import and smoke tests after any install or replacement before
   classifying the framework path as healthy

## MindSpore Path

Run:

```bash
uv run --python <selected_python_path> python -c "import mindspore as ms; print(ms.__version__)" 2>/dev/null
uv run --python <selected_python_path> python -c "import mindspore as ms; ms.set_context(device_target='Ascend'); print('mindspore_ascend_ok')" 2>/dev/null
```

Missing package handling:
- remind the user to verify the Ascend system layer first
- resolve the compatible MindSpore target version from the exact local
  CANN-keyed MindSpore table and the current Python version
- if the local table cannot classify the tuple, check the official `https://www.mindspore.cn/versions` page for the detected release and print the detected MindSpore version, the official CANN pairing, and whether the Python support range is still unclear
- check the official `https://www.mindspore.cn/versions` page before
  recommending MindSpore installation when the local table cannot classify the
  tuple
- if the official page still leaves Python support or replacement safety
  unclear, keep the MindSpore path as `WARN` and ask the user to confirm the
  tuple before recommending installation
- install that version directly inside the selected `uv` environment, for
  example with `uv pip install --python <selected_python_path> mindspore==<resolved_version>`
- after installation, re-run the MindSpore import and Ascend context smoke test

Installed package handling:
- compare the installed version against the exact local MindSpore
  compatibility row for the detected CANN version and current Python version
- if the local table cannot classify the tuple, check the official
  `https://www.mindspore.cn/versions` page before deciding whether the tuple
  is unknown
- classify installed-and-compatible MindSpore as `PASS`
- classify installed-but-incompatible MindSpore as `FAIL`
- if a compatible replacement can be derived from the local table:
  - print the detected CANN version, current MindSpore version, and the
    recommended compatible MindSpore version
  - ask for confirmation before replacing the package inside the selected `uv`
    environment
  - after replacement, re-run the import and Ascend context smoke test before
    reclassifying MindSpore as healthy
- if the official page confirms the CANN pairing but does not clearly confirm
  Python support or replacement safety:
  - print the detected MindSpore version, the official CANN pairing, and the
    missing Python confirmation
  - keep the MindSpore path as `WARN`
  - ask the user to confirm the tuple before recommending installation or
    replacement
- if no compatible replacement can be derived:
  - keep the MindSpore path as `WARN`
  - tell the user the local compatibility table could not resolve a supported
    MindSpore replacement for the detected CANN version
  - tell the user to verify the official `versions` page before replacing
    MindSpore

Import or smoke-test dependency handling:
- if the import or smoke test fails because a Python package is missing:
- if the missing package name is clear from the error, install it directly
  inside the selected `uv` environment with `uv pip install --python <selected_python_path> <package>`
- if the package name cannot be identified with high confidence, stop and
  report the unresolved dependency name instead of guessing
- re-run the failed check before classifying the MindSpore path

## PTA Path

Run:

```bash
uv run --python <selected_python_path> python -c "import torch; print(torch.__version__)" 2>/dev/null
uv run --python <selected_python_path> python -c "import torch_npu; print(torch_npu.__version__)" 2>/dev/null
uv run --python <selected_python_path> python -c "import torch, torch_npu; x=torch.tensor([1.0]).npu(); print('torch_npu_ok', x)" 2>/dev/null
```

Normalize versions before lookup:
- compare `torch` on `major.minor.patch`
- compare `torch_npu` on the full version, including `.postN`
- treat local build suffixes such as `+cpu` on `torch` as packaging detail
  rather than a compatibility key

Missing package handling:
- remind the user to verify the Ascend system layer first
- resolve the compatible PTA target tuple for the detected CANN version and
  current Python version
- install the missing framework package or tuple directly inside the selected
  `uv` environment, for example with `uv pip install --python <selected_python_path> torch==<resolved_torch>` and
  `uv pip install --python <selected_python_path> torch_npu==<resolved_torch_npu>`
- after installation, re-run the PTA import and NPU tensor smoke test

Installed package handling:
- compare the installed PTA tuple against the exact CANN-keyed local PTA table
- if the local PTA table cannot classify the tuple, use the remote fallback
  path before deciding whether the tuple is unknown
- classify installed-and-compatible PTA as `PASS`
- classify installed-but-incompatible PTA as `FAIL`
- classify unresolved PTA after local and remote lookup as `WARN`
- when classifying incompatibility, explain whether the mismatch is caused by:
  - Python version
  - `torch` / `torch_npu` version pairing
  - `torch_npu` `.postN` build mismatch for the detected CANN version
- if a compatible replacement tuple can be derived locally or remotely:
  - print the detected CANN version, current PTA tuple, and the recommended
    compatible tuple
  - ask for confirmation before replacing PTA packages inside the selected `uv`
    environment
  - after replacement, re-run `torch`, `torch_npu`, and the NPU tensor smoke
    test before reclassifying PTA as healthy
- if no compatible replacement can be derived:
  - keep the PTA path as `WARN`
  - tell the user to verify the current PTA release notes

Import or smoke-test dependency handling:
- if the import or smoke test fails because a Python package is missing:
- if the missing package name is clear from the error, install it directly
  inside the selected `uv` environment with `uv pip install --python <selected_python_path> <package>`
- if the package name cannot be identified with high confidence, stop and
  report the unresolved dependency name instead of guessing
- re-run the failed PTA check before classifying the framework path

Unknown PTA tuple handling:
If the exact PTA tuple remains unresolved after local and remote lookup:
- classify the PTA path as `WARN`
- do not auto-remediate PTA packages
- tell the user to verify the current PTA release notes before installing or
  replacing packages

## Replacement Policy

- only replace packages inside the user-confirmed `uv` environment
- never modify the system interpreter
- never install, upgrade, or downgrade CANN
- never replace packages without user confirmation
- if the current Python version cannot satisfy any compatible framework tuple
  for the detected CANN version, stop package remediation and tell the user to
  recreate the `uv` environment with a compatible Python version

## Runtime Dependency Checks

Run these package checks in the selected environment:

```bash
uv run --python <selected_python_path> python -c "import transformers; print(transformers.__version__)" 2>/dev/null
uv run --python <selected_python_path> python -c "import tokenizers; print(tokenizers.__version__)" 2>/dev/null
uv run --python <selected_python_path> python -c "import datasets; print(datasets.__version__)" 2>/dev/null
uv run --python <selected_python_path> python -c "import accelerate; print(accelerate.__version__)" 2>/dev/null
uv run --python <selected_python_path> python -c "import safetensors; print(safetensors.__version__)" 2>/dev/null
uv run --python <selected_python_path> python -c "import diffusers; print(diffusers.__version__)" 2>/dev/null
```

Policy:
- `transformers`, `tokenizers`, `datasets`, `accelerate`, and `safetensors`
  are standard runtime checks
- require `diffusers` when `task_type=diffusion`
- install missing runtime dependencies directly inside the selected `uv`
  environment with `uv pip install --python <selected_python_path> <package>`
- if a framework smoke test or import fails with `ModuleNotFoundError` or
  `ImportError` for an installable Python package, install that dependency
  directly inside the selected `uv` environment with
  `uv pip install --python <selected_python_path> <package>` and re-run the failed
  check
- do not guess a package name when the import error is ambiguous
- ask for confirmation only when creating a new `uv` environment
