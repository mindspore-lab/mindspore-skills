from pathlib import Path

import yaml


SKILL_ROOT = Path(__file__).resolve().parents[1]
REFERENCES_DIR = SKILL_ROOT / "references"
SCRIPTS_DIR = SKILL_ROOT / "scripts"
SKILL_MD = SKILL_ROOT / "SKILL.md"
SKILL_YAML = SKILL_ROOT / "skill.yaml"
ROOT_AGENTS = SKILL_ROOT.parents[1] / "AGENTS.md"


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def read_yaml(path: Path):
    return yaml.safe_load(read_text(path))


def test_skill_md_stays_compact_after_refactor():
    lines = read_text(SKILL_MD).splitlines()
    assert len(lines) <= 360


def test_skill_references_required_files():
    content = read_text(SKILL_MD)
    assert "references/ascend-compat.md" in content
    assert "references/framework-remediation.md" in content
    assert "references/workspace-discovery.md" in content
    assert "references/nvidia-compat.md" not in content
    assert "references/execution-contract.md" in content
    assert "scripts/pta_compat_lookup.py" in content


def test_ascend_reference_exists_and_has_required_sections():
    path = REFERENCES_DIR / "ascend-compat.md"
    content = read_text(path)
    assert path.exists()
    assert "Driver / Firmware / CANN Matrix" in content
    assert "Compatibility Source Policy" in content
    assert "MindSpore on Ascend" in content
    assert "PyTorch + torch_npu on Ascend" in content
    assert "Local PTA Compatibility Table" in content
    assert "Official Installation Guides" in content


def test_ascend_reference_cann_matrix_covers_80rc1_through_85():
    content = read_text(REFERENCES_DIR / "ascend-compat.md")
    assert "| 8.5.0 | 25.0.X |" in content
    assert "| 8.3.RC1 | 25.3.rc1 |" in content
    assert "| 8.2.RC1 | 25.2.0 |" in content
    assert "| 8.1.RC1 | 24.1.rc3 |" in content
    assert "| 8.0.RC3 | 24.1.rc2 |" in content
    assert "| 8.0.RC2 | 24.1.rc1 |" in content
    assert "| 8.0.RC1 | 23.0.6 |" in content
    assert "| 7.3.0 | 23.0.5 |" not in content
    assert "| 7.1.0 | 23.0.3 |" not in content


def test_ascend_reference_has_exact_pta_rows_for_26_to_29():
    content = read_text(REFERENCES_DIR / "ascend-compat.md")
    assert "| 8.5.0 | 2.9.0 | 2.9.0 | 3.9-3.11 | v2.9.0-7.3.0 |" in content
    assert "| 8.5.0 | 2.8.0 | 2.8.0.post2 | 3.9-3.11 | v2.8.0-7.3.0 |" in content
    assert "| 8.5.0 | 2.7.1 | 2.7.1.post2 | 3.9-3.11 | v2.7.1-7.3.0 |" in content
    assert "| 8.5.0 | 2.6.0 | 2.6.0.post5 | 3.9-3.11 | v2.6.0-7.3.0 |" in content


def test_ascend_reference_limits_local_pta_table_to_81rc1_through_85():
    content = read_text(REFERENCES_DIR / "ascend-compat.md")
    assert "| 8.1.RC1 | 2.5.1 | 2.5.1 | 3.9-3.11 | v2.5.1-7.0.0 |" in content
    assert "| 8.0.0 | 2.4.0 | 2.4.0.post2 |" not in content
    assert "| 8.0.RC3 | 2.4.0 | 2.4.0 |" not in content
    assert "| 8.0.RC2 | 2.3.1 | 2.3.1 |" not in content
    assert "| 8.0.RC1 | 2.2.0 | 2.2.0 |" not in content
    assert "| 7.0.0 | 2.1.0 | 2.1.0 |" not in content


def test_ascend_reference_documents_pta_lookup_order_and_remote_fallback():
    content = read_text(REFERENCES_DIR / "ascend-compat.md")
    assert "1. `Local PTA Compatibility Table`" in content
    assert "2. upstream `Ascend/pytorch` README remote fallback" in content
    assert "3. if still unresolved, mark the tuple `WARN` and stop PTA auto-remediation" in content
    assert "do not mutate this local reference file during a normal `setup-agent` run" in content
    assert "verify the current PTA release" in content
    assert "notes before installation" in content


def test_ascend_reference_documents_mindspore_replacement_guidance():
    content = read_text(REFERENCES_DIR / "ascend-compat.md")
    assert "| CANN | MindSpore | Python | Typical Use |" in content
    assert "if the installed MindSpore version is incompatible but a compatible local" in content
    assert "replacement can be derived, recommend replacement inside the selected `uv`" in content
    assert "installed version incompatible but replacement available locally" in content


def test_ascend_reference_uses_local_first_mindspore_policy_with_versions_lookup():
    content = read_text(REFERENCES_DIR / "ascend-compat.md")
    assert "1. local MindSpore compatibility table" in content
    assert "2. official `https://www.mindspore.cn/versions` page for the detected release" in content
    assert "user-confirmed reference" in content
    assert "keep the MindSpore path `WARN`" in content
    assert "local table only" not in content
    assert "do not fetch remote compatibility data for MindSpore during a normal" not in content


def test_ascend_reference_has_exact_mindspore_rows_for_270rc1_to_280():
    content = read_text(REFERENCES_DIR / "ascend-compat.md")
    assert "| 8.5.0 | 2.8.0 | 3.9-3.12 |" in content
    assert "| 8.5.0 | 2.7.2 | 3.9-3.12 |" in content
    assert "| 8.3.RC1 | 2.8.0 | 3.9-3.12 |" in content
    assert "| 8.3.RC1 | 2.7.1 | 3.9-3.11 |" in content
    assert "| 8.2.RC1 | 2.7.0 | 3.9-3.11 |" in content
    assert "| 8.2.RC1 | 2.7.0-rc1 | 3.9-3.11 |" in content


def test_ascend_reference_marks_mindspore_python_unknowns_as_warn():
    content = read_text(REFERENCES_DIR / "ascend-compat.md")
    assert "if the official page does not clearly publish Python support for that row," in content
    assert "local row or official lookup that still requires manual Python confirmation:" in content
    assert "`WARN`" in content


def test_execution_reference_exists_and_has_required_sections():
    path = REFERENCES_DIR / "execution-contract.md"
    content = read_text(path)
    assert path.exists()
    assert "Streaming Console Output" in content
    assert "Console Contract" in content
    assert "Final Mailbox Summary" in content


def test_framework_remediation_reference_exists_and_has_required_sections():
    path = REFERENCES_DIR / "framework-remediation.md"
    content = read_text(path)
    assert path.exists()
    assert "## Framework Resolution Order" in content
    assert "## MindSpore Path" in content
    assert "## PTA Path" in content
    assert "## Replacement Policy" in content
    assert "## Runtime Dependency Checks" in content


def test_workspace_discovery_reference_exists_and_has_required_sections():
    path = REFERENCES_DIR / "workspace-discovery.md"
    content = read_text(path)
    assert path.exists()
    assert "## Model-First Policy" in content
    assert "### Find local model directories" in content
    assert "### Download only when no local model directory is selected" in content
    assert "## Script and Checkpoint Discovery" in content


def test_pta_lookup_script_exists():
    path = SCRIPTS_DIR / "pta_compat_lookup.py"
    content = read_text(path)
    assert path.exists()
    assert "REMOTE_README" in content
    assert "parse_local_table" in content
    assert "parse_remote_table" in content
    assert "--remote-fallback" in content


def test_skill_no_longer_mentions_gpu_or_nvidia_path():
    content = read_text(SKILL_MD)
    assert "This skill is Ascend-only." in content
    assert "Nvidia or CUDA environment setup" in content
    assert "remote SSH workflows" in content
    assert "## Remote Environments" not in content
    assert "### Step 1 — Detect Hardware" not in content


def test_skill_requires_uv_before_python_installs():
    content = read_text(SKILL_MD)
    assert "All Python package checks and installs happen only after `uv` is confirmed" in content
    assert "Never install Python packages into the system interpreter." in content
    assert "`uv` is healthy only when both `command -v uv` and `uv --version` succeed." in content
    assert "Do not maintain step-by-step run logs during environment checking." in content


def test_skill_uses_uv_scoped_python_examples():
    content = read_text(SKILL_MD)
    assert "uv run --python .venv/bin/python python -V" in content
    assert "selected_env_root" in content
    assert "selected_python_path" in content
    assert "uv run --python <selected_python_path> python -c \"import mindspore as ms; print(ms.__version__)\"" in content
    assert "uv run --python <selected_python_path> python <setup_agent_skill_root>/scripts/pta_compat_lookup.py" in content
    assert "not from the user work dir" in content
    assert "<selected_python>" not in content


def test_framework_remediation_uses_uv_scoped_python_and_installs():
    content = read_text(REFERENCES_DIR / "framework-remediation.md")
    assert "uv run --python <selected_python_path> python -c \"import mindspore as ms; print(ms.__version__)\"" in content
    assert "uv run --python <selected_python_path> python -c \"import torch; print(torch.__version__)\"" in content
    assert "uv pip install --python <selected_python_path> mindspore==<resolved_version>" in content
    assert "uv pip install --python <selected_python_path> <package>" in content
    assert "uv run --python <selected_python_path> python <setup_agent_skill_root>/scripts/pta_compat_lookup.py" in content
    assert "Do not resolve the helper path relative" in content
    assert "to the user work dir." in content
    assert "<selected_python>" not in content


def test_execution_contract_uses_uppercase_status_examples():
    content = read_text(REFERENCES_DIR / "execution-contract.md")
    assert "setup-agent : os PASS:" in content
    assert "setup-agent : npu visibility FAIL:" in content
    assert "setup-agent : training scripts PASS:" in content


def test_skill_has_explicit_confirmation_policy():
    content = read_text(SKILL_MD)
    assert "## Confirmation Policy" in content
    assert "- installing `uv`" in content
    assert "- creating a new `uv` environment" in content
    assert "- replacing an already installed `mindspore`, `torch`, or `torch_npu`" in content
    assert "- downloading a model from Hugging Face" in content
    assert "After the user has confirmed the target `uv` environment, you MAY do these" in content
    assert "- install a missing `mindspore` package inside that environment" in content
    assert "- install missing runtime Python dependencies inside that environment" in content


def test_skill_uses_current_path_as_default_workdir():
    content = read_text(SKILL_MD)
    assert "Treat the current shell path as the default work dir." in content
    assert "pwd" in content
    assert "Record and report the resolved work dir before `uv` environment discovery." in content


def test_skill_forbids_auto_installing_driver_and_cann():
    content = read_text(SKILL_MD)
    assert "Never auto-install or upgrade:" in content
    assert "- NPU driver" in content
    assert "- CANN toolkit" in content


def test_skill_requires_confirming_uv_env_choice_and_python_version():
    content = read_text(SKILL_MD)
    assert "ask the user whether to reuse an existing environment or create a new one" in content
    assert "ask which Python version to use" in content


def test_skill_requires_uv_to_be_directly_resolvable_after_install():
    content = read_text(SKILL_MD)
    assert "If `uv` is missing or `uv --version` fails" in content
    assert "official installer" in content
    assert "bash -lc 'command -v uv && uv --version'" in content
    assert "ask for confirmation before editing the shell profile" in content
    assert "Do not classify `uv` as healthy merely because files were installed." in content


def test_skill_checks_python_only_after_entering_uv():
    content = read_text(SKILL_MD)
    assert "Only after entering the selected `uv` environment, check Python-related facts:" in content
    assert "python -V" in content
    assert 'python -c "import sys; print(sys.executable)"' in content
    assert "Do not check or report Python runtime readiness before the NPU-related system" in content
    assert "python3 --version 2>/dev/null" not in content


def test_skill_stops_before_package_install_when_system_layer_fails():
    content = read_text(SKILL_MD)
    assert "If driver or CANN is missing or unusable:" in content
    assert "- stop before `uv` package remediation" in content
    assert "If sourcing fails:" in content
    assert "- report a system-layer failure" in content
    assert "- stop before framework installs" in content


def test_skill_skips_driver_and_cann_checks_when_no_npu_is_detected():
    content = read_text(SKILL_MD)
    assert "If no NPU card is detected:" in content
    assert "- skip later driver and CANN checks" in content


def test_skill_points_missing_system_components_to_hiascend_download_portal():
    skill_content = read_text(SKILL_MD)
    ref_content = read_text(REFERENCES_DIR / "ascend-compat.md")
    url = "https://www.hiascend.com/cann/download"
    assert url in skill_content
    assert url in ref_content


def test_skill_installs_missing_frameworks_inside_uv():
    content = read_text(REFERENCES_DIR / "framework-remediation.md")
    assert "Missing package handling:" in content
    assert "`uv pip install --python <selected_python_path> mindspore==<resolved_version>`" in content
    assert "`uv pip install --python <selected_python_path> torch==<resolved_torch>`" in content
    assert "`uv pip install --python <selected_python_path> torch_npu==<resolved_torch_npu>`" in content


def test_skill_uses_task_type_to_gate_runtime_checks():
    content = read_text(REFERENCES_DIR / "framework-remediation.md")
    assert "are standard runtime checks" in content
    assert "`transformers`, `tokenizers`, `datasets`, `accelerate`, and `safetensors`" in content
    assert "require `diffusers` when `task_type=diffusion`" in content
    assert "install missing runtime dependencies directly inside the selected `uv`" in content
    assert "`ModuleNotFoundError` or" in content
    assert "`ImportError` for an installable Python package" in content
    assert "`uv pip install --python <selected_python_path> <package>`" in content


def test_skill_adds_model_first_workdir_artifact_phase():
    skill_content = read_text(SKILL_MD)
    content = read_text(REFERENCES_DIR / "workspace-discovery.md")
    assert "## Gate 7. Model-First Workspace Checks" in skill_content
    assert "Always look for existing local model directories before" in content
    assert "candidate model" in content
    assert "directories exist" in content
    assert '-name "config.json"' in content
    assert '-name "tokenizer.json"' in content
    assert '-name "model.safetensors"' in content
    assert ".venv" in content
    assert "ask which model directory to use" in content
    assert "do not download from Hugging Face unless the user explicitly declines the" in content
    assert "local candidates" in content


def test_skill_uses_snapshot_download_when_no_local_model_directory_exists():
    content = read_text(REFERENCES_DIR / "workspace-discovery.md")
    assert "If no candidate model directory exists, or the user declines all candidates:" in content
    assert "- ask the user which Hugging Face model to download" in content
    assert "use `huggingface_hub.snapshot_download` inside the selected `uv` environment" in content
    assert "download into `<workdir>/models/<repo_name>` by default unless `model_root`" in content
    assert "is already specified" in content
    assert "`HF_ENDPOINT=https://hf-mirror.com`" in content
    assert "if the direct Hugging Face download fails because of DNS, timeout, proxy, or" in content
    assert "other network reachability problems, retry with a China mirror" in content
    assert "if the repo is gated or private and authentication is missing, stop and" in content
    assert "do not treat authentication or permission failures as mirror candidates" in content
    assert "report a download/auth failure" in content
    assert "snapshot_download(repo_id='org/model'" in content
    assert "HF_ENDPOINT=https://hf-mirror.com uv run --python <selected_python_path>" in content


def test_skill_checks_training_scripts_and_checkpoints_after_model_selection():
    content = read_text(REFERENCES_DIR / "workspace-discovery.md")
    assert '-iname "train*.py"' in content
    assert '-iname "finetune*.py"' in content
    assert '-iname "run*.py"' in content
    assert '<selected_model_dir>' in content
    assert '-name "*.ckpt"' in content
    assert '-name "*.pt"' in content
    assert '-name "*.pth"' in content
    assert '-name "*.bin"' in content
    assert '-name "*.safetensors"' in content
    assert "print and record the matched training script paths and" in content
    assert "checkpoint paths" in content
    assert "Record whether the selected model came from:" in content
    assert "do not treat arbitrary utility or test Python files as training scripts" in content


def test_skill_classifies_workspace_artifacts_by_task_type():
    content = read_text(REFERENCES_DIR / "workspace-discovery.md")
    assert "if `task_type=training`, training script check is `PASS`" in content
    assert "if `task_type=inference`, missing training scripts are `INFO` rather" in content
    assert "candidate training entry scripts exist" in content


def test_skill_guides_huggingface_download_when_artifacts_are_missing():
    content = read_text(REFERENCES_DIR / "workspace-discovery.md")
    assert "do not reclassify the Ascend driver/CANN/framework setup as failed" in content
    assert "ask the user which Hugging Face model to download" in content
    assert "tell the user exactly which artifacts are absent" in content
    assert "if multiple candidate training scripts exist, show the list and ask the user" in content


def test_skill_reports_both_framework_paths():
    content = read_text(SKILL_MD)
    assert "### MindSpore path" in content
    assert "### PTA path (`torch` + `torch_npu`)" in content
    assert "If both framework paths are unhealthy, report both independently" in content


def test_skill_uses_cann_first_framework_resolution():
    skill_content = read_text(SKILL_MD)
    content = read_text(REFERENCES_DIR / "framework-remediation.md")
    assert "Load `references/framework-remediation.md` before changing framework" in skill_content
    assert "Treat the detected CANN version as the primary selector for" in content
    assert "1. Detect the current CANN version from the system-layer evidence" in content
    assert "2. Detect the interpreter version from `selected_python_path`" in content
    assert "3. Resolve compatible framework candidates from" in content
    assert "`references/ascend-compat.md`" in content
    assert "4. For MindSpore only, if the local table does not classify the tuple, look up" in content
    assert "5. For PTA only, if the local table does not classify the tuple, prefer the" in content
    assert "6. Compare the installed framework version against the compatible candidate set" in content
    assert "7. Run the framework smoke test only after compatibility classification" in content
    assert "For each framework path, use this remediation order:" in content
    assert "1. Resolve the compatible target version from the detected CANN version and the" in content
    assert "4. If the framework is installed but incompatible, ask for confirmation before" in content


def test_skill_and_remediation_require_versions_lookup_before_mindspore_replacement():
    skill_content = read_text(SKILL_MD)
    content = read_text(REFERENCES_DIR / "framework-remediation.md")
    assert "For MindSpore only, if the local table cannot classify the tuple, check the" in skill_content
    assert "official `https://www.mindspore.cn/versions` page" in skill_content
    assert "ask the user to confirm before" in skill_content
    assert "check the official `https://www.mindspore.cn/versions` page" in content
    assert "the official CANN pairing, and whether the Python support range is still unclear" in content
    assert "official page confirms the CANN pairing but does not clearly confirm" in content
    assert "keep the MindSpore path as `WARN`" in content


def test_skill_documents_framework_replacement_after_confirmation():
    content = read_text(REFERENCES_DIR / "framework-remediation.md")
    assert "ask for confirmation before replacing the package inside the selected `uv`" in content
    assert "ask for confirmation before replacing PTA packages inside the selected `uv`" in content
    assert "never replace packages without user confirmation" in content
    assert "recreate the `uv` environment with a compatible Python version" in content
    assert "print the detected CANN version, current PTA tuple, and the recommended" in content
    assert "compatible tuple" in content


def test_skill_classifies_unknown_pta_after_remote_fallback_as_warn():
    content = read_text(REFERENCES_DIR / "framework-remediation.md")
    assert "If the exact PTA tuple remains unresolved after local and remote lookup:" in content
    assert "- classify the PTA path as `WARN`" in content
    assert "- do not auto-remediate PTA packages" in content
    assert "verify the current PTA release notes before installing or" in content


def test_skill_installs_missing_python_deps_during_framework_checks():
    content = read_text(REFERENCES_DIR / "framework-remediation.md")
    assert "if the import or smoke test fails because a Python package is missing:" in content
    assert "if the missing package name is clear from the error, install it directly" in content
    assert "if the package name cannot be identified with high confidence, stop and" in content
    assert "re-run the failed check before classifying the MindSpore path" in content
    assert "re-run the failed PTA check before classifying the framework path" in content


def test_skill_documents_console_only_contract():
    content = read_text(REFERENCES_DIR / "execution-contract.md")
    assert "Do not write `.md`, `.json`, or other result artifacts under `runs/`" in content
    assert "streamed console output plus the" in content
    assert "final boxed mailbox summary" in content
    assert "runs/<run_id>/out/" not in content
    assert "report.json" not in content
    assert "report.md" not in content
    assert "env_summary.md" not in content
    assert "- current work dir" in content
    assert "- `datasets`" in content
    assert "- `diffusers`" in content
    assert "- direct shell resolution status" in content
    assert "- local model directory findings" in content
    assert "- selected model path" in content
    assert "- selected model source (`local` or `huggingface`)" in content
    assert "- training scripts" in content
    assert "- checkpoint files" in content
    assert "- matched training script paths" in content
    assert "- matched checkpoint paths" in content
    assert "- China mirror fallback guidance using `HF_ENDPOINT=https://hf-mirror.com`" in content
    assert "- download/auth failure reason" in content
    assert "- detected CANN version used for framework compatibility resolution" in content
    assert "- framework compatibility reasoning" in content
    assert "- recommended compatible version(s)" in content
    assert "- whether a replacement was offered and whether the user confirmed it" in content
    assert "- selected environment root" in content
    assert "- selected interpreter path" in content
    assert "- direct `uv pip install --python <selected_python_path> ...` remediation inside the selected `uv` environment" in content
    assert "- Python packages installed to recover a failed framework import or smoke test" in content
    assert "- framework package installs or replacements performed inside the selected `uv`" in content
    assert "driver or" in content
    assert "toolkit is missing" in content


def test_ascend_reference_limits_cann_download_guidance_to_system_layer():
    content = read_text(REFERENCES_DIR / "ascend-compat.md")
    assert "If the Ascend driver or toolkit is missing" in content
    assert "If the Ascend driver, framework, or toolkit is missing" not in content
    assert "## Framework Package Remediation Policy" in content
    assert "install the compatible version resolved" in content
    assert "report the unresolved package name instead of guessing" in content


def test_skill_moves_workspace_and_framework_detail_out_of_skill_md():
    content = read_text(SKILL_MD)
    assert "Load `references/framework-remediation.md`" in content
    assert "Load `references/workspace-discovery.md`" in content
    assert '`pip install torch==<resolved_torch>`' not in content
    assert 'find "<selected_model_dir>" -type f' not in content


def test_skill_requires_streaming_console_output():
    content = read_text(REFERENCES_DIR / "execution-contract.md")
    assert "## Streaming Console Output" in content
    assert "emit a `checking ...` line before every major step" in content
    assert "emit a `PASS`, `FAIL`, `WARN`, or `SKIP` line after each step" in content
    assert "Major steps that must stream:" in content
    assert "setup-agent : checking work dir..." in content
    assert "setup-agent : work dir PASS: /path/to/current/workdir" in content
    assert "- local model directories" in content
    assert "- model selection" in content
    assert "- hugging face download" in content
    assert "- training scripts" in content
    assert "- checkpoint files" in content
    assert "- final mailbox summary" in content
    assert "setup-agent : training scripts PASS: ./train.py, ./scripts/finetune.py" in content
    assert "setup-agent : checkpoint files PASS: ./weights/model.safetensors" in content


def test_skill_requires_fixed_boxed_mailbox_summary():
    content = read_text(REFERENCES_DIR / "execution-contract.md")
    assert "print a final boxed mailbox summary to the console even" in content
    assert "use an ASCII box" in content
    assert "keep labels aligned" in content
    assert "use the title `setup-agent : Success` or `setup-agent : Fail`" in content
    assert "The final mailbox summary must include these fields in this exact order:" in content
    assert "- `workdir`" in content
    assert "- `device`" in content
    assert "- `uv`" in content
    assert "- `framework`" in content
    assert "- `model_deps`" in content
    assert "- `model`" in content
    assert "- `script`" in content
    assert "- `ckpt`" in content
    assert "- `fixed`" in content
    assert "- `failed`" in content
    assert "- `suggestion`" in content
    assert "| setup-agent : Success" in content
    assert "| workdir    :" in content
    assert "| device     :" in content
    assert "| framework  :" in content
    assert "| model_deps :" in content
    assert "| suggestion :" in content


def test_execution_contract_uses_env_summary_instead_of_run_logs():
    content = read_text(REFERENCES_DIR / "execution-contract.md")
    assert "`logs/run.log`" not in content
    assert "`logs/verify.log`" not in content
    assert "Do not require intermediate `run.log` or `verify.log` files" in content
    assert "the final mailbox summary." in content


def test_skill_uses_minimal_system_baseline_commands():
    content = read_text(SKILL_MD)
    assert "uname -a" in content
    assert "cat /etc/os-release 2>/dev/null" in content
    assert "ls /dev/davinci* 2>/dev/null" in content
    assert "npu-smi info 2>/dev/null" in content
    assert "npu-smi info -t board 2>/dev/null" not in content


def test_skill_points_final_output_to_fixed_mailbox_example():
    content = read_text(SKILL_MD)
    assert "a final boxed mailbox summary using the fixed example format from" in content
    assert "`references/execution-contract.md`" in content
    assert "do not write `.md` or `.json` result files during" in content


def test_manifest_matches_ascend_only_scope_and_permissions():
    manifest = read_yaml(SKILL_YAML)
    assert manifest["permissions"]["network"] == "required"
    assert manifest["permissions"]["filesystem"] == "workspace-write"
    assert manifest["composes"] == []
    assert "torch_npu" in manifest["tags"]
    assert "nvidia" not in manifest["tags"]
    assert "outputs" not in manifest
    choices = manifest["inputs"][0]["choices"]
    assert choices == ["local"]


def test_manifest_declares_uv_and_framework_inputs():
    manifest = read_yaml(SKILL_YAML)
    input_names = {item["name"] for item in manifest["inputs"]}
    assert {"target", "frameworks", "task_type", "uv_env_mode", "python_version", "model_id", "model_root", "hf_endpoint"} <= input_names


def test_root_agents_exposes_setup_agent():
    content = read_text(ROOT_AGENTS)
    assert "| setup-agent | skills/setup-agent/ |" in content
    assert "**setup-agent**" in content
    assert "torch_npu" in content
