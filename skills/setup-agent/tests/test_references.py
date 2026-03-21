import re
from pathlib import Path

import yaml


SKILL_ROOT = Path(__file__).resolve().parents[1]
REFERENCES_DIR = SKILL_ROOT / "references"
SKILL_MD = SKILL_ROOT / "SKILL.md"
SKILL_YAML = SKILL_ROOT / "skill.yaml"
ROOT_AGENTS = SKILL_ROOT.parents[1] / "AGENTS.md"


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def read_yaml(path: Path):
    return yaml.safe_load(read_text(path))


def test_skill_references_only_ascend_compat():
    content = read_text(SKILL_MD)
    assert "references/ascend-compat.md" in content
    assert "references/nvidia-compat.md" not in content
    assert "references/execution-contract.md" in content


def test_ascend_reference_exists_and_has_required_sections():
    path = REFERENCES_DIR / "ascend-compat.md"
    content = read_text(path)
    assert path.exists()
    assert "Driver / Firmware / CANN Matrix" in content
    assert "MindSpore on Ascend" in content
    assert "PyTorch + torch_npu on Ascend" in content
    assert "Official Installation Guides" in content


def test_ascend_reference_has_torch_npu_rows():
    content = read_text(REFERENCES_DIR / "ascend-compat.md")
    rows = re.findall(r"^\|\s*2\.\d+\.x\s*\|\s*2\.\d+\.x\s*\|", content, re.MULTILINE)
    assert len(rows) >= 3, f"Expected >=3 torch/torch_npu matrix rows, found {len(rows)}"


def test_execution_reference_exists_and_has_required_sections():
    path = REFERENCES_DIR / "execution-contract.md"
    content = read_text(path)
    assert path.exists()
    assert "Streaming Console Output" in content
    assert "Report Artifacts" in content
    assert "Final Mailbox Summary" in content


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


def test_skill_points_missing_ascend_components_to_hiascend_download_portal():
    skill_content = read_text(SKILL_MD)
    ref_content = read_text(REFERENCES_DIR / "ascend-compat.md")
    url = "https://www.hiascend.com/cann/download"
    assert url in skill_content
    assert url in ref_content
    assert "If MindSpore is missing:" in skill_content
    assert "If `torch` or `torch_npu` is missing:" in skill_content


def test_skill_uses_task_type_to_gate_runtime_checks():
    content = read_text(SKILL_MD)
    assert "are standard runtime checks" in content
    assert "`transformers`, `tokenizers`, `datasets`, `accelerate`, and `safetensors`" in content
    assert "require `diffusers` when `task_type=diffusion`" in content


def test_skill_adds_model_first_workdir_artifact_phase():
    content = read_text(SKILL_MD)
    assert "## Gate 7. Model-First Workspace Checks" in content
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
    content = read_text(SKILL_MD)
    assert "If no candidate model directory exists, or the user declines all candidates:" in content
    assert "- ask the user which Hugging Face model to download" in content
    assert "use `huggingface_hub.snapshot_download` inside the selected `uv` environment" in content
    assert "download into `<workdir>/models/<repo_name>` by default unless `model_root`" in content
    assert "is already specified" in content
    assert "if the repo is gated or private and authentication is missing, stop and" in content
    assert "report a download/auth failure" in content
    assert "snapshot_download(repo_id='org/model'" in content


def test_skill_checks_training_scripts_and_checkpoints_after_model_selection():
    content = read_text(SKILL_MD)
    assert 'find . -type f -name "*.py" 2>/dev/null' in content
    assert '-name "*.ckpt"' in content
    assert '-name "*.pt"' in content
    assert '-name "*.pth"' in content
    assert '-name "*.bin"' in content
    assert '-name "*.safetensors"' in content
    assert "print and record the matched training script paths and" in content
    assert "checkpoint paths" in content
    assert "Record whether the selected model came from:" in content


def test_skill_classifies_workspace_artifacts_by_task_type():
    content = read_text(SKILL_MD)
    assert "if `task_type=training`, training script check is `PASS`" in content
    assert "if `task_type=inference`, missing training scripts are `INFO` rather" in content


def test_skill_guides_huggingface_download_when_artifacts_are_missing():
    content = read_text(SKILL_MD)
    assert "do not reclassify the Ascend driver/CANN/framework setup as failed" in content
    assert "ask the user which Hugging Face model to download" in content
    assert "tell the user exactly which artifacts are absent" in content


def test_skill_reports_both_framework_paths():
    content = read_text(SKILL_MD)
    assert "### MindSpore path" in content
    assert "### PTA path (`torch` + `torch_npu`)" in content
    assert "If both framework paths are unhealthy, report both independently" in content


def test_skill_documents_standard_reporting_contract():
    content = read_text(REFERENCES_DIR / "execution-contract.md")
    assert "runs/<run_id>/out/" in content
    assert "report.json" in content
    assert "report.md" in content
    assert "meta/env.json" in content
    assert "meta/inputs.json" in content
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
    assert "- download/auth failure reason" in content


def test_skill_requires_streaming_console_output():
    content = read_text(REFERENCES_DIR / "execution-contract.md")
    assert "## Streaming Console Output" in content
    assert "emit a `checking ...` line before every major step" in content
    assert "emit a `passed`, `failed`, `warn`, or `skip` line after each step" in content
    assert "Major steps that must stream:" in content
    assert "setup-agent : checking work dir..." in content
    assert "setup-agent : work dir passed: /path/to/current/workdir" in content
    assert "- local model directories" in content
    assert "- model selection" in content
    assert "- hugging face download" in content
    assert "- training scripts" in content
    assert "- checkpoint files" in content
    assert "setup-agent : training scripts passed: ./train.py, ./scripts/finetune.py" in content
    assert "setup-agent : checkpoint files passed: ./weights/model.safetensors" in content


def test_skill_requires_mailbox_style_final_summary():
    content = read_text(REFERENCES_DIR / "execution-contract.md")
    assert "print a mailbox-style final summary to the console even" in content
    assert "The final summary must include:" in content
    assert "- current work dir" in content
    assert "- which components are already installed" in content
    assert "- selected model path when present" in content
    assert "- whether the selected model was reused locally or downloaded" in content
    assert "- matched training script paths when present" in content
    assert "- matched checkpoint paths when present" in content
    assert "- the failure reason if the run failed" in content
    assert "setup-agent : final summary" in content
    assert "- /path/to/current/workdir" in content
    assert "selected_model:" in content
    assert "model_source:" in content
    assert "training_scripts:" in content
    assert "checkpoint_files:" in content
    assert "download missing scripts or checkpoints from Hugging Face into the current work dir" in content


def test_manifest_matches_ascend_only_scope_and_permissions():
    manifest = read_yaml(SKILL_YAML)
    assert manifest["permissions"]["network"] == "required"
    assert manifest["permissions"]["filesystem"] == "workspace-write"
    assert manifest["composes"] == []
    assert "torch_npu" in manifest["tags"]
    assert "nvidia" not in manifest["tags"]
    choices = manifest["inputs"][0]["choices"]
    assert choices == ["local"]


def test_manifest_declares_uv_and_framework_inputs():
    manifest = read_yaml(SKILL_YAML)
    input_names = {item["name"] for item in manifest["inputs"]}
    assert {"target", "frameworks", "task_type", "uv_env_mode", "python_version", "model_id", "model_root"} <= input_names


def test_root_agents_exposes_setup_agent():
    content = read_text(ROOT_AGENTS)
    assert "| setup-agent | skills/setup-agent/ |" in content
    assert "**setup-agent**" in content
    assert "torch_npu" in content
