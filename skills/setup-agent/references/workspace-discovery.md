# Workspace Discovery Workflow

Load this reference during Gate 7.

Use it when:
- searching for existing local model directories
- deciding whether to reuse a local model or download from Hugging Face
- scanning for candidate training scripts
- scanning for checkpoints
- classifying workspace completeness for training or inference tasks

Use `references/execution-contract.md` for reporting requirements.

## Model-First Policy

Always look for existing local model directories before considering any
Hugging Face download.

### Find local model directories

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

### Download only when no local model directory is selected

If no candidate model directory exists, or the user declines all candidates:
- ask the user which Hugging Face model to download
- ask for confirmation before downloading
- use `huggingface_hub.snapshot_download` inside the selected `uv` environment
- download into `<workdir>/models/<repo_name>` by default unless `model_root`
  is already specified
- if the direct Hugging Face download fails because of DNS, timeout, proxy, or
  other network reachability problems, retry with a China mirror by setting
  `HF_ENDPOINT=https://hf-mirror.com`
- if the repo is gated or private and authentication is missing, stop and
  report a download/auth failure instead of guessing
- do not treat authentication or permission failures as mirror candidates
- after download, verify that the target directory exists and contains model
  markers before classifying it as usable

Example download pattern:

```bash
uv run --python <selected_python_path> python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='org/model', local_dir='./models/model', local_dir_use_symlinks=False)"
```

Mirror retry pattern:

```bash
HF_ENDPOINT=https://hf-mirror.com uv run --python <selected_python_path> python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='org/model', local_dir='./models/model', local_dir_use_symlinks=False)"
```

Record whether the selected model came from:
- a local directory
- a Hugging Face download

## Script and Checkpoint Discovery

Use two separate roots:
- current work dir for training script discovery
- selected local or downloaded model directory for checkpoint discovery

Run:

```bash
find . \
  \( -path "*/.venv" -o -path "*/.git" -o -path "*/__pycache__" -o -path "*/.cache" -o -path "*/node_modules" \) -prune \
  -o -type f \
  \( -iname "train*.py" -o -iname "finetune*.py" -o -iname "run*.py" -o -iname "main*.py" -o -path "*/scripts/train*.py" -o -path "*/scripts/finetune*.py" \) \
  -print 2>/dev/null
find "<selected_model_dir>" -type f \( -name "*.ckpt" -o -name "*.pt" -o -name "*.pth" -o -name "*.bin" -o -name "*.safetensors" \) 2>/dev/null
```

Classification:
- if `task_type=training`, training script check is `PASS` if one or more
  candidate training entry scripts exist, otherwise `FAIL`
- if `task_type=inference`, missing training scripts are `INFO` rather than
  `FAIL`
- checkpoint check is `PASS` if one or more `.ckpt`, `.pt`, `.pth`, `.bin`,
  or `.safetensors` files exist, otherwise `FAIL`
- do not treat arbitrary utility or test Python files as training scripts
- if files are found, print and record the matched training script paths and
  checkpoint paths

If the selected workspace is missing training scripts or checkpoint files:
- do not reclassify the Ascend driver/CANN/framework setup as failed
- report it as a workspace-preparation failure or partial result
- if multiple candidate training scripts exist, show the list and ask the user
  which one is the intended entry script
- if a selected model directory exists but required artifacts are still
  missing, tell the user exactly which artifacts are absent
