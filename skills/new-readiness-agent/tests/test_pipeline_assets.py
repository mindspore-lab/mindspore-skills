import json
from pathlib import Path

from .helpers import check_by_id, current_field, run_pipeline, stdout_payload


def test_pipeline_surfaces_hf_asset_options_for_script_managed_dataset(tmp_path: Path, fake_selected_python: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "train_qwen3.py").write_text(
        "\n".join(
            [
                "from datasets import load_dataset",
                "from transformers import AutoModelForCausalLM, TrainingArguments",
                'TrainingArguments(output_dir="qwen3-finetuned")',
                'dataset = load_dataset("karthiksagarn/astro_horoscope", split="train")',
                'model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (workspace / "huggingface-cache" / "datasets" / "karthiksagarn___astro_horoscope").mkdir(parents=True)

    summary = stdout_payload(
        run_pipeline(
            "--working-dir",
            str(workspace),
            "--output-dir",
            str(tmp_path / "out"),
            "--target",
            "training",
            "--framework-hint",
            "pta",
            "--launcher-hint",
            "python",
            "--selected-python",
            str(fake_selected_python),
            "--entry-script",
            "train_qwen3.py",
            "--confirm",
            "config_asset=inline_config",
            "--confirm",
            "model_asset=hf_hub:Qwen/Qwen3-0.6B",
            cwd=workspace,
        )
    )

    assert current_field(summary) == "dataset_asset"
    options = summary["current_confirmation"]["options"]
    portable_options = summary["current_confirmation"]["portable_question"]["options"]
    source_types = {str(option.get("source_type")) for option in options if option.get("source_type")}
    assert "hf_cache" in source_types
    assert "hf_hub" in source_types
    assert "script_managed_remote" not in source_types
    assert any((option.get("locator") or {}).get("repo_id") == "karthiksagarn/astro_horoscope" for option in options if isinstance(option.get("locator"), dict))
    assert any(
        "Detected from the entry script." in str(option.get("description") or "")
        for option in portable_options
        if str(option.get("value") or "").startswith("dataset-")
    )
    unknown_portable = next(
        option for option in portable_options if str(option.get("value") or "") == "__unknown__"
    )
    assert len(portable_options) <= 4
    assert "__manual__" not in {str(option.get("value")) for option in portable_options}
    assert "__unknown__" in {str(option.get("value")) for option in portable_options}
    assert unknown_portable["label"] == "Skip for now"


def test_pipeline_treats_hf_cache_dataset_as_satisfied_in_final_verdict(tmp_path: Path, fake_selected_python: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "train_qwen3.py").write_text(
        "\n".join(
            [
                "from datasets import load_dataset",
                "from transformers import AutoModelForCausalLM, TrainingArguments",
                'TrainingArguments(output_dir="qwen3-finetuned")',
                'dataset = load_dataset("karthiksagarn/astro_horoscope", split="train")',
                'model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (workspace / "huggingface-cache" / "datasets" / "karthiksagarn___astro_horoscope").mkdir(parents=True)
    (workspace / "huggingface-cache" / "hub" / "models--Qwen--Qwen3-0.6B").mkdir(parents=True)
    cann_root = tmp_path / "cann"
    cann_root.mkdir()
    (cann_root / "version.cfg").write_text("version=8.5.0\n", encoding="utf-8")
    output_dir = tmp_path / "out"

    run_pipeline(
        "--working-dir",
        str(workspace),
        "--output-dir",
        str(output_dir),
        "--target",
        "training",
        "--framework-hint",
        "pta",
        "--launcher-hint",
        "python",
        "--selected-python",
        str(fake_selected_python),
        "--entry-script",
        "train_qwen3.py",
        "--model-hub-id",
        "Qwen/Qwen3-0.6B",
        "--dataset-hub-id",
        "karthiksagarn/astro_horoscope",
        "--cann-path",
        str(cann_root),
        "--confirm",
        "config_asset=inline_config",
        cwd=workspace,
    )

    verdict = json.loads((output_dir / "meta" / "readiness-verdict.json").read_text(encoding="utf-8"))
    dataset_check = check_by_id(verdict, "workspace-dataset-asset")
    model_check = check_by_id(verdict, "workspace-model-asset")

    assert verdict["status"] in {"WARN", "READY"}
    assert dataset_check["status"] == "ok"
    assert model_check["status"] == "ok"
    assert not any("dataset asset is required but unresolved" in item for item in verdict["missing_items"])


def test_pipeline_treats_inline_config_as_a_valid_asset(tmp_path: Path, fake_selected_python: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "train.py").write_text(
        "\n".join(
            [
                "from transformers import TrainingArguments",
                'TrainingArguments(output_dir="out")',
                "print('train')",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (workspace / "model").mkdir()
    (workspace / "dataset").mkdir()
    (workspace / "dataset" / "sample.txt").write_text("hello\n", encoding="utf-8")
    cann_root = tmp_path / "cann"
    cann_root.mkdir()
    (cann_root / "version.cfg").write_text("version=8.5.0\n", encoding="utf-8")
    output_dir = tmp_path / "out"

    run_pipeline(
        "--working-dir",
        str(workspace),
        "--output-dir",
        str(output_dir),
        "--target",
        "training",
        "--framework-hint",
        "pta",
        "--launcher-hint",
        "python",
        "--selected-python",
        str(fake_selected_python),
        "--entry-script",
        "train.py",
        "--model-path",
        "model",
        "--dataset-path",
        "dataset",
        "--cann-path",
        str(cann_root),
        "--confirm",
        "config_asset=inline_config",
        cwd=workspace,
    )

    verdict = json.loads((output_dir / "meta" / "readiness-verdict.json").read_text(encoding="utf-8"))
    config_check = check_by_id(verdict, "workspace-config-asset")
    assert config_check["status"] == "ok"


def test_pipeline_refreshes_asset_catalog_after_confirming_target_and_entry(tmp_path: Path, fake_selected_python: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    entry_script = workspace / "train_qwen3.py"
    entry_script.write_text(
        "\n".join(
            [
                "from datasets import load_dataset",
                "from transformers import AutoModelForCausalLM, TrainingArguments",
                'TrainingArguments(output_dir="qwen3-finetuned")',
                'dataset = load_dataset("karthiksagarn/astro_horoscope", split="train")',
                'model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    common_args = (
        "--working-dir",
        str(workspace),
        "--framework-hint",
        "pta",
        "--launcher-hint",
        "python",
        "--selected-python",
        str(fake_selected_python),
    )

    first = stdout_payload(run_pipeline(*common_args, cwd=workspace))
    second = stdout_payload(run_pipeline(*common_args, "--confirm", "target=training", cwd=workspace))
    third = stdout_payload(
        run_pipeline(
            *common_args,
            "--confirm",
            "target=training",
            "--confirm",
            f"entry_script={entry_script}",
            cwd=workspace,
        )
    )
    fourth = stdout_payload(
        run_pipeline(
            *common_args,
            "--confirm",
            "target=training",
            "--confirm",
            f"entry_script={entry_script}",
            "--confirm",
            "config_asset=inline_config",
            cwd=workspace,
        )
    )
    fifth = stdout_payload(
        run_pipeline(
            *common_args,
            "--confirm",
            "target=training",
            "--confirm",
            f"entry_script={entry_script}",
            "--confirm",
            "config_asset=inline_config",
            "--confirm",
            "model_asset=hf_hub:Qwen/Qwen3-0.6B",
            cwd=workspace,
        )
    )

    assert current_field(first) == "target"
    assert current_field(second) == "entry_script"
    assert current_field(third) == "config_asset"
    assert current_field(fourth) == "model_asset"
    assert any(
        str(option.get("source_type")) == "hf_hub"
        for option in fourth["current_confirmation"]["options"]
    )
    assert current_field(fifth) == "dataset_asset"


def test_pipeline_uses_manual_fallback_when_portable_question_has_no_detected_entry_candidate(tmp_path: Path, fake_selected_python: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    summary = stdout_payload(
        run_pipeline(
            "--working-dir",
            str(workspace),
            "--framework-hint",
            "pta",
            "--launcher-hint",
            "python",
            "--selected-python",
            str(fake_selected_python),
            "--confirm",
            "target=training",
            cwd=workspace,
        )
    )

    assert current_field(summary) == "entry_script"
    assert {
        str(option.get("value"))
        for option in summary["current_confirmation"]["portable_question"]["options"]
    } == {"__manual__", "__unknown__"}


def test_pipeline_detects_variable_backed_model_tokenizer_and_dataset_assets(tmp_path: Path, fake_selected_python: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "train_qwen3.py").write_text(
        "\n".join(
            [
                'MODEL_REPO_ID = "Qwen/Qwen3-0.6B"',
                'DATASET_REPO_ID = "karthiksagarn/astro_horoscope"',
                'DATASET_SPLIT = "train"',
                "from datasets import load_dataset",
                "from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments",
                'TrainingArguments(output_dir="qwen3-finetuned")',
                "tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO_ID)",
                "train_source = load_dataset(DATASET_REPO_ID, split=DATASET_SPLIT)",
                'model = AutoModelForCausalLM.from_pretrained(MODEL_REPO_ID, dtype="auto")',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (workspace / "huggingface-cache" / "datasets" / "karthiksagarn___astro_horoscope").mkdir(parents=True)
    (workspace / "huggingface-cache" / "hub" / "models--Qwen--Qwen3-0.6B").mkdir(parents=True)

    common_args = (
        "--working-dir",
        str(workspace),
        "--target",
        "training",
        "--framework-hint",
        "pta",
        "--launcher-hint",
        "python",
        "--selected-python",
        str(fake_selected_python),
        "--entry-script",
        "train_qwen3.py",
    )

    model_summary = stdout_payload(run_pipeline(*common_args, "--confirm", "config_asset=inline_config", cwd=workspace))
    assert current_field(model_summary) == "model_asset"
    model_options = model_summary["current_confirmation"]["options"]
    assert any((option.get("locator") or {}).get("repo_id") == "Qwen/Qwen3-0.6B" for option in model_options if isinstance(option.get("locator"), dict))
    model_hub_options = [
        option
        for option in model_options
        if str(option.get("source_type")) == "hf_hub"
        and (option.get("locator") or {}).get("repo_id") == "Qwen/Qwen3-0.6B"
    ]
    assert len(model_hub_options) == 1

    tokenizer_summary = stdout_payload(
        run_pipeline(
            *common_args,
            "--confirm",
            "config_asset=inline_config",
            "--confirm",
            "model_asset=hf_hub:Qwen/Qwen3-0.6B",
            cwd=workspace,
        )
    )
    assert current_field(tokenizer_summary) == "tokenizer_asset"
    tokenizer_options = tokenizer_summary["current_confirmation"]["options"]
    assert any((option.get("locator") or {}).get("repo_id") == "Qwen/Qwen3-0.6B" for option in tokenizer_options if isinstance(option.get("locator"), dict))
    tokenizer_hub_options = [
        option
        for option in tokenizer_options
        if str(option.get("source_type")) == "hf_hub"
        and (option.get("locator") or {}).get("repo_id") == "Qwen/Qwen3-0.6B"
    ]
    assert len(tokenizer_hub_options) == 1
    tokenizer_portable_options = tokenizer_summary["current_confirmation"]["portable_question"]["options"]
    assert any(
        "Detected from the entry script." in str(option.get("description") or "")
        for option in tokenizer_portable_options
        if "Qwen/Qwen3-0.6B" in str(option.get("label") or "")
    )

    dataset_summary = stdout_payload(
        run_pipeline(
            *common_args,
            "--confirm",
            "config_asset=inline_config",
            "--confirm",
            "model_asset=hf_hub:Qwen/Qwen3-0.6B",
            "--confirm",
            "tokenizer_asset=hf_hub:Qwen/Qwen3-0.6B",
            cwd=workspace,
        )
    )
    assert current_field(dataset_summary) == "dataset_asset"
    dataset_options = dataset_summary["current_confirmation"]["options"]
    assert any((option.get("locator") or {}).get("repo_id") == "karthiksagarn/astro_horoscope" for option in dataset_options if isinstance(option.get("locator"), dict))
    assert any((option.get("locator") or {}).get("split") == "train" for option in dataset_options if isinstance(option.get("locator"), dict))
    dataset_hub_options = [
        option
        for option in dataset_options
        if str(option.get("source_type")) == "hf_hub"
        and (option.get("locator") or {}).get("repo_id") == "karthiksagarn/astro_horoscope"
    ]
    assert len(dataset_hub_options) == 1


def test_pipeline_surfaces_script_derived_config_and_checkpoint_assets(tmp_path: Path, fake_selected_python: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "configs").mkdir()
    (workspace / "configs" / "train.yaml").write_text("name: demo\n", encoding="utf-8")
    (workspace / "resume" / "checkpoint-1").mkdir(parents=True)
    (workspace / "model").mkdir()
    (workspace / "dataset").mkdir()
    (workspace / "run.py").write_text(
        "\n".join(
            [
                'CONFIG_PATH = "configs/train.yaml"',
                'RESUME_FROM_CHECKPOINT = "resume/checkpoint-1"',
                "def load_runtime(**kwargs):",
                "    return kwargs",
                "config_payload = load_runtime(config_path=CONFIG_PATH)",
                "train_state = load_runtime(resume_from_checkpoint=RESUME_FROM_CHECKPOINT)",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    common_args = (
        "--working-dir",
        str(workspace),
        "--target",
        "training",
        "--framework-hint",
        "pta",
        "--launcher-hint",
        "python",
        "--selected-python",
        str(fake_selected_python),
        "--entry-script",
        "run.py",
        "--model-path",
        "model",
        "--dataset-path",
        "dataset",
    )

    config_summary = stdout_payload(run_pipeline(*common_args, cwd=workspace))
    assert current_field(config_summary) == "config_asset"
    config_options = config_summary["current_confirmation"]["options"]
    config_portable_options = config_summary["current_confirmation"]["portable_question"]["options"]
    assert any(
        str((option.get("locator") or {}).get("path", "")).replace("\\", "/").endswith("configs/train.yaml")
        for option in config_options
        if isinstance(option.get("locator"), dict)
    )
    assert any(
        "Runtime uses this local path as the config asset." in str(option.get("description") or "")
        for option in config_portable_options
    )

    checkpoint_summary = stdout_payload(
        run_pipeline(
            *common_args,
            "--confirm",
            "config_asset=configs/train.yaml",
            cwd=workspace,
        )
    )
    assert current_field(checkpoint_summary) == "checkpoint_asset"
    checkpoint_options = checkpoint_summary["current_confirmation"]["options"]
    checkpoint_portable_options = checkpoint_summary["current_confirmation"]["portable_question"]["options"]
    assert any(
        str((option.get("locator") or {}).get("path", "")).replace("\\", "/").endswith("resume/checkpoint-1")
        for option in checkpoint_options
        if isinstance(option.get("locator"), dict)
    )
    assert any(
        "Runtime references this local path for checkpoint resume or load behavior." in str(option.get("description") or "")
        for option in checkpoint_portable_options
    )
