import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
VERDICT_REF = Path("meta/readiness-verdict.json")


def test_shared_envelope_and_new_readiness_verdict_validate_against_their_schemas(tmp_path: Path, fake_selected_python: Path):
    jsonschema = pytest.importorskip("jsonschema")

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "train.py").write_text("import torch\nimport torch_npu\nfrom transformers import Trainer\n", encoding="utf-8")
    (workspace / "train.yaml").write_text("model_name_or_path: model\ntrain_file: dataset/sample.txt\n", encoding="utf-8")
    (workspace / "model").mkdir()
    (workspace / "dataset").mkdir()
    (workspace / "dataset" / "sample.txt").write_text("hello\n", encoding="utf-8")
    cann_root = tmp_path / "cann"
    cann_root.mkdir()
    (cann_root / "version.cfg").write_text("version=8.5.0\n", encoding="utf-8")

    report_dir = tmp_path / "out"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "run_new_readiness_pipeline.py"),
            "--working-dir",
            str(workspace),
            "--output-dir",
            str(report_dir),
            "--target",
            "training",
            "--framework-hint",
            "pta",
            "--launcher-hint",
            "torchrun",
            "--selected-python",
            str(fake_selected_python),
            "--entry-script",
            "train.py",
            "--config-path",
            "train.yaml",
            "--model-path",
            "model",
            "--dataset-path",
            "dataset",
            "--cann-path",
            str(cann_root),
            "--launch-command",
            "torchrun train.py --config train.yaml",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    shared_schema = json.loads((ROOT.parent / "_shared" / "contract" / "report.schema.json").read_text(encoding="utf-8"))
    verdict_schema = json.loads((ROOT / "contract" / "new-readiness-verdict.schema.json").read_text(encoding="utf-8"))
    report = json.loads((report_dir / "report.json").read_text(encoding="utf-8"))
    verdict = json.loads((report_dir / VERDICT_REF).read_text(encoding="utf-8"))

    jsonschema.validate(instance=report, schema=shared_schema)
    jsonschema.validate(instance=verdict, schema=verdict_schema)

