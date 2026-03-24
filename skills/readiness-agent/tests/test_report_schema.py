import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
READINESS_VERDICT_REF = Path("meta/readiness-verdict.json")


def run_script(script_name: str, *args: str) -> subprocess.CompletedProcess[str]:
    script = SCRIPTS / script_name
    return subprocess.run(
        [sys.executable, str(script), *args],
        check=True,
        text=True,
        capture_output=True,
    )


def test_shared_envelope_and_readiness_verdict_validate_against_their_schemas(tmp_path: Path):
    jsonschema = pytest.importorskip("jsonschema")

    target_path = tmp_path / "target.json"
    normalized_path = tmp_path / "normalized.json"
    checks_path = tmp_path / "checks.json"
    report_json = tmp_path / "report.json"
    report_md = tmp_path / "report.md"
    verdict_json = tmp_path / READINESS_VERDICT_REF

    target_path.write_text(
        json.dumps(
            {
                "target_type": "inference",
                "entry_script": "infer.py",
                "framework_path": "pta",
            }
        ),
        encoding="utf-8",
    )
    normalized_path.write_text(
        json.dumps(
            {
                "blockers": [],
                "warnings": [],
                "blockers_detailed": [],
                "warnings_detailed": [],
            }
        ),
        encoding="utf-8",
    )
    checks_path.write_text(
        json.dumps(
            [
                {
                    "id": "framework-smoke-prerequisite",
                    "status": "ok",
                    "summary": "framework smoke prerequisite passed",
                }
            ]
        ),
        encoding="utf-8",
    )

    run_script(
        "build_readiness_report.py",
        "--target-json",
        str(target_path),
        "--normalized-json",
        str(normalized_path),
        "--checks-json",
        str(checks_path),
        "--output-json",
        str(report_json),
        "--output-md",
        str(report_md),
    )

    shared_schema = json.loads((ROOT.parent / "_shared" / "contract" / "report.schema.json").read_text(encoding="utf-8"))
    verdict_schema = json.loads((ROOT / "contract" / "readiness-verdict.schema.json").read_text(encoding="utf-8"))
    report = json.loads(report_json.read_text(encoding="utf-8"))
    verdict = json.loads(verdict_json.read_text(encoding="utf-8"))

    jsonschema.validate(instance=report, schema=shared_schema)
    jsonschema.validate(instance=verdict, schema=verdict_schema)
