import json
import os
import subprocess
import sys
from pathlib import Path


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


def load_report_pair(report_json: Path) -> tuple[dict, dict]:
    envelope = json.loads(report_json.read_text(encoding="utf-8"))
    verdict_json = report_json.parent / READINESS_VERDICT_REF
    verdict = json.loads(verdict_json.read_text(encoding="utf-8"))
    return envelope, verdict


def fake_python_source() -> str:
    real_python = json.dumps(sys.executable)
    return f"""#!/usr/bin/env python3
import json
import subprocess
import sys

REAL_PYTHON = {real_python}

if len(sys.argv) >= 3 and sys.argv[1] == "-c":
    code = sys.argv[2]
    if "platform.python_version" in code and "version_info" in code:
        print(json.dumps({{"version_info": [3, 10, 0], "version": "3.10.0"}}))
        raise SystemExit(0)
    if "importlib.util" in code and len(sys.argv) >= 5:
        mode = sys.argv[3]
        payload = json.loads(sys.argv[4])
        if mode == "import":
            packages = payload.get("packages", [])
            print(json.dumps({{name: True for name in packages}}))
            raise SystemExit(0)
        if mode == "framework_smoke":
            print(json.dumps({{"success": True, "details": ["fake framework smoke ok"], "error": None}}))
            raise SystemExit(0)
    completed = subprocess.run([REAL_PYTHON, *sys.argv[1:]])
    raise SystemExit(completed.returncode)

if len(sys.argv) >= 3 and sys.argv[1] == "-m" and sys.argv[2] == "py_compile":
    raise SystemExit(0)

completed = subprocess.run([REAL_PYTHON, *sys.argv[1:]])
raise SystemExit(completed.returncode)
"""


def fake_uv_source() -> str:
    return """#!/usr/bin/env python3
import os
import shutil
import sys
from pathlib import Path

FAKE_PYTHON = r'''__FAKE_PYTHON__'''
REAL_PYTHON = r'''__REAL_PYTHON__'''


def write_fake_python(env_root: Path) -> None:
    if os.name == "nt":
        target = env_root / "Scripts" / "python.exe"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REAL_PYTHON, target)
        return

    target = env_root / "bin" / "python"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(FAKE_PYTHON, encoding="utf-8")
    target.chmod(target.stat().st_mode | 0o111)


def main() -> int:
    args = sys.argv[1:]
    if not args:
        return 1

    if args[0] == "venv":
        env_root = None
        for item in args[1:]:
            if item.startswith("-"):
                continue
            env_root = Path(item)
            break
        if env_root is None:
            return 2
        write_fake_python(env_root)
        return 0

    if len(args) >= 2 and args[0] == "pip" and args[1] == "install":
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""


def install_fake_uv(tmp_path: Path, monkeypatch) -> Path:
    bin_dir = tmp_path / "fake-bin"
    bin_dir.mkdir()

    uv_py = bin_dir / "uv"
    uv_py.write_text(
        fake_uv_source()
        .replace("__FAKE_PYTHON__", fake_python_source().replace("'''", "\\'\\'\\'"))
        .replace("__REAL_PYTHON__", sys.executable.replace("'''", "\\'\\'\\'")),
        encoding="utf-8",
    )
    uv_py.chmod(uv_py.stat().st_mode | 0o111)

    uv_cmd = bin_dir / "uv.cmd"
    uv_cmd.write_text(
        f'@echo off\r\n"{sys.executable}" "%~dp0uv" %*\r\n',
        encoding="utf-8",
    )

    monkeypatch.setenv("PATH", str(bin_dir) + os.pathsep + os.environ.get("PATH", ""))
    return bin_dir


def make_workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "infer.py").write_text(
        "import torch\nimport torch_npu\nimport transformers\nprint('infer')\n",
        encoding="utf-8",
    )
    (workspace / "model").mkdir()
    return workspace


def test_run_readiness_pipeline_check_does_not_create_workspace_env(tmp_path: Path, monkeypatch):
    install_fake_uv(tmp_path, monkeypatch)
    workspace = make_workspace(tmp_path)
    output_dir = tmp_path / "out"

    run_script(
        "run_readiness_pipeline.py",
        "--working-dir",
        str(workspace),
        "--output-dir",
        str(output_dir),
        "--target",
        "inference",
        "--mode",
        "check",
        "--model-path",
        "model",
    )

    _, verdict = load_report_pair(output_dir / "report.json")
    fix_applied = json.loads((output_dir / "meta" / "fix-applied.json").read_text(encoding="utf-8"))

    assert verdict["status"] == "BLOCKED"
    assert fix_applied["execute"] is False
    assert fix_applied["executed_actions"] == []
    assert not (workspace / ".venv").exists()


def test_run_readiness_pipeline_auto_creates_default_env_and_reruns(tmp_path: Path, monkeypatch):
    install_fake_uv(tmp_path, monkeypatch)
    workspace = make_workspace(tmp_path)
    output_dir = tmp_path / "out"

    run_script(
        "run_readiness_pipeline.py",
        "--working-dir",
        str(workspace),
        "--output-dir",
        str(output_dir),
        "--target",
        "inference",
        "--auto",
        "--model-path",
        "model",
    )

    env_json = json.loads((output_dir / "meta" / "env.json").read_text(encoding="utf-8"))
    checks = json.loads((output_dir / "meta" / "checks.json").read_text(encoding="utf-8"))
    fix_applied = json.loads((output_dir / "meta" / "fix-applied.json").read_text(encoding="utf-8"))
    selected_python = json.loads((output_dir / "meta" / "selected-python.json").read_text(encoding="utf-8"))
    by_id = {item["id"]: item for item in checks}

    assert env_json["pipeline_passes"] == 2
    assert fix_applied["execute"] is True
    assert (workspace / ".venv").exists()
    assert fix_applied["executed_actions"]
    assert selected_python["selection_status"] == "selected"
    assert str(workspace / ".venv") in str(selected_python["selected_env_root"])
    assert by_id["python-selected-python"]["status"] == "ok"
