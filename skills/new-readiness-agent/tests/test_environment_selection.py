import json
import os
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from environment_selection import build_environment_candidates  # noqa: E402


def create_fake_env_python(env_root: Path) -> None:
    if os.name == "nt":
        python_path = env_root / "Scripts" / "python.exe"
    else:
        python_path = env_root / "bin" / "python"
    python_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if os.name == "nt":
            shutil.copy2(sys.executable, python_path)
        else:
            python_path.symlink_to(Path(sys.executable))
    except (OSError, NotImplementedError):
        shutil.copy2(sys.executable, python_path)
    if os.name != "nt":
        python_path.chmod(python_path.stat().st_mode | 0o111)


def create_fake_conda(bin_dir: Path, env_roots: list[Path], default_prefix: Path) -> None:
    payload = json.dumps(
        {
            "envs": [str(path) for path in env_roots],
            "default_prefix": str(default_prefix),
        }
    )
    script_path = bin_dir / "fake-conda.py"
    script_path.write_text(
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "import json",
                "import sys",
                f"PAYLOAD = {payload!r}",
                "if sys.argv[1:] == ['env', 'list', '--json']:",
                "    print(PAYLOAD)",
                "    raise SystemExit(0)",
                "raise SystemExit(2)",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    if os.name == "nt":
        launcher = bin_dir / "conda.cmd"
        launcher.write_text(f'@echo off\r\n"{sys.executable}" "%~dp0fake-conda.py" %*\r\n', encoding="utf-8")
    else:
        launcher = bin_dir / "conda"
        launcher.write_text(script_path.read_text(encoding="utf-8"), encoding="utf-8")
        launcher.chmod(launcher.stat().st_mode | 0o111)


def test_workspace_environment_candidates_include_workspace_venv_and_conda_clues(tmp_path: Path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "environment.yml").write_text("name: project-conda\nchannels:\n  - defaults\n", encoding="utf-8")
    create_fake_env_python(workspace / ".venv")

    conda_root = tmp_path / "conda-envs"
    project_conda = conda_root / "project-conda"
    other_conda = conda_root / "other-conda"
    create_fake_env_python(project_conda)
    create_fake_env_python(other_conda)

    fake_conda_bin = tmp_path / "fake-conda-bin"
    fake_conda_bin.mkdir()
    create_fake_conda(fake_conda_bin, [project_conda, other_conda], default_prefix=other_conda)
    monkeypatch.setenv("PATH", str(fake_conda_bin) + os.pathsep + os.environ.get("PATH", ""))
    monkeypatch.delenv("CONDA_PREFIX", raising=False)

    result = build_environment_candidates(
        workspace,
        launch_command=None,
        selected_python=None,
        selected_env_root=None,
    )

    labels = [str(item.get("label")) for item in result["candidates"]]
    conda_candidates = [item for item in result["candidates"] if str(item.get("kind")) == "workspace-conda"]
    env_names = [str(item.get("env_name")) for item in conda_candidates if item.get("env_name")]

    assert any(label.startswith(".venv") for label in labels)
    assert "project-conda" in env_names
    assert len(conda_candidates) >= 2
