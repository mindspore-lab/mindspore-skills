import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def run_pipeline(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPTS / "run_new_readiness_pipeline.py"), *args],
        cwd=str(cwd),
        check=True,
        text=True,
        capture_output=True,
    )


def stdout_payload(completed: subprocess.CompletedProcess[str]) -> dict:
    return json.loads(completed.stdout)


def current_field(summary: dict) -> Optional[str]:
    current_confirmation = summary.get("current_confirmation")
    if not isinstance(current_confirmation, dict):
        return None
    return current_confirmation.get("field")


def current_options(summary: dict) -> list[str]:
    current_confirmation = summary.get("current_confirmation")
    if not isinstance(current_confirmation, dict):
        return []
    return [str(option.get("value")) for option in current_confirmation.get("options", [])]


def check_by_id(verdict: dict, check_id: str) -> dict:
    for item in verdict.get("checks", []):
        if item.get("id") == check_id:
            return item
    raise AssertionError(f"missing check: {check_id}")


def make_fake_selected_python_with_import_error(tmp_path: Path, failing_package: str, error_message: str) -> Path:
    real_python = json.dumps(sys.executable)
    script = tmp_path / "fake-import-error-python.py"
    script.write_text(
        f"""#!/usr/bin/env python3
import json
import subprocess
import sys

REAL_PYTHON = {real_python}
FAILING_PACKAGE = {json.dumps(failing_package)}
ERROR_MESSAGE = {json.dumps(error_message)}
VERSION_OVERRIDES = {{
    "torch": "2.9.0",
    "torch_npu": "2.9.0",
    "mindspore": "2.6.0",
}}

if len(sys.argv) >= 3 and sys.argv[1] == "-c":
    code = sys.argv[2]
    if "platform.python_version" in code and "version_info" in code:
        print(json.dumps({{"version_info": [3, 10, 20], "version": "3.10.20"}}))
        raise SystemExit(0)
    if len(sys.argv) >= 5:
        mode = sys.argv[3]
        payload = json.loads(sys.argv[4])
        if mode == "import":
            packages = payload.get("packages", [])
            result = {{"imports": {{}}, "errors": {{}}}}
            for name in packages:
                if name == FAILING_PACKAGE:
                    result["imports"][name] = False
                    result["errors"][name] = ERROR_MESSAGE
                else:
                    result["imports"][name] = True
            print(json.dumps(result))
            raise SystemExit(0)
        if mode == "package_versions":
            packages = payload.get("packages", [])
            print(json.dumps({{"versions": {{name: VERSION_OVERRIDES.get(name, "1.0.0") for name in packages}}, "errors": {{}}}}))
            raise SystemExit(0)
    completed = subprocess.run([REAL_PYTHON, *sys.argv[1:]])
    raise SystemExit(completed.returncode)

completed = subprocess.run([REAL_PYTHON, *sys.argv[1:]])
raise SystemExit(completed.returncode)
""",
        encoding="utf-8",
    )
    if os.name == "nt":
        launcher = tmp_path / "fake-import-error-python.cmd"
        launcher.write_text(f'@echo off\r\n"{sys.executable}" "%~dp0fake-import-error-python.py" %*\r\n', encoding="utf-8")
        return launcher
    script.chmod(script.stat().st_mode | 0o111)
    return script


def make_fake_selected_python_requiring_runtime_env(
    tmp_path: Path,
    required_env_var: str,
    required_env_value: str,
) -> Path:
    real_python = json.dumps(sys.executable)
    script = tmp_path / "fake-runtime-env-python.py"
    script.write_text(
        f"""#!/usr/bin/env python3
import json
import os
import subprocess
import sys

REAL_PYTHON = {real_python}
REQUIRED_ENV_VAR = {json.dumps(required_env_var)}
REQUIRED_ENV_VALUE = {json.dumps(required_env_value)}
VERSION_OVERRIDES = {{
    "torch": "2.8.0",
    "torch_npu": "2.8.0.post2",
}}

if len(sys.argv) >= 3 and sys.argv[1] == "-c":
    code = sys.argv[2]
    if "platform.python_version" in code and "version_info" in code:
        print(json.dumps({{"version_info": [3, 10, 20], "version": "3.10.20"}}))
        raise SystemExit(0)
    if len(sys.argv) >= 5:
        mode = sys.argv[3]
        payload = json.loads(sys.argv[4])
        runtime_ready = os.environ.get(REQUIRED_ENV_VAR) == REQUIRED_ENV_VALUE
        if mode == "import":
            packages = payload.get("packages", [])
            result = {{"imports": {{}}, "errors": {{}}}}
            for name in packages:
                if runtime_ready:
                    result["imports"][name] = True
                else:
                    result["imports"][name] = False
                    result["errors"][name] = "ImportError: libhccl.so: cannot open shared object file"
            print(json.dumps(result))
            raise SystemExit(0)
        if mode == "package_versions":
            packages = payload.get("packages", [])
            if runtime_ready:
                print(json.dumps({{"versions": {{name: VERSION_OVERRIDES.get(name, "1.0.0") for name in packages}}, "errors": {{}}}}))
            else:
                print(json.dumps({{"versions": {{}}, "errors": {{name: "runtime environment is incomplete" for name in packages}}}}))
            raise SystemExit(0)
    completed = subprocess.run([REAL_PYTHON, *sys.argv[1:]])
    raise SystemExit(completed.returncode)

completed = subprocess.run([REAL_PYTHON, *sys.argv[1:]])
raise SystemExit(completed.returncode)
""",
        encoding="utf-8",
    )
    if os.name == "nt":
        launcher = tmp_path / "fake-runtime-env-python.cmd"
        launcher.write_text(f'@echo off\r\n"{sys.executable}" "%~dp0fake-runtime-env-python.py" %*\r\n', encoding="utf-8")
        return launcher
    script.chmod(script.stat().st_mode | 0o111)
    return script


def make_fake_selected_python_with_torch_autoload_conflict(tmp_path: Path) -> Path:
    real_python = json.dumps(sys.executable)
    script = tmp_path / "fake-torch-autoload-python.py"
    script.write_text(
        f"""#!/usr/bin/env python3
import json
import subprocess
import sys

REAL_PYTHON = {real_python}
VERSION_OVERRIDES = {{
    "torch": "2.8.0",
    "torch_npu": "2.8.0.post2",
}}

if len(sys.argv) >= 3 and sys.argv[1] == "-c":
    code = sys.argv[2]
    if "platform.python_version" in code and "version_info" in code:
        print(json.dumps({{"version_info": [3, 10, 20], "version": "3.10.20"}}))
        raise SystemExit(0)
    if len(sys.argv) >= 5:
        mode = sys.argv[3]
        payload = json.loads(sys.argv[4])
        if mode == "import":
            packages = payload.get("packages", [])
            result = {{"imports": {{}}, "errors": {{}}}}
            if packages == ["torch", "torch_npu"]:
                result["imports"]["torch"] = True
                result["imports"]["torch_npu"] = False
                result["errors"]["torch_npu"] = "RuntimeError: duplicate triton TORCH_LIBRARY registration"
            else:
                for name in packages:
                    result["imports"][name] = True
            print(json.dumps(result))
            raise SystemExit(0)
        if mode == "package_versions":
            packages = payload.get("packages", [])
            print(json.dumps({{"versions": {{name: VERSION_OVERRIDES.get(name, "1.0.0") for name in packages}}, "errors": {{}}}}))
            raise SystemExit(0)
    completed = subprocess.run([REAL_PYTHON, *sys.argv[1:]])
    raise SystemExit(completed.returncode)

completed = subprocess.run([REAL_PYTHON, *sys.argv[1:]])
raise SystemExit(completed.returncode)
""",
        encoding="utf-8",
    )
    if os.name == "nt":
        launcher = tmp_path / "fake-torch-autoload-python.cmd"
        launcher.write_text(f'@echo off\r\n"{sys.executable}" "%~dp0fake-torch-autoload-python.py" %*\r\n', encoding="utf-8")
        return launcher
    script.chmod(script.stat().st_mode | 0o111)
    return script
