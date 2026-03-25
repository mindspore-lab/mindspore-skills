#!/usr/bin/env python3
import os
import shlex
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple


ASCEND_ENV_HINT_VARS = (
    "ASCEND_HOME_PATH",
    "ASCEND_TOOLKIT_HOME",
    "ASCEND_TOOLKIT_PATH",
)


def environment_has_ascend_runtime(environ: Optional[Dict[str, str]] = None) -> bool:
    env = environ or os.environ
    for key in ("LD_LIBRARY_PATH", "PYTHONPATH", "PATH", "ASCEND_OPP_PATH", "TBE_IMPL_PATH"):
        value = env.get(key)
        if value and "Ascend" in value:
            return True

    home_value = env.get("ASCEND_HOME_PATH") or env.get("ASCEND_TOOLKIT_HOME") or env.get("ASCEND_TOOLKIT_PATH")
    opp_value = env.get("ASCEND_OPP_PATH")
    if home_value and opp_value and ("Ascend" in home_value or "Ascend" in opp_value):
        return True

    return False


def add_candidate_path(path: Path, seen: set, candidates: List[Path]) -> None:
    normalized = str(path)
    if normalized in seen:
        return
    seen.add(normalized)
    candidates.append(path)


def candidate_ascend_env_scripts() -> List[Path]:
    candidates: List[Path] = []
    seen = set()

    for var_name in ASCEND_ENV_HINT_VARS:
        value = os.environ.get(var_name)
        if not value:
            continue
        hint_path = Path(value)
        add_candidate_path(hint_path / "set_env.sh", seen, candidates)

    fixed_paths = (
        Path("/usr/local/Ascend/ascend-toolkit/set_env.sh"),
        Path("/usr/local/Ascend/ascend-toolkit/latest/set_env.sh"),
    )
    for path in fixed_paths:
        add_candidate_path(path, seen, candidates)

    base = Path("/usr/local/Ascend")
    if base.exists():
        for pattern in ("cann-*/set_env.sh", "*/set_env.sh", "*/*/set_env.sh"):
            for path in sorted(base.glob(pattern)):
                add_candidate_path(path, seen, candidates)

    existing = [path for path in candidates if path.exists()]
    return sorted(existing, key=rank_ascend_env_script)


def rank_ascend_env_script(path: Path) -> Tuple[int, int, str]:
    text = str(path).replace("\\", "/").lower()
    if text.endswith("/ascend-toolkit/set_env.sh"):
        return (0, len(path.parts), text)
    if "/ascend-toolkit/latest/" in text:
        return (1, len(path.parts), text)
    if "/cann-" in text:
        return (2, len(path.parts), text)
    return (10, len(path.parts), text)


def detect_ascend_runtime() -> dict:
    candidates = candidate_ascend_env_scripts()
    script_path = str(candidates[0]) if candidates else None
    return {
        "requires_ascend": True,
        "device_paths_present": any(Path("/dev").glob("davinci*")),
        "ascend_env_script_present": bool(script_path),
        "ascend_env_script_path": script_path,
        "ascend_env_candidate_paths": [str(path) for path in candidates[:10]],
        "ascend_env_active": environment_has_ascend_runtime(),
    }


def source_environment_from_script(script_path: str) -> Tuple[Optional[Dict[str, str]], Optional[str]]:
    command = [
        "bash",
        "-lc",
        "source {script} >/dev/null 2>&1 && env -0".format(script=shlex.quote(script_path)),
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return None, str(exc)

    payload = completed.stdout or b""
    env: Dict[str, str] = {}
    for item in payload.split(b"\0"):
        if not item or b"=" not in item:
            continue
        key, value = item.split(b"=", 1)
        env[key.decode("utf-8", errors="replace")] = value.decode("utf-8", errors="replace")

    if not env:
        return None, "sourced environment payload was empty"

    return env, None


def resolve_runtime_environment(system_layer: dict) -> Tuple[Dict[str, str], str, Optional[str]]:
    if system_layer.get("ascend_env_active"):
        return dict(os.environ), "current_environment", None

    script_path = system_layer.get("ascend_env_script_path")
    if script_path:
        sourced_env, error = source_environment_from_script(str(script_path))
        if sourced_env is not None:
            return sourced_env, "sourced_script", None
        return dict(os.environ), "current_environment", error

    return dict(os.environ), "current_environment", None
