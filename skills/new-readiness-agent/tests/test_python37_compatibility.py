from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def test_scripts_avoid_runtime_evaluated_python39_style_generics():
    forbidden_patterns = [
        re.compile(r"[:(]\s*list\["),
        re.compile(r"[:(]\s*dict\["),
        re.compile(r"[:(]\s*tuple\["),
        re.compile(r"[:(]\s*set\["),
        re.compile(r"[:(]\s*re\.Pattern\["),
        re.compile(r"->\s*list\["),
        re.compile(r"->\s*dict\["),
        re.compile(r"->\s*tuple\["),
        re.compile(r"->\s*set\["),
        re.compile(r"->\s*re\.Pattern\["),
    ]
    offenders = []
    for path in SCRIPTS.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        for pattern in forbidden_patterns:
            if pattern.search(text):
                offenders.append(f"{path.name}: {pattern.pattern}")
    assert offenders == []
