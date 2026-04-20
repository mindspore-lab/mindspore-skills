#!/usr/bin/env python3
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Optional
from uuid import uuid4

from new_readiness_core import build_run_state
from new_readiness_report import write_report_bundle


READINESS_OUTPUT_DIRNAME = "readiness-output"
LATEST_CACHE_DIR = Path(READINESS_OUTPUT_DIRNAME) / "latest" / "new-readiness-agent"
ATTEMPTS_DIR = Path(READINESS_OUTPUT_DIRNAME) / "attempts"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the read-only new-readiness-agent workflow.")
    parser.add_argument("--working-dir", help="workspace root (defaults to the current shell path)")
    parser.add_argument("--output-dir", help="output directory for readiness artifacts")
    parser.add_argument("--run-id", help="explicit run id")
    parser.add_argument("--target", default="auto", help="training, inference, or auto")
    parser.add_argument("--framework-hint", default="auto", help="mindspore, pta, mixed, or auto")
    parser.add_argument("--launcher-hint", default="auto", help="python, bash, torchrun, accelerate, deepspeed, msrun, llamafactory-cli, make, or auto")
    parser.add_argument("--selected-python", help="explicit Python interpreter for runtime certification")
    parser.add_argument("--selected-env-root", help="explicit environment root for runtime certification")
    parser.add_argument("--cann-path", help="explicit CANN root or set_env.sh path")
    parser.add_argument("--entry-script", help="explicit entry script")
    parser.add_argument("--config-path", help="explicit config path")
    parser.add_argument("--model-path", help="explicit model path")
    parser.add_argument("--model-hub-id", help="explicit Hugging Face model repo ID")
    parser.add_argument("--dataset-path", help="explicit dataset path")
    parser.add_argument("--dataset-hub-id", help="explicit Hugging Face dataset repo ID")
    parser.add_argument("--dataset-split", help="optional dataset split for Hugging Face datasets")
    parser.add_argument("--checkpoint-path", help="explicit checkpoint path")
    parser.add_argument("--launch-command", help="explicit launch command template")
    parser.add_argument("--extra-context", help="additional free-text context")
    parser.add_argument("--confirm", action="append", help="confirmation override in the form field=value; repeat as needed")
    return parser


def default_run_id() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8]


def load_latest_cache_payload(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def resolve_attempt_id(root: Path, args: argparse.Namespace) -> str:
    if args.run_id:
        return args.run_id
    if args.confirm:
        latest_root = root / LATEST_CACHE_DIR
        confirmation_payload = load_latest_cache_payload(latest_root / "confirmation-latest.json")
        pending_fields = confirmation_payload.get("pending_confirmation_fields")
        run_ref_payload = load_latest_cache_payload(latest_root / "run-ref.json")
        cached_attempt_id = str(run_ref_payload.get("run_id") or "").strip()
        if cached_attempt_id and isinstance(pending_fields, list) and pending_fields:
            return cached_attempt_id
    return default_run_id()


def compute_output_dir(root: Path, attempt_id: str, explicit_output_dir: Optional[str], phase: str) -> Path:
    if explicit_output_dir:
        return Path(explicit_output_dir).resolve()
    phase_dir = "current" if phase == "awaiting_confirmation" else "final"
    return (root / ATTEMPTS_DIR / attempt_id / phase_dir).resolve()


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    working_dir = Path(args.working_dir or ".").resolve()
    run_id = resolve_attempt_id(working_dir, args)
    state = build_run_state(working_dir, args)
    phase = "awaiting_confirmation" if state["validation"]["status"] == "NEEDS_CONFIRMATION" else "validated"
    output_dir = compute_output_dir(working_dir, run_id, args.output_dir, phase)
    output_dir.mkdir(parents=True, exist_ok=True)

    inputs_snapshot = {
        "skill": "new-readiness-agent",
        "run_id": run_id,
        "parameters": {
            "working_dir": str(working_dir),
            "target": args.target,
            "framework_hint": args.framework_hint,
            "launcher_hint": args.launcher_hint,
            "selected_python": args.selected_python,
            "selected_env_root": args.selected_env_root,
            "cann_path": args.cann_path,
            "entry_script": args.entry_script,
            "config_path": args.config_path,
            "model_path": args.model_path,
            "model_hub_id": args.model_hub_id,
            "dataset_path": args.dataset_path,
            "dataset_hub_id": args.dataset_hub_id,
            "dataset_split": args.dataset_split,
            "checkpoint_path": args.checkpoint_path,
            "launch_command": args.launch_command,
            "extra_context": args.extra_context,
            "confirm": args.confirm or [],
            "output_dir": str(output_dir),
        },
    }

    bundle = write_report_bundle(
        root=working_dir,
        run_id=run_id,
        output_dir=output_dir,
        inputs_snapshot=inputs_snapshot,
        state=state,
    )
    verdict = bundle["verdict"]
    artifact_refs = bundle.get("artifact_refs") if isinstance(bundle.get("artifact_refs"), dict) else {}

    print(
        json.dumps(
            {
                "phase": verdict["phase"],
                "status": verdict["status"],
                "confirmation_required": verdict["confirmation_required"],
                "pending_confirmation_fields": verdict["pending_confirmation_fields"],
                "can_run": verdict["can_run"],
                "target": verdict["target"],
                "summary": verdict["summary"],
                "cann_path": verdict.get("cann_path"),
                "ascend_env_script_path": verdict.get("ascend_env_script_path"),
                "output_dir": str(output_dir),
                "confirmation_step_ref": "artifacts/confirmation-step.json",
                "current_confirmation": verdict["current_confirmation"] if verdict["confirmation_required"] else None,
                "artifact_refs": artifact_refs,
                "latest_cache_ref": verdict.get("latest_cache_ref"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
