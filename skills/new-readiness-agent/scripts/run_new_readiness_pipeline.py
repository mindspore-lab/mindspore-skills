#!/usr/bin/env python3
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Optional
from uuid import uuid4

from new_readiness_core import build_run_state
from new_readiness_report import write_report_bundle


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
    parser.add_argument("--dataset-path", help="explicit dataset path")
    parser.add_argument("--checkpoint-path", help="explicit checkpoint path")
    parser.add_argument("--launch-command", help="explicit launch command template")
    parser.add_argument("--extra-context", help="additional free-text context")
    return parser


def default_run_id() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8]


def compute_output_dir(root: Path, run_id: str, explicit_output_dir: Optional[str]) -> Path:
    if explicit_output_dir:
        return Path(explicit_output_dir).resolve()
    return (root / "runs" / run_id / "out").resolve()


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    working_dir = Path(args.working_dir or ".").resolve()
    run_id = args.run_id or default_run_id()
    output_dir = compute_output_dir(working_dir, run_id, args.output_dir)
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
            "dataset_path": args.dataset_path,
            "checkpoint_path": args.checkpoint_path,
            "launch_command": args.launch_command,
            "extra_context": args.extra_context,
            "output_dir": str(output_dir),
        },
    }

    state = build_run_state(working_dir, args)
    bundle = write_report_bundle(
        root=working_dir,
        run_id=run_id,
        output_dir=output_dir,
        inputs_snapshot=inputs_snapshot,
        state=state,
    )
    verdict = bundle["verdict"]

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
                "output_dir": str(output_dir),
                "confirmation_form_ref": "artifacts/confirmation-options.json",
                "confirmation_form": verdict["confirmation_form"] if verdict["confirmation_required"] else None,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
