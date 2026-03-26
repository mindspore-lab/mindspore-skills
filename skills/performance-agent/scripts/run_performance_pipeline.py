#!/usr/bin/env python3
import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from perf_common import read_json, write_json


SCRIPTS_DIR = Path(__file__).resolve().parent


def default_out_dir(working_dir: Path) -> Path:
    run_id = datetime.utcnow().strftime("performance-%Y%m%d-%H%M%S")
    return working_dir / "runs" / run_id / "out"


def write_log(log_path: Path, content: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(content, encoding="utf-8")


def run_command(
    command: list[str],
    cwd: Path,
    log_path: Path,
    allow_fail: bool = False,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    write_log(log_path, result.stdout or "")
    if result.returncode != 0 and not allow_fail:
        raise SystemExit(result.stdout or f"Command failed: {' '.join(command)}")
    return result


def run_python_helper(
    script_name: str,
    args: list[str],
    cwd: Path,
    log_path: Path,
    allow_fail: bool = False,
) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, str(SCRIPTS_DIR / script_name)] + args
    return run_command(command, cwd, log_path, allow_fail=allow_fail)


def pick_entry_script(working_dir: Path, explicit: Optional[str], context_report: dict) -> Optional[Path]:
    if explicit:
        return Path(explicit).resolve()

    candidates = ((context_report.get("candidates") or {}).get("script") or [])
    preferred = ["inference.py", "train.py", "run.py", "main.py"]
    for name in preferred:
        if name in candidates:
            return (working_dir / name).resolve()
    if candidates:
        return (working_dir / candidates[0]).resolve()
    return None


def locate_profiler_root(
    working_dir: Path,
    trace_path: Optional[str],
    locator_json: Path,
    logs_dir: Path,
) -> dict:
    args = ["--output-json", str(locator_json)]
    if trace_path:
        args.extend(["--trace-path", trace_path])
    else:
        args.extend(["--working-dir", str(working_dir)])
    run_python_helper("locate_profiler_output.py", args, working_dir, logs_dir / "locate_profiler.log")
    return read_json(locator_json)


def maybe_run_collect(
    working_dir: Path,
    stack: str,
    script_path: Path,
    output_dir: Path,
    python_bin: str,
    script_args: list[str],
    logs_dir: Path,
) -> Path:
    command = [
        "bash",
        str(SCRIPTS_DIR / "collect_msprof.sh"),
        "--stack",
        stack,
        "--script",
        str(script_path),
        "--output-dir",
        str(output_dir),
        "--python",
        python_bin,
    ]
    if script_args:
        command.append("--")
        command.extend(script_args)
    run_command(command, working_dir, logs_dir / "collect_msprof.log")
    locator_path = output_dir / "locator.json"
    if not locator_path.exists():
        raise SystemExit("Profiler collection finished, but locator.json was not produced.")
    locate = read_json(locator_path)
    selected_root = locate.get("selected_root")
    if not selected_root:
        raise SystemExit("Profiler collection finished, but no profiler export root was recovered.")
    return Path(selected_root).resolve()


def run_optional_summary(
    script_name: str,
    args: list[str],
    cwd: Path,
    log_path: Path,
    output_path: Path,
) -> Optional[Path]:
    result = run_python_helper(script_name, args, cwd, log_path, allow_fail=True)
    if result.returncode == 0 and output_path.exists():
        return output_path
    return None


def build_context_json(working_dir: Path, logs_dir: Path, output_json: Path) -> dict:
    result = run_python_helper(
        "find_run_context.py",
        ["--working-dir", str(working_dir)],
        working_dir,
        logs_dir / "find_run_context.log",
    )
    payload = json.loads(result.stdout)
    write_json(output_json, payload)
    return payload


def build_summary_refs(trace_root: Path, working_dir: Path, summaries_dir: Path, logs_dir: Path) -> dict[str, Optional[Path]]:
    summaries_dir.mkdir(parents=True, exist_ok=True)
    refs: dict[str, Optional[Path]] = {}
    refs["step"] = run_optional_summary(
        "summarize_step_breakdown.py",
        ["--trace-root", str(trace_root), "--output-json", str(summaries_dir / "step.json")],
        working_dir,
        logs_dir / "summarize_step.log",
        summaries_dir / "step.json",
    )
    refs["communication"] = run_optional_summary(
        "summarize_communication.py",
        ["--trace-root", str(trace_root), "--output-json", str(summaries_dir / "communication.json")],
        working_dir,
        logs_dir / "summarize_communication.log",
        summaries_dir / "communication.json",
    )
    refs["memory"] = run_optional_summary(
        "summarize_memory_pressure.py",
        ["--trace-root", str(trace_root), "--output-json", str(summaries_dir / "memory.json")],
        working_dir,
        logs_dir / "summarize_memory.log",
        summaries_dir / "memory.json",
    )
    refs["input"] = run_optional_summary(
        "summarize_input_pipeline.py",
        ["--trace-root", str(trace_root), "--output-json", str(summaries_dir / "input.json")],
        working_dir,
        logs_dir / "summarize_input.log",
        summaries_dir / "input.json",
    )
    refs["trace_gaps"] = run_optional_summary(
        "summarize_trace_gaps.py",
        ["--trace-root", str(trace_root), "--output-json", str(summaries_dir / "trace_gaps.json")],
        working_dir,
        logs_dir / "summarize_trace_gaps.log",
        summaries_dir / "trace_gaps.json",
    )
    hotspot_md = summaries_dir / "hotspot.md"
    hotspot_json = summaries_dir / "hotspot.json"
    refs["hotspot"] = run_optional_summary(
        "summarize_msprof_hotspots.py",
        ["--input-dir", str(trace_root), "--output-md", str(hotspot_md), "--output-json", str(hotspot_json)],
        working_dir,
        logs_dir / "summarize_hotspot.log",
        hotspot_json,
    )
    if refs["hotspot"]:
        run_optional_summary(
            "build_hotspot_brief.py",
            [
                "--input-json",
                str(hotspot_json),
                "--output-json",
                str(summaries_dir / "hotspot_brief.json"),
                "--output-md",
                str(summaries_dir / "hotspot_brief.md"),
            ],
            working_dir,
            logs_dir / "build_hotspot_brief.log",
            summaries_dir / "hotspot_brief.json",
        )
    return refs


def maybe_add_script_args(base_args: list[str], script_args: list[str]) -> list[str]:
    if not script_args:
        return base_args
    return base_args + ["--"] + script_args


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the end-to-end performance-agent pipeline, including optimization trial and rerun validation.")
    parser.add_argument("--working-dir", default=".", help="workspace root")
    parser.add_argument("--user-problem", required=True, help="performance problem description")
    parser.add_argument("--script", help="explicit workload entry script")
    parser.add_argument("--stack", choices=["ms", "pta"], help="optional runtime stack override")
    parser.add_argument("--trace-path", help="optional existing profiler export root")
    parser.add_argument("--python", default=sys.executable, help="Python interpreter for workload runs")
    parser.add_argument("--output-dir", help="output directory for structured artifacts")
    parser.add_argument("script_args", nargs=argparse.REMAINDER, help="arguments passed to the workload script after --")
    args = parser.parse_args()

    script_args = list(args.script_args)
    if script_args and script_args[0] == "--":
        script_args = script_args[1:]

    working_dir = Path(args.working_dir).resolve()
    out_root = Path(args.output_dir).resolve() if args.output_dir else default_out_dir(working_dir)
    meta_dir = out_root / "meta"
    tmp_dir = meta_dir / "tmp"
    summaries_dir = meta_dir / "summaries"
    artifacts_dir = out_root / "artifacts"
    logs_dir = artifacts_dir / "logs"
    script_dir = artifacts_dir / "scripts"
    profiler_dir = artifacts_dir / "profiler"
    for path in (meta_dir, tmp_dir, summaries_dir, artifacts_dir, logs_dir, script_dir, profiler_dir):
        path.mkdir(parents=True, exist_ok=True)

    context_json = meta_dir / "context.json"
    locator_json = meta_dir / "locator.json"
    profile_json = tmp_dir / "performance-profile.json"
    bottlenecks_json = tmp_dir / "bottlenecks.json"
    before_json = meta_dir / "before-metrics.json"
    after_json = meta_dir / "after-metrics.json"
    validation_json = meta_dir / "validation-comparison.json"
    optimization_json = meta_dir / "optimization-trial.json"
    report_json = out_root / "report.json"
    report_md = out_root / "report.md"

    context_report = build_context_json(working_dir, logs_dir, context_json)
    script_path = pick_entry_script(working_dir, args.script, context_report)
    if script_path is None or not script_path.exists():
        raise SystemExit("Could not resolve a runnable workload entry script for performance-agent.")

    stack = args.stack or ((context_report.get("recovered_context") or {}).get("stack"))
    if not stack:
        raise SystemExit("Could not infer the runtime stack. Provide --stack ms or --stack pta.")

    run_python_helper(
        "capture_run_metrics.py",
        maybe_add_script_args(
            [
                "--python",
                args.python,
                "--script",
                str(script_path),
                "--working-dir",
                str(working_dir),
                "--output-json",
                str(before_json),
                "--output-log",
                str(artifacts_dir / "baseline.log"),
                "--label",
                "baseline",
            ],
            script_args,
        ),
        working_dir,
        logs_dir / "capture_baseline.log",
    )

    locate = locate_profiler_root(working_dir, args.trace_path, locator_json, logs_dir)
    trace_root = locate.get("selected_root")
    if not trace_root:
        trace_root_path = maybe_run_collect(
            working_dir,
            stack,
            script_path,
            profiler_dir,
            args.python,
            script_args,
            logs_dir,
        )
        shutil.copyfile(profiler_dir / "locator.json", locator_json)
        locate = read_json(locator_json)
        trace_root = str(trace_root_path)

    summary_refs = build_summary_refs(Path(trace_root), working_dir, summaries_dir, logs_dir)

    profile_args = [
        "--working-dir",
        str(working_dir),
        "--user-problem",
        args.user_problem,
        "--locate-json",
        str(locator_json),
        "--output-json",
        str(profile_json),
    ]
    if summary_refs.get("step"):
        profile_args.extend(["--step-json", str(summary_refs["step"])])
    if summary_refs.get("communication"):
        profile_args.extend(["--communication-json", str(summary_refs["communication"])])
    if summary_refs.get("memory"):
        profile_args.extend(["--memory-json", str(summary_refs["memory"])])
    if summary_refs.get("input"):
        profile_args.extend(["--input-json", str(summary_refs["input"])])
    if summary_refs.get("trace_gaps"):
        profile_args.extend(["--trace-gaps-json", str(summary_refs["trace_gaps"])])
    if summary_refs.get("hotspot"):
        profile_args.extend(["--hotspot-json", str(summary_refs["hotspot"])])
    run_python_helper("build_performance_profile.py", profile_args, working_dir, logs_dir / "build_profile.log")

    classify_args = [
        "--profile-json",
        str(profile_json),
        "--output-json",
        str(bottlenecks_json),
    ]
    if summary_refs.get("step"):
        classify_args.extend(["--step-json", str(summary_refs["step"])])
    if summary_refs.get("communication"):
        classify_args.extend(["--communication-json", str(summary_refs["communication"])])
    if summary_refs.get("memory"):
        classify_args.extend(["--memory-json", str(summary_refs["memory"])])
    if summary_refs.get("input"):
        classify_args.extend(["--input-json", str(summary_refs["input"])])
    if summary_refs.get("trace_gaps"):
        classify_args.extend(["--trace-gaps-json", str(summary_refs["trace_gaps"])])
    if summary_refs.get("hotspot"):
        classify_args.extend(["--hotspot-json", str(summary_refs["hotspot"])])
    run_python_helper("classify_bottlenecks.py", classify_args, working_dir, logs_dir / "classify_bottlenecks.log")

    optimized_script = script_dir / f"{script_path.stem}-optimized.py"
    apply_args = [
        "--profile-json",
        str(profile_json),
        "--bottlenecks-json",
        str(bottlenecks_json),
        "--source-script",
        str(script_path),
        "--output-script",
        str(optimized_script),
        "--output-json",
        str(optimization_json),
    ]
    if summary_refs.get("hotspot"):
        apply_args.extend(["--hotspot-json", str(summary_refs["hotspot"])])
    run_python_helper("apply_performance_features.py", apply_args, working_dir, logs_dir / "apply_features.log")

    run_python_helper(
        "capture_run_metrics.py",
        maybe_add_script_args(
            [
                "--python",
                args.python,
                "--script",
                str(optimized_script),
                "--working-dir",
                str(working_dir),
                "--output-json",
                str(after_json),
                "--output-log",
                str(artifacts_dir / "optimized.log"),
                "--label",
                "optimized",
            ],
            script_args,
        ),
        working_dir,
        logs_dir / "capture_optimized.log",
    )

    run_python_helper(
        "compare_validation_metrics.py",
        [
            "--before-json",
            str(before_json),
            "--after-json",
            str(after_json),
            "--output-json",
            str(validation_json),
        ]
        + sum([["--metric", item] for item in read_json(optimization_json).get("metrics_to_watch", [])], []),
        working_dir,
        logs_dir / "compare_validation.log",
    )

    report_args = [
        "--working-dir",
        str(working_dir),
        "--user-problem",
        args.user_problem,
        "--locate-json",
        str(locator_json),
        "--profile-json",
        str(profile_json),
        "--bottlenecks-json",
        str(bottlenecks_json),
        "--validation-json",
        str(validation_json),
        "--optimization-json",
        str(optimization_json),
        "--before-metrics-json",
        str(before_json),
        "--after-metrics-json",
        str(after_json),
        "--output-json",
        str(report_json),
        "--output-md",
        str(report_md),
    ]
    run_python_helper("build_performance_report.py", report_args, working_dir, logs_dir / "build_report.log")

    report = read_json(report_json)
    print(json.dumps({"status": report["status"], "report_json": str(report_json), "report_md": str(report_md)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
