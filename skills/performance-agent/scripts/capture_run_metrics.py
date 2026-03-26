#!/usr/bin/env python3
import argparse
import json
import re
import subprocess
from pathlib import Path

from perf_common import write_json


STAT_RE = re.compile(
    r"Stats:\s*(?P<tokens>\d+)\s+tokens in\s*(?P<seconds>\d+(?:\.\d+)?)s\s*\((?P<throughput>\d+(?:\.\d+)?)\s*tok/s\)",
    re.IGNORECASE,
)
AVG_THROUGHPUT_RE = re.compile(
    r"Average throughput:\s*(?P<value>\d+(?:\.\d+)?)\s*tok/s",
    re.IGNORECASE,
)
TOTAL_TOKENS_RE = re.compile(r"Total tokens:\s*(?P<value>\d+)", re.IGNORECASE)
TOTAL_TIME_RE = re.compile(r"Total time:\s*(?P<value>\d+(?:\.\d+)?)s", re.IGNORECASE)
LOAD_TIME_RE = re.compile(r"loaded in\s*(?P<value>\d+(?:\.\d+)?)s", re.IGNORECASE)


def parse_metrics(output: str) -> dict:
    prompt_stats = []
    for match in STAT_RE.finditer(output):
        prompt_stats.append(
            {
                "tokens": int(match.group("tokens")),
                "seconds": float(match.group("seconds")),
                "throughput": float(match.group("throughput")),
            }
        )

    average_throughput = None
    avg_match = AVG_THROUGHPUT_RE.search(output)
    if avg_match:
        average_throughput = float(avg_match.group("value"))

    total_tokens = None
    total_tokens_match = TOTAL_TOKENS_RE.search(output)
    if total_tokens_match:
        total_tokens = float(total_tokens_match.group("value"))
    elif prompt_stats:
        total_tokens = float(sum(item["tokens"] for item in prompt_stats))

    total_time = None
    total_time_match = TOTAL_TIME_RE.search(output)
    if total_time_match:
        total_time = float(total_time_match.group("value"))
    elif prompt_stats:
        total_time = float(sum(item["seconds"] for item in prompt_stats))

    if average_throughput is None and total_tokens and total_time:
        average_throughput = round(total_tokens / total_time, 6)

    load_time = None
    load_match = LOAD_TIME_RE.search(output)
    if load_match:
        load_time = float(load_match.group("value"))

    if average_throughput is None and not prompt_stats:
        raise ValueError("Could not parse throughput metrics from workload output.")

    prompt_throughputs = [item["throughput"] for item in prompt_stats]
    prompt_latencies = [item["seconds"] for item in prompt_stats]
    steady_state = prompt_throughputs[1:] if len(prompt_throughputs) > 1 else prompt_throughputs

    metrics = {
        "throughput": round(average_throughput, 6) if average_throughput is not None else None,
        "prompt_count": len(prompt_stats),
        "total_tokens": round(total_tokens, 6) if total_tokens is not None else None,
        "total_time": round(total_time, 6) if total_time is not None else None,
        "first_prompt_throughput": round(prompt_throughputs[0], 6) if prompt_throughputs else None,
        "min_prompt_throughput": round(min(prompt_throughputs), 6) if prompt_throughputs else None,
        "max_prompt_throughput": round(max(prompt_throughputs), 6) if prompt_throughputs else None,
        "steady_state_throughput": round(sum(steady_state) / len(steady_state), 6) if steady_state else None,
        "mean_prompt_latency": round(sum(prompt_latencies) / len(prompt_latencies), 6) if prompt_latencies else None,
        "load_time": round(load_time, 6) if load_time is not None else None,
    }
    metrics = {key: value for key, value in metrics.items() if value is not None}
    return {
        "metrics": metrics,
        "prompt_stats": prompt_stats,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a workload entry script and extract validation metrics from its stdout.")
    parser.add_argument("--script", required=True, help="Python entry script path")
    parser.add_argument("--python", default="python", help="Python interpreter to run the entry script")
    parser.add_argument("--working-dir", default=".", help="working directory for the workload run")
    parser.add_argument("--output-json", required=True, help="path to write parsed metric JSON")
    parser.add_argument("--output-log", help="optional path to write combined stdout/stderr")
    parser.add_argument("--label", default="run", help="label for the captured run")
    parser.add_argument("--timeout-sec", type=int, default=1800, help="timeout in seconds")
    parser.add_argument("script_args", nargs=argparse.REMAINDER, help="arguments passed to the workload script after --")
    args = parser.parse_args()

    script_args = list(args.script_args)
    if script_args and script_args[0] == "--":
        script_args = script_args[1:]

    command = [args.python, str(Path(args.script).resolve())] + script_args
    result = subprocess.run(
        command,
        cwd=str(Path(args.working_dir).resolve()),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=args.timeout_sec,
        check=False,
    )

    combined_output = result.stdout or ""
    if args.output_log:
        output_log = Path(args.output_log).resolve()
        output_log.parent.mkdir(parents=True, exist_ok=True)
        output_log.write_text(combined_output, encoding="utf-8")

    if result.returncode != 0:
        raise SystemExit(combined_output or f"Workload run failed with exit code {result.returncode}")

    parsed = parse_metrics(combined_output)
    payload = {
        "schema_version": "performance-agent/0.1",
        "skill": "performance-agent",
        "label": args.label,
        "command": command,
        "working_dir": str(Path(args.working_dir).resolve()),
        "script": str(Path(args.script).resolve()),
        "metrics": parsed["metrics"],
        "prompt_stats": parsed["prompt_stats"],
        "log_ref": str(Path(args.output_log).resolve()) if args.output_log else None,
    }
    write_json(Path(args.output_json), payload)
    print(json.dumps({"label": payload["label"], "throughput": payload["metrics"].get("throughput")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
