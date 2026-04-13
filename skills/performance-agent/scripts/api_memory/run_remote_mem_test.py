#!/usr/bin/env python3
"""
Remote memory test tool — run memory benchmarks in parallel on Ascend/GPU servers.

Uses SSH_ASKPASS + subprocess (same approach as remote_deploy_build.py),
no third-party dependencies like paramiko required.

Features:
  - Connect to Ascend (NPU) and GPU servers in parallel
  - Upload test scripts, execute them, and capture memory stats JSON
  - [Ascend] Enable plog, record PID, filter with filter_plog_memory.py, and download
  - Generate comparison report mem_results.md

Usage:
    python run_remote_mem_test.py <npu_script> <gpu_script> [options]

Examples:
    python run_remote_mem_test.py torchapi_id0299_nanmean.py torchapi_id0299_nanmean_gpu.py
    python run_remote_mem_test.py npu.py gpu.py --skip-gpu --gpu-json '{"target_api":"torch.nanmean","total_driver_GB":4.27}'
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
SKILL_ROOT = SCRIPT_DIR.parent.parent
SERVERS_JSON = SKILL_ROOT / "references" / "servers.json"
FILTER_NPU_SCRIPT = SCRIPT_DIR / "filter_plog_memory.py"

SSH_BIN = os.environ.get("SSH_BIN", "ssh")
SCP_BIN = os.environ.get("SCP_BIN", "scp")
IS_WINDOWS = sys.platform.startswith("win")

_print_lock = threading.Lock()


def log(tag, msg):
    with _print_lock:
        print(f"{tag} {msg}", flush=True)


# ─── config helpers ────────────────────────────────────────────────────────

def load_servers(config_path):
    """Load servers.json and return the 'servers' dict."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg.get("servers", {})


def _env_source_cmd(cfg):
    """Build the shell snippet that sources the environment setup script.

    If the server config contains an ``env_script`` path, use it;
    otherwise fall back to ``source ~/.bashrc``.
    """
    script = cfg.get("env_script", "").strip()
    if script:
        return f"source {script}"
    return "source ~/.bashrc"


def extract_api_name(script_path):
    """Extract the value of TARGET_API = '...' from the script."""
    with open(script_path, "r", encoding="utf-8") as f:
        for line in f:
            m = re.match(r'^TARGET_API\s*=\s*["\'](.+?)["\']', line)
            if m:
                return m.group(1)
    return None


def extract_key_code_lines(script_path, api_name):
    """Extract the key API-call code lines from a test script.

    Extraction logic: inside the calculate_xxx_non32aligned() function,
    start from the first non-empty line after `device = torch.device(...)`,
    end at `output = torch.<api>...`.

    Return: (key_lines, start_line, end_line) or (None, None, None).
    """
    if not script_path or not os.path.isfile(script_path):
        return None, None, None
    
    with open(script_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    in_non32aligned_func = False
    device_line_idx = None
    start_idx = None
    end_idx = None
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        if re.match(r'^def\s+calculate_\w+_non32aligned\s*\(', stripped):
            in_non32aligned_func = True
            continue
        
        if in_non32aligned_func:
            if stripped.startswith("def ") and "non32aligned" not in stripped:
                break
            
            if device_line_idx is None and re.match(r'^device\s*=\s*torch\.device\s*\(', stripped):
                device_line_idx = i
                continue
            
            if device_line_idx is not None and start_idx is None and stripped:
                start_idx = i
            
            if start_idx is not None and re.match(r'^output\s*=\s*torch\.\w+', stripped):
                end_idx = i
                break
    
    if start_idx is not None and end_idx is not None:
        key_lines = "".join(lines[start_idx:end_idx + 1]).rstrip()
        return key_lines, start_idx + 1, end_idx + 1
    
    return None, None, None


# ─── SSH helpers (SSH_ASKPASS, same as remote_deploy_build.py) ─────────────

def _make_askpass(password: str) -> str:
    """Create a helper script that outputs the password for SSH_ASKPASS.

    Writes the raw password to a companion file and uses ``type`` (Windows)
    or ``cat`` (Linux/macOS) to output it, so the shell never interprets
    special characters (&, |, >, $, etc.) in the password.
    Each thread gets a unique pair of files to avoid races.
    """
    tid = threading.current_thread().ident
    pw_file = os.path.join(tempfile.gettempdir(), f"mem_test_pw_{tid}.txt")
    with open(pw_file, "w", newline="") as f:
        f.write(password)

    if IS_WINDOWS:
        script = os.path.join(tempfile.gettempdir(), f"mem_test_askpass_{tid}.bat")
        with open(script, "w") as f:
            f.write("@echo off\n")
            f.write(f'type "{pw_file}"\n')
    else:
        script = os.path.join(tempfile.gettempdir(), f"mem_test_askpass_{tid}.sh")
        with open(script, "w") as f:
            f.write("#!/bin/sh\n")
            f.write(f'cat "{pw_file}"\n')
        os.chmod(script, 0o700)
    return script


def _cleanup_askpass(script_path: str):
    """Remove the askpass script and its companion password file."""
    tid_match = re.search(r"mem_test_askpass_(\d+)", script_path)
    if tid_match:
        pw_path = os.path.join(
            tempfile.gettempdir(), f"mem_test_pw_{tid_match.group(1)}.txt"
        )
    else:
        pw_path = None
    for p in (script_path, pw_path):
        if p:
            try:
                os.remove(p)
            except OSError:
                pass


def _ssh_env(askpass_script: str) -> dict:
    env = os.environ.copy()
    env["SSH_ASKPASS"] = askpass_script
    env["SSH_ASKPASS_REQUIRE"] = "force"
    env["DISPLAY"] = ":0"
    return env


_SSH_OPTS = [
    "-o", "StrictHostKeyChecking=no",
    "-o", "PreferredAuthentications=keyboard-interactive,password",
]


def ssh_run(target, cmd, env, timeout=600, keep_alive=False):
    """Execute a remote command via SSH. Return (stdout, stderr, returncode)."""
    ssh_cmd = [SSH_BIN] + _SSH_OPTS
    if keep_alive:
        ssh_cmd += ["-o", "ServerAliveInterval=30",
                    "-o", "ServerAliveCountMax=10"]
    ssh_cmd += [target, cmd]
    r = subprocess.run(
        ssh_cmd, env=env, stdin=subprocess.DEVNULL,
        capture_output=True, text=True, timeout=timeout,
        start_new_session=(not IS_WINDOWS),
    )
    return r.stdout, r.stderr, r.returncode


def scp_upload(target, local_path, remote_path, env, timeout=120):
    """Upload a file to the remote server via SCP."""
    r = subprocess.run(
        [SCP_BIN] + _SSH_OPTS + [str(local_path), f"{target}:{remote_path}"],
        env=env, stdin=subprocess.DEVNULL,
        capture_output=True, text=True, timeout=timeout,
        start_new_session=(not IS_WINDOWS),
    )
    return r.returncode == 0, r.stderr


def scp_download(target, remote_path, local_path, env, timeout=120):
    """Download a file from the remote server via SCP."""
    r = subprocess.run(
        [SCP_BIN] + _SSH_OPTS + [f"{target}:{remote_path}", str(local_path)],
        env=env, stdin=subprocess.DEVNULL,
        capture_output=True, text=True, timeout=timeout,
        start_new_session=(not IS_WINDOWS),
    )
    return r.returncode == 0, r.stderr


# ─── Ascend (NPU) test ────────────────────────────────────────────────────

def run_ascend_test(cfg, npu_script, api_name, out_dir, results):
    T = "[Ascend]"
    host, user, pw = cfg["host"], cfg["user"], cfg["password"]
    base = cfg["remote_dir"]
    wdir = f"{base}/{api_name}"
    target = f"{user}@{host}"

    askpass = _make_askpass(pw)
    env = _ssh_env(askpass)

    try:
        log(T, f"Connecting to {host} ...")

        # Create working directory
        ssh_run(target, f"mkdir -p {wdir}", env)
        log(T, f"Remote work dir: {wdir}")

        # Upload files
        sn = os.path.basename(npu_script)
        ok, err = scp_upload(target, npu_script, f"{wdir}/{sn}", env)
        if not ok:
            log(T, f"Upload {sn} failed: {err}")
            results["ascend_error"] = f"SCP upload failed: {err}"
            return

        fn = FILTER_NPU_SCRIPT.name
        scp_upload(target, str(FILTER_NPU_SCRIPT), f"{wdir}/{fn}", env)
        log(T, f"Uploaded: {sn}, {fn}")

        env_src = _env_source_cmd(cfg)
        remote_cmd = (
            f"cd {wdir} && "
            f"{{ {env_src}; true; }} && "
            f"export ASCEND_GLOBAL_LOG_LEVEL=0 && "
            f"export ASCEND_PROCESS_LOG_PATH={wdir} && "
            f"python {sn} > {wdir}/_stdout.txt 2>&1 & "
            f"SCRIPT_PID=$! && "
            f"echo SCRIPT_PID=$SCRIPT_PID && "
            f"wait $SCRIPT_PID; "
            f"echo EXIT_CODE=$?; "
            f"cat {wdir}/_stdout.txt"
        )

        log(T, "Running NPU test script ...")
        out, err, _ = ssh_run(target, remote_cmd, env, timeout=600)
        lines = out.strip().split("\n")

        pid, exit_code, json_result = None, -1, None
        for l in lines:
            l = l.strip()
            m = re.match(r"SCRIPT_PID=(\d+)", l)
            if m:
                pid = m.group(1)
            m2 = re.match(r"EXIT_CODE=(\d+)", l)
            if m2:
                exit_code = int(m2.group(1))
            if l.startswith("{") and "target_api" in l:
                try:
                    json_result = json.loads(l)
                except json.JSONDecodeError:
                    pass

        log(T, f"PID={pid}, exit_code={exit_code}")

        if exit_code != 0:
            log(T, f"Execution failed!\nstdout:\n{out[-1000:]}\nstderr:\n{err[-500:]}")
            results["ascend_error"] = f"exit_code={exit_code}"
            return

        if json_result:
            results["ascend"] = json_result
            log(T, f"Memory result: {json.dumps(json_result, ensure_ascii=False)}")
        else:
            log(T, f"Warning: no JSON output captured\n{out}")

        # Find plog log
        log(T, "Locating plog log ...")
        if pid:
            find_cmd = f"find {wdir} -name '*plog*{pid}*' -type f 2>/dev/null"
        else:
            find_cmd = f"find {wdir} -path '*/plog*' -type f 2>/dev/null"
        plog_out, _, _ = ssh_run(target, find_cmd, env)
        plog_files = [p.strip() for p in plog_out.strip().split("\n") if p.strip()]

        if not plog_files:
            find_cmd2 = f"find {wdir} -name 'plog*' 2>/dev/null"
            plog_out2, _, _ = ssh_run(target, find_cmd2, env)
            candidates = [p.strip() for p in plog_out2.strip().split("\n") if p.strip()]
            for cand in candidates:
                chk, _, _ = ssh_run(target, f"[ -f '{cand}' ] && echo FILE || echo DIR", env)
                if "FILE" in chk:
                    plog_files.append(cand)
                elif "DIR" in chk:
                    sub_out, _, _ = ssh_run(target, f"find '{cand}' -type f 2>/dev/null", env)
                    plog_files.extend(
                        s.strip() for s in sub_out.strip().split("\n") if s.strip()
                    )

        if plog_files:
            # Pick largest plog file
            best, best_sz = plog_files[0], 0
            for pf in plog_files:
                sz_out, _, _ = ssh_run(target, f"wc -c < '{pf}' 2>/dev/null", env)
                try:
                    sz = int(sz_out.strip())
                    if sz > best_sz:
                        best_sz, best = sz, pf
                except ValueError:
                    pass

            log(T, f"plog file: {best} ({best_sz:,} bytes)")

            filt_name = f"filtered_plog_{api_name.replace('.', '_')}.log"
            remote_filt = f"{wdir}/{filt_name}"
            filter_cmd = (
                f"cd {wdir} && "
                f"{{ {env_src}; true; }} && "
                f"python {fn} '{best}' -o '{remote_filt}' 2>&1"
            )
            fo, fe, frc = ssh_run(target, filter_cmd, env)

            if frc == 0:
                local_filt = os.path.join(out_dir, filt_name)
                ok2, err2 = scp_download(target, remote_filt, local_filt, env)
                if ok2:
                    results["ascend_plog"] = local_filt
                    log(T, f"Filtered plog downloaded: {local_filt}")
                else:
                    log(T, f"Download plog failed: {err2}")
            else:
                log(T, f"Filter execution failed:\n{fo}")
        else:
            log(T, "No plog file found")

    except Exception as e:
        log(T, f"Exception: {e}")
        results["ascend_error"] = str(e)
    finally:
        _cleanup_askpass(askpass)
        log(T, "Done")


def run_local_ascend_test(npu_script, api_name, out_dir, results):
    """Run NPU test script locally (no SSH). Sets plog env, parses JSON, filters plog."""
    T = "[Ascend]"
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["ASCEND_GLOBAL_LOG_LEVEL"] = "0"
    env["ASCEND_PROCESS_LOG_PATH"] = str(out_path.resolve())

    try:
        npu_path = Path(npu_script).resolve()
        log(T, "Running NPU test script locally ...")
        r = subprocess.run(
            [sys.executable, str(npu_path)],
            env=env,
            cwd=str(npu_path.parent),
            capture_output=True,
            text=True,
            timeout=600,
        )
        out = r.stdout or ""
        err = r.stderr or ""
        lines = out.strip().split("\n")

        pid, exit_code, json_result = None, -1, None
        for l in lines:
            l = l.strip()
            m = re.match(r"SCRIPT_PID=(\d+)", l)
            if m:
                pid = m.group(1)
            m2 = re.match(r"EXIT_CODE=(\d+)", l)
            if m2:
                exit_code = int(m2.group(1))
            if l.startswith("{") and "target_api" in l:
                try:
                    json_result = json.loads(l)
                except json.JSONDecodeError:
                    pass

        if exit_code == -1:
            exit_code = r.returncode

        log(T, f"PID={pid}, exit_code={exit_code}")

        if exit_code != 0:
            log(T, f"Execution failed!\nstdout:\n{out[-1000:]}\nstderr:\n{err[-500:]}")
            results["ascend_error"] = f"exit_code={exit_code}"
            return

        if json_result:
            results["ascend"] = json_result
            log(T, f"Memory result: {json.dumps(json_result, ensure_ascii=False)}")
        else:
            log(T, f"Warning: no JSON output captured\n{out}")

        log(T, "Locating plog log ...")
        root = out_path.resolve()
        plog_files = []
        if pid:
            for p in root.rglob("*"):
                if p.is_file() and "plog" in p.name and pid in p.name:
                    plog_files.append(p)
        if not plog_files:
            for p in root.rglob("*"):
                if p.is_file() and "plog" in p.name:
                    plog_files.append(p)
        if not plog_files:
            for p in root.rglob("plog*"):
                if p.is_file():
                    plog_files.append(p)
                elif p.is_dir():
                    for sub in p.rglob("*"):
                        if sub.is_file():
                            plog_files.append(sub)

        if plog_files:
            best = max(plog_files, key=lambda p: p.stat().st_size)
            best_sz = best.stat().st_size
            log(T, f"plog file: {best} ({best_sz:,} bytes)")

            filt_name = f"filtered_plog_{api_name.replace('.', '_')}.log"
            local_filt = out_path / filt_name
            fr = subprocess.run(
                [
                    sys.executable,
                    str(FILTER_NPU_SCRIPT),
                    str(best),
                    "-o",
                    str(local_filt),
                ],
                env=os.environ.copy(),
                capture_output=True,
                text=True,
                timeout=600,
            )
            if fr.returncode == 0:
                results["ascend_plog"] = str(local_filt.resolve())
                log(T, f"Filtered plog written: {local_filt}")
            else:
                log(T, f"Filter execution failed:\n{fr.stdout}")
        else:
            log(T, "No plog file found")

    except Exception as e:
        log(T, f"Exception: {e}")
        results["ascend_error"] = str(e)
    finally:
        log(T, "Done")


# ─── GPU test ──────────────────────────────────────────────────────────────

def run_gpu_test(cfg, gpu_script, api_name, out_dir, results):
    T = "[GPU]"
    host, user, pw = cfg["host"], cfg["user"], cfg["password"]
    base = cfg["remote_dir"]
    wdir = f"{base}/{api_name}"
    target = f"{user}@{host}"

    askpass = _make_askpass(pw)
    env = _ssh_env(askpass)

    try:
        log(T, f"Connecting to {host} ...")
        ssh_run(target, f"mkdir -p {wdir}", env)
        log(T, f"Remote work dir: {wdir}")

        sn = os.path.basename(gpu_script)
        ok, err = scp_upload(target, gpu_script, f"{wdir}/{sn}", env)
        if not ok:
            log(T, f"Upload {sn} failed: {err}")
            results["gpu_error"] = f"SCP upload failed: {err}"
            return
        log(T, f"Uploaded: {sn}")

        env_src = _env_source_cmd(cfg)
        remote_cmd = (
            f"cd {wdir} && "
            f"{{ {env_src}; true; }} && "
            f"python {sn} 2>&1; echo EXIT_CODE=$?"
        )

        log(T, "Running GPU test script ...")
        out, err, _ = ssh_run(target, remote_cmd, env, timeout=600)
        lines = out.strip().split("\n")

        exit_code, json_result = -1, None
        for l in lines:
            l = l.strip()
            m = re.match(r"EXIT_CODE=(\d+)", l)
            if m:
                exit_code = int(m.group(1))
            if l.startswith("{") and "target_api" in l:
                try:
                    json_result = json.loads(l)
                except json.JSONDecodeError:
                    pass

        log(T, f"exit_code={exit_code}")

        if exit_code != 0:
            log(T, f"Execution failed!\nstdout:\n{out[-800:]}\nstderr:\n{err[-500:]}")
            results["gpu_error"] = f"exit_code={exit_code}"
            return

        if json_result:
            results["gpu"] = json_result
            log(T, f"Memory result: {json.dumps(json_result, ensure_ascii=False)}")
        else:
            log(T, f"Warning: no JSON output captured\n{out}")

    except Exception as e:
        log(T, f"Exception: {e}")
        results["gpu_error"] = str(e)
    finally:
        _cleanup_askpass(askpass)
        log(T, "Done")


# ─── results ───────────────────────────────────────────────────────────────

def write_results(results, path, key_code=None, script_path=None, line_range=None):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    api = (results.get("ascend") or results.get("gpu") or {}).get(
        "target_api", "unknown"
    )
    L = [f"# Memory Benchmark: {api}", f"Time: {ts}\n"]
    
    if key_code:
        L.append("## Key Code")
        L.append("```python")
        L.append(key_code)
        L.append("```\n")

    if "ascend" in results:
        r = results["ascend"]
        L += [
            "## Ascend (NPU)",
            f"- target_api: {r['target_api']}",
            f"- 32aligned: {r.get('32aligned')}",
            f"- total_driver_GB: {r.get('total_driver_GB')}",
            f"- pta_reserved_GB: {r.get('pta_reserved_GB')}",
            f"- pta_activated_GB: {r.get('pta_activated_GB')}",
            "",
        ]
    if "ascend_error" in results:
        L += ["## Ascend (NPU) - ERROR", results["ascend_error"], ""]

    if "gpu" in results:
        r = results["gpu"]
        L += [
            "## GPU (CUDA)",
            f"- target_api: {r['target_api']}",
            f"- 32aligned: {r.get('32aligned')}",
            f"- total_driver_GB: {r.get('total_driver_GB')}",
            f"- gpu_reserved_GB: {r.get('gpu_reserved_GB')}",
            f"- gpu_activated_GB: {r.get('gpu_activated_GB')}",
            "",
        ]
    if "gpu_error" in results:
        L += ["## GPU (CUDA) - ERROR", results["gpu_error"], ""]

    if "ascend" in results and "gpu" in results:
        a, g = results["ascend"], results["gpu"]
        L += [
            "## Comparison (NPU vs GPU)",
            "| Metric | NPU | GPU | Delta | Ratio |",
            "|--------|-----|-----|-------|-------|",
        ]
        pairs = [
            ("total_driver_GB", a.get("total_driver_GB", 0), g.get("total_driver_GB", 0)),
            ("reserved_GB", a.get("pta_reserved_GB", 0), g.get("gpu_reserved_GB", 0)),
            ("activated_GB", a.get("pta_activated_GB", 0), g.get("gpu_activated_GB", 0)),
        ]
        for name, nv, gv in pairs:
            d = nv - gv
            ratio = nv / gv if gv else float("inf")
            L.append(
                f"| {name} | {nv:.4f} | {gv:.4f} | {d:+.4f} | {ratio:.2f}x |"
            )
        L.append("")

    if "ascend_plog" in results:
        L += ["## Plog", f"- Filtered plog: {results['ascend_plog']}", ""]

    content = "\n".join(L)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"\n{'=' * 60}")
    print(content)
    print(f"{'=' * 60}")
    print(f"Results saved: {path}")


# ─── main ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Remote memory test tool – NPU vs GPU",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python run_remote_mem_test.py npu_test.py gpu_test.py
              python run_remote_mem_test.py npu_test.py gpu_test.py --api-name torch.nanmean
              python run_remote_mem_test.py npu_test.py gpu_test.py --skip-gpu
              python run_remote_mem_test.py npu.py gpu.py --skip-gpu --gpu-json '{"target_api":"torch.x","total_driver_GB":1.0,"gpu_reserved_GB":1.0,"gpu_activated_GB":1.0}'
        """),
    )
    ap.add_argument("npu_script", help="NPU test script path")
    ap.add_argument("gpu_script", help="GPU test script path")
    ap.add_argument("--api-name", help="API name (default: extract TARGET_API from script)")
    ap.add_argument(
        "--servers",
        default=str(SERVERS_JSON),
        help=f"Path to servers.json (default: {SERVERS_JSON})",
    )
    ap.add_argument("--output-dir", help="Local output directory (default: same as NPU script)")
    ap.add_argument("--ascend-key", default="npu", help="Ascend server key (default: npu)")
    ap.add_argument("--gpu-key", default="gpu", help="GPU server key (default: gpu)")
    ap.add_argument("--skip-ascend", action="store_true", help="Skip Ascend test")
    ap.add_argument("--skip-gpu", action="store_true", help="Skip GPU test")
    ap.add_argument(
        "--local-npu",
        action="store_true",
        help="Run NPU test locally (when running on the Ascend server itself)",
    )
    ap.add_argument(
        "--gpu-json",
        help="Provide GPU result JSON manually (when skipping GPU remote test)",
    )
    args = ap.parse_args()

    if not os.path.isfile(args.npu_script):
        sys.exit(f"Error: file not found {args.npu_script}")
    if not args.skip_gpu and not os.path.isfile(args.gpu_script):
        sys.exit(f"Error: file not found {args.gpu_script}")
    need_servers_json = (not args.skip_gpu) or (
        not args.skip_ascend and not args.local_npu
    )
    if need_servers_json and not os.path.isfile(args.servers):
        sys.exit(f"Error: servers.json not found {args.servers}")
    if not FILTER_NPU_SCRIPT.is_file():
        sys.exit(f"Error: filter script not found {FILTER_NPU_SCRIPT}")

    api_name = args.api_name or extract_api_name(args.npu_script)
    if not api_name:
        sys.exit("Error: could not extract API name; use --api-name to specify")
    print(f"Target API: {api_name}")

    servers = load_servers(args.servers) if need_servers_json else {}
    out_dir = args.output_dir or os.path.dirname(os.path.abspath(args.npu_script))
    os.makedirs(out_dir, exist_ok=True)

    results = {}
    threads = []

    if not args.skip_ascend and not args.local_npu:
        if args.ascend_key not in servers:
            sys.exit(f"Error: server '{args.ascend_key}' not found")
        threads.append(
            threading.Thread(
                target=run_ascend_test,
                args=(servers[args.ascend_key], args.npu_script, api_name, out_dir, results),
                name="ascend",
            )
        )

    if not args.skip_gpu:
        if args.gpu_key not in servers:
            sys.exit(f"Error: server '{args.gpu_key}' not found")
        threads.append(
            threading.Thread(
                target=run_gpu_test,
                args=(servers[args.gpu_key], args.gpu_script, api_name, out_dir, results),
                name="gpu",
            )
        )
    elif args.gpu_json:
        gpu_json_str = args.gpu_json
        if os.path.isfile(gpu_json_str):
            with open(gpu_json_str, "r", encoding="utf-8") as f:
                gpu_json_str = f.read().strip()
        try:
            results["gpu"] = json.loads(gpu_json_str)
            print(f"Using manually provided GPU result: {gpu_json_str}")
        except json.JSONDecodeError as e:
            sys.exit(f"Error: invalid --gpu-json format: {e}")

    will_run_local_ascend = not args.skip_ascend and args.local_npu
    if not threads and "gpu" not in results and not will_run_local_ascend:
        sys.exit("Error: no test to run")

    t0 = time.time()
    for t in threads:
        t.start()
    if not args.skip_ascend and args.local_npu:
        run_local_ascend_test(args.npu_script, api_name, out_dir, results)
    for t in threads:
        t.join()
    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")

    result_file = os.path.join(out_dir, f"mem_results_{api_name.replace('.', '_')}.md")
    
    key_code, line_start, line_end = extract_key_code_lines(args.npu_script, api_name)
    if not key_code and not args.skip_gpu and os.path.isfile(args.gpu_script):
        key_code, line_start, line_end = extract_key_code_lines(args.gpu_script, api_name)
    
    script_src = args.npu_script if key_code else None
    line_range = (line_start, line_end) if key_code else None
    
    write_results(results, result_file, key_code, script_src, line_range)


if __name__ == "__main__":
    main()
