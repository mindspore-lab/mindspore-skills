#!/usr/bin/env python3
"""
Remote build & verify — push local source changes to the remote torch_npu repo,
compile, and re-run a memory test script to check if the fix resolves the issue.

Usage:
    python remote_build_verify.py <local_file> <remote_pta_path> <verify_cmd> [options]

Arguments:
    local_file       Local file that was modified (absolute path)
    remote_pta_path  Remote torch_npu source root (e.g. /home/user/pytorch)
    verify_cmd       Command to run after build for verification
                     (executed from remote_dir/<api_name>/ context)

Examples:
    python remote_build_verify.py ^
        d:/open_source/pytorch_npu/third_party/op-plugin/.../SomeKernel.cpp ^
        /home/user/pytorch ^
        "python torchapi_id0299_nanmean.py" ^
        --api-name torch.nanmean ^
        --container build_container
"""

import argparse
import base64
import os
import re
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path

from run_remote_mem_test import (
    load_servers, _make_askpass, _ssh_env, _cleanup_askpass,
    scp_upload, log,
)

SCRIPT_DIR = Path(__file__).parent.resolve()
SKILL_ROOT = SCRIPT_DIR.parent.parent
SERVERS_JSON = SKILL_ROOT / "references" / "servers.json"

SSH_BIN = os.environ.get("SSH_BIN", "ssh")
IS_WINDOWS = sys.platform.startswith("win")


def _local_run(cmd, timeout=600):
    """Run a shell command locally. Return (stdout, stderr, returncode)."""
    kwargs = {
        "shell": True,
        "stdin": subprocess.DEVNULL,
        "capture_output": True,
        "text": True,
        "timeout": timeout,
    }
    if os.name != "nt":
        kwargs["executable"] = "/bin/bash"
    r = subprocess.run(cmd, **kwargs)
    return r.stdout, r.stderr, r.returncode


def ssh_run(target, cmd, env, timeout=600, keep_alive=False):
    """Execute a remote command via SSH. Return (stdout, stderr, returncode).

    When *keep_alive* is True, ServerAliveInterval is set to prevent the
    connection from dropping during long-running builds.
    """
    opts = [
        "-o", "StrictHostKeyChecking=no",
        "-o", "PreferredAuthentications=keyboard-interactive,password",
    ]
    if keep_alive:
        opts += ["-o", "ServerAliveInterval=60", "-o", "ServerAliveCountMax=120"]
    r = subprocess.run(
        [SSH_BIN] + opts + [target, cmd],
        env=env, stdin=subprocess.DEVNULL,
        capture_output=True, text=True, timeout=timeout,
        start_new_session=(not IS_WINDOWS),
    )
    return r.stdout, r.stderr, r.returncode


def setup_ssh_auth(password):
    """Create SSH authentication env. Return (env_dict, askpass_path)."""
    askpass = _make_askpass(password)
    return _ssh_env(askpass), askpass


def _ssh_upload_b64(target, local_path, remote_path, env, chunk_size=800,
                    max_retries=3):
    """Upload a file via base64-encoded chunks over SSH.

    Avoids SCP which can be unreliable on some network/server combinations.
    """
    with open(local_path, "rb") as f:
        raw = f.read()
    b64 = base64.b64encode(raw).decode("ascii")
    local_size = len(raw)
    tmp = "/tmp/_build_upload.b64"

    ssh_run(target, f"rm -f {tmp}", env, timeout=15)

    chunks = [b64[i:i + chunk_size] for i in range(0, len(b64), chunk_size)]
    for idx, chunk in enumerate(chunks, 1):
        ok = False
        for attempt in range(1, max_retries + 1):
            out, _, rc = ssh_run(
                target,
                f"echo -n '{chunk}' >> {tmp} && echo CHUNK_OK",
                env, timeout=30,
            )
            if "CHUNK_OK" in out:
                ok = True
                break
            log("[UPLOAD]", f"chunk {idx}/{len(chunks)} attempt {attempt} failed, retrying ...")
            time.sleep(2)
        if not ok:
            log("[UPLOAD]", f"chunk {idx}/{len(chunks)} failed after {max_retries} retries")
            return False

    out, _, rc = ssh_run(
        target,
        f"base64 -d {tmp} > {remote_path} && wc -c < {remote_path}",
        env, timeout=30,
    )
    remote_size = 0
    for line in out.strip().split("\n"):
        line = line.strip()
        if line.isdigit():
            remote_size = int(line)
    if remote_size != local_size:
        log("[UPLOAD]", f"size mismatch: local={local_size}, remote={remote_size}")
        return False

    log("[UPLOAD]", f"OK ({len(chunks)} chunks, {local_size} bytes)")
    return True


def detect_python_version(target, env, container=None):
    """Detect the Python major.minor version on the remote host, locally, or inside container.

    When *target* is None, run detection on this machine (no SSH).
    """
    py_cmd = "python3 --version 2>&1 || python --version 2>&1"
    if container:
        full_cmd = f"docker exec {container} bash -c '{py_cmd}'"
    else:
        full_cmd = py_cmd

    if target is None:
        out, _, rc = _local_run(full_cmd, timeout=30)
    else:
        out, _, rc = ssh_run(target, full_cmd, env, timeout=30)
    m = re.search(r"Python\s+(\d+\.\d+)", out)
    if m:
        return m.group(1)
    return "3.10"


def resolve_relative_path(local_file, local_pta_root):
    """Compute the file's path relative to the local torch_npu repo root."""
    local_file = os.path.normpath(os.path.abspath(local_file))

    if local_pta_root:
        root = os.path.normpath(os.path.abspath(local_pta_root))
        if local_file.startswith(root):
            return local_file[len(root):].lstrip(os.sep).replace("\\", "/")

    parts = Path(local_file).parts
    for i, p in enumerate(parts):
        if p.lower() in ("pytorch_npu", "pytorch", "torch_npu"):
            return "/".join(parts[i + 1:])

    return Path(local_file).name


def main():
    ap = argparse.ArgumentParser(
        description="Push source changes, build torch_npu remotely, and verify memory fix",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python remote_build_verify.py ./SomeKernel.cpp /home/user/pytorch \\
                  "python torchapi_id0299_nanmean.py" --api-name torch.nanmean
        """),
    )
    ap.add_argument("local_file", help="Local modified source file")
    ap.add_argument("remote_pta_path", help="Remote torch_npu source root")
    ap.add_argument("verify_cmd", help="Verification command to run after build")
    ap.add_argument("--api-name", default="", help="API name for workdir resolution")
    ap.add_argument("--container", default="", help="Docker container name for build")
    ap.add_argument("--local-pta-root", default="",
                    help="Local torch_npu repo root (to compute relative path)")
    ap.add_argument("--servers", default=str(SERVERS_JSON))
    ap.add_argument("--server-key", default="npu")
    ap.add_argument("--env-script", default="",
                    help="Environment setup command (default: from servers.json env_script or source ~/.bashrc)")
    ap.add_argument("--local-build", action="store_true",
                    help="Build and verify locally (when running on the Ascend server itself)")
    args = ap.parse_args()

    local_file = os.path.abspath(args.local_file)
    if not os.path.isfile(local_file):
        sys.exit(f"Error: local file not found: {local_file}")

    remote_pta = args.remote_pta_path.rstrip("/")
    rel_path = resolve_relative_path(local_file, args.local_pta_root)
    remote_file = f"{remote_pta}/{rel_path}"

    T = "[BUILD]"
    t0 = time.time()

    if args.local_build:
        servers = load_servers(args.servers)
        if args.server_key not in servers:
            sys.exit(f"Error: server key '{args.server_key}' not found")

        cfg = servers[args.server_key]
        remote_dir = cfg["remote_dir"]

        if not args.env_script:
            es = cfg.get("env_script", "").strip()
            args.env_script = f"source {es}" if es else "source ~/.bashrc"

        log(T, "Mode          : local (no SSH)")
        log(T, f"Local file    : {local_file}")
        log(T, f"Relative path : {rel_path}")
        log(T, f"Local PTA dest: {remote_file}")
        log(T, f"Local PTA root: {remote_pta}")
        log(T, f"Container     : {args.container or '(none)'}")
        log(T, f"Verify cmd    : {args.verify_cmd}")

        try:
            # 1. Push: copy into local PTA tree
            log(T, "Copying modified source file ...")
            dest_path = os.path.join(remote_pta, *rel_path.split("/"))
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            shutil.copy2(local_file, dest_path)
            log(T, f"Copied to {dest_path}")

            # 2. Detect Python version
            py_ver = detect_python_version(None, None, args.container or None)
            log(T, f"Python version: {py_ver}")

            # 3. Build
            if args.container:
                build_cmd = (
                    f"docker exec {args.container} "
                    f"bash -c 'cd {remote_pta} && bash ci/build.sh --python={py_ver}'"
                )
            else:
                build_cmd = (
                    f"cd {remote_dir} && {args.env_script} && "
                    f"cd {remote_pta} && bash ci/build.sh --python={py_ver}"
                )

            log(T, "Building torch_npu (this may take several minutes) ...")
            log(T, f"  cmd: {build_cmd[:200]}...")
            out, err, rc = _local_run(build_cmd, timeout=3600)

            if rc != 0:
                log(T, f"Build FAILED (exit={rc})")
                log(T, "--- stdout (last 80 lines) ---")
                for ln in out.strip().split("\n")[-80:]:
                    print(f"  {ln}")
                log(T, "--- stderr (last 80 lines) ---")
                for ln in err.strip().split("\n")[-80:]:
                    print(f"  {ln}")
                sys.exit(1)

            log(T, "Build succeeded")

            build_tail = out.strip().split("\n")[-5:]
            for ln in build_tail:
                log(T, f"  {ln}")

            # 4. Verify
            api_dir = f"{remote_dir}/{args.api_name}" if args.api_name else remote_dir
            work_dir = api_dir
            check_cmd = f"test -d {api_dir}"
            _, _, chk_rc = _local_run(check_cmd, timeout=15)
            if chk_rc != 0:
                work_dir = remote_dir

            verify_full = (
                f"cd {work_dir} && {args.env_script} && "
                f"{args.verify_cmd} 2>&1"
            )

            log(T, "Running verification ...")
            out, err, rc = _local_run(verify_full, timeout=600)

            print("\n" + "=" * 72)
            print("  VERIFICATION OUTPUT")
            print("=" * 72)
            print(out)
            if err.strip():
                print("--- stderr ---")
                print(err)
            print("=" * 72)

            elapsed = time.time() - t0
            log(T, f"Verify exit_code={rc}")
            log(T, f"Total elapsed: {elapsed:.1f}s")

            return rc

        except Exception as e:
            log(T, f"Exception: {e}")
            raise

    servers = load_servers(args.servers)
    if args.server_key not in servers:
        sys.exit(f"Error: server key '{args.server_key}' not found")

    cfg = servers[args.server_key]
    host, user, pw = cfg["host"], cfg["user"], cfg["password"]
    remote_dir = cfg["remote_dir"]
    target = f"{user}@{host}"

    if not args.env_script:
        es = cfg.get("env_script", "").strip()
        args.env_script = f"source {es}" if es else "source ~/.bashrc"

    log(T, f"Target server : {host}")
    log(T, f"Local file    : {local_file}")
    log(T, f"Relative path : {rel_path}")
    log(T, f"Remote dest   : {remote_file}")
    log(T, f"Remote PTA    : {remote_pta}")
    log(T, f"Container     : {args.container or '(none)'}")
    log(T, f"Verify cmd    : {args.verify_cmd}")

    env, askpass = setup_ssh_auth(pw)

    try:
        # ── 1. Push modified file ─────────────────────────────────────────
        log(T, "Uploading modified source file ...")
        ok = _ssh_upload_b64(target, local_file, remote_file, env)
        if not ok:
            log(T, "base64 upload failed, falling back to SCP ...")
            ok, err = scp_upload(target, local_file, remote_file, env)
            if not ok:
                sys.exit(f"Error: upload failed: {err}")
        log(T, f"Uploaded to {remote_file}")

        # ── 2. Detect Python version ──────────────────────────────────────
        py_ver = detect_python_version(target, env, args.container or None)
        log(T, f"Python version: {py_ver}")

        # ── 3. Build ──────────────────────────────────────────────────────
        if args.container:
            ssh_build = (
                f"docker exec {args.container} "
                f"bash -c 'cd {remote_pta} && bash ci/build.sh --python={py_ver}'"
            )
        else:
            ssh_build = (
                f"cd {remote_dir} && {args.env_script} && "
                f"cd {remote_pta} && bash ci/build.sh --python={py_ver}"
            )

        log(T, "Building torch_npu (this may take several minutes) ...")
        log(T, f"  cmd: {ssh_build[:200]}...")
        out, err, rc = ssh_run(target, ssh_build, env, timeout=3600, keep_alive=True)

        if rc != 0:
            log(T, f"Build FAILED (exit={rc})")
            log(T, "--- stdout (last 80 lines) ---")
            for ln in out.strip().split("\n")[-80:]:
                print(f"  {ln}")
            log(T, "--- stderr (last 80 lines) ---")
            for ln in err.strip().split("\n")[-80:]:
                print(f"  {ln}")
            sys.exit(1)

        log(T, "Build succeeded")

        build_tail = out.strip().split("\n")[-5:]
        for ln in build_tail:
            log(T, f"  {ln}")

        # ── 4. Verify ─────────────────────────────────────────────────────
        api_dir = f"{remote_dir}/{args.api_name}" if args.api_name else remote_dir
        work_dir = api_dir
        check_cmd = f"test -d {api_dir}"
        _, _, chk_rc = ssh_run(target, check_cmd, env, timeout=15)
        if chk_rc != 0:
            work_dir = remote_dir

        verify_full = (
            f"cd {work_dir} && {args.env_script} && "
            f"{args.verify_cmd} 2>&1"
        )

        log(T, "Running verification ...")
        out, err, rc = ssh_run(target, verify_full, env, timeout=600, keep_alive=True)

        print("\n" + "=" * 72)
        print("  VERIFICATION OUTPUT")
        print("=" * 72)
        print(out)
        if err.strip():
            print("--- stderr ---")
            print(err)
        print("=" * 72)

        elapsed = time.time() - t0
        log(T, f"Verify exit_code={rc}")
        log(T, f"Total elapsed: {elapsed:.1f}s")

        return rc

    except Exception as e:
        log(T, f"Exception: {e}")
        raise
    finally:
        if askpass:
            _cleanup_askpass(askpass)


if __name__ == "__main__":
    sys.exit(main() or 0)
