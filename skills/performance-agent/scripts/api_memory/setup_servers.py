#!/usr/bin/env python3
"""
Interactive server configuration generator for API memory consistency analysis.

Checks for an existing servers.json in the references directory, offers to
reuse it, and otherwise collects NPU and GPU server details from the user.

When running on an Ascend (NPU) server (auto-detected), only GPU server
information is collected since NPU tests run locally.

Usage:
    python setup_servers.py [--force]

Options:
    --force   Skip the existing-config check and always prompt for new input
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
SKILL_ROOT = SCRIPT_DIR.parent.parent
REFERENCES_DIR = SKILL_ROOT / "references"
SERVERS_JSON = REFERENCES_DIR / "servers.json"


def is_npu_server():
    """Detect whether this machine is an Ascend (NPU) server."""
    try:
        r = subprocess.run(
            ["npu-smi", "info"],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode == 0:
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    try:
        r = subprocess.run(
            [sys.executable, "-c", "import torch_npu; print(torch_npu.__version__)"],
            capture_output=True, text=True, timeout=15,
        )
        if r.returncode == 0 and r.stdout.strip():
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return False


def read_input(prompt, default=""):
    """Read a line from stdin with an optional default value."""
    if default:
        value = input(f"  {prompt} [{default}]: ").strip()
        return value if value else default
    return input(f"  {prompt}: ").strip()


def collect_server_info(label):
    """Interactively collect host, user, password, remote_dir, env_script for one server."""
    print(f"\n--- {label} ---")
    host = read_input("host (IP or hostname)")
    user = read_input("user")
    password = read_input("password")
    remote_dir = read_input("remote_dir (remote working directory)", "/home/user/pta_mem_issue")
    env_script = read_input("env_script (absolute path to env setup script, optional)")
    info = {
        "host": host,
        "user": user,
        "password": password,
        "remote_dir": remote_dir,
    }
    if env_script:
        info["env_script"] = env_script
    return info


def show_config(config, local_npu=False):
    """Pretty-print the server configuration."""
    print("\n" + "=" * 56)
    print("  Server Configuration")
    print("=" * 56)
    servers = config.get("servers", {})
    if "npu" in servers:
        s = servers["npu"]
        if local_npu:
            print(f"  NPU : local (this machine)")
        else:
            print(f"  NPU : {s['user']}@{s['host']}")
            print(f"    remote_dir: {s['remote_dir']}")
    if "gpu" in servers:
        s = servers["gpu"]
        print(f"  GPU         : {s['user']}@{s['host']}")
        print(f"    remote_dir: {s['remote_dir']}")
    print("=" * 56)


def build_config(local_npu):
    """Collect server info from the user and build the config dict."""
    while True:
        servers = {}

        if not local_npu:
            print("\nPlease enter NPU (Ascend) server information:")
            npu_info = collect_server_info("NPU Server")
            npu_info["description"] = "Ascend 910B dev server"
            servers["npu"] = npu_info
        else:
            print("\n[INFO] Running on NPU server — NPU tests will run locally.")
            print("[INFO] Only GPU server information is needed.\n")

        print("Please enter GPU server information:")
        gpu_info = collect_server_info("GPU Server")
        gpu_info["description"] = "GPU dev server"
        servers["gpu"] = gpu_info

        config = {
            "servers": servers,
            "default": "npu",
        }

        show_config(config, local_npu=local_npu)

        choice = input("\nConfirm this configuration? (y = save / n = re-enter): ").strip().lower()
        if choice in ("y", "yes", ""):
            return config


def main():
    ap = argparse.ArgumentParser(
        description="Generate servers.json for API memory consistency analysis",
    )
    ap.add_argument("--force", action="store_true",
                    help="Skip existing-config check and always prompt for new input")
    args = ap.parse_args()

    print("=" * 56)
    print("  API Memory Consistency — Server Configuration Setup")
    print("=" * 56)

    local_npu = is_npu_server()
    if local_npu:
        print("\n[DETECTED] This machine is an Ascend (NPU) server.")
    else:
        print("\n[DETECTED] This machine is NOT an NPU server.")
        print("           Both NPU and GPU remote server info are required.")

    if not args.force and SERVERS_JSON.is_file():
        print(f"\n[INFO] Existing configuration found: {SERVERS_JSON}")
        try:
            with open(SERVERS_JSON, "r", encoding="utf-8") as f:
                existing = json.load(f)
            show_config(existing, local_npu=local_npu)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"[WARN] Failed to parse existing config: {e}")
            existing = None

        if existing:
            choice = input("\nUse this existing configuration? (y = use / n = re-enter): ").strip().lower()
            if choice in ("y", "yes", ""):
                print("[INFO] Using existing configuration.")
                return

    config = build_config(local_npu)

    REFERENCES_DIR.mkdir(parents=True, exist_ok=True)
    with open(SERVERS_JSON, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"\n[SUCCESS] Configuration saved to: {SERVERS_JSON}")


if __name__ == "__main__":
    main()
