#!/usr/bin/env python3
"""
Automated memory test environment preparation — locate NPU/GPU scripts,
generate GPU scripts, run remote tests, and validate result files in one step.

Pipeline:
  1. Locate NPU test script (torchapi_id{num}_{api_name}.py)
  2. Find or generate GPU test script (torchapi_id{num}_{api_name}_gpu.py)
  3. Find or generate mem_results_{api_name}.md and filtered_plog_{api_name}.log
  4. Validate all 4 required files are present

Usage:
    python execute_mem_test.py <file_or_directory_path>

Examples:
    python execute_mem_test.py /home/user/tests/torchapi_id0299_nanmean.py
    python execute_mem_test.py /home/user/tests/
"""

import argparse
import os
import re
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NPU_PATTERN = re.compile(r"^torchapi_id(\d+)_(.+)\.py$")


def find_npu_script(user_path: str):
    """Locate the NPU test script from the user-provided path. Return (abs_path, num, api_name)."""
    user_path = os.path.abspath(user_path)

    if os.path.isfile(user_path):
        basename = os.path.basename(user_path)
        if basename.endswith("_gpu.py"):
            print(f"[ERROR] The given file is a GPU script; please provide the NPU script path: {basename}")
            sys.exit(1)
        m = NPU_PATTERN.match(basename)
        if not m:
            print(f"[ERROR] Filename does not match 'torchapi_id{{num}}_{{api_name}}.py' format: {basename}")
            print("        Please check the filename and try again.")
            sys.exit(1)
        return user_path, m.group(1), m.group(2)

    if os.path.isdir(user_path):
        matches = []
        for fname in os.listdir(user_path):
            if fname.endswith("_gpu.py"):
                continue
            m = NPU_PATTERN.match(fname)
            if m:
                matches.append((os.path.join(user_path, fname), m.group(1), m.group(2)))
        if not matches:
            print(f"[ERROR] No file matching 'torchapi_id{{num}}_{{api_name}}.py' found in directory: {user_path}")
            print("        Please provide a valid path.")
            sys.exit(1)
        if len(matches) > 1:
            print(f"[ERROR] Multiple matching files found in directory:")
            for p, n, a in matches:
                print(f"        - {os.path.basename(p)}")
            print("        Please provide the absolute path to the specific file.")
            sys.exit(1)
        return matches[0]

    print(f"[ERROR] Path does not exist: {user_path}")
    sys.exit(1)


def ensure_gpu_script(npu_script: str, num: str, api_name: str):
    """Find or generate the GPU test script. Return the GPU script path."""
    script_dir = os.path.dirname(npu_script)
    gpu_script = os.path.join(script_dir, f"torchapi_id{num}_{api_name}_gpu.py")

    if os.path.isfile(gpu_script):
        print(f"[INFO] GPU script already exists: {gpu_script}")
        return gpu_script

    print(f"[INFO] GPU script not found, generating...")
    converter = os.path.join(SCRIPT_DIR, "convert_npu_to_gpu.py")
    ret = subprocess.run(
        [sys.executable, converter, npu_script],
        capture_output=True, text=True,
    )
    if ret.returncode != 0:
        print(f"[SCRIPT_ERROR] GPU script generation failed (possible NPU script content issue):")
        print(f"[SCRIPT_ERROR] NPU script path: {npu_script}")
        if ret.stderr:
            print(ret.stderr)
        if ret.stdout:
            print(ret.stdout)
        sys.exit(2)

    if not os.path.isfile(gpu_script):
        print(f"[ERROR] GPU script was not found at expected path after generation: {gpu_script}")
        sys.exit(1)

    print(f"[INFO] GPU script generated: {gpu_script}")
    return gpu_script


def ensure_mem_results(npu_script: str, gpu_script: str, api_name: str, local_npu: bool = False):
    """Find or generate mem_results and filtered_plog files."""
    script_dir = os.path.dirname(npu_script)
    mem_results = os.path.join(script_dir, f"mem_results_{api_name}.md")
    filtered_plog = os.path.join(script_dir, f"filtered_plog_{api_name}.log")

    if os.path.isfile(mem_results) and os.path.isfile(filtered_plog):
        print(f"[INFO] Test result files already exist:")
        print(f"       - {mem_results}")
        print(f"       - {filtered_plog}")
        return mem_results, filtered_plog

    print(f"[INFO] Test result files incomplete, running test...")
    runner = os.path.join(SCRIPT_DIR, "run_remote_mem_test.py")
    cmd = [sys.executable, runner, npu_script, gpu_script, "--api-name", api_name]
    if local_npu:
        cmd.append("--local-npu")
    ret = subprocess.run(
        cmd,
        capture_output=True, text=True,
    )
    if ret.returncode != 0:
        print(f"[SCRIPT_ERROR] Remote test execution failed (possible test script runtime error):")
        print(f"[SCRIPT_ERROR] NPU script: {npu_script}")
        print(f"[SCRIPT_ERROR] GPU script: {gpu_script}")
        if ret.stderr:
            print(ret.stderr)
        if ret.stdout:
            print(ret.stdout)
        sys.exit(2)

    if ret.stdout:
        print(ret.stdout)

    return mem_results, filtered_plog


def final_check(npu_script: str, gpu_script: str, api_name: str):
    """Validate that all 4 required output files are present."""
    script_dir = os.path.dirname(npu_script)
    required = {
        "NPU test script": npu_script,
        "GPU test script": gpu_script,
        "Memory report": os.path.join(script_dir, f"mem_results_{api_name}.md"),
        "Filtered plog": os.path.join(script_dir, f"filtered_plog_{api_name}.log"),
    }

    missing = []
    for label, path in required.items():
        if not os.path.isfile(path):
            missing.append(f"  - {label}: {path}")

    if missing:
        print("[ERROR] The following files are missing:")
        for m in missing:
            print(m)
        sys.exit(1)

    print("[SUCCESS] All required files are ready:")
    for label, path in required.items():
        print(f"  [OK] {label}: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare memory test environment (locate script → generate GPU script → run test → validate results)",
    )
    parser.add_argument(
        "path",
        help="NPU test script file path or directory containing the script",
    )
    parser.add_argument("--local-npu", action="store_true",
                        help="Run NPU test locally (when running on the Ascend server itself)")
    args = parser.parse_args()
    local_npu = args.local_npu

    print("=" * 60)
    print("  Memory test environment preparation")
    print("=" * 60)

    # Step 1
    print("\n[Step 1/4] Locating NPU test script...")
    npu_script, num, api_name = find_npu_script(args.path)
    print(f"[INFO] NPU script: {npu_script}")
    print(f"[INFO] API id: {num}, API name: {api_name}")

    # Step 2
    print(f"\n[Step 2/4] Preparing GPU test script...")
    gpu_script = ensure_gpu_script(npu_script, num, api_name)

    # Step 3
    print(f"\n[Step 3/4] Preparing test result files...")
    ensure_mem_results(npu_script, gpu_script, api_name, local_npu=local_npu)

    # Step 4
    print(f"\n[Step 4/4] Final file validation...")
    final_check(npu_script, gpu_script, api_name)

    print("\n" + "=" * 60)
    print("  Preparation complete. Proceed to analysis stage.")
    print("=" * 60)


if __name__ == "__main__":
    main()
