---
name: compile-macos
description: Use when compiling MindSpore from source on macOS Apple Silicon, troubleshooting build failures, or setting up MindSpore development environment
---

# MindSpore macOS Compilation

## Overview

Systematic workflow for compiling MindSpore from source on macOS Apple Silicon. Core principle: verify environment at each stage before proceeding to avoid cascading failures.

## When to Use

**Use this skill when:**
- User requests MindSpore compilation from source
- Building on macOS Apple Silicon (M1/M2/M3)
- Troubleshooting build failures or dependency errors
- Setting up development environment for MindSpore
- Keywords: "compile", "build from source", "编译", "源码编译", "compilation error"

**Don't use for:**
- Installing pre-built MindSpore packages (use pip/conda instead)
- Linux or Windows compilation (different toolchain)
- Runtime errors after successful installation

## Quick Reference

| Stage | Key Command | Verification |
|-------|-------------|--------------|
| 1. Environment (FIRST) | `conda activate <env_name>` | `which python` points to conda env, version 3.9-3.12 |
| 2. Dependencies | `conda install cmake=3.22.3 patch autoconf -y` + `pip install ... pybind11` | `which cmake` points to conda env |
| 3. Source | `git clone https://gitcode.com/mindspore/mindspore.git` | `build.sh` exists |
| 4. Build | `bash build.sh -e cpu -S on -j8` | Check `output/` directory |
| 5. Install | `pip install output/mindspore-*.whl` | `import mindspore` works |
| 6. Verify | `mindspore.run_check()` | Prints success message |

**Typical build time:** 30-60 minutes (first build)
**Disk space required:** 20GB minimum
**Troubleshooting:** See `reference/troubleshooting.md` for error patterns

## Prerequisites

- **OS**: macOS (Apple Silicon)
- **Compiler**: Apple Clang
- **Disk Space**: At least 20GB

## Compilation Steps

### Step 1: Set Up and Activate Conda Environment (REQUIRED FIRST)

**CRITICAL**: All subsequent steps MUST run within an activated conda environment with Python 3.9-3.12.

**First, check if conda is installed:**
```bash
conda --version
```

**Check existing environments:**
```bash
conda env list
```

**MUST ask the user to choose:**
1. Create a new conda environment (MUST ask which Python version: 3.9, 3.10, 3.11, or 3.12)
2. Use an existing conda environment (MUST ask for environment name)

**For new environment:**
```bash
# Create environment with user-specified Python version
# Example: conda create -n mindspore_py310 python=3.10 -y
conda create -n <env_name> python=<version> -y
conda activate <env_name>

# Verify activation and Python version
python --version
which python
```

**For existing environment:**
```bash
# Activate existing environment
conda activate <existing_env_name>

# Verify Python version is supported (3.9-3.12)
python --version
which python
```

**STOP HERE if environment is not activated.** All following commands assume you are in the activated conda environment.

### Step 2: Check Dependencies (Within Activated Environment)

**PREREQUISITE**: Conda environment must be activated (Step 1).

#### System Tools

**Xcode Command Line Tools** (Required)
```bash
xcode-select -p
# If not installed, prompt user to run:
# xcode-select --install
```

**Install build tools via conda** (within activated environment)
```bash
# These install into the active conda environment
conda install cmake=3.22.3 patch autoconf scipy -y

# Verify cmake is from conda environment
which cmake
cmake --version
```

#### Python Packages

```bash
# Install within activated conda environment
pip install wheel==0.46.3 PyYAML==6.0.2 numpy==1.26.4 pybind11 -i https://repo.huaweicloud.com/repository/pypi/simple/

# Verify packages are installed in conda environment
pip list | grep -E "wheel|PyYAML|numpy|pybind11"
```

### Step 3: Prepare Source Code

**Logic**:
1. Check if already in MindSpore source directory → Success
2. Check if `./mindspore` exists in current directory → `cd mindspore` → Success
3. Otherwise → Clone to current directory → `cd mindspore`

```bash
# Check if in MindSpore source directory (check for build.sh)
if [ -f "build.sh" ]; then
    echo "Already in MindSpore source directory"
elif [ -d "mindspore" ] && [ -f "mindspore/build.sh" ]; then
    echo "Found MindSpore in ./mindspore"
    cd mindspore
else
    echo "Cloning MindSpore source code..."
    git clone -b master https://gitcode.com/mindspore/mindspore.git ./mindspore
    cd mindspore
fi

# Ask user whether to update source code
git fetch origin
git checkout master
git pull origin master
```

### Step 4: Compile MindSpore (Within Activated Environment)

**PREREQUISITE**: Conda environment must be activated with all dependencies installed.

Set environment variables:

```bash
# Use .mslib in current directory for cache
export MSLIBS_CACHE_PATH=$(pwd)/.mslib
export CC=/usr/bin/clang
export CXX=/usr/bin/clang++

# Verify environment
echo "Python: $(which python)"
echo "CMake: $(which cmake)"
echo "CC: $CC"
echo "CXX: $CXX"
```

Execute compilation:

```bash
# Ensure in MindSpore source directory and conda environment is active
bash build.sh -e cpu -S on -j8
```

**Parameters**:
- `MSLIBS_CACHE_PATH`: Cache path for third-party libraries
- `-e cpu`: CPU-only build
- `-S on`: Enable symbol table
- `-j8`: Use 8 threads (adjust based on CPU cores)

### Step 5: Install MindSpore

```bash
# Re-install wheel package
pip uninstall mindspore -y
pip install output/mindspore-*.whl -i https://repo.huaweicloud.com/repository/pypi/simple/
```

### Step 6: Verify Installation

**Basic check:**
```bash
python -c "import mindspore;print(mindspore.__version__)"
python -c "import mindspore;mindspore.set_device(device_target='CPU');mindspore.run_check()"
```

**Expected:**
```
MindSpore version: [version number]
The result of multiplication calculation is correct, MindSpore has been installed on platform [CPU] successfully!
```

**CI test suite (Ask the user to run or not):**
```bash
cd /path/to/mindspore/tests/st
pip install pytest torch torchvision torchaudio
export KMP_DUPLICATE_LIB_OK=TRUE

# Collect tests with markers: (cpu_macos OR platform_arm_cpu) AND level0
pytest --collect-only -m "(cpu_macos or platform_arm_cpu) and level0" -q 2>/dev/null | grep "::" > /tmp/ci_tests.txt

while IFS= read -r test; do
    echo "Running: $test"
    if pytest "$test" -v; then
        echo "✅ PASSED: $test"
    else
        echo "❌ FAILED: $test"
    fi
done < /tmp/ci_tests.txt
```

**Expected:** All tests pass. If any failures occur, analyze root cause and guide user to fix.

## Common Mistakes

| Mistake | Symptom | Fix |
|---------|---------|-----|
| Wrong Python version | Import errors, ABI mismatch | Use Python 3.9-3.12 |
| Missing Xcode tools | `clang: command not found` | Run `xcode-select --install` |
| Insufficient disk space | Build fails mid-compilation | Free up 20GB+ before starting |
| Skipping environment variables | Linker errors, missing symbols | Set CC, CXX, LIBRARY_PATH, LDFLAGS |
| Not in source directory | `build.sh: No such file` | Verify `pwd` shows mindspore/ |
| Reusing old cache with new source | Cryptic build failures | Clear `.mslib/` and rebuild |
| Installing without uninstalling old version | Version conflicts | `pip uninstall mindspore -y` first |
| Missing pybind11 | `'pybind11/pybind11.h' file not found` | Install: `pip install pybind11`, then clean CMake cache and rebuild |
| Stale CMake cache | Same error persists after installing missing package | Delete `build/mindspore/CMakeCache.txt` and `build/mindspore/CMakeFiles/`, then rebuild |
| Missing PyTorch (tests) | Many tests skipped/not collected | `pip install torch torchvision torchaudio` |
| OpenMP conflict (tests) | `OMP: Error #15: Initializing libiomp5.dylib` | `export KMP_DUPLICATE_LIB_OK=TRUE` |

**When build fails:**
1. Check `reference/troubleshooting.md` for matching error pattern
2. Verify all environment variables are set
3. Confirm you're in the correct directory
4. Check disk space: `df -h .`

## User Interaction Guidelines

- **ALWAYS activate conda environment FIRST**: Before any dependency checks or compilation steps
- **Verify environment activation**: Use `which python` and `python --version` to confirm
- **Always check conda first**: Run `conda --version` to verify installation before proceeding
- **Always ask first**: Whether to create a new conda environment (with Python version 3.9-3.12) or use an existing one
- Explain each major step before execution
- Ask user whether to update if source directory exists
- Wait for user to install Xcode Command Line Tools if missing
- Display version info and verification results after completion
- **When compilation fails**: First consult `reference/troubleshooting.md` for matching error patterns and solutions before suggesting generic fixes
- Provide error log location and context-specific solutions based on troubleshooting history
