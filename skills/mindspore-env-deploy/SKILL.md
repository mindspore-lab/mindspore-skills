---
name: mindspore-env-deploy
description: Deploy a complete MindSpore development environment with CANN support for Huawei Ascend NPU. Handles package management, CANN installation, MindSpore setup, and environment verification.
---

# MindSpore Environment Deployment

Deploy a complete MindSpore development environment with CANN (Compute Architecture for Neural Networks) support for Huawei Ascend NPU.

## When to Use

Use this skill when:
- Setting up a new MindSpore development environment
- Installing MindSpore with Ascend NPU support
- Deploying CANN packages for Huawei Ascend chips
- Verifying MindSpore installation and runtime

## Quick Start

1. Choose package manager (uv or conda)
2. Download and install CANN packages
3. Install MindSpore wheel package
4. Verify environment with runcheck

## Instructions

### Step 1: Package Management Environment

First, determine which package manager to use and the Python version.

**Ask the user:**
- Which package manager to use: `uv` (recommended, faster) or `conda`
- Which Python version to use (default: 3.12)

**For uv (recommended):**

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment with specified Python version
uv venv --python 3.12 mindspore-env
source mindspore-env/bin/activate  # Linux/macOS
# or
mindspore-env\Scripts\activate  # Windows
```

**For conda:**

```bash
# Create conda environment
conda create -n mindspore-env python=3.12
conda activate mindspore-env
```

**Network troubleshooting:**

If network issues occur, ask the user about proxy configuration:

```bash
# For uv
export HTTP_PROXY=http://proxy.example.com:8080
export HTTPS_PROXY=http://proxy.example.com:8080

# For conda
conda config --set proxy_servers.http http://proxy.example.com:8080
conda config --set proxy_servers.https http://proxy.example.com:8080

# For pip (used by both)
pip config set global.proxy http://proxy.example.com:8080
```

Common network solutions:
- Use a mirror/registry (e.g., Tsinghua, Aliyun for China users)
- Configure corporate proxy settings
- Check firewall rules
- Verify DNS resolution

### Step 2: Download and Install CANN Packages

CANN (Compute Architecture for Neural Networks) is required for Ascend NPU support.

**Download CANN packages:**

Visit the Ascend CANN official website:
- URL: https://www.hiascend.com/software/cann/community
- Select the appropriate version based on:
  - Operating system (Linux x86_64, Linux aarch64)
  - CANN version (recommend latest stable, e.g., 8.0.RC3)
  - Python version (must match your environment)

**Required CANN packages:**
1. `Ascend-cann-toolkit` - Core toolkit
2. `Ascend-cann-kernels-{arch}` - Kernel libraries (910 for training, 310 for inference)

**Installation steps:**

```bash
# Extract packages
tar -xzf Ascend-cann-toolkit_*.tar.gz
tar -xzf Ascend-cann-kernels-*.tar.gz

# Install toolkit
cd Ascend-cann-toolkit
./install.sh --install-path=/usr/local/Ascend

# Install kernels
cd ../Ascend-cann-kernels-*
./install.sh --install-path=/usr/local/Ascend

# Set environment variables
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

**Add to shell profile for persistence:**

```bash
echo "source /usr/local/Ascend/ascend-toolkit/set_env.sh" >> ~/.bashrc
# or for zsh
echo "source /usr/local/Ascend/ascend-toolkit/set_env.sh" >> ~/.zshrc
```

**Verify CANN installation:**

```bash
npu-smi info  # Check NPU device status
```

### Step 3: Install MindSpore

Download and install the MindSpore wheel package matching your CANN version.

**Download MindSpore:**

Visit MindSpore official website:
- URL: https://www.mindspore.cn/install
- Select:
  - Hardware platform: Ascend
  - CANN version (must match Step 2)
  - Python version (must match Step 1)
  - MindSpore version (recommend latest stable)

**Installation:**

```bash
# Install from downloaded wheel
pip install mindspore-*.whl

# Or install from PyPI (if available for your configuration)
pip install mindspore
```

**For development/nightly builds:**

```bash
# Install from MindSpore daily builds
pip install mindspore --index-url https://pypi.mindspore.cn/simple
```

**Network troubleshooting for pip:**

```bash
# Use mirror (China users)
pip install mindspore -i https://pypi.tuna.tsinghua.edu.cn/simple

# Increase timeout
pip install mindspore --timeout 300

# Use proxy
pip install mindspore --proxy http://proxy.example.com:8080
```

### Step 4: Verify Installation with runcheck

Run MindSpore's built-in verification to ensure the environment is correctly configured.

**Basic verification:**

```python
import mindspore as ms
print(ms.__version__)

# Run environment check
ms.run_check()
```

**Expected output:**
```
MindSpore version: 2.x.x
The result of multiplication calculation is correct, MindSpore has been installed on platform [Ascend] successfully!
```

**If runcheck fails, troubleshoot:**

1. **Import error:**
   ```bash
   # Check Python path
   which python
   python -c "import sys; print(sys.path)"

   # Reinstall MindSpore
   pip uninstall mindspore
   pip install mindspore-*.whl
   ```

2. **CANN not found:**
   ```bash
   # Verify CANN environment variables
   echo $ASCEND_TOOLKIT_HOME
   echo $LD_LIBRARY_PATH

   # Re-source CANN environment
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   ```

3. **NPU device not detected:**
   ```bash
   # Check NPU status
   npu-smi info

   # Check driver
   cat /usr/local/Ascend/driver/version.info

   # Restart NPU service (requires root)
   sudo systemctl restart ascend-hccl.service
   ```

4. **Version mismatch:**
   - Ensure CANN version matches MindSpore requirements
   - Check compatibility matrix: https://www.mindspore.cn/install
   - Reinstall matching versions

5. **Permission issues:**
   ```bash
   # Add user to HwHiAiUser group
   sudo usermod -a -G HwHiAiUser $USER

   # Re-login or use newgrp
   newgrp HwHiAiUser
   ```

**Advanced verification:**

```python
import mindspore as ms
from mindspore import Tensor, ops

# Set device target
ms.set_context(device_target="Ascend", device_id=0)

# Test basic operation
x = Tensor([1.0, 2.0, 3.0])
y = Tensor([4.0, 5.0, 6.0])
result = ops.add(x, y)
print(f"Add result: {result}")

# Test NPU allocation
print(f"Device: {ms.get_context('device_target')}")
print(f"Device ID: {ms.get_context('device_id')}")
```

## Common Issues and Solutions

### Network Issues

| Issue | Solution |
|-------|----------|
| Slow download | Use mirror sites or download manager |
| Connection timeout | Configure proxy, increase timeout |
| SSL certificate error | Use `--trusted-host` for pip |
| DNS resolution failure | Check `/etc/resolv.conf`, use public DNS |

### CANN Installation Issues

| Issue | Solution |
|-------|----------|
| Permission denied | Use `sudo` or check install path permissions |
| Library not found | Source `set_env.sh` correctly |
| Version conflict | Uninstall old CANN, clean `/usr/local/Ascend` |
| Driver mismatch | Update Ascend driver to match CANN version |

### MindSpore Installation Issues

| Issue | Solution |
|-------|----------|
| Wheel not compatible | Check Python version, platform, CANN version |
| Dependency conflict | Create fresh virtual environment |
| Import error | Verify `LD_LIBRARY_PATH` includes CANN libs |
| Segmentation fault | Check CANN installation, driver version |

## Environment Variables Reference

Key environment variables for MindSpore + CANN:

```bash
# CANN toolkit path
export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/latest
export LD_LIBRARY_PATH=${ASCEND_TOOLKIT_HOME}/lib64:$LD_LIBRARY_PATH
export PATH=${ASCEND_TOOLKIT_HOME}/bin:$PATH
export PYTHONPATH=${ASCEND_TOOLKIT_HOME}/python/site-packages:$PYTHONPATH

# CANN OPP path (operator packages)
export ASCEND_OPP_PATH=${ASCEND_TOOLKIT_HOME}/opp

# CANN AICPU path
export ASCEND_AICPU_PATH=${ASCEND_TOOLKIT_HOME}/

# Optional: Enable debug logging
export GLOG_v=1
export ASCEND_SLOG_PRINT_TO_STDOUT=1
```

## Quick Reference

### Package Manager Comparison

| Feature | uv | conda |
|---------|----|----|
| Speed | Very fast | Moderate |
| Disk usage | Minimal | Higher |
| Python version management | Yes | Yes |
| Popularity | Growing | Established |
| Recommendation | ✅ Preferred | ✅ Also good |

### Version Compatibility Matrix

Check official compatibility at:
- MindSpore: https://www.mindspore.cn/install
- CANN: https://www.hiascend.com/software/cann/community

Example compatibility:
- MindSpore 2.3.x → CANN 8.0.RC3
- MindSpore 2.2.x → CANN 7.0.x
- Python 3.9-3.12 supported

### Useful Commands

```bash
# Check versions
python --version
pip list | grep mindspore
npu-smi info

# Environment info
ms.run_check()
ms.get_context('device_target')

# Clean reinstall
pip uninstall mindspore
rm -rf ~/.cache/pip
pip install mindspore-*.whl
```

## Next Steps

After successful installation:
1. Try MindSpore tutorials: https://www.mindspore.cn/tutorials
2. Explore operator development with other skills in this repository
3. Run sample models to verify performance
4. Set up development tools (IDE, debugger, profiler)
