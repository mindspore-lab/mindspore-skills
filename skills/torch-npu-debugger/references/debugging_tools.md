# torch_npu 调试工具指南

## 目录

1. [环境变量调试开关](#1-环境变量调试开关)
2. [OpHook (C++ 算子 Hook)](#2-ophook-c-算子-hook)
3. [NPU Trace (Python 层 Callback)](#3-npu-trace-python-层-callback)
4. [算子 Dump 工具](#4-算子-dump-工具)
5. [溢出检测 (OverflowUtils)](#5-溢出检测-overflowutils)
6. [环境信息收集](#6-环境信息收集)
7. [内存调试](#7-内存调试)
8. [性能分析](#8-性能分析)
9. [常用调试命令速查](#9-常用调试命令速查)

---

## 1. 环境变量调试开关

### ASCEND_LAUNCH_BLOCKING

最重要的调试开关。NPU 默认异步执行，错误报告可能延迟，堆栈不准确。

```bash
# Enable synchronous execution - errors reported at the exact call site
export ASCEND_LAUNCH_BLOCKING=1
```

**何时使用**: 遇到 SIGSEGV、stream sync 错误、或错误堆栈不指向实际出错位置时。

### ASCEND_GLOBAL_LOG_LEVEL

控制 Ascend 日志级别：

```bash
# 0=DEBUG, 1=INFO, 2=WARNING, 3=ERROR
export ASCEND_GLOBAL_LOG_LEVEL=1

# Log output directory
export ASCEND_SLOG_PRINT_TO_STDOUT=1
```

**plog 路径**: `$HOME/ascend/log/debug/plog/plog-pid_*.log`，运行前清理旧日志避免混淆。

### TORCH_NPU_LOG_LEVEL

torch_npu 自身的日志级别：

```bash
export TORCH_NPU_LOG_LEVEL=DEBUG
```

### ACL_DUMP_DATA

启用算子输入输出数据 dump：

```bash
export ACL_DUMP_DATA=1
```

### CLOSE_MATMUL_K_SHIFT

关闭 MatMul K 轴 shift 优化（910B4 上可能引入数值误差）：

```bash
# When matmul precision is abnormal (split vs whole computation inconsistent)
export CLOSE_MATMUL_K_SHIFT=1
```

**何时使用**: float16 下 matmul 切分计算与整体计算结果不一致时。

### ATB 算子日志

ATB 算子（`libop_plugin_atb.so`）的错误信息在 Python 层只显示 "setup failed"，必须开启日志：

```bash
export ASDOPS_LOG_LEVEL=INFO
export ASDOPS_LOG_TO_STDOUT=1
```

**何时使用**: ATB 算子报 "setup failed" 时。日志中 `CheckIniMatch Failed! Supported Combs:` 会列出合法 dtype 组合。

### TORCH_DEVICE_BACKEND_AUTOLOAD

禁用设备后端自动加载（解决 triton 冲突）：

```bash
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
```

**何时使用**: `import torch_npu` 失败且调用栈含 triton 时。

---

## 2. OpHook (C++ 算子 Hook)

OpHook 是 torch_npu 框架层提供的 C++ 算子 hook 机制，定义在 `torch_npu/csrc/framework/OpHook.cpp/h`。

### 用途

- 追踪算子执行顺序
- 检查算子输入输出的 shape、dtype、format
- 在算子执行前后插入自定义逻辑
- 性能计时

### 使用方式

OpHook 通过注册回调函数来拦截算子执行：

```cpp
// In torch_npu/csrc/framework/OpHook.h
class OpHook {
public:
    static OpHook& GetInstance();
    void RegisterHook(const std::string& name, HookFunc func);
    void UnregisterHook(const std::string& name);
};
```

### Python 层使用

```python
import torch_npu

# Enable op execution tracing
torch_npu.npu.set_option({"ACL_OP_DEBUG_LEVEL": "3"})
```

---

## 3. NPU Trace (Python 层 Callback)

NPU Trace 提供 Python 层的算子追踪能力，定义在 `torch_npu/csrc/sanitizer/NPUTrace.h` 和 `torch_npu/npu/_sanitizer.py`。

### 使用方式

```python
import torch
import torch_npu

# Register trace callback
def trace_handler(op_name, inputs, outputs):
    print(f"Op: {op_name}")
    for i, inp in enumerate(inputs):
        if isinstance(inp, torch.Tensor):
            print(f"  Input[{i}]: shape={inp.shape}, dtype={inp.dtype}")
    for i, out in enumerate(outputs):
        if isinstance(out, torch.Tensor):
            print(f"  Output[{i}]: shape={out.shape}, dtype={out.dtype}")

# Use torch profiler with NPU support
with torch.autograd.profiler.profile(use_npu=True) as prof:
    output = model(input)
print(prof.key_averages().table())
```

### _npu_trace 环境变量

```bash
# Enable NPU operation tracing
export NPU_TRACE=1
```

---

## 4. 算子 Dump 工具

### AclopStartDumpArgs

在 C++ 层 dump 算子的输入参数，用于精确对比算子行为：

```cpp
// Enable dump in code
#include "torch_npu/csrc/framework/OpCommand.h"

// The dump is controlled via environment variables
// ACL_DUMP_DATA=1 enables data dump
// Dump files are saved to the specified dump path
```

### Python 层 Dump

```python
import torch
import torch_npu

# Dump tensor to file for offline analysis
def dump_tensor(tensor, name):
    """Dump tensor data for debugging."""
    cpu_tensor = tensor.cpu().detach()
    print(f"{name}: shape={cpu_tensor.shape}, dtype={cpu_tensor.dtype}, "
          f"min={cpu_tensor.min():.6f}, max={cpu_tensor.max():.6f}, "
          f"mean={cpu_tensor.float().mean():.6f}")
    # Check for special values
    print(f"  nan_count={torch.isnan(cpu_tensor).sum()}, "
          f"inf_count={torch.isinf(cpu_tensor).sum()}")
    return cpu_tensor

# Compare NPU vs CPU results
def compare_op(op_func, *args, rtol=1e-3, atol=1e-3):
    """Compare operator results between NPU and CPU."""
    cpu_args = [a.cpu() if isinstance(a, torch.Tensor) else a for a in args]
    npu_args = [a.npu() if isinstance(a, torch.Tensor) else a for a in args]

    cpu_result = op_func(*cpu_args)
    npu_result = op_func(*npu_args)

    if isinstance(cpu_result, torch.Tensor):
        npu_cpu = npu_result.cpu()
        close = torch.allclose(cpu_result, npu_cpu, rtol=rtol, atol=atol)
        if not close:
            diff = (cpu_result - npu_cpu).abs()
            print(f"Max diff: {diff.max():.6e}, Mean diff: {diff.mean():.6e}")
        return close
    return True
```

---

## 5. 溢出检测 (OverflowUtils)

OverflowUtils 定义在 `torch_npu/csrc/core/OverflowUtils.h`，用于检测 NPU 计算中的数值溢出。

### 使用方式

```python
import torch_npu

# Enable overflow detection
torch_npu.npu.set_option({"ACL_OP_DEBUG_LEVEL": "3"})

# Check for overflow after computation
x = torch.randn(100, 100, dtype=torch.float16).npu()
y = x * 65504  # May overflow in fp16

# Use torch_npu overflow detection
overflow = torch_npu.npu.utils.npu_check_overflow(y)
print(f"Overflow detected: {overflow}")
```

### C++ 层接口

```cpp
// torch_npu/csrc/core/OverflowUtils.h
class OverflowUtil {
public:
    static OverflowUtil& GetInstance();
    void EnableOverflowNpu();
    bool CheckOverflowNpu();
    void ClearOverflowNpu();
};
```

---

## 6. 环境信息收集

### collect_env.py

torch_npu 提供环境信息收集脚本：

```bash
# Collect full environment info
python -c "import torch_npu; torch_npu.npu.collect_env()"

# Or use PyTorch's collect_env with NPU support
python -m torch.utils.collect_env
```

### 手动检查关键信息

```python
import torch
import torch_npu

# PyTorch version
print(f"PyTorch: {torch.__version__}")
print(f"torch_npu: {torch_npu.__version__}")

# NPU device info
print(f"NPU available: {torch.npu.is_available()}")
print(f"NPU count: {torch.npu.device_count()}")
print(f"Current device: {torch.npu.current_device()}")
print(f"Device name: {torch.npu.get_device_name()}")

# CANN version
print(f"CANN version: {torch_npu.version.cann}")
```

```bash
# Check CANN version from system
cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg

# Check NPU driver
npu-smi info
```

---

## 7. 内存调试

### 显存使用监控

```python
import torch
import torch_npu

# Memory summary
print(torch.npu.memory_summary())

# Detailed memory stats
print(f"Allocated: {torch.npu.memory_allocated() / 1024**2:.1f} MB")
print(f"Reserved: {torch.npu.memory_reserved() / 1024**2:.1f} MB")
print(f"Max allocated: {torch.npu.max_memory_allocated() / 1024**2:.1f} MB")

# Reset peak stats
torch.npu.reset_peak_memory_stats()
```

### 内存泄漏检测

```python
# Enable memory snapshot for leak detection
torch.npu.memory._record_memory_history()

# ... run your code ...

# Dump memory snapshot
snapshot = torch.npu.memory._snapshot()
```

---

## 8. 性能分析

### torch profiler with NPU

```python
import torch
from torch.profiler import profile, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.NPU],
    record_shapes=True,
    profile_memory=True,
) as prof:
    output = model(input_data)

# Print summary
print(prof.key_averages().table(sort_by="npu_time_total", row_limit=20))

# Export chrome trace
prof.export_chrome_trace("trace.json")
```

### Task Queue 模式

```python
# Enable task queue for better performance (reduces host-device sync)
torch_npu.npu.set_option({"TASK_QUEUE_ENABLE": "1"})
```

---

## 9. 常用调试命令速查

```bash
# === Environment ===
# Check CANN version
cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg

# Check NPU status
npu-smi info

# Check torch_npu version and config
python -c "import torch; import torch_npu; print(torch_npu.__version__); print(torch.__config__.show())"

# Check driver loaded
lspci | grep -i ascend

# Check GCC version (recommend 9.x for compilation)
gcc --version

# === Synchronous debugging ===
export ASCEND_LAUNCH_BLOCKING=1

# === Verbose logging ===
export ASCEND_GLOBAL_LOG_LEVEL=1
export ASCEND_SLOG_PRINT_TO_STDOUT=1

# === plog path (check after errors) ===
ls -lt $HOME/ascend/log/debug/plog/plog-pid_*.log | head -5

# === ATB operator debugging ===
export ASDOPS_LOG_LEVEL=INFO
export ASDOPS_LOG_TO_STDOUT=1

# === MatMul precision debugging ===
export CLOSE_MATMUL_K_SHIFT=1

# === triton conflict workaround ===
export TORCH_DEVICE_BACKEND_AUTOLOAD=0

# === Data dump ===
export ACL_DUMP_DATA=1

# === Quick precision check ===
python -c "
import torch, torch_npu
x = torch.randn(4, 4).npu()
y = torch.xxx(x)
y_cpu = torch.xxx(x.cpu())
print(torch.allclose(y.cpu(), y_cpu, rtol=1e-3, atol=1e-3))
print(f'max_diff={(y.cpu() - y_cpu).abs().max():.6e}')
"

# === Search for operator implementation ===
# Find opapi implementation
rg -l "op_name" third_party/op-plugin/op_plugin/ops/opapi/

# Find aclops implementation
rg -l "op_name" third_party/op-plugin/op_plugin/ops/aclops/

# Find registration
rg "TORCH_LIBRARY_IMPL.*op_name" third_party/op-plugin/

# Find DO_COMPATIBILITY usage
rg "DO_COMPATIBILITY.*aclnnOpName" third_party/op-plugin/
```
