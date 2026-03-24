# torch_npu 源码架构与导航

> **路径约定**: 本文档中的路径相对于 torch_npu 仓库根目录。
> 框架层代码在 `torch_npu/csrc/`，算子实现在 `third_party/op-plugin/op_plugin/`。

## 目录

1. [顶层目录结构](#1-顶层目录结构)
2. [算子两层架构](#2-算子两层架构)
3. [算子实现层 (op-plugin)](#3-算子实现层-op-plugin)
4. [框架层 (torch_npu/csrc)](#4-框架层-torch_npucsrc)
5. [ACLNN 执行宏](#5-aclnn-执行宏)
6. [算子注册与分发](#6-算子注册与分发)
7. [Python 绑定层](#7-python-绑定层)
8. [构建系统](#8-构建系统)
9. [源码搜索速查表](#9-源码搜索速查表)
10. [Reduce 类算子排查入口](#10-reduce-类算子排查入口)

---

## 1. 顶层目录结构

```
torch_npu/
├── torch_npu/              # Python 包 + C++ 框架层
│   ├── csrc/               # C++ 核心代码
│   │   ├── aten/           # ATen 算子绑定和实现
│   │   ├── core/           # NPU 运行时（内存、Stream、Event）
│   │   ├── framework/      # 算子执行框架（OpCommand、FormatHelper）
│   │   ├── distributed/    # 分布式训练（HCCL）
│   │   ├── profiler/       # 性能分析
│   │   └── sanitizer/      # 内存/溢出检测
│   ├── npu/                # Python NPU 模块
│   ├── dynamo/             # TorchDynamo NPU 后端
│   └── testing/            # 测试工具
├── third_party/
│   └── op-plugin/          # 算子实现插件（git submodule）
├── torchnpugen/            # 算子注册代码生成
├── test/                   # 测试套件
├── ci/                     # 构建和 CI 脚本
├── benchmarks/             # 性能基准
└── docs/                   # 文档
```

## 2. 算子两层架构

torch_npu 的算子实现分为两层：

```
PyTorch Dispatcher (PrivateUse1 dispatch key)
    │
    ▼
算子注册层 (TORCH_LIBRARY_IMPL)
    │
    ├── opapi/ (新路径 - ACLNN)
    │   └── EXEC_NPU_CMD(aclnnXxx, ...) → 直接调用 ACLNN 接口
    │
    ├── DO_COMPATIBILITY fallback
    │   └── 当 opapi 不可用时回退到 aclops
    │
    └── aclops/ (旧路径 - ACL)
        └── OpCommand().Name("Xxx").Input(...).Run() → ACL 算子调用
```

### 两条路径的区别

| 特性 | opapi (ACLNN) | aclops (ACL) |
|------|---------------|--------------|
| 路径 | `op-plugin/ops/opapi/` | `op-plugin/ops/aclops/` |
| 调用方式 | `EXEC_NPU_CMD(aclnnXxx, ...)` | `OpCommand().Name().Input().Run()` |
| 性能 | 更优（直接调用 ACLNN） | 较慢（经过 ACL 适配层） |
| CANN 依赖 | 需要较新 CANN 版本 | 兼容旧版 CANN |
| 状态 | 主推路径，新算子都用这个 | 维护模式，逐步迁移 |

## 3. 算子实现层 (op-plugin)

### 目录结构

```
third_party/op-plugin/op_plugin/
├── ops/
│   ├── opapi/              # ACLNN 算子实现（~355 个）
│   │   ├── AddKernelNpu.cpp
│   │   ├── SigmoidKernelNpu.cpp
│   │   └── ...
│   ├── aclops/             # ACL 算子实现（~418 个）
│   │   ├── AddKernelNpu.cpp
│   │   ├── SigmoidKernelNpu.cpp
│   │   └── ...
│   └── atb/                # ATB 算子（~25 个，高级张量后端）
├── utils/
│   ├── op_api_common.h     # ACLNN 执行宏定义（核心）
│   ├── OpAdapter.h         # 算子参数适配
│   ├── KernelNpuOutputSize.h   # 输出 shape 推导
│   └── KernelNpuOutputDtype.h  # 输出 dtype 推导
├── python/                 # Python 绑定
└── codegen/                # 算子注册代码生成
```

### opapi 算子实现模式

典型的 opapi 算子实现：

```cpp
// op_plugin/ops/opapi/AddKernelNpu.cpp
#include "op_plugin/utils/op_api_common.h"

namespace op_plugin {
at::Tensor add(const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha) {
    DO_COMPATIBILITY(aclnnAdd, acl_op::add(self, other, alpha));

    // Prepare output tensor
    auto output_size = op_infer::broadcast_ops_npu_output_size(self, other);
    at::Tensor result = npu_preparation::apply_tensor_without_format(output_size, self.options());

    // Execute ACLNN operator
    EXEC_NPU_CMD(aclnnAdd, self, other, alpha, result);
    return result;
}
} // namespace op_plugin
```

### aclops 算子实现模式

典型的 aclops 算子实现：

```cpp
// op_plugin/ops/aclops/AddKernelNpu.cpp
namespace acl_op {
at::Tensor add(const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha) {
    auto output_size = op_infer::broadcast_ops_npu_output_size(self, other);
    at::Tensor result = npu_preparation::apply_tensor(self, output_size);

    at_npu::native::OpCommand cmd;
    cmd.Name("Add")
       .Input(self)
       .Input(other)
       .Input(alpha, self.scalar_type())
       .Output(result)
       .Run();
    return result;
}
} // namespace acl_op
```

### Reduce 类算子的特殊入口

reduce / index reduction 类算子（如 `argmax`、`argmin`、`sum`、`mean`）除了普通算子入口外，还应额外关注以下位置：

- `third_party/op-plugin/op_plugin/ops/opapi/*Reduce*`
- `third_party/op-plugin/op_plugin/ops/opapi/ArgmaxKernelNpuOpApi.cpp`
- `third_party/op-plugin/op_plugin/ops/opapi/ArgminKernelNpuOpApi.cpp`
- `third_party/op-plugin/op_plugin/ops/aclops/ArgmaxKernelNpu.cpp`
- `third_party/op-plugin/op_plugin/ops/aclops/ArgminKernelNpu.cpp`
- `third_party/op-plugin/op_plugin/utils/KernelNpuOutputSize.h`

这类算子经常不是“算不出来”，而是“语义对不上”，尤其是在以下场景：

- `dim=None`
- `keepdim=True`
- 输入先被 `reshape({-1})` 扁平化
- 输出 shape 由 `reduce_ops_npu_output_size(...)` 推导

如果问题涉及 shape、rank-preserving keepdim、默认 dim 语义或 CPU/NPU 结果结构不一致，优先检查：

1. `dim=None` 时是否被强制改写成扁平 tensor 上的 `dim=0`
2. 执行可以扁平化，但输出 shape 推导是否错误沿用了扁平 tensor 的 rank
3. `keepdim=True` 时是否应该按原始输入 rank 构造全 1 维输出
4. opapi 与 aclops 两条路径是否都实现了相同语义
5. `DO_COMPATIBILITY` 与显式 fallback 是否覆盖了 ACLNN 不兼容场景

## 4. 框架层 (torch_npu/csrc)

### 核心组件

#### OpCommand (`framework/OpCommand.cpp/h`)

ACL 算子调用的 builder 模式封装：

```cpp
at_npu::native::OpCommand cmd;
cmd.Name("OpName")           // ACL 算子名
   .Input(tensor)            // 输入张量
   .Input(scalar, dtype)     // 标量输入
   .Output(result)           // 输出张量
   .Attr("attr_name", value) // 属性
   .Run();                   // 执行
```

#### FormatHelper (`framework/FormatHelper.cpp/h`)

NPU 张量格式管理。NPU 有私有格式（NZ、FRACTAL_Z、NC1HWC0 等），FormatHelper 负责格式转换和推导：

- `GetBaseFormat()` — 获取基础格式（NCHW/NHWC）
- `GetFormatFromTensor()` — 从张量获取当前格式
- `IsBaseFormatType()` — 判断是否为基础格式
- `GetPermittedFormats()` — 获取算子允许的格式列表

#### OpHook (`framework/OpHook.cpp/h`)

C++ 层算子 hook 机制，用于调试和追踪算子执行：

```cpp
// Register a hook
OpHook::GetInstance().RegisterHook("my_hook", hook_func);
```

#### NPUCachingAllocator (`core/npu/NPUCachingAllocator.cpp/h`)

NPU 显存分配器，类似 PyTorch 的 CUDA caching allocator。

#### NPUStream (`core/npu/NPUStream.h`)

NPU stream 管理，控制算子执行的异步流。

### 框架层目录

```
torch_npu/csrc/
├── framework/
│   ├── OpCommand.cpp/h         # ACL 算子调用 builder
│   ├── OpParamMaker.cpp/h      # 参数转换
│   ├── FormatHelper.cpp/h      # 格式管理
│   ├── OpHook.cpp/h            # 算子 hook
│   ├── InferFormat.cpp/h       # 格式推导
│   ├── utils/
│   │   ├── NpuUtils.cpp/h      # NPU 工具函数
│   │   ├── CalcuOpUtil.cpp/h   # 计算工具
│   │   └── OpAdapter.h         # 算子适配器
│   └── interface/
│       ├── AclOpCompileInterface.cpp/h  # ACL 编译接口
│       └── EnvVariables.cpp/h           # 环境变量管理
├── core/npu/
│   ├── NPUCachingAllocator.cpp/h  # 显存分配
│   ├── NPUStream.h                # Stream 管理
│   ├── NPUEvent.cpp/h             # Event 管理
│   ├── NPUException.cpp/h         # 异常处理
│   ├── interface/
│   │   ├── AclInterface.cpp/h     # ACL 接口封装
│   │   └── HcclInterface.cpp/h    # HCCL 接口封装
│   └── register/                  # 选项注册
├── aten/
│   ├── ops/                       # ATen 算子绑定
│   └── common/                    # 通用张量操作（copy、format cast）
└── distributed/
    ├── HcclOps.cpp                # HCCL 集合通信
    └── ProcessGroupHCCL.cpp       # 进程组
```

## 5. ACLNN 执行宏

ACLNN 执行宏定义在 `op-plugin/op_plugin/utils/op_api_common.h`，是 opapi 算子的核心执行机制。

### DO_COMPATIBILITY

版本兼容宏，当 ACLNN 接口不可用时回退到 aclops：

```cpp
#define DO_COMPATIBILITY(aclnn_api, acl_op_call)          \
    do {                                                   \
        static const auto aclnn_func = GET_OP_API_FUNC(aclnn_api); \
        if (aclnn_func == nullptr) {                       \
            return acl_op_call;  /* fallback to aclops */  \
        }                                                  \
    } while (0)
```

使用方式：
```cpp
at::Tensor add(...) {
    // If aclnnAdd is not available in current CANN, fallback to acl_op::add
    DO_COMPATIBILITY(aclnnAdd, acl_op::add(self, other, alpha));
    // ... ACLNN implementation
}
```

### EXEC_NPU_CMD

执行 ACLNN 算子的核心宏：

```cpp
#define EXEC_NPU_CMD(aclnn_api, ...)                      \
    do {                                                   \
        static const auto getWorkspaceSizeFuncAddr =       \
            GET_OP_API_FUNC(aclnn_api##GetWorkspaceSize);  \
        static const auto opApiFuncAddr =                  \
            GET_OP_API_FUNC(aclnn_api);                    \
        /* 1. Get workspace size */                        \
        /* 2. Allocate workspace */                        \
        /* 3. Execute operator */                          \
    } while (0)
```

### EXEC_NPU_NO_FORMAT_CHECK_CMD

跳过格式检查的执行宏，用于已知格式正确的场景：

```cpp
EXEC_NPU_NO_FORMAT_CHECK_CMD(aclnnAdd, self, other, alpha, result);
```

## 6. 算子注册与分发

### PyTorch Dispatcher 机制

torch_npu 通过 `PrivateUse1` dispatch key 注册算子：

```cpp
// 注册到 PrivateUse1 backend
TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("add.Tensor", TORCH_FN(op_plugin::add));
    m.impl("sigmoid", TORCH_FN(op_plugin::sigmoid));
}
```

### 注册代码生成

`torchnpugen/` 目录包含代码生成工具，自动生成算子注册代码：

```
torchnpugen/
├── gen.py                  # 主生成脚本
├── native_functions.yaml   # 算子声明
└── templates/              # 代码模板
```

### op-plugin 注册

`op-plugin/codegen/` 也包含注册相关代码：

```
op-plugin/codegen/
├── gen_op_plugin.py        # op-plugin 代码生成
└── op_plugin_functions.yaml # 算子函数声明
```

### 分发流程

```
torch.add(x, y)  (x is NPU tensor)
    │
    ▼
PyTorch Dispatcher → PrivateUse1 key
    │
    ▼
op_plugin::add()  (registered via TORCH_LIBRARY_IMPL)
    │
    ├── DO_COMPATIBILITY check
    │   ├── aclnnAdd available → EXEC_NPU_CMD(aclnnAdd, ...)
    │   └── aclnnAdd not found → acl_op::add() (fallback)
    │
    └── Return result
```

## 7. Python 绑定层

### NPU 模块 (`torch_npu/npu/`)

```
torch_npu/npu/
├── __init__.py             # NPU 模块初始化
├── _kernel_check.py        # Kernel 可用性检查
├── _sanitizer.py           # 内存检测工具
├── utils/
│   └── collect_env.py      # 环境信息收集
└── ...
```

### C++ Python 绑定 (`torch_npu/csrc/npu/`)

```
torch_npu/csrc/npu/
├── Module.cpp              # Python 模块初始化、设备属性
├── Stream.cpp              # Stream Python 绑定
├── Event.cpp               # Event Python 绑定
└── Graph.cpp               # NPU Graph 执行
```

### 自定义算子 Python 接口

```python
# torch_npu 提供的自定义算子
import torch_npu

# NPU 特有算子
torch_npu.npu_format_cast(tensor, format)
torch_npu.npu_confusion_transpose(tensor, ...)
torch_npu.npu_bmmV2(tensor1, tensor2, ...)
```

## 8. 构建系统

### 编译命令

```bash
# 标准编译
cd /home/lch/work && source env_ms.sh mindspore/ && cd torch_npu && bash ci/build.sh --python=3.9

# ci/build.sh 主要步骤：
# 1. 检查 Python 版本和依赖
# 2. 编译 C++ 扩展（torch_npu/csrc/）
# 3. 编译 op-plugin（third_party/op-plugin/）
# 4. 生成算子注册代码
# 5. 打包 wheel
```

### 关键构建文件

```
ci/build.sh                 # 主构建脚本
setup.py                    # Python 包构建配置
CMakeLists.txt              # CMake 构建配置
third_party/op-plugin/CMakeLists.txt  # op-plugin 构建
```

## 9. 源码搜索速查表

### 按算子名搜索

| 目标 | 搜索命令 |
|------|---------|
| opapi 实现 | `rg -l "{OpName}" third_party/op-plugin/op_plugin/ops/opapi/` |
| aclops 实现 | `rg -l "{OpName}" third_party/op-plugin/op_plugin/ops/aclops/` |
| 算子注册 | `rg "m\.impl.*{op_name}" torch_npu/csrc/ third_party/` |
| DO_COMPATIBILITY | `rg "DO_COMPATIBILITY.*aclnn{OpName}" third_party/op-plugin/` |
| EXEC_NPU_CMD | `rg "EXEC_NPU_CMD.*aclnn{OpName}" third_party/op-plugin/` |
| 测试 | `rg -l "{op_name}" test/` |

### 按文件类型搜索

```bash
# Search ACLNN macro definitions
rg "EXEC_NPU_CMD\|EXEC_NPU_NO_FORMAT_CHECK_CMD" op-plugin/op_plugin/utils/op_api_common.h

# Search format conversion
rg "FormatHelper\|TransData\|NpuFormat" torch_npu/csrc/framework/

# Search memory allocation
rg "NPUCachingAllocator\|allocate" torch_npu/csrc/core/npu/

# Search dispatch registration
rg "TORCH_LIBRARY_IMPL" torch_npu/csrc/ third_party/

# Search Python custom ops
rg "npu_\w+" torch_npu/npu/__init__.py
```

### 按问题类型搜索

| 问题类型 | 搜索方向 |
|---------|---------|
| 算子未注册 | `rg "TORCH_LIBRARY_IMPL" + rg "m.impl"` 检查注册 |
| ACLNN 符号缺失 | `rg "GET_OP_API_FUNC.*aclnn{Name}"` 检查动态加载 |
| Format 错误 | `rg "FormatHelper\|InferFormat\|format_cast"` |
| dtype 错误 | `rg "scalar_type\|dtype\|to\(" ` 在算子实现中搜索 |
| 反向传播 | `rg "backward\|grad_fn\|autograd"` |
| Stream 同步 | `rg "NPUStream\|synchronize\|ASCEND_LAUNCH_BLOCKING"` |
| reduce 语义问题 | `rg "reshape\(\{-1\}\)\|reduce_ops_npu_output_size\|keepdim\|!dim.has_value" third_party/op-plugin/` |

## 10. Reduce 类算子排查入口

当问题集中在 `argmax`、`argmin`、`sum`、`mean` 等 reduce 类算子的 shape、rank、keepdim 或默认 dim 语义时，建议按以下顺序排查。

### 10.1 先找两条实现路径

```bash
rg -l "Argmax|ArgMin|Reduce" third_party/op-plugin/op_plugin/ops/opapi/
rg -l "Argmax|ArgMin|Reduce" third_party/op-plugin/op_plugin/ops/aclops/
```

优先关注：

- `third_party/op-plugin/op_plugin/ops/opapi/ArgmaxKernelNpuOpApi.cpp`
- `third_party/op-plugin/op_plugin/ops/opapi/ArgminKernelNpuOpApi.cpp`
- `third_party/op-plugin/op_plugin/ops/aclops/ArgmaxKernelNpu.cpp`
- `third_party/op-plugin/op_plugin/ops/aclops/ArgminKernelNpu.cpp`
- `third_party/op-plugin/op_plugin/utils/KernelNpuOutputSize.h`

### 10.2 再看语义敏感点

对 reduce 类算子，重点检查以下代码模式：

- `self.reshape({-1})`
- `int64_t real_dim = 0`
- `bool real_keep_dim = false`
- `op_infer::reduce_ops_npu_output_size(...)`
- 自定义 `get_keepdim_output_size(...)`
- `if (!dim.has_value() && keepdim) { ... fallback ... }`

这些模式通常对应以下风险：

- 执行路径为了实现全量归约而扁平化输入
- 但输出 shape 仍错误按扁平 tensor 推导
- `keepdim=True` 时丢失原始 rank
- ACLNN 能执行，但无法 materialize 正确语义输出

### 10.3 最后做三方对比归因

定位这类问题时，不要只比较“有没有报错”，而要比较语义：

- PyTorch CPU 是否正常
- ACLOP 路径是否正常
- ACLNN 路径是否正常

推荐归因规则：

- **CPU 正常 + ACLOP 正常 + ACLNN 异常** → 优先归因为 ACLNN 语义不一致或实现缺陷
- **CPU 正常 + ACLNN/ACLOP 都异常** → 优先归因为 torch_npu 公共适配逻辑问题
- **CPU 本身异常** → 不能直接归因 torch_npu
