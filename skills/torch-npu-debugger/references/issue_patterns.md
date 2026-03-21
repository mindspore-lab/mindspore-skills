# torch_npu 常见问题模式与根因分类

## 目录

1. [根因分类总览](#1-根因分类总览)
2. [精度与数值问题](#2-精度与数值问题)
3. [算子注册与分发问题](#3-算子注册与分发问题)
4. [Shape 与 Format 问题](#4-shape-与-format-问题)
5. [ACLNN 适配问题](#5-aclnn-适配问题)
6. [编译与版本配套问题](#6-编译与版本配套问题)
7. [运行时问题](#7-运行时问题)
8. [性能问题](#8-性能问题)
9. [分布式与 HCCL 问题](#9-分布式与-hccl-问题)
10. [环境兼容性问题](#10-环境兼容性问题)
11. [自定义算子与 ATB 问题](#11-自定义算子与-atb-问题)
12. [快速定界决策树](#12-快速定界决策树)

---

## 1. 根因分类总览

| 分类 | 占比(估) | 涉及组件 |
|------|---------|---------|
| 精度/数值 | ~20% | opapi/aclops kernel, CANN |
| 算子注册/分发 | ~10% | codegen, TORCH_LIBRARY_IMPL |
| Shape/Format | ~10% | FormatHelper, npu_preparation |
| ACLNN 适配 | ~10% | op_api_common.h, CANN headers |
| 编译/版本配套 | ~15% | ci/build.sh, CMake, CANN/op-plugin 版本 |
| 运行时/驱动 | ~10% | Stream, Allocator, ACL interface, 驱动/固件 |
| 分布式/HCCL | ~8% | ProcessGroupHCCL, HCCL 通信域 |
| 环境兼容性 | ~7% | GCC ABI, SoC 版本, triton 冲突 |
| 性能 | ~5% | Format 转换, task queue, MLIR fallback |
| 自定义算子/ATB | ~5% | gen_opapi, ATB 算子, 结构化适配 |

---

## 2. 精度与数值问题

### 诊断特征

**典型错误信息**:
- `AssertionError` in `assertRtolEqual` / `torch.allclose`
- 输出全 NaN / Inf / 全零
- 与 CPU/GPU 结果不一致
- 特定 dtype（fp16/bf16）下精度偏差

**触发条件**:
- Ascend fp16 计算（NPU fp16 精度低于 GPU）
- 对比 CPU/GPU 基准结果
- 特定 CANN 版本更新后
- 特殊值输入（inf、nan、极大/极小值）

### 常见根因

| 根因 | 说明 | 定位方向 |
|------|------|---------|
| **fp16 精度不足** | NPU fp16 计算溢出或精度丢失 | 检查是否需要 dtype 提升 |
| **MatMul K-shift 优化** | 910B4 上 MatMul K 轴 shift 优化引入数值误差，切分计算与整体计算结果不一致 | 设置 `CLOSE_MATMUL_K_SHIFT=1` 关闭优化验证 |
| **非连续 tensor** | `narrow/stride/transpose` 后的 tensor 作为输入或 `out` 参数时，写入位置计算错误 | 专门构造非连续 tensor 对比 CPU/NPU 结果 |
| **ACLNN 特殊值处理** | aclnn 算子对 inf/nan 处理不正确 | 对比 CPU 结果，检查 ACLNN 实现 |
| **dtype 未正确传递** | 输入 dtype 在转换过程中丢失 | 检查 npu_preparation 和 dtype 推导 |
| **CANN 内核变更** | CANN 升级后算子计算行为变化 | 对比不同 CANN 版本结果 |
| **反向传播精度** | backward 中累加误差或 dtype 降级 | 检查 grad 函数的 dtype 处理 |
| **Format 转换精度损失** | NZ/FRACTAL_Z 格式转换引入误差 | 检查 FormatHelper 转换路径 |
| **GE 初始化顺序** | `set_device()` 提前触发 GEInitialize，precision_mode 参数未传入，默认低精度 | plog 搜 `PrecisionMode`，避免在配置完成前调用 `set_device()` |
| **CPU fallback 误导** | 算子 NPU 未实现回退 CPU 执行，但 CPU 实现本身有 bug | 观察 warning 区分"NPU 未实现"和"CPU 本身有 bug" |

### 诊断步骤

1. 确认环境: torch_npu 版本、CANN 版本、设备型号
2. 对比基准: 同样输入在 CPU 上的结果
3. 隔离 dtype: 用 float32 测试是否仍有偏差
4. 检查特殊值: 输入是否包含 inf/nan/极值
5. 检查非连续: 输入/输出 tensor 是否为 `narrow/stride/transpose` 后的非连续 tensor
6. 检查 MatMul: 如涉及 matmul，试 `CLOSE_MATMUL_K_SHIFT=1`
7. 检查 CANN: 对比不同 CANN 版本的结果

### 二级定界决策

```
allclose 失败
├─ 输出全零 → 算子未正确执行或 format 错误
├─ 输出全 NaN → dtype 溢出或未初始化，检查 fp16 计算链路
├─ 小幅偏差 (< 1e-3) → 累加精度或 CANN 变更
├─ 大幅偏差 → 逻辑错误或参数传递错误
├─ 仅 fp16/bf16 偏差 → dtype 提升缺失
├─ 仅反向偏差，正向正常 → 检查 backward 实现
├─ 切分计算与整体计算不一致 → MatMul K-shift 优化，试 CLOSE_MATMUL_K_SHIFT=1
├─ 非连续 tensor 结果偏移 → 检查 out 参数写入逻辑（如 logspace.out）
├─ set_device() 后精度降低 → GE 初始化顺序问题，plog 搜 PrecisionMode
└─ warning 提示 CPU 执行但结果仍错 → CPU fallback 路径本身有 bug
```

---

## 3. 算子注册与分发问题

### 诊断特征

**典型错误信息**:
- `NotImplementedError: Could not run 'aten::xxx' with arguments from the 'PrivateUse1' backend`
- `No kernel found for 'aten::xxx' on 'PrivateUse1'`
- 算子执行了错误的实现（走了 CPU fallback）

### 常见根因

| 根因 | 说明 | 定位方向 |
|------|------|---------|
| **算子未注册** | TORCH_LIBRARY_IMPL 中缺少注册 | 检查 codegen/ 和注册文件 |
| **dispatch key 错误** | 注册到了错误的 dispatch key | 检查 PrivateUse1 vs AutogradPrivateUse1 |
| **DO_COMPATIBILITY fallback 失败** | opapi 回退到 aclops 但 aclops 也没实现 | 检查两条路径的实现 |
| **Autograd 注册缺失** | 有 forward 但没注册 autograd | 检查 AutogradPrivateUse1 注册 |
| **codegen 遗漏** | 代码生成时遗漏了某个算子 | 检查 torchnpugen/ 配置 |

### 诊断步骤

1. 确认算子名: PyTorch 中的 aten 算子名（如 `aten::add.Tensor`）
2. 搜索注册: `rg "TORCH_LIBRARY_IMPL.*PrivateUse1" -l` 查找注册文件
3. 检查 dispatch: 确认注册的 dispatch key 是否正确
4. 检查 fallback: DO_COMPATIBILITY 是否正确配置

---

## 4. Shape 与 Format 问题

### 诊断特征

**典型错误信息**:
- `format mismatch` / `TransData failed`
- `shape mismatch` / `invalid shape`
- 输出 shape 不符合预期
- `ACL_ERROR_INVALID_PARAM` 伴随 shape 信息

### NPU 私有格式

NPU 使用私有张量格式以优化计算性能：

| 格式 | 说明 | 典型用途 |
|------|------|---------|
| NCHW | 标准 4D 格式 | 通用 |
| NHWC | Channel-last 格式 | 部分卷积算子 |
| NZ (FRACTAL_NZ) | 分形格式，16x16 分块 | MatMul 等计算密集算子 |
| FRACTAL_Z | 权重分形格式 | 卷积权重 |
| NC1HWC0 | 5D 格式，C 轴分块 | 部分 NPU 算子 |

### 常见根因

| 根因 | 说明 | 定位方向 |
|------|------|---------|
| **Format 不匹配** | 算子输入 format 与期望不符 | 检查 FormatHelper 和 npu_preparation |
| **Format 转换失败** | TransData 不支持某种格式转换 | 检查 CANN 支持的格式转换路径 |
| **输出 shape 推导错误** | KernelNpuOutputSize 计算错误 | 检查 op_infer 中的 shape 推导 |
| **非连续张量** | 非连续内存布局导致 format 异常 | 检查是否需要 contiguous() |
| **动态 shape** | 动态 shape 场景下 format 推导失败 | 检查是否支持动态 shape |

### 诊断步骤

1. 打印张量信息: `tensor.storage_offset()`, `tensor.stride()`, `tensor.is_contiguous()`
2. 检查 format: 通过 `torch_npu.get_npu_format(tensor)` 获取当前格式
3. 检查 shape 推导: 查看 `KernelNpuOutputSize.h` 中的推导逻辑
4. 尝试基础格式: 用 `npu_format_cast` 转为 NCHW 后重试

---

## 5. ACLNN 适配问题

### 诊断特征

**典型错误信息**:
- `undefined symbol: aclnnXxx` / `aclnnXxxGetWorkspaceSize`
- `ACL_ERROR_*` 错误码
- `EZ9999: Inner Error` / `EE9999`
- 参数数量或类型不匹配

### 常见根因

| 根因 | 说明 | 定位方向 |
|------|------|---------|
| **ACLNN 符号缺失** | CANN 版本不包含该 aclnn 接口 | 检查 CANN 版本，添加 DO_COMPATIBILITY |
| **参数不匹配** | aclnn 接口参数与调用不一致 | 对比 CANN 头文件中的函数签名 |
| **Workspace 计算错误** | GetWorkspaceSize 返回错误大小 | 检查 workspace 分配逻辑 |
| **dtype 不支持** | aclnn 算子不支持某种 dtype | 添加 dtype 转换或 fallback |
| **CANN 版本不兼容** | 新 aclnn 接口在旧 CANN 上不存在 | 使用 DO_COMPATIBILITY 宏 |
| **ACLNN / ACLOP 语义不一致** | CPU 语义明确、ACLOP 正常，但 ACLNN 在同场景报错或行为不同 | 对比 opapi/aclops 同名算子的 shape 推导、dim 处理、keepdim 处理与 fallback 路径 |

### 诊断步骤

1. 确认 CANN 版本: `cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg`
2. 检查符号: `nm -D /usr/local/Ascend/ascend-toolkit/latest/lib64/libopapi.so | grep aclnnXxx`
3. 检查头文件: 查看 CANN 头文件中 aclnn 函数的签名
4. 检查 DO_COMPATIBILITY: 确认 fallback 路径是否正确
5. 对比 CPU / ACLOP / ACLNN: 若 CPU 与 ACLOP 一致而 ACLNN 异常，优先判定为 ACLNN 语义不一致或底层缺陷
6. 对 reduce / index 类算子重点检查: `dim=None`、`keepdim=True`、flatten 后的 output_size 推导是否仍保留上游语义

---

## 6. 编译与版本配套问题

### 诊断特征

**典型错误信息**:
- `fatal error: xxx.h: No such file or directory`
- `undefined reference to 'xxx'` / `undefined reference to op_api::*`
- `error: no matching function for call to 'xxx'`
- `'RunOpApiV2' is not a member of 'at_npu::native::OpCommand'`
- `ld: cannot find -lstdc++fs`
- `ld: cannot find -ltorch_npu`（更常见于链接路径或构建产物缺失）
- `is_convertible_v was not declared in this scope`
- `import torch_npu` / `_c10d_npu_init` 阶段 `SIGSEGV`

### 常见根因

| 根因 | 说明 | 定位方向 |
|------|------|---------|
| **版本三元组不匹配** | CANN + torch_npu + op-plugin 三者版本必须严格匹配 | 检查版本配套表 |
| **op-plugin submodule 不对齐** | op-plugin commit 与 torch_npu tag 不匹配，导致大量 `undefined reference to op_api::*` | 检查 `git submodule status` |
| **OpCommand API 版本不匹配** | op-plugin 引用了 `OpCommand::RunOpApiV2` 等新方法，但 torch_npu 尚未实现 | 不能混用更新的 op-plugin |
| **GCC 版本不兼容** | 宿主机重编出的 torch_npu 与当前 Python / torch 二进制工具链不一致，可能在 import 或 pybind 绑定阶段直接崩溃 | 比较 `libpython`、`torch/_C`、good/bad `torch_npu` 的 `.comment`，再检查宿主机与容器编译器来源 |
| **C++ 标准降级** | CMakeLists.txt 中 `CMAKE_CXX_STANDARD` 被错误设为 14 | 检查 `is_convertible_v`、`if constexpr` 等 C++17 特性报错 |
| **opp_kernel 包缺失** | CANN 环境缺少对应芯片的 opp_kernel 包 | 错误码 500002 = GE 图编译失败，查 plog 首报错 |
| **头文件前向声明缺失** | `AclOpCompileInterface.h` 中 `executor` 未声明 | 添加 `struct aclopExecute;` 前向声明 |
| **头文件依赖** | CANN 头文件路径变更或缺失 | 检查 CMakeLists.txt include 路径 |
| **增量编译失败** | 缓存的 .o 文件与新代码不兼容 | 清理 build/ 目录后重新编译 |
| **链接路径/产物缺失** | `-ltorch_npu` 等库找不到，通常是 build/install 结果或链接路径问题，而非 GCC 本身 | 检查 `build/`、`dist/`、库搜索路径和安装结果 |
| **Python 版本不匹配** | 编译时和运行时 Python 版本不同 | 检查 `--python=3.9` 参数 |

### 版本配套决策树

```
编译/导入失败
├─ `import torch_npu` / `_c10d_npu_init` 直接 SIGSEGV
│  ├─ 同环境旧包正常、重编包崩溃 → 比较 good/bad 包 `.comment`
│  ├─ `libpython`、`torch/_C`、good 包为 GCC 11.x，bad 包为 GCC 10.x → 工具链不兼容
│  └─ GCC 11 只存在于容器 / overlay → 回到原容器内 clean rebuild，不要直接复用 overlay 编译器
├─ "undefined reference to op_api::*" → op-plugin submodule commit 与 torch_npu tag 不对齐
├─ "is not a member of OpCommand" → op-plugin 比 torch_npu 新，降级 op-plugin 或升级 torch_npu
├─ "No such file or directory: aclnn*.h" → CANN 版本过旧，缺少新 ACLNN 头文件
├─ "ld: cannot find -lstdc++fs" → GCC 版本不兼容，可先尝试与 torch / Python 相同的大版本工具链
├─ "is_convertible_v was not declared" → C++ 标准被降级为 C++14，需改回 C++17
├─ error code 500002 → GE 图编译失败，查 plog，检查 opp_kernel 包
├─ "Failed to load the backend extension: torch_npu" → 版本三元组不兼容，需完整 traceback
└─ "executor was not declared" → 头文件前向声明缺失
```

### 调试技巧

- 修改 `setup.py` 将 `subprocess.check_call` 改为 `subprocess.check_output` 可捕获详细编译错误
- plog 路径: `$HOME/ascend/log/debug/plog/plog-pid_*.log`
- EulerOS 上可能需要 `ln -s /usr/lib64/libpthread.a /usr/lib64/libpthread_nonshared.a`
- `torch.compile` 的 inductor 后端 C++ 编译失败（`CppCompileError`）首先检查 GCC 版本

### 诊断步骤

1. 检查版本三元组: CANN 版本、torch_npu 版本、op-plugin commit
2. 检查子模块: `git submodule status`
3. 检查 Python / torch / torch_npu 编译器来源: `platform.python_compiler()`、`readelf -p .comment`
4. 检查宿主机与容器 GCC: `gcc --version`，确认是否存在与 good 包一致的 GCC 11.x 容器工具链
5. 仅当 `ld: cannot find -lstdc++fs` 或明确依赖旧 ABI 时，再考虑固定推荐版本
6. 清理重编: `rm -rf build/ && bash ci/build.sh --python=3.9`
7. 检查 CMake 日志: `build/CMakeFiles/CMakeError.log`
8. 检查 plog: `$HOME/ascend/log/debug/plog/plog-pid_*.log`

---

## 7. 运行时问题

### 诊断特征

**典型错误信息**:
- `SIGSEGV` / `Segmentation fault`
- `ACL_ERROR_RT_*` 运行时错误
- `stream sync failed`
- `device memory not enough` / OOM
- `EZ9999` / `error code = 0x800000`（AICore/MTE 硬件级错误）
- `_lazy_init` 卡死 / import 阶段崩溃

### 常见根因

| 根因 | 说明 | 定位方向 |
|------|------|---------|
| **Stream 同步缺失** | 异步执行时数据未就绪 | 启用 ASCEND_LAUNCH_BLOCKING=1 |
| **内存越界** | 输出 tensor 大小不足 | 检查 output shape 计算 |
| **显存不足** | NPU 显存 OOM | 检查 batch size 和模型大小 |
| **设备未初始化** | 未正确初始化 NPU 设备 | 检查 torch.npu.set_device() |
| **多 stream 竞争** | 多个 stream 访问同一内存 | 检查 stream 同步点 |
| **驱动/固件异常** | 驱动未加载或版本不匹配 | `lspci \| grep -i ascend` 验证驱动，gdb 采集堆栈 |
| **多框架共卡** | torch_npu 和 MindSpore 同时使用同一张卡 | 用 `ASCEND_RT_VISIBLE_DEVICES` 做设备隔离 |
| **ACL API 返回值越界** | ACL API 返回的 count/size 未做上界保护 | 加 `MAX_*` 上界保护 |
| **HBM 内存泄漏** | 进程异常退出未释放 HBM | `npu-smi info` 检查，重启服务或重置设备 |
| **异步错误延迟** | 异步执行导致错误报告位置不准确 | 在关键点插入 `synchronize()` |

### EZ9999 / AICore 错误诊断

EZ9999 是 AICore 硬件级错误，不一定是算子 bug：

```
EZ9999 / error code = 0x800000
├─ 多框架共卡 → 先做单框架隔离测试
├─ 日志中 vec/mte/ifu/ccu error info → 区分哪个计算单元出错
├─ pc start/current 地址 → 配合 dump 工具定位出错算子
└─ 单独运行可复现 → 算子本身 bug，提 CANN issue
```

### 诊断步骤

1. 同步执行: `export ASCEND_LAUNCH_BLOCKING=1` 定位异步错误
2. 检查内存: `torch.npu.memory_summary()` 查看显存使用
3. 检查驱动: `lspci | grep -i ascend`，`npu-smi info`
4. 最小复现: 缩小输入规模，确认是否为 OOM
5. 检查堆栈: 使用 gdb 或 ASAN 定位 SIGSEGV
6. 隔离测试: 确认是否有多框架/多进程共卡

---

## 8. 性能问题

### 诊断特征

- 算子执行时间远超预期
- 大量 TransData（格式转换）操作
- GPU 上快但 NPU 上慢
- profiler 显示大量 format cast

### 常见根因

| 根因 | 说明 | 定位方向 |
|------|------|---------|
| **Format 转换开销** | 频繁的 NCHW ↔ NZ 转换 | 检查算子链的 format 一致性 |
| **Task queue 未启用** | 未使用 task queue 模式 | 检查 `export TASK_QUEUE_ENABLE=1` |
| **同步执行** | ASCEND_LAUNCH_BLOCKING=1 未关闭 | 确认环境变量 |
| **算子回退到 CPU** | 算子未注册 NPU 实现，fallback 到 CPU | 检查算子注册和 dispatch |
| **非最优 ACLNN 路径** | 走了 aclops 而非 opapi | 检查 DO_COMPATIBILITY fallback |
| **MLIR fallback 冗余算子** | `transform_args` 引入的 BroadcastTo/Cast 在 fallback 路径中不会被消除 | profiling `op_statistic` 表对比 eager vs fallback |
| **隐式 D2H 同步** | 某些操作触发隐式 Device-to-Host 同步 | Profiler 定位热点 |

### 诊断步骤

1. Profiler: 使用 `torch.npu.profiler` 分析算子耗时
2. 检查 format: 关注 TransData 操作的数量和耗时
3. 检查 dispatch: 确认算子走的是 opapi 还是 aclops
4. 检查环境变量: TASK_QUEUE_ENABLE, ASCEND_LAUNCH_BLOCKING
5. 检查 op_statistic: 对比 eager 和 fallback 路径的算子数量

---

## 9. 分布式与 HCCL 问题

### 诊断特征

**典型错误信息**:
- `EJ0001: Failed to initialize the HCCP process`
- `hcclCommInitRootInfoConfig` 报 `Failed to allocate memory`
- `ProcessGroup does not support xxx`
- 多卡训练卡死或超时

### 常见根因

| 根因 | 说明 | 定位方向 |
|------|------|---------|
| **残留进程** | 上次训练进程未完全退出，HCCP 资源未释放 | `ps aux \| grep python`，kill 后等 10 秒 |
| **HCCL 内存不足** | 每个通信域初始化占用数百 MB 设备内存 | `npu-smi info` 确认显存，HCCL error code 2 = 内存不足 |
| **通信域重复创建** | aclgraph 捕获期间重复初始化通信域 | 检查 capture 循环中的显存增长 |
| **集合通信原语缺失** | HCCL 后端未实现某个原语（如 `allgather_coalesced`） | 升级 torch_npu 版本 |
| **版本不匹配** | 升级 CANN 后 HCCL 库不兼容 | 检查 CANN + 驱动版本配套 |

### 诊断步骤

1. 检查残留进程: `ps aux | grep python`，kill 后等待 10 秒
2. 检查显存: `npu-smi info`
3. 单卡测试: 单卡可以跑、多卡报错 → 直接指向 HCCL 初始化问题
4. 检查版本: CANN + 驱动 + torch_npu 版本配套

---

## 10. 环境兼容性问题

### 诊断特征

**典型错误信息**:
- `undefined symbol: _ZNK5torch8autograd4Node4nameB5cxx11Ev`
- `Unsupported soc version: AscendXXX`
- `import torch_npu` 失败，调用栈含 triton
- `_lazy_init` 卡死 / SIGSEGV（import 阶段）

### 常见根因

| 根因 | 说明 | 定位方向 |
|------|------|---------|
| **GCC ABI 不匹配** | 典型表现为 `undefined symbol` 含 `cxx11`，多见于预编译 wheel 与 torch ABI 不匹配；若是本地重编后 import 阶段直接崩溃，应优先转到第 6 节的工具链诊断 | `torch.__config__.show()` 查 GCC 版本，结合 wheel / manylinux 信息判断 |
| **SoC 版本未注册** | torch_npu 版本过旧，不支持该 SoC 型号 | 升级 torch_npu 版本 |
| **triton 冲突** | triton 包与 torch_npu 冲突导致 import 失败 | 卸载 triton 或设 `TORCH_DEVICE_BACKEND_AUTOLOAD=0` |
| **驱动未加载** | 驱动安装异常，`lspci` 无输出 | 重装驱动/固件/CANN |
| **transfer_to_npu 冲突** | `transfer_to_npu` 与 `torch.jit.script` 不兼容 | 不要同时使用，升级版本 |
| **torch.compile 兼容性** | `torch._dynamo.exc.Unsupported` | 找 `from user code` 调用栈，检查 torch_npu wrapper 是否有 side effect |

### 诊断步骤

1. 检查 ABI: `torch.__config__.show()` 查看 GCC 版本
2. 检查驱动: `lspci | grep -i ascend`
3. 检查冲突包: `pip list | grep triton`
4. 最小 import 测试: `python -c "import torch; import torch_npu; print(torch_npu.__version__)"`

---

## 11. 自定义算子与 ATB 问题

### 诊断特征

**典型错误信息**:
- ATB 算子 `setup failed`
- 错误码 `507015`（FFTS+ aivector 错误）
- `0x800000`（MTE DDR 地址越界）
- `CheckIniMatch Failed! Supported Combs:`

### 常见根因

| 根因 | 说明 | 定位方向 |
|------|------|---------|
| **ATB dtype 约束** | ATB 算子对 int32/int64 有严格区分，比 ACLNN 更严格 | 开启 `ASDOPS_LOG_LEVEL=INFO` 查看合法 dtype 组合 |
| **gen_opapi dtype 硬编码** | `gen_opapi` 配置中 `out` 的 dtype 硬编码（如 `at::kFloat`），与实际输入不一致 | 检查 out dtype 是否动态推断 |
| **opapi infershape 错误** | 输出 tensor 的 size/shape 计算有误 | ACLNN 报 `161002` + `CheckAxisRange/CheckShapeValid failed` |
| **算子使用场景错误** | IFA 只用于 decode（S=1），FA 用于 prefill | tiling 报错通常意味着输入 shape 不符合算子使用约束 |

### ATB 算子调试

ATB 算子（`libop_plugin_atb.so`）的错误信息在 Python 层只显示 "setup failed"，必须开启日志才能看到真正原因：

```bash
export ASDOPS_LOG_LEVEL=INFO
export ASDOPS_LOG_TO_STDOUT=1
```

日志中 `CheckIniMatch Failed! Supported Combs:` 会列出所有合法的 dtype 组合，直接对比入参即可定位。

### 诊断步骤

1. 开启 ATB 日志: `ASDOPS_LOG_LEVEL=INFO` + `ASDOPS_LOG_TO_STDOUT=1`
2. 检查 dtype 组合: 对比日志中的 Supported Combs 与实际入参
3. 检查 gen_opapi 配置: out 的 dtype 是否与输入对齐
4. 区分 aclnn 直调 vs torch_npu 调用: aclnn 成功而 torch_npu 失败 → 问题在适配层

---

## 12. 快速定界决策树

```
torch_npu 报错
│
├─ import 失败
│  ├─ "undefined symbol" 含 cxx11 → GCC ABI 不匹配，torch.__config__.show() 查版本
│  ├─ "Unsupported soc version" → torch_npu 版本过旧，升级
│  ├─ 调用栈含 triton → 卸载 triton 或设 TORCH_DEVICE_BACKEND_AUTOLOAD=0
│  ├─ `import torch_npu` / `_c10d_npu_init` 直接 SIGSEGV
│  │  ├─ 同环境旧包正常、重编包崩溃 → 比较 good/bad 包 `.comment`
│  │  ├─ `libpython`、`torch/_C`、good 包为 GCC 11.x，bad 包为 GCC 10.x → 工具链不兼容
│  │  ├─ `_lazy_init` 卡死且未进入 pybind 绑定 → 再排查驱动异常，`lspci | grep ascend`，gdb 采集堆栈
│  │  └─ GCC 11 只存在于容器 / overlay → 回到原容器内 clean rebuild，不要直接复用 overlay 编译器
│  └─ "Failed to load the backend extension" → 版本三元组不兼容
│
├─ 编译期错误
│  ├─ "undefined reference to op_api::*" → op-plugin submodule 与 torch_npu tag 不对齐
│  ├─ "is not a member of OpCommand" → op-plugin 比 torch_npu 新
│  ├─ "No such file or directory: aclnn*.h" → CANN 版本过旧
│  ├─ "ld: cannot find -lstdc++fs" → GCC 版本不兼容，可先尝试与 torch / Python 相同的大版本工具链
│  ├─ "is_convertible_v was not declared" → C++ 标准被降级为 C++14
│  ├─ error code 500002 → GE 图编译失败，查 plog，检查 opp_kernel 包
│  └─ "executor was not declared" → 头文件前向声明缺失
│
├─ 运行期错误
│  ├─ "not implemented for 'PrivateUse1'" → 算子未注册
│  │  ├─ 新算子 → 需要添加 TORCH_LIBRARY_IMPL 注册
│  │  ├─ 稀疏 tensor 输入 → NPU 不支持稀疏算子，改用密集等价实现
│  │  └─ 已有算子 → 检查 dispatch key 和 codegen
│  │
│  ├─ "undefined symbol: aclnn*" → CANN 版本不支持
│  │  ├─ 添加 DO_COMPATIBILITY fallback
│  │  └─ 或升级 CANN 版本
│  │
│  ├─ "ACL_ERROR_*" / "EZ9999" → ACLNN 执行错误
│  │  ├─ 161002 + CheckAxisRange/CheckShapeValid → opapi 层 infershape 错误
│  │  ├─ 0x800000 MTE 越界 → 检查 gen_opapi 的 out dtype 配置
│  │  ├─ CPU 正常 + ACLOP 正常 + ACLNN 异常 → 优先判定 ACLNN / ACLOP 语义不一致
│  │  ├─ 多框架共卡 → 用 ASCEND_RT_VISIBLE_DEVICES 隔离
│  │  ├─ 参数错误 → 检查 dtype/shape/format 是否匹配
│  │  ├─ 内部错误 → 可能是 CANN bug，尝试不同版本
│  │  └─ 资源错误 → 检查显存和设备状态
│  │
│  ├─ ATB "setup failed" → 开启 ASDOPS_LOG_LEVEL=INFO 查看真正原因
│  │  └─ CheckIniMatch Failed → dtype 组合不合法，对比 Supported Combs
│  │
│  ├─ "format mismatch" / TransData 失败 → Format 问题
│  │  ├─ 检查 allow_internal_format 开关
│  │  ├─ 检查 tensor 维度是否满足目标格式约束
│  │  └─ 尝试 npu_format_cast 转为基础格式
│  │
│  ├─ SIGSEGV / 段错误 → 内存问题
│  │  ├─ 启用 ASCEND_LAUNCH_BLOCKING=1 同步执行
│  │  ├─ gdb 采集堆栈定位
│  │  └─ 检查输出 tensor shape 是否正确
│  │
│  ├─ OOM → 显存不足
│  │  ├─ 减小 batch size
│  │  ├─ 检查 HCCL 通信域数量（每个占数百 MB）
│  │  └─ 检查是否有显存泄漏（npu-smi info）
│  │
│  ├─ EJ0001 HCCL 初始化失败 → 检查残留进程，kill 后等 10 秒
│  │
│  ├─ "torch._dynamo.exc.Unsupported" → torch.compile 兼容性
│  │  └─ 找 "from user code" 调用栈，检查 torch_npu wrapper side effect
│  │
│  └─ NpuFusedOptimizer 报错
│     ├─ "set_to_none is not supported" → 改 set_to_none=False
│     ├─ aclnnInplaceAdd tensor size 不匹配 → 检查参数分组逻辑
│     └─ KeyError on state_dict["state"][0] → optimizer state 为空，添加非空守卫
│
└─ 精度问题
   ├─ 全零/全 NaN → 算子执行异常或 format 错误
   ├─ 小幅偏差 → fp16 精度或 CANN 变更
   ├─ 大幅偏差 → 逻辑错误或参数传递错误
   ├─ 仅特定 dtype → dtype 提升或转换缺失
   ├─ 切分计算不一致 → MatMul K-shift，试 CLOSE_MATMUL_K_SHIFT=1
   ├─ 非连续 tensor 结果偏移 → 检查 out 参数写入逻辑
   ├─ set_device() 后精度降低 → GE 初始化顺序，plog 搜 PrecisionMode
   └─ warning 提示 CPU 执行但结果仍错 → CPU fallback 路径本身有 bug
```
