# torch_npu 常见问题模式与根因分类

## 目录

1. [根因分类总览](#1-根因分类总览)
2. [精度与数值问题](#2-精度与数值问题)
3. [算子注册与分发问题](#3-算子注册与分发问题)
4. [Shape 与 Format 问题](#4-shape-与-format-问题)
5. [ACLNN 适配问题](#5-aclnn-适配问题)
6. [编译问题](#6-编译问题)
7. [运行时问题](#7-运行时问题)
8. [性能问题](#8-性能问题)
9. [快速定界决策树](#9-快速定界决策树)

---

## 1. 根因分类总览

| 分类 | 占比(估) | 涉及组件 |
|------|---------|---------|
| 精度/数值 | ~25% | opapi/aclops kernel, CANN |
| 算子注册/分发 | ~15% | codegen, TORCH_LIBRARY_IMPL |
| Shape/Format | ~15% | FormatHelper, npu_preparation |
| ACLNN 适配 | ~15% | op_api_common.h, CANN headers |
| 编译 | ~10% | ci/build.sh, CMake, headers |
| 运行时 | ~10% | Stream, Allocator, ACL interface |
| 性能 | ~10% | Format 转换, task queue |

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
| **ACLNN 特殊值处理** | aclnn 算子对 inf/nan 处理不正确 | 对比 CPU 结果，检查 ACLNN 实现 |
| **dtype 未正确传递** | 输入 dtype 在转换过程中丢失 | 检查 npu_preparation 和 dtype 推导 |
| **CANN 内核变更** | CANN 升级后算子计算行为变化 | 对比不同 CANN 版本结果 |
| **反向传播精度** | backward 中累加误差或 dtype 降级 | 检查 grad 函数的 dtype 处理 |
| **Format 转换精度损失** | NZ/FRACTAL_Z 格式转换引入误差 | 检查 FormatHelper 转换路径 |

### 诊断步骤

1. 确认环境: torch_npu 版本、CANN 版本、设备型号
2. 对比基准: 同样输入在 CPU 上的结果
3. 隔离 dtype: 用 float32 测试是否仍有偏差
4. 检查特殊值: 输入是否包含 inf/nan/极值
5. 检查 CANN: 对比不同 CANN 版本的结果

### 二级定界决策

```
allclose 失败
├─ 输出全零 → 算子未正确执行或 format 错误
├─ 输出全 NaN → dtype 溢出或未初始化，检查 fp16 计算链路
├─ 小幅偏差 (< 1e-3) → 累加精度或 CANN 变更
├─ 大幅偏差 → 逻辑错误或参数传递错误
├─ 仅 fp16/bf16 偏差 → dtype 提升缺失
└─ 仅反向偏差，正向正常 → 检查 backward 实现
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

### 诊断步骤

1. 确认 CANN 版本: `cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg`
2. 检查符号: `nm -D /usr/local/Ascend/ascend-toolkit/latest/lib64/libopapi.so | grep aclnnXxx`
3. 检查头文件: 查看 CANN 头文件中 aclnn 函数的签名
4. 检查 DO_COMPATIBILITY: 确认 fallback 路径是否正确

---

## 6. 编译问题

### 诊断特征

**典型错误信息**:
- `fatal error: xxx.h: No such file or directory`
- `undefined reference to 'xxx'`
- `error: no matching function for call to 'xxx'`
- 增量编译后链接失败

### 常见根因

| 根因 | 说明 | 定位方向 |
|------|------|---------|
| **头文件依赖** | CANN 头文件路径变更或缺失 | 检查 CMakeLists.txt include 路径 |
| **增量编译失败** | 缓存的 .o 文件与新代码不兼容 | 清理 build/ 目录后重新编译 |
| **CANN 版本不匹配** | 代码使用了新 CANN API 但环境是旧版 | 检查 CANN 版本兼容性 |
| **op-plugin 子模块未更新** | third_party/op-plugin 版本不对 | `git submodule update --init` |
| **Python 版本不匹配** | 编译时和运行时 Python 版本不同 | 检查 `--python=3.9` 参数 |

### 诊断步骤

1. 清理重编: `rm -rf build/ && bash ci/build.sh --python=3.9`
2. 检查子模块: `git submodule status`
3. 检查 CANN: 确认 CANN toolkit 路径和版本
4. 检查 CMake 日志: `build/CMakeFiles/CMakeError.log`

---

## 7. 运行时问题

### 诊断特征

**典型错误信息**:
- `SIGSEGV` / `Segmentation fault`
- `ACL_ERROR_RT_*` 运行时错误
- `stream sync failed`
- `device memory not enough` / OOM

### 常见根因

| 根因 | 说明 | 定位方向 |
|------|------|---------|
| **Stream 同步缺失** | 异步执行时数据未就绪 | 启用 ASCEND_LAUNCH_BLOCKING=1 |
| **内存越界** | 输出 tensor 大小不足 | 检查 output shape 计算 |
| **显存不足** | NPU 显存 OOM | 检查 batch size 和模型大小 |
| **设备未初始化** | 未正确初始化 NPU 设备 | 检查 torch.npu.set_device() |
| **多 stream 竞争** | 多个 stream 访问同一内存 | 检查 stream 同步点 |

### 诊断步骤

1. 同步执行: `export ASCEND_LAUNCH_BLOCKING=1` 定位异步错误
2. 检查内存: `torch.npu.memory_summary()` 查看显存使用
3. 最小复现: 缩小输入规模，确认是否为 OOM
4. 检查堆栈: 使用 gdb 或 ASAN 定位 SIGSEGV

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

### 诊断步骤

1. Profiler: 使用 `torch.npu.profiler` 分析算子耗时
2. 检查 format: 关注 TransData 操作的数量和耗时
3. 检查 dispatch: 确认算子走的是 opapi 还是 aclops
4. 检查环境变量: TASK_QUEUE_ENABLE, ASCEND_LAUNCH_BLOCKING

---

## 9. 快速定界决策树

```
torch_npu 报错
│
├─ 编译期错误
│  ├─ "No such file or directory" → 头文件缺失，检查 CANN 路径
│  ├─ "undefined reference" → 链接错误，检查库路径和 CANN 版本
│  └─ "no matching function" → API 签名变更，检查 CANN 头文件
│
├─ 运行期错误
│  ├─ "not implemented for 'PrivateUse1'" → 算子未注册
│  │  ├─ 新算子 → 需要添加 TORCH_LIBRARY_IMPL 注册
│  │  └─ 已有算子 → 检查 dispatch key 和 codegen
│  │
│  ├─ "undefined symbol: aclnn*" → CANN 版本不支持
│  │  ├─ 添加 DO_COMPATIBILITY fallback
│  │  └─ 或升级 CANN 版本
│  │
│  ├─ "ACL_ERROR_*" / "EZ9999" → ACLNN 执行错误
│  │  ├─ 参数错误 → 检查 dtype/shape/format 是否匹配
│  │  ├─ 内部错误 → 可能是 CANN bug，尝试不同版本
│  │  └─ 资源错误 → 检查显存和设备状态
│  │
│  ├─ "format mismatch" / TransData 失败 → Format 问题
│  │  ├─ 检查输入 format 是否为算子支持的格式
│  │  └─ 尝试 npu_format_cast 转为基础格式
│  │
│  ├─ SIGSEGV / 段错误 → 内存问题
│  │  ├─ 启用 ASCEND_LAUNCH_BLOCKING=1 同步执行
│  │  └─ 检查输出 tensor shape 是否正确
│  │
│  └─ OOM → 显存不足
│     ├─ 减小 batch size
│     └─ 检查是否有显存泄漏
│
└─ 精度问题
   ├─ 全零/全 NaN → 算子执行异常或 format 错误
   ├─ 小幅偏差 → fp16 精度或 CANN 变更
   ├─ 大幅偏差 → 逻辑错误或参数传递错误
   └─ 仅特定 dtype → dtype 提升或转换缺失
```
