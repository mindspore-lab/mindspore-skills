# 高频问题算子画像

基于 gitcode 100 个问题单 + gitee 35K+ 问题单统计，选出高频问题算子，提供源码路径、已知问题和快速诊断提示。

所有路径以 `mindspore/mindspore/` 为前缀（即 `{工作目录}/mindspore/mindspore/`）。

---

## matmul

| 字段 | 内容 |
|------|------|
| YAML | `ops/op_def/yaml/matmul_op.yaml` |
| Infer | `ops/infer/ops_func_impl/matmul_ext.cc` |
| Kernel (Ascend) | `ops/kernel/ascend/aclnn/kernel_mod_impl/customize/batch_matmul_aclnn_kernel.cc` |
| Bprop | `ccsrc/frontend/expander/grad/grad_math_ops.cc` |
| Python API | `python/mindspore/ops/function/math_func.py` |

**已知问题**:
- CANN 升级后 matmul 计算结果微调，需重新基线 (#41977)
- 大 k 轴累加精度问题，fp16 下尤为明显 (#41931)
- 半自动并行场景偶现 core dump (EnableDvmComm 多线程冲突) (#41961)

**快速诊断**:
- 精度偏差 < 1e-3 → 先检查 CANN 版本是否升级
- 偶现 core dump + 多卡 → 检查通信初始化的线程安全

---

## conv2d

| 字段 | 内容 |
|------|------|
| YAML | `ops/op_def/yaml/conv2d_ext_op.yaml` |
| Infer | `ops/infer/conv2d.cc` |
| Kernel (Ascend) | `ops/kernel/ascend/aclnn/kernel_mod_impl/customize/conv2d_ext_aclnn_kernel.cc` |
| Bprop | `ccsrc/frontend/expander/grad/grad_nn_ops.cc` |
| Python API | `python/mindspore/ops/function/nn_func.py` |

**已知问题**:
- 动态 shape 场景 Resize 中需正确处理 -1 维度
- fp16 精度在特定 kernel size 下可能不足

**快速诊断**:
- 仅 Ascend 出错 → 检查 ACLNN kernel 的 dtype 支持
- 动态 shape 报错 → 检查 InferShape 中的 -1 处理

---

## concat

| 字段 | 内容 |
|------|------|
| YAML | `ops/op_def/yaml/concat_op.yaml` |
| Infer | `ops/infer/ops_frontend_func_impl/concat.cc` |
| Kernel (Ascend) | `ops/kernel/ascend/aicpu/aicpu_ops/cpu_kernel/ms_kernel/concat.h` |
| Bprop | `ccsrc/frontend/expander/grad/grad_array_ops.cc` |
| Python API | `python/mindspore/ops/function/array_func.py` |

**已知问题**:
- 通信算子 AllGather + concat 场景偶现精度问题 (#42155)
- 动态 shape 下 axis 为负数时推导可能出错

**快速诊断**:
- 通信场景 → 检查 complex dtype 支持
- 动态 shape → 检查 axis 负数处理

---

## gather / gather_d

| 字段 | 内容 |
|------|------|
| YAML | `ops/op_def/yaml/gather_op.yaml` |
| Infer | `ops/infer/ops_frontend_func_impl/gather_d.cc` |
| Kernel (Ascend) | `ops/kernel/ascend/aicpu/aicpu_ops/cpu_kernel/ms_kernel/gather_nd.cc` |
| Bprop | `ccsrc/frontend/expander/grad/grad_array_ops.cc` |
| Python API | `python/mindspore/ops/function/array_func.py` |

**已知问题**:
- GatherD 算子在 gitee 中出现频率极高 (15 次)
- 反向 scatter 操作在特定 index 分布下精度问题

**快速诊断**:
- 精度问题 → 检查 bprop 中的 scatter 实现
- 动态 shape → 检查 index 维度推导

---

## reshape

| 字段 | 内容 |
|------|------|
| YAML | `ops/op_def/yaml/reshape_op.yaml` |
| Infer | `ops/infer/ops_frontend_func_impl/reshape.cc` |
| Kernel (Ascend) | `ops/kernel/ascend/internal/reshape_and_cache.cc` |
| Bprop | `ccsrc/frontend/expander/grad/grad_array_ops.cc` |
| Python API | `python/mindspore/ops/function/array_func.py` |

**已知问题**:
- 动态 shape 下 -1 维度推导失败
- 与 view 操作混用时内存布局问题

**快速诊断**:
- `product of shape should be equal` → 检查 InferShape 中的 -1 推导
- 仅动态 shape 出错 → 检查 `IsDynamic` 判断

---

## pow

| 字段 | 内容 |
|------|------|
| YAML | `ops/op_def/yaml/pow_op.yaml` |
| Infer | `ops/infer/grad/power_grad.cc` |
| Kernel (Ascend) | `ops/kernel/ascend/aclnn/kernel_mod_impl/customize/pow_tensor_scalar_aclnn_kernel.h` |
| Bprop | `ccsrc/frontend/expander/grad/grad_math_ops.cc` |
| Python API | `python/mindspore/ops/function/math_func.py` |

**已知问题**:
- 4D broadcast 场景反向梯度为零 (GE 内存踩踏) (#41932, CS-001)
- 反向 Select 操作在 A2 后端有内存安全风险

**快速诊断**:
- 反向梯度为零 → 检查 bprop 中的 Select 操作，考虑用 Mul+Cast 替代
- 仅 Ascend 出错 → 检查 GE 场景下的内存管理

---

## select

| 字段 | 内容 |
|------|------|
| YAML | `ops/op_def/yaml/select_op.yaml` |
| Infer | `ops/infer/ops_frontend_func_impl/select.cc` |
| Kernel (Ascend) | `ops/kernel/ascend/aicpu/aicpu_ops/cpu_kernel/ms_kernel/select.cc` |
| Bprop | `ccsrc/frontend/expander/grad/grad_array_ops.cc` |
| Python API | `python/mindspore/ops/function/array_func.py` |

**已知问题**:
- condition 与 input shape 广播规则检查不充分 (#41936)
- 在反向图中使用 Select 可能导致 GE 内存问题

**快速诊断**:
- `shapes can not broadcast` → 检查 InferShape 中的广播兼容性检查
- 反向图中的 Select → 考虑用 Mul+Cast 替代

---

## sort / argsort

| 字段 | 内容 |
|------|------|
| YAML | `ops/op_def/yaml/argsort_op.yaml` / `ops/op_def/yaml/sort_ext_op.yaml` |
| Infer | `ops/infer/sort.cc` |
| Kernel (Ascend) | `ops/kernel/ascend/aclnn/kernel_mod_impl/customize/sort_ext_aclnn_kernel.cc` |
| Bprop | `ccsrc/frontend/expander/grad/grad_array_ops.cc` |
| Python API | `python/mindspore/ops/function/array_func.py` |

**已知问题**:
- 非确定性计算场景 MD5 对比偶现失败 (#41956)
- stable sort 与 unstable sort 行为差异

**快速诊断**:
- MD5 对比失败 → CANN 算子非确定性，不应做 MD5 对比
- 精度偏差 → 检查 stable 参数设置

---

## topk

| 字段 | 内容 |
|------|------|
| YAML | `ops/op_def/yaml/topk_ext_op.yaml` |
| Infer | `ops/infer/topk.cc` |
| Kernel (Ascend) | `ops/kernel/ascend/aclnn/kernel_mod_impl/customize/topk_ext_aclnn_kernel.cc` |
| Bprop | `ccsrc/frontend/expander/grad/grad_array_ops.cc` |
| Python API | `python/mindspore/ops/function/array_func.py` |

**已知问题**:
- 动态 shape 下 k 值推导
- 与 MoE 路由结合时的精度问题

**快速诊断**:
- 动态 shape → 检查 k 是否为动态值
- 精度 → 检查 largest/sorted 参数

---

## dropout

| 字段 | 内容 |
|------|------|
| YAML | `ops/op_def/yaml/dropout_op.yaml` |
| Infer | `ops/infer/dropout_do_mask.cc` |
| Kernel (Ascend) | `ops/kernel/ascend/aicpu/aicpu_ops/cpu_kernel/ms_kernel/dropout2d.h` |
| Bprop | `ccsrc/frontend/expander/grad/grad_nn_ops.cc` |
| Python API | `python/mindspore/nn/layer/dropout.py` |

**已知问题**:
- 随机数种子在 ops 接口和 mint 接口下行为不同
- 分布式训练中 dropout mask 同步问题

**快速诊断**:
- 精度不一致 → 检查是否使用 mint 接口（推荐）
- 分布式 → 检查 dropout mask 是否在各卡同步

---

## tile

| 字段 | 内容 |
|------|------|
| YAML | `ops/op_def/yaml/tile_op.yaml` |
| Infer | `ops/infer/ops_frontend_func_impl/tile.cc` |
| Kernel (Ascend) | `ops/kernel/ascend/aicpu/aicpu_ops/cpu_kernel/ms_kernel/tile.cc` |
| Bprop | `ccsrc/frontend/expander/grad/grad_array_ops.cc` |
| Python API | `python/mindspore/ops/function/array_func.py` |

**已知问题**:
- multiples 参数为动态值时推导
- 高维 tile 的内存效率

**快速诊断**:
- 动态 shape → 检查 multiples 是否为动态
- 内存 OOM → 检查 tile 倍数是否合理

---

## pad / pad_v3

| 字段 | 内容 |
|------|------|
| YAML | `ops/op_def/yaml/pad_v3_op.yaml` |
| Infer | `ops/infer/pad.cc` / `ops/infer/pad_v3.cc` |
| Kernel (Ascend) | `ops/kernel/ascend/aicpu/aicpu_ops/cpu_kernel/ms_kernel/pad_v3.h` |
| Bprop | `ccsrc/frontend/expander/grad/grad_nn_ops.cc` |
| Python API | `python/mindspore/ops/function/nn_func.py` |

**已知问题**:
- pad_v3 在 gitee 中出现频率高 (9 次)
- 动态 shape 下 padding 值推导

**快速诊断**:
- 动态 shape → 检查 padding 是否为动态值
- 精度 → 检查 mode (constant/reflect/replicate) 是否正确

---

## stack / unstack

| 字段 | 内容 |
|------|------|
| YAML | `ops/op_def/yaml/stack_ext_op.yaml` |
| Infer | `ops/infer/stack.cc` |
| Kernel (Ascend) | `ops/kernel/ascend/aclnn/kernel_mod_impl/customize/stack_ext_aclnn_kernel.cc` |
| Bprop | `ccsrc/frontend/expander/grad/grad_array_ops.cc` |
| Python API | `python/mindspore/ops/function/array_func.py` |

**已知问题**:
- 输入 tensor 数量动态时推导
- 与 sequence 操作混用时类型问题

**快速诊断**:
- 动态输入数量 → 检查 sequence_stack 实现
- 类型错误 → 检查 list vs tuple 输入处理

---

## split

| 字段 | 内容 |
|------|------|
| YAML | `ops/op_def/yaml/split_op.yaml` |
| Infer | `ops/infer/ops_frontend_func_impl/split_tensor.cc` |
| Kernel (Ascend) | `ops/kernel/ascend/aicpu/aicpu_ops/cpu_kernel/ms_kernel/split.cc` |
| Bprop | `ccsrc/frontend/expander/grad/grad_array_ops.cc` |
| Python API | `python/mindspore/ops/function/array_func.py` |

**已知问题**:
- split_size 为动态值时推导
- 不均匀分割时的边界处理

**快速诊断**:
- 动态 shape → 检查 split_size 是否为动态
- 形状错误 → 检查 dim 参数和 total_size 整除性

---

## norm (LayerNorm / BatchNorm / GroupNorm)

| 字段 | 内容 |
|------|------|
| YAML | `ops/op_def/yaml/layer_norm_op.yaml` / `ops/op_def/yaml/group_norm_op.yaml` |
| Infer | `ops/infer/LayerNormBetaGammaBackprop.cc` |
| Kernel (Ascend) | `ops/kernel/ascend/aclnn/kernel_mod_impl/customize/layer_norm_ext_aclnn_kernel.cc` |
| Bprop | `ccsrc/frontend/expander/grad/grad_nn_ops.cc` |
| Python API | `python/mindspore/nn/layer/normalization.py` |

**已知问题**:
- global_norm 在优化器并行场景偶现问题 (#42129)
- LayerNorm 反向 beta/gamma 梯度精度

**快速诊断**:
- 分布式 + norm → 检查 global_norm 的 lazy_inline 导入
- 精度 → 检查 eps 参数和 dtype

---

## adam / adamw

| 字段 | 内容 |
|------|------|
| YAML | `ops/op_def/yaml/adamw_op.yaml` |
| Infer | `ops/infer/adam.cc` |
| Kernel (Ascend) | `ops/kernel/ascend/aicpu/aicpu_ops/cpu_kernel/ms_kernel/fused_sparse_adam.h` |
| Bprop | N/A (优化器无反向) |
| Python API | `python/mindspore/nn/optim/adam.py` |

**已知问题**:
- TF 基准版本升级导致 dtype 自动提升行为变化 (#41934, CS-002)
- FusedAdamW amsgrad 参数冲突 (#42227, CS-008)
- add_param_group 后 amsgrad 状态未初始化

**快速诊断**:
- 与 TF 不一致 → 先检查 TF 版本，确认基准代码显式指定 dtype
- TypeError += NoneType → 检查 amsgrad 状态初始化

---

## trunc / fix

| 字段 | 内容 |
|------|------|
| YAML | `ops/op_def/yaml/trunc_op.yaml` |
| Infer | `ops/infer/ops_func_impl/trunc.h` (继承 `EltwiseOpFuncImpl`) |
| Kernel (Ascend) | `LAUNCH_ACLNN(aclnnTrunc)` — 自动生成，`pyboost_ascend_ops_2.cc` |
| Kernel (CPU) | `ops/kernel/cpu/native/trunc_cpu_kernel.cc` — 使用 `std::trunc()` |
| Kernel (GPU) | `ops/kernel/gpu/cuda/math/elementwise_ops_gpu_kernel.cc` |
| ACLNN 注册 | `aclnn_kernel_register_auto.cc`: `MS_ACLNN_COMMON_KERNEL_FACTORY_REG(Trunc, aclnnTrunc, 2)` |
| Python API | `mint.trunc` → `ops.auto_generate.trunc_op`; `mint.fix` → `mint.trunc` 的别名 |
| numpy API | `numpy.fix` → 用 `floor` + `ceil` + `select` 组合实现（不依赖 Trunc 算子） |

**已知问题**:
- #IC3M0Q: dtype 支持不对齐 — MindSpore 不支持 int8/int16/uint8，PyTorch 支持
- #42295: 910A 上 `aclnnTrunc` 对大数值返回 INT32_MAX — aclnn 内部 float→int32 溢出（CANN 缺陷）
- `aclnnTrunc` 不支持 float64 (DT_DOUBLE)，仅支持 DT_FLOAT/DT_FLOAT16/DT_BFLOAT16

**快速诊断**:
- 返回值为 2.14748e+09 或 -2.14748e+09 → INT32 溢出，检查 aclnn 平台实现 (M-012)
- dtype 不支持报错 → 检查 aclnnTrunc 支持的 dtype 列表
- 910A vs 910B 结果不一致 → 平台特有的 aclnn 实现差异
