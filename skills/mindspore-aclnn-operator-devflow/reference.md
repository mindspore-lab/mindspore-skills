# ACLNN 算子开发全流程参考（细节版）

本文件给 `mindspore-aclnn-operator-devflow` 提供"可按图索骥"的细节与模板。需要时再读取，避免把 `SKILL.md` 撑得过长。

> **关于"来自/来源"标注**：部分章节标题中的 `来自 算子流程/.../xxx.md` 是知识溯源标记，
> 表明该节内容提炼自哪份原始文档。**这些原始文档不随 skill 分发**，不影响本文件的独立使用。

## 目录

- [1. 目录/文件定位](#1-目录文件定位建议用搜索而非硬编码路径)
- [2. YAML 设计模板](#2-yaml-设计模板前向反向各一份)
- [3. gen_ops.py 常见问题定位](#3-genopspy-常见问题定位)
- [4. GeneralInfer（C++）推导约定](#4-generalinferc推导约定)
- [5. PyBoost（Pynative）实现要点](#5-pyboostpynative实现要点)
- [6. KBK（Graph）kernel 要点](#6-kbkgraphkernel-要点)
- [7. BPROP 接线要点](#7-bprop-接线要点)
- [8. 测试策略（UT + ST）](#8-测试策略ut--st)
- [9. 文档与导出](#9-文档与导出)
- [10. 交付/转测共识要点](#10-交付转测共识要点)
- [11. 资料开发要点](#11-资料开发要点)
- [12. 性能自验工具 apitimewrapper](#12-性能自验工具-apitimewrapper)
- [13. 开源运作（RFC）流程要点](#13-开源运作rfc流程要点)
- [14. 反向实现注意事项](#14-反向实现注意事项)
- [15. 安全编码与代码检视](#15-安全编码与代码检视)
- [16. Resize/Launch 优化要点](#16-resizelaunch-优化要点)
- [17. 精度零偏差与显存对齐自验](#17-精度零偏差与显存对齐自验)
- [18. 用 Cursor 辅助分析 PyTorch 算子](#18-用-cursor-辅助分析-pytorch-算子)
- [19. 接口开发要点（functional / nn / Tensor）](#19-接口开发要点functional--nn--tensor)
- [20. 问题处理与"书面结论"](#20-问题处理与书面结论)
- [21. 质量门禁与格式要求](#21-质量门禁与格式要求)
- [22. 当 ACLNN/PTA 文档不完善：用"探测脚本"补齐事实范围](#22-当-aclnnpta-文档不完善用探测脚本补齐事实范围)
- [23. vmap 支持（按需）](#23-vmap-支持按需)
- [24. 代码骨架模板（可直接复制改造）](#24-代码骨架模板可直接复制改造)
- [25. PTA 源码审查方法（必做）](#25-pta-源码审查方法必做)
- [26. InferValue 常量折叠（可选优化）](#26-infervalue-常量折叠可选优化)
- [27. 动态 shape 分类与处理策略](#27-动态-shape-分类与处理策略)
- [28. ACLNN 调用链分析与子算子盘点（组合场景）](#28-aclnn-调用链分析与子算子盘点组合场景)
- [29. 组合实现模式（PyBoost/KBK 多 ACLNN 串联）](#29-组合实现模式pyboostkbk-多-aclnn-串联)
- [§30. Feature 文档（评审与交付必须产物）](#30-feature-文档评审与交付必须产物)

---

## 1. 目录/文件定位（建议用搜索而非硬编码路径）

MindSpore / op-plugin 的目录在不同分支可能不一致，优先用搜索定位：
- 通过字符串搜索：`gen_ops.py`、`LAUNCH_ACLNN`、`MS_ACLNN_KERNEL_FACTORY_REG`、`REG_BPROP_BUILDER`。
- 通过相似算子对照：按目标算子特征分类，在仓库中搜索已接入的同类算子（详见 §2.4）。

常见的"目标区域"（仅作方向提示）：
- **YAML**：`mindspore/ops/op_def/yaml/`
- **推导/元实现**：`mindspore/` 下 `ops` / `infer` / `ops_func_impl` 等目录（以实际仓库为准）
- **Ascend kernel / PyBoost / KBK**：`mindspore/ccsrc/` 与 `op-plugin-*/` 内的 `ascend`/`kernel`/`aclnn`/`customize`
- **bprop**：`mindspore/ccsrc/` 下 `bprop` / `grad_*ops.cc`
- **测试**：`tests/ut/`、`tests/st/`
- **文档**：英文 function_doc 的 YAML + 中文 `docs/api/api_python/ops/*.rst`

## 2. YAML 设计模板（前向/反向各一份）

### 2.1 最小一致性原则
同一个参数（例如 `actual_seq_len`）必须在以下位置**一致**：
- YAML（op_def + api_def + function_doc）
- GeneralInfer（C++ 推导）
- PyBoost（Pynative 调用）
- KBK（Graph kernel 取参/Launch）
- 文档（中英文）
- UT/ST（覆盖参数边界与异常路径）

### 2.2 Customize 后缀
若你走的是项目默认 ACLNN kernel 机制，**一般不需要在 YAML 手工加 Customize 后缀**（框架会自动处理）。

### 2.3 两条接入路径（核心决策——决定整个开发工作量）

ACLNN 算子接入的**最关键决策**是：MindSpore API 的参数能否原样透传给 ACLNN 接口？
这决定了走**自动生成路径**还是**手动 Customize 路径**，直接影响需要写哪些文件。

#### 路径 1：自动生成（参数直通，不需要 Customize）

**适用条件**：MindSpore API 参数与 ACLNN 接口参数完全一致——
参数个数、顺序、类型、默认值都不需要在调用前做任何转换。

**YAML 配置关键**：`op_def` 中 `dispatch: enable: True`，**不写** `Ascend:` 字段。
框架内部 `Ascend` 默认为 `'default'`，走自动生成路径。

```yaml
# 路径 1 示例（如 abs、mul、trunc）
dispatch:
  enable: True
  # 不写 Ascend 字段 → 自动生成
```

**编译时自动生成**：
- PyBoost 调用代码（`pyboost_ascend_call_template.tpl` → `LAUNCH_ACLNN(aclnnXxx, ...)`）
- KBK 注册（`MS_ACLNN_COMMON_KERNEL_FACTORY_REG` → `aclnn_kernel_register_auto.cc`）
- Python 接口包装（`functional_overload.py` 等）

**开发者需要手写的文件**：
| 文件 | 对应步骤 |
| --- | --- |
| `op_def/yaml/xxx_op.yaml` | Step 1 |
| `api_def/xxx.yaml` | Step 1 |
| `op_def/yaml/doc/xxx_doc.yaml`（`_ext` 风格）或 `api_def/function_doc/xxx_doc.yaml`（旧风格） | Step 1 |
| `infer/ops_func_impl/xxx.h` + `.cc` | Step 3 |
| `aclnn_config.yaml` 添加映射（修改） | Step 2 |
| `math_func.py` / `mint/__init__.py` / `tensor_method.py` 导出（修改） | Step 7 |
| `tests/ut/cpp/ops/test_xxx.cc` | Step 8 |
| `tests/st/ops/ascend/test_xxx.py`（+ `tests/st/mint/`、`tests/st/tensor/overload/`） | Step 8 |
| 英文 function_doc + 中文 RST（每个接口形态一份） | Step 9 |

**不需要写**：PyBoost customize 文件、KBK customize 文件（**跳过 Step 4 和 Step 5**）。

**实例**：`abs`、`mul`、`trunc`、`xlogy`、`div`（基础算术）。

#### 路径 2：手动 Customize（参数需要预处理）

**适用条件**：调用 ACLNN 前需要做参数转换，常见情况：
- `tuple[int]` → `std::vector<int64_t>`（如 `actual_seq_qlen`）
- `Optional[Tensor]` 的 None 语义需特殊处理
- `str` → enum/int 转换（如 `layout: "BSND"` → 整型编码）
- 标量参数提取（从 Value 中取值）
- 多个输入需要重排序/合并后传入 ACLNN
- 输出 Tensor 需要手动分配（shape 与输入不同）

**YAML 配置关键**：`dispatch: enable: True` + `Ascend: XxxAscend`（显式指定 Customize 类名）。

```yaml
# 路径 2 示例（如 dense_lightning_indexer_grad_kl_loss）
dispatch:
  enable: True
  Ascend: DenseLightningIndexerGradKlLossAscend
```

**编译时**：gen_ops.py 生成包装代码，该包装代码调用你手写的 Customize 类
（`pyboost_ascend_customize_call_template.tpl` → `XxxAscendCustomize(...)`）。

**开发者需要额外手写的文件**（在路径 1 基础上）：
| 文件 | 对应步骤 |
| --- | --- |
| `kernel/.../pyboost_impl/customize/xxx.h` + `.cc` | Step 4 |
| `kernel/.../kernel_mod_impl/customize/xxx_aclnn_kernel.h` + `.cc` | Step 5 |
| （如有反向）上述文件的 `_grad` 版本 | Step 4/5 |

**实例**：`dense_lightning_indexer_grad_kl_loss`、`multi_scale_deformable_attn`、`conv2d_ext`、`add`。

#### 路径决策流程图

```
分析 MindSpore API 参数 vs ACLNN 接口参数
                │
      参数能否原样透传？
       ╱              ╲
      是               否
      │                │
  路径 1（自动）    路径 2（Customize）
      │                │
  YAML 不写           YAML 写
  Ascend 字段     Ascend: XxxAscend
      │                │
  跳过 Step 4/5    必须写 Step 4/5
      │                │
  编译自动生成     编译调用你的
  PyBoost/KBK      Customize 类
```

#### "对接类型"三分类与路径的对应关系

| 对接类型 | 描述 | 对应路径 |
| --- | --- | --- |
| **类型 1** | API 定义与 ACLNN 完全一致 | **路径 1**（自动生成） |
| **类型 2** | 名称不同但功能一致 | 通常**路径 1**（通过 YAML 的 `class` 字段做名称映射） |
| **类型 3** | 原型/语义不一致 | **路径 2**（必须手动 Customize） |

> **注意**：类型 2 是否需要 Customize 取决于"名称不同"是否仅限于算子名映射。
> 如果只是名字不同但参数完全一致，路径 1 即可（YAML 的 `class` 字段做映射）；
> 如果还涉及参数顺序/类型差异，仍需走路径 2。

### 2.4 相似算子查找策略（不要硬编码算子名）

开发中需要参照"已接入的相似算子"来确认代码风格、目录结构、宏用法。**不要默认指定某几个算子名**，
而是先分析目标算子的特征，再在仓库中搜索匹配的同类算子。

#### 分类维度（按优先级排列）

#### A. 功能/算法类别（最直觉的分类——同一类算子的实现模式往往高度相似）

| 类别 | 典型算子 | 共性特征 |
| --- | --- | --- |
| **Attention 族** | flash_attention、nsa_compress_attention、paged_attention、incre_flash_attention | TND/BSND 布局、多输出（softmax_max/sum）、带 mask/actual_seq_len、独立 Grad 算子 |
| **Loss 族** | cross_entropy、cosine_embedding_loss、ctc_loss、nll_loss | 前向输出 loss + 中间缓存（log_sum_exp 等）、reduction 参数（none/mean/sum）、反向需中间值 |
| **Norm 族** | layer_norm、group_norm、rms_norm、batch_norm | 输入 + weight + bias 三件套、running_mean/var 状态、rstd 中间输出、反向输出 dx/dw/db |
| **Optimizer 族** | adam、sgd、lamb、adamw | 就地更新（副作用算子）、lr/beta/epsilon 标量参数、多 Tensor 输入（param/grad/m/v）、通常无反向 |
| **激活函数族** | relu、gelu、silu、swish、leaky_relu | 逐元素、单输入单输出、反向简单（乘 mask 或导函数）、通常类型 1 直连 |
| **逐元素算术族** | add、mul、div、eq、ne、gt | 逐元素、支持广播、支持 Tensor-Scalar 重载、符号重载（`__add__`/`__eq__`）、多态分发 |
| **Reduce 族** | sum、mean、prod、amax、argmax | 沿指定 axis 缩减、keepdim 参数、输出 shape 少一个或多个维度、部分有反向（sum/mean）部分无（argmax） |
| **矩阵运算族** | matmul、bmm、linear、baddbmm | 2D/3D 矩阵乘、transpose 参数、alpha/beta 系数、输出 shape 由矩阵乘法规则决定 |
| **索引/gather 族** | index_select、gather、scatter、embedding | 索引 Tensor 输入、不规则 shape 推导、反向是 scatter/zero-fill 模式 |
| **变形/排列族** | reshape、transpose、permute、contiguous | 通常不涉及 ACLNN 计算（纯 shape 变换）、不需要反向或反向是逆变换 |
| **卷积/池化族** | conv2d、avg_pool2d、max_pool2d | kernel_size/stride/padding/dilation 四参数组、NCHW/NHWC 布局、反向有独立 Grad 算子 |
| **通信/并行族** | all_reduce、all_gather、reduce_scatter | 集合通信、group 参数、副作用算子、通常无标准 ACLNN（走 HCCL） |

> **用法**：先判断目标算子属于哪个族，然后在仓库中搜索同族已接入的算子。
> 同族算子的 Infer 推导逻辑、PyBoost/KBK 调用模式、bprop 接线方式、测试覆盖策略往往高度相似，
> 是最有价值的参照对象。

#### B. 技术实现特征（辅助筛选——在同族内进一步缩小范围）

| 维度 | 典型分类 | 搜索关键词/方法 |
| --- | --- | --- |
| **输入布局** | TND / BSND / BNSD / 标准逐元素 | 在 `op_def/yaml/` 中 grep 相同 shape 注释 |
| **ACLNN 对接方式** | 单 ACLNN 直连 / 多 ACLNN 组合 / 无 ACLNN（纯 Python 组合） | grep `LAUNCH_ACLNN` 数量；组合算子看 customize 目录 |
| **是否有反向** | 有独立 Grad 算子 / 自动微分 / 无反向 | grep `REG_BPROP_BUILDER` + grep `_grad` YAML |
| **接口形态** | functional only / functional + nn / functional + tensor / 符号重载 | 看 api_def YAML 的 `interface` 字段 |
| **参数特殊性** | 含 Optional[Tensor] / tuple[int] / 枚举(layout/mode) / 标量 | 看 YAML 的 `default: None` / `type_cast` / `arg_handler` |
| **对接类型** | 类型 1（完全一致）/ 类型 2（名称映射）/ 类型 3（需 customize） | 对照 §2.3 判断 |

#### 查找流程

1. **判断功能/算法类别**：目标算子属于上面哪个族？
   - 例：`nsa_compress_attention` → **Attention 族**
   - 例：`cosine_embedding_loss` → **Loss 族**
   - 例：`eq`（== 重载）→ **逐元素算术族**

2. **确定技术特征标签**：从 B 表中选出 2-3 个显著特征，在同族内进一步筛选。
   - 例：`nsa_compress_attention` → Attention 族 + TND 布局 + 单 ACLNN 直连 + 有独立 Grad + 含 tuple[int]
   - 例：`cosine_embedding_loss` → Loss 族 + 多 ACLNN 组合 + 无单独 Primitive + functional + nn + reduction 参数
   - 例：`adamw` → Optimizer 族 + 就地更新（副作用）+ 多 Tensor 输入 + 无反向

3. **在仓库中搜索同类**：
   ```bash
   # 按功能族名找：搜索同族算子（如 attention 族）
   grep -rl "attention" mindspore/ops/op_def/yaml/ --include="*.yaml"

   # 按布局找：搜索含相同 shape 模式的算子
   grep -r "TND" mindspore/ops/op_def/yaml/ --include="*.yaml" -l

   # 按 ACLNN 组合找：搜索 customize 目录下含多个 LAUNCH_ACLNN 的文件
   grep -rl "LAUNCH_ACLNN" mindspore/ops/kernel/.../customize/

   # 按反向模式找：搜索有 Grad 后缀 YAML 的算子
   ls mindspore/ops/op_def/yaml/*_grad_op.yaml

   # 按接口形态找：搜索同时有 tensor + function 接口的算子
   grep -l "interface:.*tensor.*function" mindspore/ops/api_def/*.yaml

   # 按 reduction 参数找（loss 族常见）
   grep -l "reduction" mindspore/ops/op_def/yaml/*.yaml
   ```

4. **选择 2-3 个最匹配的算子**，逐目录对照其 YAML/Infer/PyBoost/KBK/bprop/测试/文档的写法。
   优先选**同族 + 技术特征最接近**的；其次选**不同族但技术特征（对接类型/参数模式）相似**的。

5. **如果搜不到高度匹配的同类**（全新类型算子），退而选择"对接类型"（§2.3）相同的任意算子作为
   代码风格参考，同时在实现过程中更谨慎地逐步验证。

> **原则**：相似算子是"代码风格和结构的参照"，不是"功能逻辑的抄写对象"。
> 功能逻辑以 PTA 源码 + ACLNN 文档为准，相似算子只用来确认目录结构、宏名、注册方式、测试写法等。

### 2.5 dispatch + "先自动生成，再拷贝改造"的实用套路
当你需要自定义 PyBoost/KBK 时，一个高效做法是：
1. 在 YAML 里打开 `dispatch.enable: True`。
2. **临时注释掉** YAML 中 `dispatch.Ascend: XxxAscend` 这类自定义声明，让 `gen_ops.py` 先生成一份可编译骨架。
3. 将生成目录里的 `.h/.cc` **拷贝**到 `customize` 目录（或对应自定义目录）。
4. 按 ACLNN 实际签名调整入参（例如删除 ACLNN 不需要的 dtype、处理 tuple→vector 等）。
5. 按项目约定重命名入口（常见模式：`OpNameAscendCustomize` / `OpNameGradAscendCustomize`），恢复 YAML 声明。
6. 删除临时自动生成文件，只保留自定义实现。

## 3. gen_ops.py 常见问题定位

典型报错与方向：
- **keys 结构不匹配**：对照已有基础算子 YAML（如 add）调整字段层级。
- **缺 `py_method`**：补齐 python 暴露相关字段。
- **function_doc 缺条目**：补齐对应的 doc 节点，保持参数一致。

提示：Windows 下英文 YAML 文档尽量不要混入中文字符，避免编码问题。

## 4. GeneralInfer（C++）推导约定

### 4.1 职责边界
- 只做**形状/类型推导**；不要做运行时合法性校验（交给 ACLNN/运行时）。
- 报错使用框架异常宏，错误信息要包含：参数名、期望、实际。

### 4.2 动态 shape / 动态 rank
> 动态 shape 的完整三分类（InputDynamic / OutputDynamic）见 §27。本节侧重 Infer 推导时的快速回退策略。

推荐策略（与 `算子流程/ACLNN_nsa_compress_适配开发经验.md` 一致）：
- 动态 rank：返回动态秩（`kShapeRankAny` 或项目等价常量）。
- 推导依赖的关键参数（如 block/stride/seq_len）只要出现 unknown：
  - 输出对应维度回退为动态维（`kShapeDimAny` 或项目等价常量）
  - 其余维度沿用输入推导
- 当关键参数都已知时：尽可能返回精确 shape。

### 4.3 常用 InferInfo API（以项目已有实现为准）
- `GetScalarValueWithCheck<T>()`：取标量（带检查）
- `GetArrayValue<T>()` + `HasUnknownValue()`：取 tuple/list
- `IsNone()`：判断 None

不要凭空使用项目里不存在的 API。

## 5. PyBoost（Pynative）实现要点

### 5.1 输入参数转换
- tuple/list：建议统一转为 `std::vector<int64_t>` 再传给 ACLNN。
- 可选输入：若允许 None，需定义"None 语义"，并在 PyBoost/Infer/KBK 同步处理。

### 5.2 调用惯例
以项目已有 ACLNN 封装为准（例如 `LAUNCH_ACLNN`/`RunOp`），保持风格一致。

## 6. KBK（Graph）kernel 要点

> Init/Resize/Launch 职责分离、无意义输出、compute-depend 输出等优化要点见 §16。

推荐固定结构：
- `GetWorkSpaceInfo()`：取参 + `GetWorkspaceForResize`
- `Launch()`：调用 `RunOp` 或等价执行路径
- 注册：`MS_ACLNN_KERNEL_FACTORY_REG`（或项目等价宏）

强约束：
- 前向/反向分文件、分注册
- 头/实现命名空间保持一致（否则易出现"未声明/未定义"）

### 6.1 KBK 自动生成骨架位置提示
从经验文档示例看，KBK 的自动生成代码常落在类似目录（以实际仓库为准）：
- `.../ops/kernel/ascend/opapi/aclnn_auto_gen/`
你可以先让 `gen_ops.py` 自动生成，再拷贝到自定义目录改造（见 §2.5）。

## 7. BPROP 接线要点

> 反向实现的进阶注意事项（OutZeros/ZerosLikeExt/inplace/Depend）另见 §14。

在 bprop builder 中：
- 只为需要梯度的输入构建反向子图
- 非张量/不需要梯度的输入返回零梯度占位
- 使用 `need_compute_grad_out()`（或等价接口）做必要性判断

### 7.1 反向输入/输出个数经验规则（来自 `算子流程/.../aclnn开发示例.md`）
- **反向输入个数**：等于"正向输入个数 + 2"（`out` 与 `dout`）。
- **反向输出个数**：等于"正向输入个数"（每个输入一个梯度）。
- 多输出正向算子：`out` 在反向侧通常是 tuple，需要通过 `TupleGetItem` 取对应输出。

### 7.2 SetUnusedInputs 的使用场景
当反向不依赖某些输入的 tensor value（只依赖 shape/type 或完全没用到）时，可标记为 unused，以便
Pynative 异步场景更早释放正向 kernel 内存，降低峰值。

## 8. 测试策略（UT + ST）

### 8.1 Python UT（ops 层）
- 推导正确性：shape/type、动态/边界
- 错误路径：非法参数、None 语义（若不支持 None，要覆盖抛错）
- 固定随机种子：例如 `np.random.default_rng(seed)`

### 8.2 C++ UT（GeneralInfer）
典型构造（按项目现有 UT 工具）：
- 标量：`ShapeVector{}` + `CreateScalar<T>(value)`
- tuple：`ShapeArray{{}}` + `ValuePtrList{CreateScalar<...>(...)}`
- None：`kMetaTypeNone` + `kNone`
- unknown：使用 `kValueAny` 或项目等价占位

### 8.3 Ascend ST（对齐 torch_npu/参考实现）
- 优先"形状/类型"再比数值
- 需要严格对齐时可设 `atol/rtol=0`（以算子数值特性为准）
- 避免引入额外算子导致误差累积（例如反向里不要多余的 sum）
- bfloat16 比较前升精到 float32（避免 numpy bf16 限制）

## 9. 文档与导出

> 资料开发的详细规范（文件命名、接口列表字母序、中英文一致等）见 §11。

强一致性：
- 英文 function_doc（YAML）与中文 RST 参数名、默认值、必填/可选、示例必须一致。
- ops 包显式导出算子 API；非 Ascend 设备提供占位实现并给出清晰错误。

## 10. 交付/转测"共识要点"（来自 `算子流程/.../Aclnn算子对接开发整体流程.md`）

### 10.1 适配方案的基本原则
- **优先对标 PyTorch/PTA**：PTA 不支持的功能可以不开发。
- CANN 不支持且 PTA 也不支持的功能可以不开发。
- 正反向尽量采用与 PTA 相同的 ACLNN/aten 组合，便于达成"精度 0 偏差"的目标。

### 10.2 影响面评估
若可能影响 GEOP / GPU / CPU / Lite 现有流程，需要给出消除影响方案（例如通过 Pass/Expander）。

### 10.3 交付件与验证范围（摘要）
转测时通常要求覆盖：
- 接口形态：NN / functional / Tensor（若完全一致可只验一个入口）
- 后端：Ascend +（GPU/CPU 不回退，若本来不支持则说明）
- 模式：动态图 / 静态图 / 静态图 KernelByKernel
- shape：动态/静态
- 维度：泛化性（dtype/shape）+ 精度 + 性能（按项目门槛要求）

## 11. 资料开发要点（来自 `算子流程/.../5. 资料开发指导.md`）

> 文档导出的基本要求见 §9。

### 11.1 总原则
- **中英文严格一致**：参数、默认值、必选/可选、约束、示例等必须一致。
- **接口列表按字母序添加**：减少冲突与重复。
- **文件名 / 文件内标题 / 文件内接口定义三者一致**：不一致会导致页面生成失败。
- 示例需要**完整 import**、确保可运行；必要时打印输出或 shape 便于理解。

### 11.2 常见场景与落点（摘要）
- 新增 functional：英文注释在实现 `.py`；中文在 `docs/api/api_python/ops/` 下 `func_*.rst`；并更新接口列表。
- 新增 mint：需要同时处理 mint 的中英文列表与中文 rst（若是 import 原有接口可复用）。
- 新增 Tensor 方法：英文在 `tensor.py`，中文在 `docs/api/api_python/mindspore/Tensor/`，并更新列表。

## 12. 性能自验工具 apitimewrapper（来自 `算子流程/.../7. 接口性能自验工具.md`）

### 12.1 用途
对 MindSpore / PyTorch 脚本里的 API 做 wrap 打点，得到端到端耗时与（可选的）接口内部耗时分解。

### 12.2 使用要点（摘要）
- 安装：`pip install apitimewrapper-0.0.3-py3-none-any.whl`
- 整网打点：在网络入口启动 `start_hook_net(hook_inside=False)`，需要在网络执行前调用。
- 单 API：可同时启用 `start_hook_net` 与 `start_hook_torch_net`，用 `start_analysis()/end_analysis()` 包住循环。

## 13. 开源运作（RFC）流程要点（来自 `算子流程/.../11.算子开发开源运作规范.md`）

### 13.1 核心变化
需求分析/方案评审等传统会议流程简化为 RFC：在 Issue/RFC 里把信息链写全，通过评论完成讨论与共识。

### 13.2 RFC 内容建议（摘要）
- 背景与目标、交付范围、使用约束、遗留问题
- 方案设计（必要时外链归档）
- 验收与自测依据：UT/ST 覆盖、稳定性（多次运行无偶现）
- 代码 PR 与测试仓 PR 链接，@maintainers 检视

## 14. 反向实现注意事项（来自 `算子流程/.../算子反向注意事项.md`）

> 基础 BPROP 接线规则（I/O 个数、need_compute_grad_out、SetUnusedInputs）见 §7。

### 14.1 不可微分入参
Torch 里不可微分的入参（如 index/mode 等），MS 侧反向输出必须与输入个数一致：
- 对不可微分入参返回 `ib->OutZeros(x)`。
- 若全部入参都不可微分，可用 `ReturnZeros`（以框架现状为准）。

### 14.2 "梯度就是 0"时的实现选择
当某输入梯度理论上为 0，建议使用 `ib->ZerosLikeExt()`，确保走到 ACLNN/后端期望路径。

### 14.3 inplace 算子反向
- 若反向需要用到更新前的 self，需要注册 `CloneInplaceInput(...)` 让框架保留旧值。
- KBK 动态 shape 场景下在反向里使用 inplace 可能不保序时，可用 `ib->Depend(target, inplace_call)` 规避。

## 15. 安全编码与代码检视（来自 `算子流程/.../安全编码培训-算子代码检视.md`）

建议把"要检视的代码范围"当作改动面 checklist：
- Python 原语 + NN/functional/tensor + vmap
- C++ Infer
- bprop（Python/C++）
- 后端 kernel（CPU/GPU/AICPU/Ascend）及 Grad 单算子

## 16. Resize/Launch 优化要点（来自 `算子流程/.../ResizeKernelLaunch实现优化.md`）

> KBK 的基础结构与注册见 §6。

### 16.1 禁止在 InferShape 中修改属性
不要在 InferShape/InferType 内设置或修改算子属性（Pynative 下会引入问题）。

### 16.2 Resize/Launch 职责分离
- **能在 Init 确定的放 Init**；与 shape 强相关的放 Resize；Launch 尽量只做发射/调用。
- 运行期不要做 device 内存申请（例如 GPU 的 `cudaMalloc/cudaFree`），统一通过 workspace 由框架管理。

### 16.3 无意义输出忽略
对于预留/无意义输出，覆写 `GetUseLessOutputIdx()`（或等价接口）避免 dump/溢出误检/确定性副作用。

### 16.4 计算依赖（compute-depend）输出
按框架要求：分配最大可能输出 + 同步/更新输出 shape（如 NonZero 类模式）。

## 17. 精度零偏差与显存对齐自验

> 合并自原"精度零偏差自验指导"与"显存占用情况自验指导"。

### 17.1 精度零偏差（bitwise 一致）
当目标是与 PTA 输出二进制一致时：
- 固定随机种子，保存输出为 `.npy`
- 用 `md5sum`（或等价方式）对比两个输出文件的哈希确保一致
- 来源：`算子流程/.../精度零偏差自验指导.md`

### 17.2 显存占用对齐
关键点：MS 与 PTA 在**相同阶段**统计 max memory（避免把初始化/编译混入）。
- MS：`mindspore.runtime.max_memory_allocated()`
- PTA：`torch_npu.npu.max_memory_allocated()`
- 来源：`算子流程/.../显存占用情况自验指导.md`

## 18. 用 Cursor 辅助分析 PyTorch 算子（来自 `算子流程/.../基于 AI 工具Cursor进行pytorch算子分析.md`）

可用于"对标实现定位/反向路径查找"的提示词模板：
- `torch.<op> 算子的正反向是如何实现的？代码在哪里？`
配合查找：正向实现、autograd/derivatives 注册、NPU 插件映射等线索。

## 19. 接口开发要点（functional / nn / Tensor）（来自 `算子流程/.../2. 接口开发.md`）

### 19.1 functional 接口（强约束）
- functional 内部**务必使用** `_get_cache_prim` 获取 Primitive 实例，避免反复 __init__ 造成性能问题。
- 复杂接口允许"一对多映射/组合映射"：按参数分支选择不同 Primitive 或组合算子实现。

### 19.2 nn 接口
- nn 接口是 `Cell` 子类：在 `__init__` 初始化算子与属性，在 `construct` 做执行路径。
- `construct` 类似编译器入口：不要在其中直接 `raise`；需要编译期校验/抛错时，用 `@constexpr` 的辅助函数。

### 19.3 Tensor 方法（含 GE 映射要点）
- Tensor 方法需要覆盖不同模式：PyNative/KBK 与 GE（若项目要求）。
- GE 模式往往需要：
  - 在 `resource.cc` 注册映射；
  - 在 `standard_method.py` 实现（该处校验函数不能接收 Tensor 作为入参，需用对应封装）。

### 19.4 原语与接口接入策略（必须在 Pre-B 阶段确定）

在开始 YAML 定义之前，必须完成接口分析并确定原语/接口策略。

#### 19.4.1 接口分析五要素（来自 `aclnn开发示例.md` §1.2）

通过对比 MindSpore 与 PTA/torch 的文档和实现，搞清以下问题：
1. 功能是否一致
2. 参数定义是否一致
3. 支持的数据类型是否一致
4. **是否要新增原语**（Primitive）
5. **是新增接口还是复用原有接口**

> PTA/torch 可能存在**同名接口重载**（同函数名、不同参数签名），需逐一分析。

#### 19.4.2 YAML 三种场景（来自 `aclnn开发示例.md` §2.1）

| 场景 | YAML 操作 | 示例 |
| --- | --- | --- |
| **已有 YAML + 复用原有原语** | 在现有 YAML 上加 `dispatch` 字段 | `eye`：已有原语，加 `dispatch.Ascend: EyeAscend` |
| **已有 YAML + 新增原语** | 新建 YAML，加 `_ext` 后缀 | `zeros_like_ext`：已有 `zeros_like` 但参数不兼容 |
| **没有 YAML** | 新建 YAML，通常不加 `_ext` | 全新算子直接创建 |

**复用原有原语**示例：
```yaml
# 在已有 eye YAML 上加 dispatch 字段即可
dispatch:
  enable: True
  Ascend: EyeAscend
```

**新增原语 + `_ext`**示例：
```yaml
zeros_like_ext:
    args:
        input:
            dtype: tensor
        dtype:
            dtype: TypeId
            arg_handler: dtype_to_type_id
            default: None
    returns:
        y:
            dtype: tensor
    function:
        disable: True
    dispatch:
        enable: True
        Ascend: ZerosLikeExtAscend
```

#### 19.4.3 `ops.extend` 命名空间（来自 `Aclnn算子对接开发整体流程.md` §1.4.6）

> 若 aclnn 功能与存量 `ops.xx` 方法不一致，且不能对 ops 存量方法做兼容性修改 → 需要新增 extend 接口。

MindSpore 接口命名空间（来自 `5. 资料开发指导.md`）：
- `ops.xxx` / `ops.xxx_ext()` / `ops.auto_generate.xxx_ext()` / `ops.extend.xxx_ext()`
- `nn.xxx` / `nn.xxxExt()` / `nn.extend.xxx()`

#### 19.4.4 修改已有原语参数签名（文档缺口，需参考相似代码）

实际开发中，经常遇到已有 Primitive 需要**扩展参数**的情况（如 PTA 新版多了参数、需要支持 ACLNN 特有参数等）。原始文档提到了相关参考（"前端api接口重载开发指导"、"Tensor重载指导"），但详细内容未收录在当前知识库中。

**遇到此场景时的实践策略**：
1. **搜索 MS 仓库中相似算子**的处理方式作为参考（如同类算子是如何加参数的）
2. **具体分析兼容性**：新参数是否能设默认值、是否影响已有调用方、是否影响其他后端
3. **判断走哪条路**：
   - 可兼容修改（加可选参数不破坏已有行为）→ 直接修改现有 YAML + Infer + 接口
   - 不可兼容 → 新增原语加 `_ext` 后缀，或走 `ops.extend`
4. **确保不破坏已有功能**：修改后已有 UT/ST 全部回归通过
5. **遵循评审规则**（见 §19.4.5）

#### 19.4.5 评审规则（来自 `Aclnn算子对接开发整体流程.md` §2.2）

| 变更类型 | 评审要求 |
| --- | --- |
| 无新增接口，功能与之前完全一致 | 无需评审 |
| 无新增接口，功能有扩展 | 需要评审 |
| 新增接口 | **重点评审** |
| 存量接口功能非兼容修改 | **原则上不允许**，特殊情况评审 |
| 新增算子 | 需要评审 |
| 存量算子功能非兼容修改 | 需要评审 |

> 无论接口/算子对外与否，都需要按规则评审。评审后需发送**接口变更邮件**通知 MindSpore 各组件。

## 20. 问题处理与"书面结论"（来自 `Aclnn算子对接开发整体流程.md`）

当出现"无法修复的问题"，建议按来源分类并固化证据：

### 20.1 CANN 算子问题
- 需要拿到**正式书面记录**：规格说明书、邮件/会议纪要结论、DTS 单等。
- 聊天记录等非正式内容不作为正式结论。

### 20.2 MindSpore 框架问题 / 方案限制
- 与框架相关负责人确认后，如结论是"允许带问题转测"，需要形成会议纪要并抄送相关人员。
- 会议纪要建议包含：议题、时间、人员、背景、结论。

## 21. 质量门禁与格式要求（结合项目 `.cursorrules`）

建议在 checklist 中显式跟踪这些点：
- 行长不超过 120；避免行尾空格；尽量统一空格缩进。
- UTF-8 编码、制表符检查等基础规范。
- 代码质量检查工具（项目列出的 Check_*）在本地/CI 中应通过。

## 22. 当 ACLNN / PTA 文档不完善：用"探测脚本"补齐事实范围

现实中 ACLNN/PTA 文档可能滞后或缺失细节（尤其是不同 CANN/PTA 版本支持范围变化时）。此时不要猜：

### 22.1 必须先记录版本矩阵
让用户确认并记录（写入 RFC/验收报告/测试输出）：
- torch 版本、torch_npu 版本
- CANN 版本（或可追溯的安装路径/镜像版本信息）
- 芯片型号/驱动信息（能打印则打印）

### 22.2 生成并运行"PTA 支持范围探测脚本"
推荐做法：
- 我在本 skill 里提供了模板脚本：`scripts/probe_pta_sparse_flash_attention.py`
  （这是以 `sparse_flash_attention` 为例的**模板**，适配其他算子时需复制并修改 `run_case` 的输入构造与 API 调用）
- 用途：自动枚举一组 dtype/layout/关键参数组合，记录成功/失败与错误信息，并输出 JSON 汇总。

运行方式（示例）：

```bash
python scripts/probe_pta_sparse_flash_attention.py --device npu:0 --out pta_probe.json
# 快速模式（只跑核心组合）：
python scripts/probe_pta_sparse_flash_attention.py --device npu:0 --quick --out pta_probe_quick.json
```

你需要用户回传的证据：
- `pta_probe.json`（或其中 summary + 关键失败用例的错误信息）
- 同一份输出里的版本信息（torch/torch_npu/env/npu-smi）

### 22.3 用探测结果驱动"接口对齐与约束落地"
基于探测结果再决定：
- sparse_size 是否被固定（例如某些 CANN 版本要求 2048）
- attention_mode / return_softmax_lse / layout 组合是否可用
- dtype 支持是否确实只有 fp16/bf16，是否存在隐藏限制
并把这些结论同步到 YAML/Infer/文档/测试中。

## 23. vmap 支持（按需）

> 来源：`算子流程/.../4. 算子关键特性.md`。当前 skill 主要覆盖 ACLNN 算子的前向/反向/
> 推导/测试/文档流程，vmap 作为**可选扩展**列在此处。如果目标算子不需要 vmap 支持，可跳过本节。

### 23.1 何时需要 vmap
- 算子需要被 `vmap`/`vectorize_cell` 调用时。
- 项目要求覆盖 vmap 路径时（部分算子交付件要求 vmap 验证）。

### 23.2 关键要点（摘要）
- 注册 vmap rule：在框架指定位置注册（以项目现有 vmap 注册模式为准）。
- 测试：需单独补 vmap UT，验证批量化后的 shape/数值正确性。
- 注意 vmap 路径可能不走 ACLNN（而是退回到组合算子/循环展开），需确认性能是否可接受。

## 24. 代码骨架模板（可直接复制改造）

> 以下骨架来自"先自动生成再自定义改造"的实际经验，仅作**起步参考**。
> 真正使用时必须对照同目录下相似算子的现有代码调整宏名、命名空间、参数列表。

### 24.1 YAML 最小模板（op_def + api_def + function_doc）

```yaml
# ---- op_def ----
op_name: "OpNameCustomize"
args:
  - {name: "input_x", type: "Tensor", desc: "..."}
  - {name: "scale", type: "float", default: 1.0, desc: "..."}
outputs:
  - {name: "output", type: "Tensor", desc: "..."}
dispatch:
  enable: True
  Ascend: "OpNameAscendCustomize"
# ---- api_def ----
api:
  py_method: "op_name"
  module: "mindspore.ops"
# ---- function_doc ----
function_doc:
  desc: "Brief English description of the operator."
  args:
    input_x: "Description of input_x."
    scale: "Description of scale. Default: ``1.0``."
  returns: "Description of output."
  examples: |
    >>> import mindspore as ms
    >>> from mindspore import ops
    >>> x = ms.Tensor([1.0, 2.0, 3.0])
    >>> out = ops.op_name(x, scale=1.0)
```

### 24.2 GeneralInfer 骨架（C++）

```cpp
// op_name_general_infer.cc
#include "ops/ops_func_impl/op_name.h"
// 具体 include 路径以仓库实际为准

namespace mindspore::ops {

BaseShapePtr OpNameFuncImpl::InferShape(
    const PrimitivePtr &prim,
    const std::vector<AbstractBasePtr> &input_args) const {
  // 1. 取输入 shape
  auto x_shape = input_args[0]->GetShape()->GetShapeVector();

  // 2. 动态 rank 回退
  if (IsDynamicRank(x_shape)) {
    return std::make_shared<abstract::TensorShape>(ShapeVector{abstract::TensorShape::kShapeRankAny});
  }

  // 3. 取关键标量参数（可能 unknown）
  // auto scale_opt = GetScalarValueWithCheck<float>(input_args[1]->GetValue());
  // if (!scale_opt.has_value()) {
  //   // 关键参数 unknown -> 对应维度回退动态维
  //   out_shape[dim_idx] = abstract::TensorShape::kShapeDimAny;
  // }

  // 4. 精确推导
  ShapeVector out_shape = x_shape;  // 按算子语义计算
  return std::make_shared<abstract::TensorShape>(out_shape);
}

TypePtr OpNameFuncImpl::InferType(
    const PrimitivePtr &prim,
    const std::vector<AbstractBasePtr> &input_args) const {
  auto x_type = input_args[0]->GetType();
  // 通常输出 dtype 与输入一致或按算子语义确定
  return x_type->Clone();
}

}  // namespace mindspore::ops
```

### 24.3 PyBoost customize 骨架（C++）

```cpp
// op_name_ascend_customize.cc
#include "plugin/device/ascend/kernel/pyboost/customize/op_name_ascend_customize.h"
// 具体 include 以仓库实际为准

namespace mindspore::kernel::pyboost {

// 前向
tensor::TensorPtr OpNameAscendCustomize::Call(
    const tensor::TensorPtr &input_x,
    const std::optional<float> &scale) {
  // 1. 输出 tensor 分配
  auto output = std::make_shared<tensor::Tensor>(input_x->data_type(), out_shape);

  // 2. 参数转换（tuple->vector / None 处理等）
  // auto scale_val = scale.value_or(1.0f);

  // 3. ACLNN 两段式调用（以项目封装宏为准）
  // LAUNCH_ACLNN(aclnnOpName, stream, input_x, scale_val, output);

  return output;
}

}  // namespace mindspore::kernel::pyboost
```

### 24.4 KBK kernel 骨架（C++）

```cpp
// op_name_aclnn_kernel.cc
#include "plugin/device/ascend/kernel/opapi/aclnn_kernel/op_name_aclnn_kernel.h"
// 具体 include 以仓库实际为准

namespace mindspore::kernel {

void OpNameAclnnKernel::GetWorkSpaceInfo(
    const std::vector<KernelTensor *> &inputs,
    const std::vector<KernelTensor *> &outputs) {
  // 取参（标量/tuple 等）
  // auto scale = inputs[1]->GetValueWithCheck<float>();

  // 获取 workspace
  // GetWorkspaceForResize(aclnnOpNameGetWorkspaceSize, ...);
}

bool OpNameAclnnKernel::Launch(
    const std::vector<KernelTensor *> &inputs,
    const std::vector<KernelTensor *> &workspace,
    const std::vector<KernelTensor *> &outputs,
    void *stream_ptr) {
  // RunOp(aclnnOpName, stream, ...);
  return true;
}

// 注册
MS_ACLNN_KERNEL_FACTORY_REG(OpName, OpNameAclnnKernel);

}  // namespace mindspore::kernel
```

### 24.5 BPROP builder 骨架（C++）

```cpp
// grad_xxx_ops.cc (在合适的 bprop 注册文件中添加)

REG_BPROP_BUILDER("OpName").SetBody([](const BpropBuilder *ib) -> NodePtrList {
  // 取正向输入
  auto input_x = ib->GetInput(kIndex0);
  auto scale = ib->GetInput(kIndex1);
  // 取正向输出与上游梯度
  auto out = ib->GetInput(kIndex2);   // 正向输入个数 + 0
  auto dout = ib->GetInput(kIndex3);  // 正向输入个数 + 1

  // 构建反向子图
  NodePtr dx;
  if (ib->need_compute_grad_out(kIndex0)) {
    dx = ib->Emit("OpNameGrad", {input_x, out, dout, scale});
  } else {
    dx = ib->OutZeros(input_x);
  }

  // 非张量参数（如 scale）返回零梯度占位
  auto d_scale = ib->OutZeros(scale);

  return {dx, d_scale};
});
```

## 25. PTA 源码审查方法（必做）

> PTA 文档可能滞后或遗漏细节。开发前**必须**同时审查 op-plugin 仓库中的实际代码，
> 结合文档与源码一起参考。若两者不一致，不要猜，让用户找接口人确认（见 §25.5）。

### 25.1 需要审查的三类关键文件

| 文件 | 路径模式 | 提取什么信息 |
| --- | --- | --- |
| **函数签名 YAML** | `op_plugin/config/op_plugin_functions.yaml` | 精确的参数名、类型、默认值、返回值结构、是否走 `op_api` / `acl_op` / `gen_opapi` |
| **反向注册 YAML** | `op_plugin/config/derivatives.yaml` | 哪些输入可微、grad 函数名、参数传递顺序、`output_differentiability` |
| **C++ 实现** | `op_plugin/ops/opapi/XxxKernelNpuOpApi.cpp`（含 Grad 变体） | 实际调用的 `aclnnXxx`、参数预处理逻辑、输出 tensor 构造方式、硬编码默认值 |

### 25.2 审查时重点关注的差异点

从 PTA 代码中经常能发现文档未提及的关键细节：

**1. 前向与反向的参数命名/顺序不一致**
- 例：前向用 `actual_seq_lengths_query`，反向用 `actual_seq_qlen`
- 例：前向 `layout_query`/`layout_kv` 分开，反向退化为单个 `layout`
- **影响**：MS 侧 bprop builder 的参数传递必须按反向的实际签名，不能照搬前向

**2. 反向 ACLNN 调用的额外/隐藏参数**
- 例：反向里硬编码 `deterministic_const = true`（前向无此参数）
- 例：反向缺少 `block_table`（前向有但反向不传）
- **影响**：MS 侧 KBK/PyBoost 的反向实现要对齐这些隐藏行为

**3. 可选参数的 None 处理方式**
- 例：`query_rope` 为 None 时，PTA 构造 `at::Tensor()`（空 tensor）传给 ACLNN
- 例：`query_rope` 的梯度在 None 时输出 `at::empty({0}, ...)`（形状为 [0]，非零张量）
- **影响**：MS 侧必须同步 None 语义，否则 ACLNN 可能报错或结果不一致

**4. 输出 tensor 个数与构造**
- 例：前向返回 `(output, softmax_max, softmax_sum)` 共 3 个，反向返回 5 个
- `softmax_max`/`softmax_sum` 的 shape 推导逻辑在 C++ 里有明确的 3D/4D 分支
- **影响**：MS Infer 必须对齐输出 shape 推导逻辑，bprop 必须正确传递中间结果

**5. `derivatives.yaml` 中的梯度传递**
- 例：`result0`/`result1`/`result2` 分别对应前向的第 0/1/2 个输出
- 哪些输入标记为 `non_differentiable`，哪些参与 grad
- **影响**：MS bprop 的 `GetInput` 索引和 `OutZeros` 占位必须对齐

### 25.3 审查操作步骤

1. **在 `op_plugin_functions.yaml` 中搜索算子名**：提取前向/反向的精确签名，记录参数差异。
2. **在 `derivatives.yaml` 中搜索算子名**：提取反向注册，确认可微输入和 grad 参数传递。
3. **找到对应的 C++ 实现文件**（`ops/opapi/` 下），阅读：
   - 输出 tensor 的 shape 构造逻辑
   - 可选参数的 None 处理（`value_or` 的默认值是什么）
   - 实际传给 `EXEC_NPU_NO_FORMAT_CHECK_CMD` / `aclnnXxx` 的参数列表与顺序
   - 是否有硬编码参数（如 `deterministic`）
4. **记录发现的差异**，作为 MS 适配的依据写入验证闭环的"关键证据"部分。
5. **若发现代码与文档不一致 → 必须暂停并确认**（见 25.5）。

### 25.4 典型差异记录模板

```text
算子：npu_sparse_flash_attention

前向 vs 反向参数差异：
- actual_seq_lengths_query (fwd) → actual_seq_qlen (bwd)
- layout_query + layout_kv (fwd) → layout 单个 (bwd)
- block_table (fwd 有) → bwd 不传
- return_softmax_lse (fwd 有) → bwd 不传

反向隐藏行为：
- deterministic_const = true（硬编码）
- query_rope 为 None 时 d_query_rope = at::empty({0}, ...)

输出结构：
- 前向：(output, softmax_max, softmax_sum) = 3 个
- 反向：(d_query, d_key, d_value, d_query_rope, d_key_rope) = 5 个

derivatives.yaml 可微输入：
- query, key, value, query_rope, key_rope（5 个）
- sparse_indices, block_table 等不可微
```

### 25.5 代码与文档不一致时的处理流程

> **核心原则**：文档与源码一致时，结合两者参考，高效推进。不一致时不要猜，直接让用户去确认。

**一致时**：结合文档（了解语义/约束）和源码（了解实现细节/隐藏行为）同步参考，直接推进开发。

**不一致时**：
1. **整理差异清单**：逐条列出"文档说的是 X，代码实际是 Y"，给出文件路径和行号。
2. **立即交给用户确认**：不自行判断以哪边为准，让用户找 ACLNN/PTA 算子开发接口人确认。
3. **拿到结论后继续**：将确认结论记录到方案文档/RFC 中，据此推进 MS 适配。

差异确认输出模板：

```text
⚠️ PTA 代码与文档不一致，需要确认

差异清单：
| # | 内容 | 文档描述 | 代码实际行为 | 文件/行号 |
| - | ---- | -------- | ------------ | --------- |
| 1 | ... | ... | ... | ... |

建议找以下接口人确认以哪边为准：
- ACLNN 算子开发接口人
- PTA 算子开发接口人

请确认后告知结论，我再继续开发。
```

## 26. InferValue 常量折叠（可选优化）

> 来源：`算子流程/.../3. 算子开发.md`。当算子的输入在编译期全部已知时，可通过 InferValue 直接
> 推导出结果值，跳过运行时计算，提升整图执行性能。

### 26.1 两种实现方式
- **Python 回调**（如 concat）：在 `mindspore/python/mindspore/ops/operations/manually_defined/ops_def.py`
  中注册 InferValue 回调函数。
- **C++ 实现**（如 add）：在 `mindspore/ops/infer/ops_frontend_func_impl/` 下实现。
- **C++ 性能优于 Python**，优先使用 C++ 实现。

### 26.2 验证方法
- 增加 InferValue 的 UT 用例（全常量输入场景）。
- 运行测试脚本查看 IR 图，确认常量折叠生效（输出节点变为 ValueNode）。

### 26.3 适用场景
- 算子输入在编译期可确定（如 shape 计算、类型转换等辅助算子）。
- 大多数 ACLNN 计算算子的输入在运行时才确定，**不需要实现 InferValue**。

## 27. 动态 shape 分类与处理策略

> 来源：`算子流程/.../4. 算子关键特性.md`。Infer 推导的快速回退策略另见 §4.2。

### 27.1 动态 shape 三种类型
| 类型 | 含义 | 典型算子 | Infer 策略 |
| --- | --- | --- | --- |
| **InputDynamic** | 输入 shape 编译期未知 | 大多数算子 | 输出对应维度设为 -1（`kShapeDimAny`） |
| **OutputDynamic (Input Value Depend)** | 输出 shape 依赖输入的值 | `DynamicBroadcastTo` | 用 `GetShapeValue()` 取输入值作为输出 shape |
| **OutputDynamic (Compute Depend)** | 输出 shape 需运行时计算 | `NonZero`、`UniqueConsecutive` | 输出分配最大可能 size + 运行后 `SyncOutputShape` |

### 27.2 InputDynamic 处理要点
- 输入 shape 中 -1 维度：输出对应维度也设为 -1。
- 输入动态秩（-2）：输出回退动态秩。
- 关键标量参数 unknown（`HasUnknownValue`）：依赖该参数的输出维度回退 -1。

### 27.3 Input Value Depend 处理要点
- 使用 `GetShapeValue()` 接口提取输入 tensor 中的值作为输出 shape。
- 若输入值 unknown（`kValueAny`），回退动态维。
- 典型场景：`Reshape`（新 shape 作为输入传入）。

### 27.4 Compute Depend 处理要点
- 输出分配最大可能 size（编译期估算上界）。
- 运行后通过 `Sync` + `SyncOutputShape` 更新实际输出 shape。
- 需覆写 `GetUseLessOutputIdx()` 避免 dump/溢出误检。

## 28. ACLNN 调用链分析与子算子盘点（组合场景）

> PTA 的一个 `torch_npu.npu_xxx()` 接口，底层不一定只调用一个 `aclnnXxx` 大算子，
> 常见模式是**多个 ACLNN 小算子串联**（前向/反向均可能）。在这种场景下，MS 必须先盘点
> 所有子算子的接入情况，补齐缺失的子算子，再按相同方式组合。

### 28.1 何时需要做调用链分析

- PTA C++ 实现中出现了**多个 `EXEC_NPU_CMD` / `EXEC_NPU_NO_FORMAT_CHECK_CMD`** 调用。
- PTA C++ 实现中调用了其他 `at_npu::native::` 函数（间接组合）。
- ACLNN 文档/头文件中没有与 PTA 接口一一对应的单个大算子。
- 反向实现不是单个 `aclnnXxxGrad`，而是用多个小算子拼出梯度计算。

### 28.2 调用链提取方法

1. **找到 PTA 前向 C++ 实现**（`ops/opapi/XxxKernelNpuOpApi.cpp`），逐行标注：
   - 每个 `EXEC_NPU_CMD(aclnnYyy, ...)` 或 `OpApiFunc(aclnnYyy, ...)`
   - 中间 tensor 的构造（`at::empty(...)` / `npu_preparation::apply_tensor(...)`）
   - 参数预处理（类型转换、默认值填充、None 处理）
2. **同样分析反向 C++ 实现**（`XxxGradKernelNpuOpApi.cpp` 或 `derivatives.yaml` 指向的函数）。
3. **产出调用链图**（文本即可）：

```text
torch_npu.npu_foo(q, k, v, scale) 前向调用链：
  ① aclnnBarPrepare(q, k) → intermediate_qk     # 预处理
  ② aclnnAttentionScore(intermediate_qk, v, scale) → output  # 主计算
  ③ aclnnSoftmaxLse(output) → softmax_lse        # 辅助输出

torch_npu.npu_foo 反向调用链：
  ① aclnnAttentionScoreGrad(dout, q, k, v, softmax_lse) → (dq, dk, dv)
  （反向为单个大算子，无需拆分）
```

### 28.3 MS 侧覆盖盘点方法

对调用链中的每个 `aclnnYyy`，在 MS 仓库中搜索确认：

| 搜索对象 | 搜索关键词 | 说明 |
| --- | --- | --- |
| YAML 定义 | `aclnnYyy` 或对应 op_name | 确认 op_def 是否存在 |
| PyBoost 实现 | `LAUNCH_ACLNN(aclnnYyy` 或 `aclnnYyyGetWorkspaceSize` | 确认 Pynative 路径 |
| KBK kernel | `MS_ACLNN_KERNEL_FACTORY_REG` + 对应类名 | 确认 Graph 路径 |
| Infer | 对应 `FuncImpl` 类 | 确认推导是否存在 |
| aclnn_config.yaml | 算子名映射 | 确认调度映射 |

### 28.4 盘点结果模板

```text
目标接口：torch_npu.npu_foo → mindspore.ops.foo

ACLNN 调用链盘点：
| # | aclnnXxx | 用途 | MS 状态 | 备注 |
| - | -------- | ---- | ------- | ---- |
| 1 | aclnnBarPrepare | 前向-预处理 | ✅ 已接入 | YAML/Infer/PyBoost/KBK 齐全 |
| 2 | aclnnAttentionScore | 前向-主计算 | ⚠️ 仅有 YAML+Infer | 缺 PyBoost customize 和 KBK |
| 3 | aclnnSoftmaxLse | 前向-辅助输出 | ❌ 未接入 | 需走完整开发流程 |
| 4 | aclnnAttentionScoreGrad | 反向 | ✅ 已接入 | 无需额外工作 |

实施计划：
1. 先补 #3（aclnnSoftmaxLse）：走 YAML→Infer→PyBoost→KBK→UT 全流程
2. 再补 #2 的 PyBoost/KBK
3. 最后在 foo 的 customize 中组合 #1+#2+#3
```

### 28.5 实施顺序原则

- **叶子先、组合后**：先实现所有独立的子算子，再实现组合算子。
- **前向先、反向后**：反向可能复用前向子算子，先确保前向链完整。
- **每个子算子走完整流程**：缺失的子算子按 SKILL.md 的步骤 1-8 逐步实现
  （但通常不需要独立导出/文档，只需 YAML+Infer+PyBoost+KBK+UT）。
- **组合算子在最后实现**：确认所有子算子可用后，再写组合层的 customize。

## 29. 组合实现模式（PyBoost/KBK 多 ACLNN 串联）

> 当目标算子需要串联多个 ACLNN 调用时，PyBoost 和 KBK 的写法与单算子直连有显著差异。

### 29.1 PyBoost 组合模式

```cpp
// foo_ascend_customize.cc — 组合多个 ACLNN 调用
tensor::TensorPtr FooAscendCustomize::Call(
    const tensor::TensorPtr &query,
    const tensor::TensorPtr &key,
    const tensor::TensorPtr &value,
    const std::optional<float> &scale) {

  // ---- 阶段 1：预处理子算子 ----
  auto intermediate_qk = std::make_shared<tensor::Tensor>(
      query->data_type(), infer_qk_shape(query, key));
  // LAUNCH_ACLNN(aclnnBarPrepare, stream, query, key, intermediate_qk);

  // ---- 阶段 2：主计算子算子 ----
  auto output = std::make_shared<tensor::Tensor>(
      query->data_type(), infer_output_shape(intermediate_qk, value));
  auto scale_val = scale.value_or(1.0f);
  // LAUNCH_ACLNN(aclnnAttentionScore, stream, intermediate_qk,
  //              value, scale_val, output);

  // ---- 阶段 3：辅助输出子算子（可选）----
  // auto softmax_lse = ...;
  // LAUNCH_ACLNN(aclnnSoftmaxLse, stream, output, softmax_lse);

  return output;  // 或 MakeTuple(output, softmax_lse)
}
```

**关键注意事项**：
- 中间 tensor 必须**手动分配**（shape 需自行推导或从 Infer 获取）。
- 每个 ACLNN 调用都是两段式（workspace + launch），stream 在同一个上下文中顺序执行。
- 中间 tensor 的生命周期仅限本函数，不会暴露给框架（除非作为输出返回）。

### 29.2 KBK 组合模式

```cpp
// foo_aclnn_kernel.cc — 组合多个 ACLNN 调用
void FooAclnnKernel::GetWorkSpaceInfo(
    const std::vector<KernelTensor *> &inputs,
    const std::vector<KernelTensor *> &outputs) {
  // 每个子算子分别计算 workspace，累加到总 workspace
  // GetWorkspaceForResize(aclnnBarPrepareGetWorkspaceSize, ...);
  // GetWorkspaceForResize(aclnnAttentionScoreGetWorkspaceSize, ...);
}

bool FooAclnnKernel::Launch(
    const std::vector<KernelTensor *> &inputs,
    const std::vector<KernelTensor *> &workspace,
    const std::vector<KernelTensor *> &outputs,
    void *stream_ptr) {
  // 按顺序调用多个 RunOp
  // RunOp(aclnnBarPrepare, stream, ...);
  // RunOp(aclnnAttentionScore, stream, ...);
  return true;
}
```

**关键注意事项**：
- **workspace 管理**：多个子算子的 workspace 需要分别获取并累加，或按最大值取。
  具体策略取决于框架的 workspace 管理接口（以仓库现有组合算子实现为准）。
- **中间 tensor**：KBK 中中间 tensor 可能需要通过 workspace 分配，
  或在 `GetWorkSpaceInfo` 中额外申请。参考仓库中已有的组合算子实现。
- **错误处理**：任一子算子调用失败应立即返回 false，不继续后续调用。

### 29.3 组合场景的 Infer 要点

- 组合算子的 Infer **只需推导最终输出的 shape/type**，不需要推导中间 tensor。
- 中间 tensor 的 shape 推导逻辑放在 PyBoost/KBK 的实现代码中。
- 如果最终输出依赖于中间结果的 shape（级联推导），Infer 中需要**重复这段推导逻辑**
  或者直接按已知输入推导最终输出（跳过中间步骤）。

### 29.4 组合场景的分层验证策略

| 阶段 | 验证内容 | 方法 |
| --- | --- | --- |
| **子算子级** | 每个子算子独立正确 | 子算子各自的 UT/ST |
| **组合级-中间值** | 中间 tensor 与 PTA 对齐 | 在 customize 中临时保存中间 tensor，与 PTA 逐阶段对比 |
| **组合级-最终输出** | 最终输出与 PTA 对齐 | 标准 ST 对齐流程（shape/type/数值） |
| **反向级** | 梯度正确性 | 反向 ST + 数值梯度检查（若适用） |

> **调试技巧**：组合算子出错时，先定位是哪个子算子调用出了问题。
> 在 PyBoost 中可以临时在每个子算子调用后 dump 中间 tensor 进行排查。

---

## §30 Feature 文档（评审与交付必须产物）

> **来源**：实际交付的 Feature 文档（`==符号重载Feature.md`、`CosineEmbeddingLoss Feature.md`、`NsaCompressAttention_Feature_文档.md`、`参考feature.md`）。

### 30.1 什么是 Feature 文档

Feature 文档是算子**评审和转测交付的必须文件**，它将方案设计、接口定义、实现细节、测试计划和验收结果整合到一份标准化文档中。
评审委员会根据此文档判断算子是否可以合入主干。

### 30.2 Feature 文档标准章节

| 序号 | 章节 | 填写时机 | 说明 |
| ---- | ---- | -------- | ---- |
| 1 | 背景描述 | Pre-B | 算子来源、动机、MindSpore 为何需要 |
| 2 | 标杆与接口 | Pre-B | 标杆接口（PTA/Torch）、MindSpore 接口（functional/nn/tensor） |
| 3 | 任务清单 | Pre-B 初始化 → 开发中更新状态 | **标准 13 大类表格**（见下方 §30.3） |
| 4 | 功能与接口说明 | Pre-B | 计算公式、接口签名、参数说明 |
| 5 | YAML 定义 | Step 1 后 | `op_def` YAML 内容 |
| 6 | 约束与类型 | Pre-B | 设备、dtype、shape 约束、空 Tensor 策略 |
| 7 | 执行模式与适配 | Step 4/5 后 | PyBoost / KBK 实现说明 |
| 8 | 与 PTA 的差异与对齐 | Pre-B 初始化 → 开发中补齐 | 功能/精度/API 语义差异 |
| 9 | 动态 Shape/Rank 支持 | Step 3 后 | 动态维/动态秩推导策略 |
| 10 | 异常与校验 | Step 3/4 后 | 推导期/运行期校验 |
| 11 | 反向（BPROP） | Step 6 后 | BPROP 注册、反向接口、梯度处理 |
| 12 | 测试方案 | Step 8 后 | UT/ST/TEST_OP 覆盖说明 |
| 13 | 代码与文件改动说明 | 开发完成后 | 所有新增/修改文件的完整路径 |
| 14 | 验收报告 | 转测前 | 四张自测表：资料验证 + 功能验证 + 性能验证 + 安全编码（见 §30.4） |

### 30.3 任务清单标准 13 大类

Feature 文档中的"任务清单"是标准化表格，每个算子**必须逐项填写**：

| 序号 | 任务项 | 子项 |
| ---- | ------ | ---- |
| 1 | 接口基本功能 | Primitive / functional / nn / tensor |
| 2 | 后端及数据类型支持 | Ascend / GPU / CPU |
| 3 | 支持 vmap | — |
| 4 | 支持动态 Shape | 动态 Shape / 动态 Rank |
| 5 | 支持反向 | bprop 函数 / 复数支持 |
| 6 | 补齐资料 | API 映射 / 接口中英文资料 |
| 7 | 性能优化 | CPU / GPU / Ascend |
| 8 | 功能 | 空 Tensor / inf-nan / 0~8 维 / 其他功能点 |
| 9 | 门禁用例补齐 | UT / ST / TEST_OP |
| 10 | 支持 MS Adapter | — |
| 11 | 自动并行切分 | — |
| 12 | 混合精度（AMP） | — |
| 13 | 安全与异常 | 异常用例与报错规范 |

每项需标注：`新增` / `修改` / `无变更` / `不涉及`，并在备注中简要说明。

### 30.4 验收报告四张表

#### 资料验证表（17 项）

涵盖：接口列表、UT/ST 用例、中英文文档、接口描述、公式、参数描述、输入描述、输出描述、
输出尺寸与输入关系、Raises、平台填写、格式检查、样例提供、样例打印结果、样例可执行、API 沙盘。

#### 功能验证表（27 项）

涵盖：默认参数、空 Tensor、inf/nan、dtype 对齐、取值范围、维度覆盖 0D-8D、dtype 全覆盖、
隐式类型转换、广播、输入约束、正向精度、反向支持、反向单算子实现、异常报错信息、报错白名单、
functional 用例、动态 shape/rank、退避关闭验证、测试仓回归、bf16、bprop 按需求导、
输出 shape 计算依赖、非连续输入、PTA 0 偏差、存量接口影响、AMP、多 Tensor dtype 不一致。

#### 性能验证表（4 项）

涵盖：广播场景性能、反向显存优化（SetUnusedInputs）、多规格性能（≥3 种）、显存持平 PTA。

#### 安全编码检视表（12 项）

涵盖：指针判空、先用后校、越界、除零、内存泄露、异常路径释放、nothrow、安全函数库、
类型转换溢出、冗余代码、敏感信息、弱随机数。

### 30.5 Feature 文档生成流程

```
Pre-B 阶段：
  1. 从模板 templates/feature-document.md 复制一份
  2. 填写 §1-§4（背景/标杆/任务清单/接口说明）和 §6（约束）和 §8（PTA 差异初始化）
  3. 提交给评审委员会做方案评审

开发过程中：
  4. 每完成一个 Workflow Step，回填对应章节
     - Step 1 → §5（YAML）
     - Step 3 → §9（动态Shape）, §10（异常）
     - Step 4/5 → §7（执行模式）
     - Step 6 → §11（反向）
     - Step 8 → §12（测试方案）

转测交付前：
  5. 补齐 §13（代码改动）
  6. 填写 §14 验收报告的四张自测表（资料/功能/性能/安全编码）
  7. 更新 §3 任务清单中每项的最终状态
  8. 完整 Feature 文档随代码 PR 一起提交
```

### 30.6 不同类型算子的 Feature 文档差异

| 场景 | 差异 |
| ---- | ---- |
| **单 ACLNN 算子** | 标准流程，§7 中 PyBoost/KBK 各调用一个 ACLNN 接口 |
| **组合算子（小算子拼接）** | §4 需描述 ACLNN 调用链，§7 描述多 ACLNN 组合，§12 需分层验证 |
| **符号重载（如 ==）** | §4 需描述 MultitypeFuncGraph 适配，§3 中 functional/tensor 列为"修改" |
| **纯 Python 组合（无 Primitive）** | §3 中 Primitive 列为"不涉及"，§7 只描述 functional 层实现 |

### 30.7 模板位置

- 模板文件：`templates/feature-document.md`
- 参考实例：用户提供的已有 Feature 文档（建议在开发前找到相似算子的 Feature 作参考）
