# Workflow 8: 测试

## 目标

完成 C++ UT + Python ST（+ 可选 Python UT），确保功能、精度、动态 shape 全覆盖。

## 输入

- **算子实现**：YAML / Infer / PyBoost / KBK / BPROP
- **PTA 对标实现**：用于 ST 数值对齐

## 输出（三类测试，逐项确认）

> **⚠️ 以下三类测试文件是 Step 8 的必须产出，每一类都要明确标注状态。**
> 不允许只写其中一类就认为"测试步骤完成"。

| 类型 | 文件位置 | 必须程度 | 状态标注 |
| --- | --- | --- | --- |
| **C++ UT** | `tests/ut/cpp/ops/test_ops_{op_name}.cc` | `[MUST]` 必须新建 | ✅已写 / ❌未写（说明原因） |
| **Python ST** | `tests/st/ops/ascend/test_{op_name}.py` | `[MUST]` 新建或确认已有 | ✅已写 / ✅已有（标明路径） / ❌未写 |
| **Python UT** | `tests/ut/python/ops/test_{op_name}.py` | `[SHOULD]` 推荐 | ✅已写 / ⏭跳过（说明原因） |

### 关于"已存在"的判断

搜到已有测试文件时，**必须确认它是否覆盖新算子路径**：
- 已有测试调用的是新接口（如 `mint.acos` → `acos_ext`）→ 确认覆盖，标注路径
- 已有测试只调用旧接口（如 `ops.acos` → 旧 `ACos`）→ **不算覆盖**，必须新建
- 已有测试覆盖不完整（如只测了前向没测反向）→ 需要补充

---

## 执行步骤

### Step 1：C++ UT（`reference.md` §8.2）—— 必须新建

> agent 可以完全自主完成，不需要设备。**没有理由跳过。**

典型构造：
- 标量：`ShapeVector{}` + `CreateScalar<T>(value)`
- tuple：`ShapeArray{{}}` + `ValuePtrList{...}`
- None：`kMetaTypeNone` + `kNone`
- unknown：`kValueAny`

参照同类算子的已有 C++ UT 文件确认测试宏和参数结构。

### Step 2：Python ST（`reference.md` §8.3）—— 必须新建或确认已有

> **这是最容易遗漏的一类。** agent 必须生成完整的 ST 测试文件（即使无法在设备上运行）。

- 优先"形状/类型"再比数值
- 严格对齐时 `atol/rtol=0`（按算子特性）
- 避免引入额外算子导致误差累积
- bfloat16 比较前升精到 float32
- **必须覆盖**：Pynative + Graph 双模式、前向 + 反向（如有）、动态 shape

### Step 3：Python UT（`reference.md` §8.1）—— 推荐

- 推导正确性：shape/type、动态/边界
- 错误路径：非法参数、None 语义覆盖
- 固定随机种子：`np.random.default_rng(seed)`

### Step 4：精度零偏差验证（`reference.md` §17.1，按需）

- 固定随机种子，保存输出为 `.npy`
- `md5sum` 对比 MS/PTA 输出哈希

### Step 5：显存对齐验证（`reference.md` §17.2，按需）

- MS：`mindspore.runtime.max_memory_allocated()`
- PTA：`torch_npu.npu.max_memory_allocated()`
- 在相同阶段统计

### Step 6：组合场景分层验证（`reference.md` §29.4）

| 阶段 | 验证内容 |
| --- | --- |
| 子算子级 | 每个子算子独立 UT/ST |
| 组合级-中间值 | 临时 dump 中间 tensor 与 PTA 对比 |
| 组合级-最终输出 | 标准 ST 对齐 |
| 反向级 | 反向 ST + 数值梯度检查 |

---

## 需要用户配合的环节

| 环节 | 原因 | 向用户说明 |
| --- | --- | --- |
| Ascend ST 执行 | 需要 Ascend 设备 | "ST 测试需要在 Ascend 设备上运行，请在设备上执行以下命令并回传结果" |
| 精度零偏差验证 | 需同时跑 MS 和 PTA | "请在相同环境下分别运行 MS 和 PTA 脚本，回传输出 .npy 文件" |
| 性能/显存对比 | 需要真实设备 | "请在 Ascend 设备上运行性能脚本并回传耗时和显存数据" |
| 稳定性 100 次验证 | 耗时较长 | "请在设备上执行 100 次循环脚本并回传结果" |

> agent 可以**生成测试脚本和验证命令**，但若无法直接访问 Ascend 设备，必须将脚本和运行指令交给用户执行，**等用户回传结果后再判断是否通过**。

---

## 🔒 Step 8 完成前强制检查（不可跳过）

**在标记 Step 8 为完成之前，必须逐项确认以下清单：**

```text
测试产出检查清单：

C++ UT 文件：
  - 文件路径：tests/ut/cpp/ops/test_ops_{op_name}.cc
  - 状态：✅已新建 / ❌未写（原因：___）

Python ST 文件：
  - 文件路径：tests/st/ops/ascend/test_{op_name}.py
  - 状态：✅已新建 / ✅已有且确认覆盖新路径（路径：___）/ ❌未写（原因：___）
  - 若"已有"：确认调用的是新接口而非旧接口？ 是/否

Python UT 文件（推荐）：
  - 文件路径：tests/ut/python/ops/test_{op_name}.py
  - 状态：✅已新建 / ⏭跳过（原因：___）
```

> 如果 C++ UT 或 Python ST 的状态为 ❌，**必须说明原因并暂停等用户确认后再继续**。
> 不允许静默跳过。

## 成功标准

- [ ] **C++ UT 文件已产出**（Infer 推导覆盖 unknown/None/动态shape）
- [ ] **Python ST 文件已产出或确认已有覆盖**（Pynative + Graph、前向 + 反向、动态 shape）
- [ ] Python UT 已产出或有理由跳过
- [ ] 稳定性验证：100 次运行无偶现失败（需用户在设备上验证）
- [ ] 覆盖场景：动态 shape / 静态 shape / 非连续 tensor / 空 tensor / 特殊值
- [ ] （精度零偏差）hash 对比通过（按需）
- [ ] （组合场景）分层验证通过（按需）

---

## 下一步

测试完成后，进入 **[Workflow 9: 文档](./09-docs.md)**
