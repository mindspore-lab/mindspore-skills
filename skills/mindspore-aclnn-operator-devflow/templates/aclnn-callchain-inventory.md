# ACLNN 调用链盘点表 - {OpName}

> **文档用途**：记录 PTA 接口的 ACLNN 调用链及 MS 侧覆盖状况。
> **文档状态**：本地文件，不提交 Git。
> **生成时间**：{generation_time}

---

## 一、目标接口

| 属性 | 值 |
| ---- | -- |
| **PTA 接口** | `torch_npu.npu_{op_name}` |
| **MS 目标接口** | `mindspore.ops.{op_name}` |

---

## 二、前向调用链

```text
torch_npu.npu_{op_name}(args...) 前向：
  ① aclnn{Sub1}(input1, input2) → intermediate_1    # 说明
  ② aclnn{Sub2}(intermediate_1, input3) → output     # 说明
```

---

## 三、反向调用链

```text
npu_{op_name} 反向：
  ① aclnn{SubGrad1}(dout, ...) → (d_input1, d_input2)   # 说明
```

---

## 四、MS 覆盖盘点

| # | aclnnXxx | 用途 | YAML | Infer | PyBoost | KBK | UT | 状态 | 备注 |
| - | -------- | ---- | ---- | ----- | ------- | --- | -- | ---- | ---- |
| 1 | aclnn{Sub1} | 前向-预处理 | ✅/❌ | ✅/❌ | ✅/❌ | ✅/❌ | ✅/❌ | ✅已接入 / ⚠️部分 / ❌缺失 | |
| 2 | aclnn{Sub2} | 前向-主计算 | | | | | | | |
| 3 | aclnn{SubGrad1} | 反向 | | | | | | | |

---

## 五、实施计划

### 优先级排序（叶子先、组合后；前向先、反向后）

| 序号 | 算子 | 依赖 | 工作量估计 | 状态 |
| ---- | ---- | ---- | ---------- | ---- |
| 1 | aclnn{Sub1}（缺失） | 无依赖 | YAML+Infer+PyBoost+KBK+UT | ⬜ 待开始 |
| 2 | aclnn{Sub2}（补 PyBoost） | 依赖 #1 | PyBoost+KBK | ⬜ 待开始 |
| 3 | {OpName} 组合实现 | 依赖 #1, #2 | customize+ST | ⬜ 待开始 |

### 注意事项

- 缺失的子算子按 SKILL.md 步骤 1-8 逐步实现
- 子算子通常不需要独立导出/文档，只需 YAML+Infer+PyBoost+KBK+UT
- 组合算子在所有子算子就绪后再实现

---

## 六、验证策略

| 阶段 | 验证内容 | 方法 |
| ---- | -------- | ---- |
| 子算子级 | 每个子算子独立正确 | 子算子各自 UT/ST |
| 组合级-中间值 | 中间 tensor 与 PTA 对齐 | 临时 dump 中间 tensor 对比 |
| 组合级-最终输出 | 最终输出与 PTA 对齐 | 标准 ST 对齐流程 |
| 反向级 | 梯度正确性 | 反向 ST + 数值梯度检查 |
