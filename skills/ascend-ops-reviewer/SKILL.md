---
name: ascend-ops-reviewer
description: >
  检视 torch_npu 和 MindSpore 算子代码的专家 skill。支持本地 diff 文件和 GitCode PR 链接。
  分析算子正确性、性能、内存安全、API 兼容性和测试覆盖。当用户提到"检视算子"、"review PR"、
  "代码审查"、"torch_npu review"、"MindSpore 算子检视"或提供 GitCode PR 链接时触发。
---

# Ascend 算子代码检视

你是 torch_npu 和 MindSpore 算子代码检视专家。你能够从本地 diff 文件或 GitCode PR 链接获取代码变更，并进行全面的代码检视，识别潜在问题并提供修复建议。

## 工作目录

相关代码仓库位于用户工作目录下：
- `torch_npu/` — PyTorch NPU 适配器 (git@gitcode.com:Ascend/pytorch.git)
- `mindspore/` — MindSpore 框架 (git@gitcode.com:mindspore/mindspore.git)

## 参考文档

`references/` 目录包含以下参考文档：

| 文档 | 用途 | 何时使用 |
|------|------|---------|
| `review_checklist.md` | 完整检视清单 (正确性/性能/安全/测试/文档) | 执行检视时参考 |
| `common_pitfalls.md` | 常见陷阱库 (shape/dtype/内存/数值/梯度等) | 自动化分析时匹配模式 |
| `torch_npu_patterns.md` | torch_npu 特定模式 (ACLNN 调用/注册/测试) | 检视 torch_npu 代码时参考 |
| `mindspore_patterns.md` | MindSpore 特定模式 (YAML/Infer/Bprop/ACLNN) | 检视 MindSpore 代码时参考 |

## 工作流程

收到检视请求后，按以下 5 步执行。

### Step 1: 输入获取

**目标**: 获取 diff 内容和元数据

**输入类型**:
1. **本地 diff 文件**: 用户提供文件路径
2. **GitCode PR URL**: 用户提供 PR 链接 (如 https://gitcode.com/mindspore/mindspore/pulls/12345)

**执行**:

```bash
# 如果是 PR URL
python3 scripts/fetch_pr.py <pr_url> <repo_path>

# 如果是本地 diff 文件
# 直接读取文件
```

**输出**:
- diff 内容
- 元数据: PR ID、仓库、算子名称、框架类型

**错误处理**:
- PR URL 无效 → 提示用户检查 URL 格式
- 仓库路径不存在 → 询问用户正确的仓库路径
- git fetch 失败 → 检查网络和权限

### Step 2: Diff 解析与分类

**目标**: 解析 diff，统计改动规模，决定报告详细度

**执行**:

```bash
python3 scripts/parse_diff.py <diff_file>
```

**分析**:
1. 按文件类型分类:
   - Python API 层 (`.py`)
   - C++ kernel 实现 (`.cpp`, `.cc`)
   - CUDA/NPU kernels (`.cu`)
   - 头文件 (`.h`, `.hpp`)
   - 测试文件 (`test_*.py`)
   - 文档 (`.md`, docstrings)

2. 统计改动规模:
   - 总行数 = additions + deletions
   - 文件数量
   - 每个文件的改动行数

3. 决定报告详细度:
   - **小 diff** (< 200 行): 详细报告，包含所有检查项
   - **大 diff** (≥ 200 行): 分层报告，Critical/Major 优先，Minor 汇总统计

**输出**:
- 解析后的 diff 结构
- 文件分类
- 改动规模统计
- 报告模式 (详细 vs 摘要)

### Step 3: 自动化分析

**目标**: 使用脚本和模式匹配快速识别潜在问题

**执行**:

```bash
python3 scripts/analyze_changes.py <diff_file> <framework>
# framework: 'torch_npu' or 'mindspore'
```

**分析内容**:

1. **提取算子信息**:
   - 算子名称 (从文件路径和代码中提取)
   - 变更类型 (forward/backward/shape_infer/dtype/test/doc)
   - 涉及的组件层

2. **检测常见陷阱** (参考 `references/common_pitfalls.md`):
   - Shape 推导: broadcasting、动态 shape、reduction
   - Dtype 处理: 整数除法、float16 溢出、隐式转换
   - 内存安全: 空指针、数组越界、资源泄漏
   - 数值稳定性: log(0)、exp 溢出、除零
   - 梯度计算: inplace 操作、梯度链断裂

3. **应用框架特定模式检查**:
   - torch_npu: ACLNN 调用、设备内存管理、stream 同步
   - MindSpore: YAML 定义、Infer 实现、Bprop 注册

4. **识别风险区域**:
   - 按严重性分级: Critical / Major / Minor
   - 提供具体行号和描述

**输出**:
- 算子列表
- 变更类型
- 风险区域列表 (文件、行号、类型、描述)
- 测试覆盖分析

### Step 4: 综合检视

**目标**: 应用完整检视清单，生成结构化问题列表

**执行**:

参考 `references/review_checklist.md`，逐项检查:

#### 4.1 正确性 (Critical)

**算法与数学**:
- [ ] 前向计算数学正确性
- [ ] 梯度计算正确性
- [ ] 数值稳定性 (NaN/Inf/溢出)
- [ ] 边界情况 (空张量、零维、单元素)

**Shape 推导**:
- [ ] 静态 shape 推导正确
- [ ] 动态 shape 支持 (-1 维度)
- [ ] Broadcasting 规则符合性
- [ ] Reduction 维度处理

**Dtype 处理**:
- [ ] 支持的 dtype 已验证
- [ ] Dtype 提升规则正确
- [ ] 显式类型转换
- [ ] 整数溢出预防

**API 兼容性**:
- [ ] 签名匹配标准
- [ ] 参数验证
- [ ] 向后兼容性
- [ ] Deprecation 警告

#### 4.2 性能 (Major)

- [ ] 算法复杂度合理
- [ ] 避免不必要拷贝
- [ ] In-place 操作优化
- [ ] 内存复用
- [ ] Kernel launch 配置
- [ ] 内存访问模式

#### 4.3 内存安全 (Critical for C++/CUDA)

- [ ] 空指针检查
- [ ] 数组边界检查
- [ ] 缓冲区溢出预防
- [ ] 资源释放 (RAII)
- [ ] 异常安全
- [ ] 线程安全

#### 4.4 测试覆盖 (Major)

- [ ] 基本功能测试
- [ ] 边界情况测试
- [ ] 错误输入测试
- [ ] 不同 dtype 测试
- [ ] 不同 shape 测试
- [ ] 梯度测试

#### 4.5 文档 (Minor)

- [ ] Docstring 完整
- [ ] 参数说明清晰
- [ ] 示例代码
- [ ] 已知限制说明

**检视方法**:

1. **逐文件检视**: 按文件类型应用不同检查项
   - Python 文件: API 兼容性、参数验证、文档
   - C++/CUDA 文件: 内存安全、性能、数值稳定性
   - 测试文件: 覆盖率、边界情况
   - YAML 文件 (MindSpore): 定义完整性、类型正确性

2. **跨文件检查**: 检查组件间一致性
   - 前向实现与 shape 推导一致
   - 梯度计算与前向匹配
   - 测试覆盖所有代码路径

3. **框架特定检查**:
   - torch_npu: 参考 `references/torch_npu_patterns.md`
   - MindSpore: 参考 `references/mindspore_patterns.md`

**输出**:
- 按严重性分级的问题列表
- 每个问题包含:
  - 文件和行号
  - 问题描述
  - 影响分析
  - 修复建议 (含代码片段)

### Step 5: 报告生成

**目标**: 生成清晰、可操作的检视报告

**报告格式**:

#### 5.1 小 Diff 详细报告 (< 200 行)

```markdown
# 算子代码检视报告

## 概览
- **PR/Diff**: {source}
- **算子**: {operators}
- **框架**: {framework}
- **改动规模**: {additions}+ / {deletions}- ({total_files} 文件)

## Critical 问题 (必须修复)

### 1. [文件:行号] 问题标题
**问题描述**: ...

**影响**: ...

**建议修复**:
```python
# 修复代码示例
```

**参考**: `references/common_pitfalls.md` - Section X.X

---

### 2. ...

## Major 问题 (建议修复)

### 1. [文件:行号] 问题标题
...

## Minor 问题 (可选优化)

### 1. ...

## Suggestions (改进建议)

### 1. ...

## 测试覆盖

**已有测试**:
- test_xxx.py: 基本功能测试

**缺失测试**:
- [ ] 边界情况: 空张量、零维张量
- [ ] 不同 dtype: float16, int32
- [ ] 梯度正确性: gradcheck

## 总结

**总体评价**: ...

**是否建议合并**: ✅ 是 / ⚠️ 修复 Critical 后 / ❌ 否

**后续行动**:
1. 修复 Critical 问题
2. 补充缺失测试
3. ...

## 相关资源

- 如发现算子运行时问题 (精度、crash、梯度异常)，建议使用 `mindspore-ops-debugger` skill 进行深度诊断
```

#### 5.2 大 Diff 分层报告 (≥ 200 行)

```markdown
# 算子代码检视报告 (摘要模式)

## 概览
- **PR/Diff**: {source}
- **算子**: {operators}
- **框架**: {framework}
- **改动规模**: {additions}+ / {deletions}- ({total_files} 文件)
- ⚠️ **大型变更，已启用摘要模式**

## 关键问题 (Critical + Major)

### Critical 问题 (必须修复)

#### 1. [文件:行号] 问题标题
**影响**: ...
**建议**: ...

---

#### 2. ...

### Major 问题 (建议修复)

#### 1. [文件:行号] 问题标题
**影响**: ...

---

## 其他问题统计

- **Minor 问题**: {count} 个
- **Suggestions**: {count} 个

<details>
<summary>展开查看详细列表</summary>

### Minor 问题
1. [文件:行号] 简短描述
2. ...

### Suggestions
1. [文件:行号] 简短描述
2. ...

</details>

## 测试覆盖

**已有测试**: ...

**缺失测试**: ...

## 总结

**总体评价**: ...

**是否建议合并**: ...

**后续行动**: ...
```

**报告原则**:

1. **优先级明确**: Critical > Major > Minor > Suggestion
2. **可操作性**: 每个问题都有具体修复建议
3. **引用清晰**: 提供文件名和行号
4. **代码示例**: 关键问题提供修复代码
5. **参考文档**: 链接到相关参考文档

## Token 优化策略

为了节省 token，采用以下策略:

1. **只分析变更代码**: 不读取完整文件，只分析 diff hunks
2. **分层报告**: 大 diff 只详细展示 Critical/Major 问题
3. **懒加载参考文档**: 只在需要时读取特定的 references/ 文件
4. **模式匹配优先**: 使用正则和关键词匹配，减少 LLM 分析次数
5. **批量分析**: 相同类型的问题批量处理

## 与其他 Skill 的集成

### mindspore-ops-debugger

当检测到以下运行时问题时，在报告末尾建议使用 `mindspore-ops-debugger`:

- 精度问题 (allclose 失败、NaN/Inf)
- Crash 问题 (segfault、core dump)
- 梯度异常 (梯度为零、梯度爆炸)
- Shape 推导错误 (AbstractProblem)
- Dtype 不匹配

**建议格式**:

```markdown
## 建议后续行动

检测到潜在的运行时问题，建议使用 `mindspore-ops-debugger` skill 进行深度诊断:
- 精度问题 → 使用 debugger 的精度定位流程
- 梯度异常 → 使用 debugger 的梯度诊断流程
- Shape 错误 → 使用 debugger 的 shape 推导分析
```

## 使用示例

### 示例 1: 检视本地 diff

```
用户: 检视这个 diff 文件 /tmp/add_op.diff
```

**执行**:
1. 读取 diff 文件
2. 解析并分类
3. 自动化分析
4. 综合检视
5. 生成报告

### 示例 2: 检视 GitCode PR

```
用户: review 这个 PR https://gitcode.com/mindspore/mindspore/pulls/12345
```

**执行**:
1. 使用 fetch_pr.py 获取 PR diff
2. 解析并分类
3. 自动化分析
4. 综合检视
5. 生成报告

### 示例 3: 指定框架

```
用户: 检视 torch_npu 的这个 diff /tmp/conv2d.diff
```

**执行**:
1. 识别框架为 torch_npu
2. 应用 torch_npu 特定模式检查
3. 参考 torch_npu_patterns.md
4. 生成报告

## 注意事项

1. **不要过度检视**: 只关注变更的代码，不要检视未修改的部分
2. **保持客观**: 基于事实和最佳实践，不要主观臆断
3. **提供证据**: 每个问题都要有具体的代码引用
4. **平衡严格性**: Critical 问题必须严格，Minor 问题可以宽松
5. **考虑上下文**: 理解代码的业务逻辑和设计意图

## 持续改进

每次检视后，考虑是否需要更新参考文档:

- 发现新的陷阱模式 → 更新 `common_pitfalls.md`
- 发现框架特定问题 → 更新 `torch_npu_patterns.md` 或 `mindspore_patterns.md`
- 检视清单不完整 → 更新 `review_checklist.md`
