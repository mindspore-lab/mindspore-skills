# Ascend Ops Reviewer Skill

专门用于检视 torch_npu 和 MindSpore 算子代码的 Claude Code skill。

## 功能特性

- ✅ 支持本地 diff 文件检视
- ✅ 支持 GitCode PR 链接自动获取
- ✅ 全面的代码检视（正确性、性能、内存安全、API 兼容性、测试覆盖）
- ✅ 自动检测常见陷阱和风险区域
- ✅ 根据 diff 大小自动调整报告详细度
- ✅ 框架特定模式检查（torch_npu / MindSpore）

## 使用方法

### 触发关键词

当用户提到以下关键词时，skill 会自动触发：
- "检视算子"
- "review PR"
- "代码审查"
- "torch_npu review"
- "MindSpore 算子检视"
- 提供 GitCode PR 链接

### 使用示例

#### 1. 检视本地 diff 文件

```
检视这个 diff 文件 /tmp/add_op.diff
```

#### 2. 检视 GitCode PR

```
review 这个 PR https://gitcode.com/mindspore/mindspore/pulls/12345
```

#### 3. 指定框架

```
检视 torch_npu 的这个 diff /tmp/conv2d.diff
```

## 工作流程

1. **输入获取** - 从本地文件或 GitCode PR 获取 diff
2. **Diff 解析** - 解析 diff，分类文件，统计改动规模
3. **自动化分析** - 提取算子信息，检测常见陷阱，识别风险区域
4. **综合检视** - 应用完整检视清单，生成结构化问题列表
5. **报告生成** - 生成详细或分层报告

## 检视内容

### 正确性 (Critical)
- 算法与数学正确性
- Shape 推导
- Dtype 处理
- API 兼容性

### 性能 (Major)
- 计算效率
- NPU/CUDA 优化
- 内存管理

### 内存安全 (Critical for C++/CUDA)
- 指针安全
- 资源管理
- 并发安全

### 测试覆盖 (Major)
- 单元测试
- 梯度测试
- 性能测试

### 文档 (Minor)
- Docstring 完整性
- 参数说明
- 示例代码

## 报告格式

### 小 Diff (< 200 行)
详细报告，包含所有检查项和修复建议

### 大 Diff (≥ 200 行)
分层报告，Critical/Major 问题优先，Minor 问题汇总统计

## 目录结构

```
ascend-ops-reviewer/
├── SKILL.md                    # Skill 定义和工作流
├── CLAUDE.md                   # 开发指南
├── README.md                   # 本文件
├── scripts/
│   ├── fetch_pr.py            # 从 GitCode 获取 PR diff
│   ├── parse_diff.py          # 解析 diff 文件
│   └── analyze_changes.py     # 分析代码变更
├── references/
│   ├── review_checklist.md    # 检视清单
│   ├── common_pitfalls.md     # 常见陷阱
│   ├── torch_npu_patterns.md  # torch_npu 特定模式
│   └── mindspore_patterns.md  # MindSpore 特定模式
└── .claude/
    └── settings.local.json    # 权限配置
```

## 脚本使用

### fetch_pr.py

从 GitCode PR 获取 diff：

```bash
python3 scripts/fetch_pr.py <pr_url> <repo_path>

# 示例
python3 scripts/fetch_pr.py \
  https://gitcode.com/mindspore/mindspore/pulls/12345 \
  /Users/claw/work/ms_debug/mindspore
```

### parse_diff.py

解析 diff 文件：

```bash
python3 scripts/parse_diff.py <diff_file>

# 示例
python3 scripts/parse_diff.py /tmp/test.diff
```

### analyze_changes.py

分析代码变更：

```bash
python3 scripts/analyze_changes.py <diff_file> <framework>

# 示例
python3 scripts/analyze_changes.py /tmp/test.diff mindspore
```

## 与其他 Skill 的集成

当检测到运行时问题时，会建议使用 `mindspore-ops-debugger` skill 进行深度诊断：
- 精度问题
- Crash 问题
- 梯度异常
- Shape 推导错误
- Dtype 不匹配

## 开发

参考 `CLAUDE.md` 了解：
- Skill 架构
- 脚本实现细节
- 参考文档维护
- 测试方法
- 成功标准

## 依赖

- Python 3.6+
- git (用于 PR 获取)
- 标准库：re, subprocess, json, pathlib

## 许可

与父项目相同
