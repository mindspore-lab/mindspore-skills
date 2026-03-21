# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

**web-fetch** 是一个网页抓取工具 skill，使用 Playwright + Chrome 持久化上下文实现对需要登录认证的网页的抓取。支持多种输出格式（MHTML、PDF、PNG、HTML）和批量抓取。

## 核心架构

### 主要组件

- **scripts/web-fetch.py** - 单个网页抓取的主脚本，包含 `WebFetcher` 类
- **scripts/web-fetch-batch.py** - 批量抓取脚本，包含 `BatchFetcher` 类
- **scripts/url_filename.py** - URL 转换为文件名的工具函数

### 关键设计决策

1. **Chrome 持久化上下文**：使用系统 Chrome 的用户数据目录（macOS: `~/Library/Application Support/Google/Chrome`，Linux: `~/.config/google-chrome`），支持复用已登录会话

2. **登录流程**：
   - 首次访问受保护页面时自动检测登录状态
   - 若检测到登录页，自动切换到可见浏览器模式
   - 用户手工完成登录后脚本自动继续
   - Cookies 按域名持久化保存到 `<user-data>/cookies/`

3. **并发模式**：
   - 不共享同一 Chrome profile（避免 ProcessSingleton 锁冲突）
   - 批量抓取时使用隔离的 profile 执行，复用 host Chrome 的 cookies/session
   - 建议并发数不超过 5

4. **弹窗处理**：
   - 默认采用"无点击"策略（隐藏/移除 DOM）
   - 支持手工兜底参数 `--manual-popup-close` 用于生产环境

## 常用命令

### 安装依赖

```bash
pip install -r requirements.txt
playwright install chromium
```

### 单个网页抓取

```bash
# 基础用法 - 保存为 MHTML
python3 scripts/web-fetch.py "https://example.com" --format mhtml

# 多种格式
python3 scripts/web-fetch.py "https://example.com" --format mhtml pdf png

# 需要登录的页面（自动弹出可见浏览器）
python3 scripts/web-fetch.py "https://example.com/protected" --format mhtml --login-timeout 600

# 指定输出目录
python3 scripts/web-fetch.py "https://example.com" --output /path/to/output --format mhtml
```

### 批量抓取

```bash
# 创建 URL 列表文件
cat > urls.txt << EOF
https://example.com/page1
https://example.com/page2
https://example.com/page3
EOF

# 批量抓取
python3 scripts/web-fetch-batch.py --urls urls.txt --format mhtml pdf

# 指定并发数
python3 scripts/web-fetch-batch.py --urls urls.txt --format mhtml --concurrency 3
```

## 关键参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--format` | 保存格式 (mhtml/pdf/png/html) | mhtml |
| `--output` | 输出目录 | 当前目录 |
| `--wait` | 最短等待渲染时间（秒） | 10 |
| `--max-wait` | 最长等待渲染时间（秒） | 30 |
| `--timeout` | 页面加载超时（秒） | 90 |
| `--login-timeout` | 手工登录最大等待时间（秒） | 300 |
| `--manual-popup-close` | 检测残留浮窗后提示手工关闭 | 关闭 |
| `--force` | 强制覆盖已存在文件 | 关闭 |
| `--concurrency` | 批量抓取并发数 | 1 |

## 开发经验沉淀

这些是在 2026-03 开发过程中总结的关键经验：

1. **并发与 Profile 隔离**：系统 Chrome 有 ProcessSingleton 锁，并发场景必须使用不同 profile，否则会直接失败

2. **登录态的复杂性**：登录状态不仅依赖 cookies，还涉及 localStorage、indexedDB、service worker 等多个存储机制的组合

3. **最优架构**：优先复用 host Chrome 的登录状态，执行时使用隔离 profile，这是"稳定 + 并发 + 可维护"的平衡点

4. **弹窗处理策略**：默认应采用"无点击"方式（隐藏/移除 DOM），点击式清理仅作为可选手段，避免误触发新窗口

5. **手工兜底通道**：登录与浮窗都必须支持可见模式下手工完成，再自动继续流水线

6. **等待策略**：并发越高，页面稳定判定越要保守，否则容易过早保存不完整内容

## 文件命名规则

输出文件名格式：`{domain}-{path}-{query}.{format}`

示例：
- `e-gitee-com-mind_spore-issues-table-q-issue-i1cevz.mhtml`
- `example-com-page.pdf`

若输出目录中已存在同名文件，脚本自动跳过该格式。使用 `--force` 参数强制覆盖。

## 故障排除

### 无法访问需要登录的页面

1. 直接运行脚本，等待自动弹出的可见浏览器窗口
2. 在页面内手工登录，脚本会自动继续
3. 如超时可提高 `--login-timeout`，如 `--login-timeout 600`

### 并发下出现登录/弹窗不一致

1. 指定正确 profile：`--profile-directory "Default"` 或 `--profile-directory "Profile 1"`
2. 开启手工浮窗兜底：`--manual-popup-close --manual-popup-timeout 300`
3. 适当降低并发（如 `--concurrency 2`）并提高等待（`--wait 12 --max-wait 45`）

### 页面内容为空

1. 增加 `--wait` 参数延长等待时间
2. 检查页面是否需要交互操作
3. 使用 `--timeout` 增加超时时间

### MHTML 保存失败

1. 确保 Playwright 版本 >= 1.40
2. 检查 CDP 会话是否可用
3. 尝试使用 HTML 格式替代

## 代码风格

遵循全局 CLAUDE.md 规范：
- 代码注释使用英文
- 文档使用中文
- 行长度限制 120 字符
- 使用空格缩进，不使用 Tab
- 使用 LF 作为行末换行符
- 文件末尾有且仅有一行空行
