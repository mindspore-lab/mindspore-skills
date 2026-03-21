# 安装到 Claude Code

本文档说明如何将 web-fetch 项目安装为 Claude Code skill。

## 项目结构

web-fetch 已改造为符合 Claude Code plugin 标准的 skill：

```
web-fetch/
├── SKILL.md                          # Skill 定义（必需）
├── plugin.json                       # Plugin 配置
├── PLUGIN.md                         # Plugin 使用说明
├── CLAUDE.md                         # 项目开发指南
├── LICENSE                           # MIT 许可证
├── requirements.txt                  # Python 依赖
├── scripts/
│   ├── web-fetch.py                 # 单个网页抓取脚本
│   ├── web-fetch-batch.py           # 批量抓取脚本
│   └── url_filename.py              # URL 转文件名工具
├── examples/
│   ├── single-fetch.sh              # 单个抓取示例
│   ├── batch-fetch.sh               # 批量抓取示例
│   └── authenticated-fetch.sh       # 认证抓取示例
└── references/
    ├── usage-guide.md               # 详细使用指南
    ├── troubleshooting.md           # 故障排除指南
    └── architecture.md              # 架构设计文档
```

## 安装方式

### 方式 1：作为独立 Skill 安装

如果你已有 Claude Code plugin 项目，可以将 web-fetch 作为 skill 添加到你的 plugin 中：

```bash
# 在你的 plugin 项目中
mkdir -p skills/web-fetch
cp -r /path/to/web-fetch/* skills/web-fetch/
```

然后在 plugin 的 `plugin.json` 中添加：

```json
{
  "skills": [
    {
      "name": "web-fetch",
      "path": "skills/web-fetch/SKILL.md"
    }
  ]
}
```

### 方式 2：作为独立 Plugin 安装

将整个 web-fetch 目录作为 Claude Code plugin 安装：

```bash
# 复制到 Claude Code plugins 目录
cp -r /path/to/web-fetch ~/.claude/plugins/web-fetch

# 或在 macOS 上
cp -r /path/to/web-fetch ~/Library/Application\ Support/Claude\ Code/plugins/web-fetch
```

然后在 Claude Code 中重新加载 plugins。

### 方式 3：从 Git 仓库安装

```bash
cd ~/.claude/plugins
git clone https://github.com/your-org/web-fetch.git
```

## 初始化

### 1. 安装 Python 依赖

```bash
cd web-fetch
pip install -r requirements.txt
```

### 2. 安装 Chromium

```bash
playwright install chromium
```

### 3. 验证安装

```bash
python3 scripts/web-fetch.py --help
```

## 使用

### 在 Claude Code 中使用

安装后，可以在 Claude Code 中直接使用 web-fetch skill：

```
用户: 帮我抓取 https://example.com 并保存为 PDF

Claude: 我会使用 web-fetch skill 来帮你完成这个任务。
```

Claude 会自动调用 web-fetch skill 中的脚本。

### 命令行使用

也可以直接使用命令行：

```bash
# 单个网页
python3 scripts/web-fetch.py "https://example.com" --format mhtml

# 批量抓取
python3 scripts/web-fetch-batch.py --urls urls.txt --format mhtml pdf

# 认证页面
python3 scripts/web-fetch.py "https://example.com/protected" --format mhtml
```

## 配置

### Chrome 用户数据目录

默认使用系统 Chrome 用户目录：

- **macOS**: `~/Library/Application Support/Google/Chrome`
- **Linux**: `~/.config/google-chrome`
- **Windows**: `%APPDATA%\Google\Chrome\User Data`

可通过 `--user-data` 参数覆盖。

### Cookies 存储

Cookies 按域名保存到：

```
<user-data>/cookies/{domain}.json
```

### 会话元数据

会话信息保存到：

```
<user-data>/session/
```

## 文档

- **SKILL.md** - Skill 概述和快速参考
- **PLUGIN.md** - Plugin 使用说明
- **CLAUDE.md** - 项目开发指南
- **references/usage-guide.md** - 详细使用指南
- **references/troubleshooting.md** - 故障排除
- **references/architecture.md** - 架构设计

## 常见问题

### Q: 如何更新 web-fetch？

```bash
cd ~/.claude/plugins/web-fetch
git pull origin main
pip install -r requirements.txt --upgrade
```

### Q: 如何卸载 web-fetch？

```bash
rm -rf ~/.claude/plugins/web-fetch
```

### Q: 支持哪些输出格式？

支持 4 种格式：
- **MHTML** - 单文件完整网页（推荐）
- **PDF** - 可打印文档
- **PNG** - 长截图预览
- **HTML** - 网页源码

### Q: 如何处理需要登录的页面？

首次抓取时会自动检测登录需求，打开可见浏览器让你手工登录。之后的同域名抓取会自动复用 cookies。

### Q: 支持并发抓取吗？

支持。使用 `web-fetch-batch.py` 进行批量抓取，可通过 `--concurrency` 参数控制并发数（建议不超过 5）。

### Q: 内存占用多少？

每个浏览器实例约占用 200-500MB 内存。批量抓取时需要根据并发数预留足够内存。

## 故障排除

### 页面内容为空

增加等待时间：

```bash
python3 scripts/web-fetch.py "https://example.com" \
  --wait 15 --max-wait 60
```

### MHTML 保存失败

更新 Playwright：

```bash
pip install --upgrade playwright
playwright install chromium
```

### 批量抓取卡住

降低并发数：

```bash
python3 scripts/web-fetch-batch.py --urls urls.txt --concurrency 1
```

更多问题见 `references/troubleshooting.md`。

## 开发

### 修改 Skill

编辑 `SKILL.md` 来修改 skill 的描述和触发条件。

### 修改脚本

编辑 `scripts/` 中的 Python 脚本来修改功能。

### 添加文档

在 `references/` 中添加新的参考文档。

### 测试

```bash
# 测试单个抓取
python3 scripts/web-fetch.py "https://example.com" --format mhtml

# 测试批量抓取
echo "https://example.com" > test_urls.txt
python3 scripts/web-fetch-batch.py --urls test_urls.txt --format mhtml
```

## 许可证

MIT License - 详见 LICENSE 文件

## 支持

- 查看 `references/troubleshooting.md` 获取常见问题解决方案
- 查看 `references/architecture.md` 了解技术细节
- 参考 Playwright 文档：https://playwright.dev/python/

---

**安装完成！现在可以在 Claude Code 中使用 web-fetch skill 了。🕸️**
