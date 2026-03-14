# Web Fetch Skill

**版本**: 1.2.0  
**作者**: OpenClaw Assistant  
**描述**: 使用 Playwright + Chrome 持久化上下文抓取需要登录的网页，支持保存为 MHTML、PDF、HTML、PNG 等多种格式。

---

## 📦 依赖

```bash
pip install playwright
playwright install chromium
```

## 🔧 配置

### Chrome 用户数据目录

默认使用系统 Chrome 用户目录：

- macOS: `~/Library/Application Support/Google/Chrome`
- Linux: `~/.config/google-chrome`

可通过 `--user-data` 覆盖，`--profile-directory` 指定具体 profile（如 `Default` / `Profile 1`）。

## 🚀 使用方法

### 基础用法 - 保存为 MHTML

```bash
python3 scripts/web-fetch.py "https://example.com/page" --format mhtml
```

### 保存为多种格式

```bash
python3 scripts/web-fetch.py "https://example.com/page" --format mhtml pdf png html
```

### 指定输出目录

```bash
python3 scripts/web-fetch.py "https://example.com/page" --output /path/to/output
```

### 使用示例 - 抓取 Gitee Issue

```bash
python3 scripts/web-fetch.py "https://e.gitee.com/mind_spore/issues/table?issue=I1CEVZ" --format mhtml pdf
```

## 📝 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `url` | 目标网页 URL | 必需 |
| `--format` | 保存格式 (mhtml/pdf/png/html) | mhtml |
| `--output` | 输出目录 | 当前目录 |
| `--wait` | 最短等待渲染时间（秒） | 10 |
| `--max-wait` | 最长等待渲染时间（秒） | 30 |
| `--timeout` | 页面加载超时（秒） | 90 |
| `--user-data` | Chrome 用户数据目录 | 系统 Chrome 默认目录 |
| `--profile-directory` | Chrome Profile 名称 | Default |
| `--chrome-path` | Chrome 可执行路径（可选） | 自动探测 |
| `--no-system-chrome` | 强制使用内置 Chromium | 关闭 |
| `--login-timeout` | 手工登录最大等待时间（秒） | 300 |
| `--manual-popup-close` | 检测残留浮窗后提示手工关闭 | 关闭 |
| `--manual-popup-timeout` | 手工关闭浮窗最大等待时间（秒） | 180 |
| `--popup-click-cleanup` | 启用点击式弹窗清理（有误触风险） | 关闭 |
| `--force` | 强制覆盖已存在文件 | 关闭 |

## 📁 输出文件命名

```
{output_dir}/{domain}-{path}-{query}.{format}
```

示例：
- `e-gitee-com-mind_spore-issues-table-q-issue-i1cevz.mhtml`
- `example-com-page.pdf`

如果输出目录中已存在同名文件，脚本会自动跳过该格式，避免重复下载。  
如需强制重新下载并覆盖，请加 `--force` 参数。

## 🔐 认证支持

本 Skill 使用 Chrome 持久化上下文，可以继承已登录的浏览器会话：

1. 抓取前会先尝试加载本地 cookies（按域名）
2. 如果检测到登录页，会自动切换到可见浏览器
3. 页面顶部会显示登录提示，请手工完成登录（含验证码/2FA）
4. 登录成功后脚本自动继续，并保存最新 cookies
5. 后续同域名抓取可直接复用 cookies，通常无需再次登录

cookies 默认保存目录（按域名）：

```bash
<user-data>/cookies/
```

会话档案（包含弹窗规则、登录缓存元信息）默认保存在：

```bash
<user-data>/session/
```

## 📊 支持格式

| 格式 | 扩展名 | 说明 | 推荐场景 |
|------|--------|------|----------|
| **MHTML** | `.mhtml` | 单文件完整网页 | ⭐ 完整保存 |
| **PDF** | `.pdf` | 可打印文档 | ⭐ 分享/打印 |
| **PNG** | `.png` | 长截图 | 快速预览 |
| **HTML** | `.html` | 网页源码 | 开发调试 |

## 🛠️ 脚本文件

- `scripts/web-fetch.py` - 主脚本
- `scripts/web-fetch-batch.py` - 批量抓取脚本

## 📋 使用示例

### 示例 1: 抓取单个网页

```bash
python3 scripts/web-fetch.py "https://e.gitee.com/mind_spore/issues/table?issue=I1CEVZ" --format mhtml
```

### 示例 2: 批量抓取

```bash
python3 scripts/web-fetch-batch.py --urls urls.txt --format mhtml pdf
```

`urls.txt` 格式：
```
https://e.gitee.com/mind_spore/issues/table?issue=I1CEVZ
https://e.gitee.com/mind_spore/issues/table?issue=I1CJ4I
https://e.gitee.com/mind_spore/issues/table?issue=I19O76
```

### 示例 3: Python API

```python
from web_fetch import WebFetcher

fetcher = WebFetcher(user_data_dir="/path/to/chrome/data")
await fetcher.save("https://example.com", formats=["mhtml", "pdf"])
```

## ⚠️ 注意事项

1. **首次使用**：首次访问受保护页面时，可能需要手工登录一次
2. **X Server**：无头模式不需要图形界面
3. **内存占用**：每个浏览器实例约占用 200-500MB 内存
4. **并发限制**：建议不要同时运行超过 5 个抓取任务
5. **并发与浏览器选择**：并发模式下建议使用内置 Chromium 执行抓取，同时复用 host Chrome 的 cookies/session

## 🔍 故障排除

### 问题：无法访问需要登录的页面

**解决**：
1. 直接运行脚本，等待自动弹出的可见浏览器窗口
2. 在页面内手工登录，脚本会自动继续
3. 如超时可提高 `--login-timeout`，如 `--login-timeout 600`

### 问题：并发下仍出现登录/弹窗不一致

**解决**：
1. 指定正确 profile，例如 `--profile-directory "Default"` 或 `--profile-directory "Profile 1"`
2. 开启手工浮窗兜底：`--manual-popup-close --manual-popup-timeout 300`
3. 适当降低并发（如 `--concurrency 2`）并提高等待（`--wait 12 --max-wait 45`）

## 🧠 本次开发经验（2026-03）

1. **并发不要共享同一 profile**：系统 Chrome 有 ProcessSingleton 锁，并发抢占会直接失败。
2. **登录态不只有 cookies**：很多站依赖 cookies + localStorage + indexedDB + service worker 的组合态。
3. **优先复用 host 状态，执行用隔离 profile**：这是“稳定 + 并发 + 可维护”三者平衡点。
4. **弹窗处理默认应“无点击”**：先隐藏/移除 DOM，点击式清理仅作为可选手段，避免误触发新窗口。
5. **必须有手工兜底通道**：登录与浮窗都要支持可见模式下手工完成，再自动继续流水线。
6. **等待策略要并发感知**：并发越高，页面稳定判定越要保守，否则容易过早保存。

### 问题：页面内容为空

**解决**：
1. 增加 `--wait` 参数延长等待时间
2. 检查页面是否需要交互操作
3. 使用 `--timeout` 增加超时时间

### 问题：MHTML 保存失败

**解决**：
1. 确保 Playwright 版本 >= 1.40
2. 检查 CDP 会话是否可用
3. 尝试使用 HTML 格式替代

## 📄 许可证

MIT License

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

---

**Happy Fetching! 🕸️**
