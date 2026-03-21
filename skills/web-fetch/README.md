# Web Fetch

🕸️ 网页抓取工具 - 使用 Playwright + Chrome 持久化上下文抓取需要登录的网页

## 特性

- ✅ 支持需要登录认证的网页
- ✅ 检测到登录页时自动切换可见浏览器，提示手工登录
- ✅ 自动保存/复用 cookies（按域名持久化）
- ✅ 多种输出格式：MHTML、PDF、PNG、HTML
- ✅ 批量抓取支持
- ✅ 保留完整网页样式和资源
- ✅ 无头模式，无需图形界面

## 安装

```bash
# 安装依赖
pip install -r requirements.txt

# 安装 Chromium
playwright install chromium
```

## 快速开始

### 抓取单个网页

```bash
# 保存为 MHTML
python3 scripts/web-fetch.py "https://example.com" --format mhtml

# 保存为多种格式
python3 scripts/web-fetch.py "https://example.com" --format mhtml pdf png

# 需要登录时，自动弹出可见浏览器并等待手工登录（默认行为）
python3 scripts/web-fetch.py "https://example.com/protected" --format mhtml --login-timeout 600
```

### 批量抓取

```bash
# 创建 URL 列表
cat > urls.txt << EOF
https://example.com/page1
https://example.com/page2
https://example.com/page3
EOF

# 批量抓取
python3 scripts/web-fetch-batch.py --urls urls.txt --format mhtml pdf
```

## 登录与 cookies

- 首次抓取受保护页面时，脚本会检测登录状态
- 如检测到登录页，会自动打开可见浏览器并在网页顶部显示登录提示
- 你手工登录后，脚本自动继续抓取
- cookies 会保存到 `<user-data>/cookies/`（按域名文件）
- 会话档案会保存到 `<user-data>/session/`（弹窗规则与登录元信息）
- 后续同域名抓取会自动加载 cookies，通常无需再次登录

## 开发经验沉淀

- 并发场景不要共享同一 Chrome profile（会触发 `ProcessSingleton` 锁）
- 登录态复用不能只看 cookies，还要考虑 localStorage/indexedDB
- 弹窗清理默认采用“无点击”策略，避免误触发新窗口
- 生产运行建议开启手工兜底参数：`--manual-popup-close`
- 完整经验总结见 `SKILL.md` 的“本次开发经验（2026-03）”

## 文档

详细文档请查看 [SKILL.md](SKILL.md)

## 许可证

MIT License
