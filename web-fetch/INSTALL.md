# Web Fetch Skill - 安装完成

## ✅ 安装成功

**web-fetch** skill 已成功安装到 OpenClaw！

### 📁 安装位置

```
~/.openclaw/skills/web-fetch/
├── SKILL.md           # 技能文档
├── README.md          # 使用说明
├── requirements.txt   # Python 依赖
└── scripts/
    ├── web-fetch.py        # 主脚本
    └── web-fetch-batch.py  # 批量抓取脚本
```

### 🚀 使用方法

#### 1. 抓取单个网页

```bash
# 使用工作区路径
cd /home/lch/.openclaw/workspace
python3 skills/web-fetch/scripts/web-fetch.py "https://example.com" --format mhtml

# 或直接从 skill 目录
cd /home/lch/.openclaw/skills/web-fetch
python3 scripts/web-fetch.py "https://example.com" --format mhtml
```

#### 2. 批量抓取

```bash
# 创建 URL 列表文件
echo "https://example.com/page1" > urls.txt
echo "https://example.com/page2" >> urls.txt

# 批量抓取
python3 scripts/web-fetch-batch.py --urls urls.txt --format mhtml pdf
```

### 📊 支持格式

| 格式 | 说明 | 推荐场景 |
|------|------|----------|
| **MHTML** | 单文件完整网页 | ⭐ 完整保存 |
| **PDF** | 可打印文档 | ⭐ 分享/打印 |
| **PNG** | 长截图 | 快速预览 |
| **HTML** | 网页源码 | 开发调试 |

### 🔐 认证支持

本 Skill 使用 Chrome 持久化上下文，可以访问需要登录的网页：

1. 首次使用前，在 Chrome 浏览器中登录目标网站
2. 浏览器会话数据保存在：`~/.openclaw/browser/openclaw/user-data`
3. 后续抓取自动使用已登录的会话

### 📝 示例

```bash
# 抓取 Gitee Issue（需要登录）
python3 scripts/web-fetch.py \
  "https://e.gitee.com/mind_spore/issues/table?issue=I1CEVZ" \
  --format mhtml pdf \
  --output ./gitee-issues \
  --wait 10

# 输出:
# ✓ MHTML: ./gitee-issues/e-gitee-com-mind_spore-issues-I1CEVZ-20260308_173000.mhtml (4.0 MB)
# ✓ PDF: ./gitee-issues/e-gitee-com-mind_spore-issues-I1CEVZ-20260308_173000.pdf (619 KB)
```

### 🛠️ 依赖

```bash
# 如果还没安装，运行：
pip install playwright
playwright install chromium
```

---

**安装时间**: 2026-03-08 17:33  
**版本**: 1.0.0  
**作者**: OpenClaw Assistant
