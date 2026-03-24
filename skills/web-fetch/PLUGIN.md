# Web Fetch Plugin for Claude Code

🕸️ Fetch and save webpages using Playwright with Chrome persistent context support.

## Features

- ✅ Fetch webpages requiring login authentication
- ✅ Multiple output formats: MHTML, PDF, PNG, HTML
- ✅ Batch processing with configurable concurrency
- ✅ Persistent cookie management per domain
- ✅ Automatic login detection and manual fallback
- ✅ Popup detection and cleanup
- ✅ Headless operation (no display required)

## Installation

### Option 1: Install from Claude Code Plugin Directory

1. Open Claude Code
2. Go to Plugins → Install Plugin
3. Search for "web-fetch"
4. Click Install

### Option 2: Manual Installation

1. Clone or download this repository
2. Place in your Claude Code plugins directory:
   - macOS: `~/.claude/plugins/`
   - Linux: `~/.claude/plugins/`
   - Windows: `%APPDATA%\.claude\plugins\`

3. Install dependencies:
```bash
cd web-fetch
pip install -r requirements.txt
playwright install chromium
```

## Quick Start

### Fetch a Single Webpage

```bash
python3 scripts/web-fetch.py "https://example.com" --format mhtml
```

### Fetch Multiple Formats

```bash
python3 scripts/web-fetch.py "https://example.com" \
  --format mhtml pdf png \
  --output ./output
```

### Fetch Authenticated Page

```bash
python3 scripts/web-fetch.py "https://example.com/protected" \
  --format mhtml \
  --login-timeout 600
```

A browser window will open for manual login if needed.

### Batch Fetch

```bash
# Create URL list
cat > urls.txt << EOF
https://example.com/page1
https://example.com/page2
https://example.com/page3
EOF

# Fetch all URLs
python3 scripts/web-fetch-batch.py --urls urls.txt --format mhtml pdf
```

## Using with Claude Code

Once installed, the web-fetch skill is available to Claude Code. Ask Claude to:

- "Fetch this webpage and save as PDF"
- "Download multiple pages from this list"
- "Capture a login-protected page"
- "Save this page in multiple formats"

Claude will use the web-fetch skill to help you accomplish these tasks.

## Documentation

- **SKILL.md** - Skill overview and quick reference
- **references/usage-guide.md** - Comprehensive usage guide
- **references/troubleshooting.md** - Common issues and solutions
- **references/architecture.md** - Technical architecture details
- **examples/** - Working example scripts

## Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--format` | Output format (mhtml/pdf/png/html) | mhtml |
| `--output` | Output directory | Current directory |
| `--wait` | Minimum render wait (seconds) | 10 |
| `--max-wait` | Maximum render wait (seconds) | 30 |
| `--timeout` | Page load timeout (seconds) | 90 |
| `--login-timeout` | Manual login timeout (seconds) | 300 |
| `--concurrency` | Batch concurrency (batch only) | 1 |
| `--force` | Force overwrite existing files | Disabled |

## Output Formats

| Format | Use Case |
|--------|----------|
| **MHTML** | Complete single-file webpage (recommended) |
| **PDF** | Printable document, sharing |
| **PNG** | Quick preview, long screenshot |
| **HTML** | Source code, development |

## Authentication

The plugin uses Chrome persistent context to maintain login sessions:

1. First fetch of protected page opens browser for manual login
2. Cookies automatically saved per domain
3. Subsequent fetches reuse saved cookies
4. No re-login needed for same domain (until cookies expire)

## System Requirements

- Python 3.7+
- pip package manager
- Chrome or Chromium browser (optional, can use built-in)
- ~200-500MB memory per browser instance

## Troubleshooting

### Empty Page Content

Increase wait times:
```bash
python3 scripts/web-fetch.py "https://example.com" \
  --wait 15 --max-wait 60
```

### MHTML Save Fails

Update Playwright:
```bash
pip install --upgrade playwright
playwright install chromium
```

### Batch Hangs

Reduce concurrency:
```bash
python3 scripts/web-fetch-batch.py --urls urls.txt --concurrency 1
```

For more troubleshooting, see `references/troubleshooting.md`.

## Development Experience

Key learnings from production use (2026-03):

1. **Concurrency & Profile Isolation** - Use different profiles for concurrent tasks
2. **Login State Complexity** - Sessions depend on cookies + localStorage + indexedDB
3. **Optimal Architecture** - Reuse host Chrome state, execute with isolated profiles
4. **Popup Handling** - Default to DOM-based removal, not click-based
5. **Manual Fallback** - Support visible mode for complex authentication
6. **Wait Strategy** - Higher concurrency requires more conservative stability detection

## Examples

Working examples in `examples/`:

- **single-fetch.sh** - Fetch single webpage
- **batch-fetch.sh** - Batch fetch multiple URLs
- **authenticated-fetch.sh** - Fetch login-protected page

Run examples:
```bash
bash examples/single-fetch.sh
bash examples/batch-fetch.sh
bash examples/authenticated-fetch.sh
```

## Performance Tips

- Reduce wait times for simple pages
- Increase wait times for dynamic content
- Use concurrency 1-3 for stability
- Monitor memory during batch processing
- Process large batches in smaller chunks

## License

MIT License - See LICENSE file for details

## Support

For issues, questions, or suggestions:

1. Check `references/troubleshooting.md`
2. Review `references/architecture.md` for technical details
3. Consult Playwright documentation: https://playwright.dev/python/

## Contributing

Contributions welcome! Please:

1. Test changes thoroughly
2. Update documentation
3. Follow existing code style
4. Submit pull requests with clear descriptions

---

**Happy Fetching! 🕸️**
