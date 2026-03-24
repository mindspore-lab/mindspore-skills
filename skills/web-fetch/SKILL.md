---
name: web-fetch
description: This skill should be used when the user asks to "fetch a webpage", "scrape a website", "download a page", "save a webpage", "capture a page as PDF", "extract webpage content", or needs to fetch pages requiring login authentication. Provides guidance for using Playwright-based web fetching with Chrome persistent context support.
version: 1.2.0
---

# Web Fetch Skill

Fetch and save webpages in multiple formats (MHTML, PDF, PNG, HTML) using Playwright and Chrome persistent context. This skill is particularly useful for capturing pages that require login authentication, as it can reuse existing browser sessions and cookies.

## Purpose

The web-fetch skill enables fetching webpages with support for:
- Authentication and login-protected pages
- Multiple output formats (MHTML, PDF, PNG, HTML)
- Batch fetching of multiple URLs
- Persistent cookie management across requests
- Headless operation without requiring a display server

## When to Use This Skill

Invoke this skill when:
- Fetching a single webpage and saving it locally
- Capturing pages that require login authentication
- Batch downloading multiple webpages
- Converting webpages to different formats (PDF, PNG, etc.)
- Automating webpage archival or documentation

## Quick Start

### Installation

Install dependencies before first use:

```bash
pip install -r requirements.txt
playwright install chromium
```

### Basic Usage

Fetch a single webpage and save as MHTML:

```bash
python3 scripts/web-fetch.py "https://example.com" --format mhtml
```

Save in multiple formats:

```bash
python3 scripts/web-fetch.py "https://example.com" --format mhtml pdf png
```

Fetch pages requiring login (automatically opens visible browser for manual login):

```bash
python3 scripts/web-fetch.py "https://example.com/protected" --format mhtml
```

### Batch Fetching

Create a URL list file:

```bash
cat > urls.txt << EOF
https://example.com/page1
https://example.com/page2
https://example.com/page3
EOF
```

Fetch all URLs:

```bash
python3 scripts/web-fetch-batch.py --urls urls.txt --format mhtml pdf
```

## Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--format` | Output format (mhtml/pdf/png/html) | mhtml |
| `--output` | Output directory | Current directory |
| `--wait` | Minimum render wait time (seconds) | 10 |
| `--max-wait` | Maximum render wait time (seconds) | 30 |
| `--timeout` | Page load timeout (seconds) | 90 |
| `--login-timeout` | Manual login max wait time (seconds) | 300 |
| `--manual-popup-close` | Prompt for manual popup closure | Disabled |
| `--force` | Force overwrite existing files | Disabled |
| `--concurrency` | Batch fetch concurrency (batch only) | 1 |

## Output Formats

| Format | Extension | Use Case |
|--------|-----------|----------|
| **MHTML** | `.mhtml` | Complete single-file webpage (recommended) |
| **PDF** | `.pdf` | Printable document, sharing |
| **PNG** | `.png` | Quick preview, long screenshot |
| **HTML** | `.html` | Source code, development debugging |

## Authentication & Cookies

The skill uses Chrome persistent context to maintain login sessions:

1. First fetch of a protected page automatically detects login requirement
2. If login page detected, opens visible browser with login prompt
3. After manual login, script continues automatically
4. Cookies saved to `<user-data>/cookies/` (per domain)
5. Subsequent same-domain fetches reuse cookies, usually no re-login needed

Session metadata saved to `<user-data>/session/` includes popup rules and login cache info.

## File Naming

Output files follow the pattern: `{domain}-{path}-{query}.{format}`

Examples:
- `e-gitee-com-mind_spore-issues-table-q-issue-i1cevz.mhtml`
- `example-com-page.pdf`

Existing files are skipped automatically. Use `--force` to overwrite.

## Scripts

Two main scripts are provided:

- **`scripts/web-fetch.py`** - Single webpage fetching
- **`scripts/web-fetch-batch.py`** - Batch URL fetching
- **`scripts/url_filename.py`** - URL to filename conversion utility

## Common Issues & Solutions

### Empty Page Content

Increase wait time or page load timeout:

```bash
python3 scripts/web-fetch.py "https://example.com" --wait 15 --max-wait 45
```

### MHTML Save Fails

Ensure Playwright >= 1.40 is installed. Try HTML format as fallback:

```bash
python3 scripts/web-fetch.py "https://example.com" --format html
```

### Login Issues in Batch Mode

Use isolated profiles and manual popup handling:

```bash
python3 scripts/web-fetch-batch.py --urls urls.txt \
  --manual-popup-close --manual-popup-timeout 300 \
  --concurrency 2 --wait 12 --max-wait 45
```

## Development Experience (2026-03)

Key learnings from production use:

1. **Concurrency & Profile Isolation** - System Chrome has ProcessSingleton lock; concurrent tasks must use different profiles
2. **Login State Complexity** - Sessions depend on cookies + localStorage + indexedDB + service workers combined
3. **Optimal Architecture** - Reuse host Chrome state, execute with isolated profiles for stability + concurrency + maintainability
4. **Popup Handling** - Default to "no-click" strategy (DOM hiding/removal); click-based cleanup only as optional fallback
5. **Manual Fallback** - Both login and popups must support manual completion in visible mode, then auto-continue
6. **Wait Strategy** - Higher concurrency requires more conservative page stability detection to avoid premature saves

## Additional Resources

For detailed usage patterns, troubleshooting, and advanced configuration, consult:

- **`references/usage-guide.md`** - Comprehensive usage guide with examples
- **`references/troubleshooting.md`** - Detailed troubleshooting and edge cases
- **`references/architecture.md`** - Technical architecture and design decisions

Working examples available in `examples/`:

- **`examples/single-fetch.sh`** - Single webpage fetch example
- **`examples/batch-fetch.sh`** - Batch fetching example
- **`examples/authenticated-fetch.sh`** - Login-protected page example

## License

MIT License
