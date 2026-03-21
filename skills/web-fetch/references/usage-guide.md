# Usage Guide

## Installation

### Prerequisites

- Python 3.7+
- pip package manager
- Chrome or Chromium browser (optional, can use built-in Chromium)

### Setup Steps

1. Install Python dependencies:

```bash
pip install -r requirements.txt
```

2. Install Chromium browser for Playwright:

```bash
playwright install chromium
```

3. Verify installation:

```bash
python3 scripts/web-fetch.py --help
```

## Single Webpage Fetching

### Basic Fetch

Fetch a webpage and save as MHTML (default format):

```bash
python3 scripts/web-fetch.py "https://example.com"
```

### Multiple Formats

Save the same page in multiple formats simultaneously:

```bash
python3 scripts/web-fetch.py "https://example.com" --format mhtml pdf png html
```

### Custom Output Directory

Specify where to save files:

```bash
python3 scripts/web-fetch.py "https://example.com" \
  --output /path/to/output \
  --format mhtml
```

### Adjust Wait Times

For pages with dynamic content, increase wait times:

```bash
python3 scripts/web-fetch.py "https://example.com" \
  --wait 15 \
  --max-wait 60 \
  --format mhtml
```

Parameters:
- `--wait`: Minimum time to wait for page rendering (default: 10s)
- `--max-wait`: Maximum time to wait for page stability (default: 30s)
- `--timeout`: Page load timeout (default: 90s)

## Authenticated Page Fetching

### First-Time Login

When fetching a page requiring authentication:

```bash
python3 scripts/web-fetch.py "https://example.com/protected" --format mhtml
```

The script will:
1. Detect login requirement
2. Open a visible browser window
3. Display login prompt at page top
4. Wait for manual login completion
5. Continue fetching automatically
6. Save cookies for future use

### Reusing Saved Credentials

Subsequent fetches from the same domain reuse saved cookies:

```bash
# Second fetch - no login needed
python3 scripts/web-fetch.py "https://example.com/protected/page2" --format mhtml
```

### Extending Login Timeout

For pages with slow login or 2FA:

```bash
python3 scripts/web-fetch.py "https://example.com/protected" \
  --login-timeout 600 \
  --format mhtml
```

## Batch Fetching

### Create URL List

Create a text file with one URL per line:

```bash
cat > urls.txt << EOF
https://example.com/page1
https://example.com/page2
https://example.com/page3
EOF
```

### Batch Fetch

Fetch all URLs in the list:

```bash
python3 scripts/web-fetch-batch.py --urls urls.txt --format mhtml pdf
```

### Concurrent Fetching

Increase concurrency for faster batch processing:

```bash
python3 scripts/web-fetch-batch.py \
  --urls urls.txt \
  --format mhtml \
  --concurrency 3
```

**Note:** Do not exceed 5 concurrent tasks; higher concurrency may cause stability issues.

### Batch with Authentication

For batch fetching authenticated pages:

```bash
python3 scripts/web-fetch-batch.py \
  --urls urls.txt \
  --format mhtml \
  --concurrency 2 \
  --wait 12 \
  --max-wait 45 \
  --manual-popup-close \
  --manual-popup-timeout 300
```

## Chrome Profile Management

### Using System Chrome

By default, the script uses system Chrome with the Default profile:

```bash
python3 scripts/web-fetch.py "https://example.com" --format mhtml
```

### Specify Different Profile

Use a different Chrome profile:

```bash
python3 scripts/web-fetch.py "https://example.com" \
  --profile-directory "Profile 1" \
  --format mhtml
```

### Custom Chrome Path

Specify a custom Chrome executable:

```bash
python3 scripts/web-fetch.py "https://example.com" \
  --chrome-path "/path/to/chrome" \
  --format mhtml
```

### Force Built-in Chromium

Use Playwright's built-in Chromium instead of system Chrome:

```bash
python3 scripts/web-fetch.py "https://example.com" \
  --no-system-chrome \
  --format mhtml
```

## Cookie Management

### Cookie Storage

Cookies are automatically saved per domain:

```
<user-data>/cookies/{domain}.json
```

Example: `<user-data>/cookies/example.com.json`

### Session Metadata

Session information (popup rules, login cache) saved to:

```
<user-data>/session/
```

### Manual Cookie Inspection

View saved cookies for a domain:

```bash
cat ~/.config/google-chrome/cookies/example.com.json
```

(Path varies by OS; see Chrome user data directory section)

## Output File Management

### File Naming Convention

Output files follow pattern: `{domain}-{path}-{query}.{format}`

Examples:
- `example-com-page.mhtml`
- `github-com-user-repo-issues.pdf`
- `api-example-com-v1-users-q-page-1.png`

### Skip Existing Files

By default, existing files are skipped:

```bash
python3 scripts/web-fetch.py "https://example.com" --format mhtml
# If file exists, it's skipped
```

### Force Overwrite

Overwrite existing files:

```bash
python3 scripts/web-fetch.py "https://example.com" \
  --format mhtml \
  --force
```

## Platform-Specific Notes

### macOS

Default Chrome user data directory:
```
~/Library/Application Support/Google/Chrome
```

### Linux

Default Chrome user data directory:
```
~/.config/google-chrome
```

### Windows

Default Chrome user data directory:
```
%APPDATA%\Google\Chrome\User Data
```

## Performance Tuning

### For Slow Networks

Increase timeouts:

```bash
python3 scripts/web-fetch.py "https://example.com" \
  --timeout 180 \
  --wait 20 \
  --max-wait 60
```

### For Dynamic Content

Increase wait times to allow JavaScript execution:

```bash
python3 scripts/web-fetch.py "https://example.com" \
  --wait 15 \
  --max-wait 45
```

### For Batch Processing

Balance concurrency and stability:

```bash
python3 scripts/web-fetch-batch.py \
  --urls urls.txt \
  --concurrency 3 \
  --wait 12 \
  --max-wait 40
```

## Memory Considerations

Each browser instance uses approximately 200-500MB of memory. For batch processing:

- Concurrency 1: ~300MB
- Concurrency 3: ~900MB
- Concurrency 5: ~1.5GB

Monitor system resources and adjust concurrency accordingly.

## Headless Mode

The script runs in headless mode by default (no display needed). For debugging, use visible mode:

```bash
# Note: Visible mode requires X Server on Linux or display on macOS
python3 scripts/web-fetch.py "https://example.com" --format mhtml
# Login prompt will appear in visible browser window
```
