# Troubleshooting Guide

## Common Issues and Solutions

### Empty Page Content

**Symptom:** Saved file is empty or contains minimal content.

**Causes:**
- Page requires JavaScript execution time
- Dynamic content loads after initial render
- Page requires user interaction

**Solutions:**

1. Increase wait times:
```bash
python3 scripts/web-fetch.py "https://example.com" \
  --wait 15 \
  --max-wait 60
```

2. Increase page load timeout:
```bash
python3 scripts/web-fetch.py "https://example.com" \
  --timeout 120
```

3. Check if page requires interaction (scroll, click, etc.)

### MHTML Save Fails

**Symptom:** Error message about MHTML format or CDP session.

**Causes:**
- Playwright version too old
- Chrome DevTools Protocol (CDP) session unavailable
- Browser crash during save

**Solutions:**

1. Update Playwright:
```bash
pip install --upgrade playwright
playwright install chromium
```

2. Verify Playwright version >= 1.40:
```bash
python3 -c "import playwright; print(playwright.__version__)"
```

3. Try alternative format:
```bash
python3 scripts/web-fetch.py "https://example.com" --format html
```

4. Use built-in Chromium:
```bash
python3 scripts/web-fetch.py "https://example.com" \
  --no-system-chrome \
  --format mhtml
```

### Login Page Not Detected

**Symptom:** Script saves login page instead of protected content.

**Causes:**
- Login detection heuristics don't match page structure
- Page uses custom authentication flow
- Session already expired

**Solutions:**

1. Manually clear cookies and retry:
```bash
rm ~/.config/google-chrome/cookies/example.com.json
python3 scripts/web-fetch.py "https://example.com/protected" --format mhtml
```

2. Increase login timeout:
```bash
python3 scripts/web-fetch.py "https://example.com/protected" \
  --login-timeout 600
```

3. Check browser console for errors during manual login

### Batch Fetch Hangs or Crashes

**Symptom:** Batch process stops responding or crashes mid-way.

**Causes:**
- Concurrency too high
- Memory exhaustion
- Network timeouts
- Browser process crashes

**Solutions:**

1. Reduce concurrency:
```bash
python3 scripts/web-fetch-batch.py \
  --urls urls.txt \
  --concurrency 1 \
  --format mhtml
```

2. Increase timeouts:
```bash
python3 scripts/web-fetch-batch.py \
  --urls urls.txt \
  --concurrency 2 \
  --timeout 120 \
  --wait 15 \
  --max-wait 45
```

3. Monitor system memory during batch processing

4. Split large URL lists into smaller batches

### Profile Lock Error (Concurrency)

**Symptom:** Error about ProcessSingleton lock or profile in use.

**Causes:**
- Multiple processes using same Chrome profile
- System Chrome already running with same profile
- Profile not properly released from previous run

**Solutions:**

1. Use different profiles for concurrent tasks:
```bash
# Task 1
python3 scripts/web-fetch.py "https://example1.com" \
  --profile-directory "Default" &

# Task 2
python3 scripts/web-fetch.py "https://example2.com" \
  --profile-directory "Profile 1" &
```

2. Use built-in Chromium for batch (avoids system Chrome lock):
```bash
python3 scripts/web-fetch-batch.py \
  --urls urls.txt \
  --no-system-chrome \
  --concurrency 3
```

3. Close system Chrome before batch processing

4. Wait for previous processes to fully exit

### Popup/Modal Not Closed

**Symptom:** Saved page contains unwanted popups or modals.

**Causes:**
- Popup detection failed
- Custom popup implementation
- Popup appears after page load

**Solutions:**

1. Enable manual popup handling:
```bash
python3 scripts/web-fetch.py "https://example.com" \
  --manual-popup-close \
  --manual-popup-timeout 180
```

2. Increase wait time to allow popup detection:
```bash
python3 scripts/web-fetch.py "https://example.com" \
  --wait 15 \
  --max-wait 45
```

3. For batch processing with popups:
```bash
python3 scripts/web-fetch-batch.py \
  --urls urls.txt \
  --manual-popup-close \
  --manual-popup-timeout 300 \
  --concurrency 2
```

### PDF Generation Fails

**Symptom:** PDF file not created or is corrupted.

**Causes:**
- Page rendering incomplete
- PDF generation timeout
- Insufficient disk space

**Solutions:**

1. Increase wait times before PDF generation:
```bash
python3 scripts/web-fetch.py "https://example.com" \
  --wait 15 \
  --max-wait 60 \
  --format pdf
```

2. Check disk space:
```bash
df -h
```

3. Try MHTML format first to verify page loads:
```bash
python3 scripts/web-fetch.py "https://example.com" --format mhtml
```

### Network Timeout

**Symptom:** Script times out waiting for page to load.

**Causes:**
- Slow network connection
- Server slow to respond
- Page has many external resources

**Solutions:**

1. Increase page load timeout:
```bash
python3 scripts/web-fetch.py "https://example.com" \
  --timeout 180
```

2. Check network connectivity:
```bash
ping example.com
```

3. Try fetching at different time

4. Check if page is accessible in browser

### Cookie Persistence Issues

**Symptom:** Cookies not saved or reused between fetches.

**Causes:**
- Cookies directory not writable
- Domain mismatch in cookie storage
- Session expired

**Solutions:**

1. Check cookie directory permissions:
```bash
ls -la ~/.config/google-chrome/cookies/
```

2. Verify cookies were saved:
```bash
cat ~/.config/google-chrome/cookies/example.com.json
```

3. Clear and re-login:
```bash
rm ~/.config/google-chrome/cookies/example.com.json
python3 scripts/web-fetch.py "https://example.com/protected" --format mhtml
```

### Memory Issues

**Symptom:** Out of memory error or system slowdown during batch.

**Causes:**
- Concurrency too high
- Large pages consuming memory
- Memory leak in long-running batch

**Solutions:**

1. Reduce concurrency:
```bash
python3 scripts/web-fetch-batch.py \
  --urls urls.txt \
  --concurrency 1
```

2. Process URLs in smaller batches:
```bash
# Split urls.txt into urls1.txt, urls2.txt, etc.
python3 scripts/web-fetch-batch.py --urls urls1.txt --format mhtml
python3 scripts/web-fetch-batch.py --urls urls2.txt --format mhtml
```

3. Monitor memory usage:
```bash
# macOS
top -l 1 | grep "PhysMem"

# Linux
free -h
```

## Debugging Tips

### Enable Verbose Output

Check script output for detailed error messages:

```bash
python3 scripts/web-fetch.py "https://example.com" --format mhtml 2>&1 | tee fetch.log
```

### Test with Simple Page

Verify basic functionality with a simple page:

```bash
python3 scripts/web-fetch.py "https://example.com" --format mhtml
```

### Check Browser Console

For login issues, check browser console during manual login for JavaScript errors.

### Verify File Output

Check if files are being created:

```bash
ls -lh *.mhtml *.pdf *.png *.html
```

### Test Network Connectivity

```bash
curl -I https://example.com
```

## Getting Help

If issues persist:

1. Check script output for error messages
2. Review this troubleshooting guide
3. Consult `references/architecture.md` for technical details
4. Check Playwright documentation: https://playwright.dev/python/
