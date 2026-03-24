# Architecture & Design

## Overview

Web-fetch is a Playwright-based web scraping tool that leverages Chrome's persistent context to maintain authentication sessions across multiple fetch operations. The architecture balances three key concerns: stability, concurrency, and maintainability.

## Core Components

### WebFetcher Class

The main single-page fetching component (`scripts/web-fetch.py`).

**Responsibilities:**
- Browser lifecycle management (start, stop)
- Page navigation and rendering
- Format conversion (MHTML, PDF, PNG, HTML)
- Cookie persistence and loading
- Login detection and handling
- Popup detection and cleanup

**Key Methods:**
- `start()` - Initialize browser and context
- `save()` - Fetch and save webpage in specified formats
- `_load_cookies()` - Load saved cookies for domain
- `_save_cookies()` - Persist cookies after fetch
- `_detect_login_page()` - Identify login requirement
- `_cleanup_popups()` - Remove detected popups

### BatchFetcher Class

Batch processing component (`scripts/web-fetch-batch.py`).

**Responsibilities:**
- URL list parsing
- Concurrent task management
- Progress tracking
- Error handling and recovery

**Key Methods:**
- `fetch_batch()` - Process multiple URLs concurrently
- `_fetch_single()` - Wrapper for single fetch with error handling

### URL Filename Utility

Helper module (`scripts/url_filename.py`).

**Responsibilities:**
- Convert URLs to safe filenames
- Maintain consistent naming across formats
- Handle special characters and long paths

## Design Decisions

### 1. Chrome Persistent Context

**Decision:** Use system Chrome with persistent user data directory instead of fresh browser instances.

**Rationale:**
- Reuses existing login sessions and cookies
- Reduces login overhead for authenticated pages
- Maintains browser state across fetches
- Leverages user's existing Chrome configuration

**Trade-offs:**
- Requires system Chrome installation
- Profile locking issues in concurrent scenarios
- Dependency on Chrome user data directory structure

### 2. Profile Isolation for Concurrency

**Decision:** Use different Chrome profiles for concurrent tasks instead of sharing single profile.

**Rationale:**
- Avoids ProcessSingleton lock conflicts
- Prevents profile corruption from concurrent access
- Enables true parallel fetching

**Implementation:**
- Batch fetcher creates isolated profiles for each concurrent task
- Each task gets unique profile directory
- Cookies still shared via host Chrome session

**Limitation:** System Chrome has hard limit on concurrent profile access (~5 tasks).

### 3. Manual Login Fallback

**Decision:** Support visible browser mode for manual login when automatic detection fails.

**Rationale:**
- Handles complex authentication flows (2FA, CAPTCHA, etc.)
- Provides user control over sensitive operations
- Enables recovery from failed automatic login

**Implementation:**
- Detect login page via heuristics (form detection, URL patterns)
- Switch to visible mode and display login prompt
- Wait for user to complete login
- Resume automatic fetching after login

### 4. No-Click Popup Handling

**Decision:** Default to DOM-based popup removal instead of click-based interaction.

**Rationale:**
- Avoids accidental window opening from click handlers
- More reliable than detecting and clicking close buttons
- Reduces side effects from popup interactions

**Implementation:**
- Detect common popup selectors (modal, overlay, popup classes)
- Hide or remove popup elements via JavaScript
- Optional click-based cleanup as fallback

### 5. Cookie Persistence by Domain

**Decision:** Store cookies in separate files per domain instead of single cookie jar.

**Rationale:**
- Enables domain-specific cookie management
- Simplifies cookie inspection and debugging
- Prevents cookie conflicts between domains

**Storage:** `<user-data>/cookies/{domain}.json`

### 6. Session Metadata Storage

**Decision:** Maintain separate session metadata directory for popup rules and login cache.

**Rationale:**
- Separates cookies (authentication) from metadata (behavior)
- Enables learning popup patterns across sessions
- Supports future session optimization

**Storage:** `<user-data>/session/`

## Data Flow

### Single Fetch Flow

```
1. User invokes web-fetch.py with URL and options
2. WebFetcher initializes browser and context
3. Load cookies for domain from persistent storage
4. Navigate to URL
5. Wait for page stability (--wait to --max-wait)
6. Detect login requirement
   - If login needed: switch to visible mode, wait for manual login
   - If login successful: save cookies, continue
7. Detect and cleanup popups
8. Convert page to requested formats
9. Save files to output directory
10. Persist updated cookies
11. Close browser context
```

### Batch Fetch Flow

```
1. User invokes web-fetch-batch.py with URL list
2. Parse URL list from file
3. For each URL (up to --concurrency in parallel):
   a. Create isolated profile for task
   b. Invoke WebFetcher with isolated profile
   c. Handle errors and continue
4. Track progress and report results
5. Cleanup temporary profiles
```

## Authentication Flow

### Cookie-Based Authentication

```
First Fetch:
1. Check for saved cookies for domain
2. Load cookies if available
3. Navigate to URL
4. If login page detected:
   - Open visible browser
   - Wait for manual login (--login-timeout)
   - Save new cookies
5. Continue with authenticated session

Subsequent Fetches:
1. Load saved cookies for domain
2. Navigate to URL
3. Usually no login needed (cookies valid)
4. If login page still appears:
   - Cookies may have expired
   - Repeat login flow
```

### Session State Complexity

Login state depends on multiple storage mechanisms:

- **Cookies** - HTTP-only authentication tokens
- **localStorage** - Client-side session data
- **indexedDB** - Persistent client storage
- **Service Workers** - Offline capability and caching
- **Browser Cache** - Resource caching

The persistent context preserves all these mechanisms, enabling complex authentication flows.

## Concurrency Architecture

### Profile Isolation Strategy

```
Host Chrome (system Chrome with Default profile)
├── Maintains login state
├── Stores cookies
└── Shared across all tasks

Concurrent Tasks (each with isolated profile)
├── Task 1: Profile "Profile 1"
├── Task 2: Profile "Profile 2"
├── Task 3: Profile "Profile 3"
└── All load cookies from host Chrome
```

**Benefits:**
- Host Chrome maintains stable login state
- Each task has isolated browser context
- Cookies shared but profiles isolated
- Avoids ProcessSingleton lock conflicts

**Limitations:**
- System Chrome has hard limit (~5 concurrent profiles)
- Each profile uses ~200-500MB memory
- Profile creation has overhead

## Format Conversion

### MHTML (Recommended)

- **Method:** Playwright's `page.save_mhtml()` via CDP
- **Advantages:** Single file, complete webpage, preserves styles
- **Requirements:** Playwright >= 1.40, CDP session available
- **Use case:** Archival, complete preservation

### PDF

- **Method:** Playwright's `page.pdf()`
- **Advantages:** Printable, widely compatible, good for sharing
- **Limitations:** May lose interactive elements
- **Use case:** Sharing, printing, documentation

### PNG

- **Method:** Playwright's `page.screenshot(full_page=True)`
- **Advantages:** Quick preview, visual verification
- **Limitations:** Large file size, not searchable
- **Use case:** Quick preview, visual documentation

### HTML

- **Method:** Playwright's `page.content()`
- **Advantages:** Source code, editable, lightweight
- **Limitations:** May lose styling, external resources not included
- **Use case:** Development, debugging, source inspection

## Error Handling

### Graceful Degradation

- If MHTML fails, try HTML format
- If PDF fails, try PNG format
- If page load times out, save partial content
- If login times out, skip and continue (batch mode)

### Retry Strategy

- No automatic retries (user controls via --force)
- Existing files skipped by default
- Manual retry via --force flag

### Batch Error Handling

- Individual fetch failures don't stop batch
- Errors logged and reported
- Batch continues with remaining URLs

## Performance Considerations

### Memory Usage

Per browser instance: 200-500MB

Batch processing memory:
- Concurrency 1: ~300MB
- Concurrency 3: ~900MB
- Concurrency 5: ~1.5GB

### Wait Time Strategy

- `--wait`: Minimum time for page rendering (default 10s)
- `--max-wait`: Maximum time for page stability detection (default 30s)
- Higher concurrency requires more conservative wait times

### Optimization Tips

1. Reduce wait times for simple pages
2. Increase wait times for dynamic content
3. Use concurrency 1-3 for stability
4. Monitor memory during batch processing
5. Process large batches in smaller chunks

## Security Considerations

### Cookie Storage

- Cookies stored in plaintext JSON files
- Located in Chrome user data directory
- Accessible to any process running as same user
- Consider file permissions for sensitive domains

### Credentials

- No credentials stored directly
- Relies on browser session cookies
- Manual login required for new sessions
- Cookies expire based on server policy

### Headless Mode

- Default headless mode doesn't require display
- Visible mode for login requires X Server (Linux) or display (macOS)
- No credentials transmitted to external services

## Future Improvements

### Potential Enhancements

1. **Proxy Support** - Route through proxy servers
2. **Custom Headers** - Add custom HTTP headers
3. **JavaScript Injection** - Execute custom scripts before save
4. **Selective Caching** - Cache specific resources
5. **Incremental Fetch** - Only fetch changed content
6. **Webhook Integration** - Notify on completion
7. **Database Storage** - Store fetched content in database
8. **API Server** - REST API for remote fetching

### Known Limitations

1. JavaScript execution limited to Playwright capabilities
2. Some sites detect and block Playwright
3. Concurrent profile limit (~5) from system Chrome
4. No support for HTTP/2 Server Push
5. Limited support for WebSocket-based content
