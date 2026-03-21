#!/usr/bin/env python3
"""
Web Fetch - 批量抓取工具

从文件读取 URL 列表，批量抓取网页。

Usage:
    python3 web-fetch-batch.py --urls urls.txt --format mhtml pdf
"""

import asyncio
import argparse
import os
import json
import tempfile
import shutil
import time
from urllib.parse import urlparse
from url_filename import generate_output_filename

try:
    from playwright.async_api import async_playwright
except ImportError:
    print("错误：请安装 playwright")
    print("  pip install playwright")
    print("  playwright install chromium")
    exit(1)


def default_chrome_user_data_dir():
    """返回系统 Chrome 用户数据目录。"""
    if os.name == "posix" and "darwin" in os.uname().sysname.lower():
        return os.path.expanduser("~/Library/Application Support/Google/Chrome")
    return os.path.expanduser("~/.config/google-chrome")


class BatchFetcher:
    """批量抓取器"""
    
    def __init__(
        self,
        user_data_dir=None,
        concurrency=1,
        chrome_path=None,
        use_system_chrome=True,
        profile_directory="Default"
    ):
        if user_data_dir is None:
            user_data_dir = default_chrome_user_data_dir()
        
        self.user_data_dir = user_data_dir
        self.concurrency = concurrency
        self.chrome_path = chrome_path
        self.use_system_chrome = use_system_chrome
        self.profile_directory = profile_directory
        self.cookies_dir = os.path.join(self.user_data_dir, "cookies")
        self.session_dir = os.path.join(self.user_data_dir, "session")
        # 登录阶段全局串行，避免并发弹多个登录窗口
        self.manual_login_lock = asyncio.Lock()
        # 断网恢复探测全局串行，避免多个 worker 同时探测
        self.network_recovery_lock = asyncio.Lock()

    @staticmethod
    def is_network_error_message(message):
        upper = message.upper()
        tokens = [
            "ERR_INTERNET_DISCONNECTED",
            "ERR_NAME_NOT_RESOLVED",
            "ERR_CONNECTION_RESET",
            "ERR_CONNECTION_REFUSED",
            "ERR_CONNECTION_CLOSED",
            "ERR_TIMED_OUT",
            "ERR_NETWORK_CHANGED",
            "ERR_ADDRESS_UNREACHABLE",
        ]
        return any(t in upper for t in tokens)

    async def check_network_reachable(self, host, port=443, timeout_seconds=5):
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port),
                timeout=timeout_seconds
            )
            writer.close()
            await writer.wait_closed()
            return True
        except Exception:
            return False

    async def wait_for_network_recovery(self, host, max_wait_seconds):
        async with self.network_recovery_lock:
            deadline = asyncio.get_event_loop().time() + max_wait_seconds
            while asyncio.get_event_loop().time() < deadline:
                if await self.check_network_reachable(host):
                    print(f"  网络恢复：{host} 可达，继续抓取。")
                    return True
                print(f"  网络不可达，等待恢复中...（目标：{host}）")
                await asyncio.sleep(3)
            return False

    async def goto_with_retries(self, page, url, timeout, max_retries=2, network_recovery_timeout=120):
        host = urlparse(url).hostname or "example.com"
        for attempt in range(max_retries + 1):
            try:
                # 先等到 domcontentloaded，后续由 wait_for_page_ready 做稳定判定
                await page.goto(url, wait_until="domcontentloaded", timeout=timeout)
                return
            except Exception as e:
                msg = str(e)
                # Playwright 超时同样视作可重试网络类问题
                if "TIMEOUT" in msg.upper() and "GOTO" in msg.upper():
                    is_retryable_timeout = True
                else:
                    is_retryable_timeout = False
                if not self.is_network_error_message(msg):
                    if not is_retryable_timeout:
                        raise
                if attempt >= max_retries:
                    raise
                print(f"  网络错误重试 {attempt + 1}/{max_retries + 1}: {msg}")
                recovered = await self.wait_for_network_recovery(host, max_wait_seconds=network_recovery_timeout)
                if not recovered:
                    raise RuntimeError(f"网络未恢复（{network_recovery_timeout} 秒）：{host}")
                await asyncio.sleep(min(2 ** attempt, 5))

    def build_launch_options(self, user_data_dir, headless):
        options = {
            "user_data_dir": user_data_dir,
            "headless": headless,
            "ignore_default_args": ['--use-mock-keychain', '--password-store=basic'],
            "args": [
                '--no-sandbox',
                '--disable-dev-shm-usage',
                '--disable-gpu',
                f'--profile-directory={self.profile_directory}'
            ]
        }
        if self.chrome_path:
            options["executable_path"] = self.chrome_path
        elif self.use_system_chrome:
            options["channel"] = "chrome"
        return options

    def seed_profile_from_host(self, target_user_data_dir):
        """把 host Chrome 的关键登录状态复制到临时 profile。"""
        profile_dir = self.profile_directory
        copy_paths = [
            "Local State",
            os.path.join(profile_dir, "Network", "Cookies"),
            os.path.join(profile_dir, "Cookies"),
            os.path.join(profile_dir, "Preferences"),
            os.path.join(profile_dir, "Secure Preferences"),
            os.path.join(profile_dir, "Local Storage"),
            os.path.join(profile_dir, "Session Storage"),
            os.path.join(profile_dir, "IndexedDB"),
            os.path.join(profile_dir, "Service Worker"),
            os.path.join(profile_dir, "WebStorage"),
        ]
        for rel in copy_paths:
            src = os.path.join(self.user_data_dir, rel)
            dst = os.path.join(target_user_data_dir, rel)
            if not os.path.exists(src):
                continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            try:
                if os.path.isdir(src):
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                else:
                    shutil.copy2(src, dst)
            except Exception:
                pass

    def cookie_file_for_url(self, url):
        base_domain = self.base_domain_for_url(url)
        safe_name = base_domain.replace(".", "_")
        return os.path.join(self.cookies_dir, f"{safe_name}.json")

    def session_file_for_url(self, url):
        base_domain = self.base_domain_for_url(url)
        safe_name = base_domain.replace(".", "_")
        return os.path.join(self.session_dir, f"{safe_name}.json")

    def load_session_profile(self, url):
        session_file = self.session_file_for_url(url)
        if not os.path.exists(session_file):
            return {}
        try:
            with open(session_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def update_session_profile(self, url, updates):
        os.makedirs(self.session_dir, exist_ok=True)
        session_file = self.session_file_for_url(url)
        data = self.load_session_profile(url)
        for key, value in updates.items():
            if key == "popup_hide_selectors":
                merged = set(data.get("popup_hide_selectors", []))
                merged.update(value or [])
                data[key] = sorted(list(merged))
            else:
                data[key] = value
        data["updated_at"] = int(time.time())
        with open(session_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    async def apply_session_profile(self, page, url):
        profile = self.load_session_profile(url)
        selectors = profile.get("popup_hide_selectors", [])
        if not selectors:
            return
        try:
            await page.evaluate(
                """
                ({ selectors }) => {
                  for (const selector of selectors) {
                    for (const node of document.querySelectorAll(selector)) {
                      node.style.setProperty('display', 'none', 'important');
                    }
                  }
                }
                """,
                {"selectors": selectors}
            )
            print(f"  已应用会话档案：{len(selectors)} 条弹窗规则")
        except Exception:
            pass

    @staticmethod
    def base_domain_for_url(url):
        hostname = (urlparse(url).hostname or "default").lower()
        parts = hostname.split(".")
        if len(parts) >= 2:
            return ".".join(parts[-2:])
        return hostname

    @staticmethod
    def cookie_belongs_to_base_domain(cookie, base_domain):
        cookie_domain = (cookie.get("domain", "") or "").lstrip(".").lower()
        return cookie_domain == base_domain or cookie_domain.endswith(f".{base_domain}")

    async def load_cookies(self, context, url):
        cookie_file = self.cookie_file_for_url(url)
        if not os.path.exists(cookie_file):
            return
        try:
            with open(cookie_file, "r", encoding="utf-8") as f:
                cookies = json.load(f)
            if cookies:
                await context.add_cookies(cookies)
                print(f"  已加载 Cookies：{cookie_file} ({len(cookies)} 条)")
        except Exception as e:
            print(f"  警告：读取 Cookies 失败，已忽略。原因：{e}")

    async def save_cookies(self, context, url):
        cookie_file = self.cookie_file_for_url(url)
        base_domain = self.base_domain_for_url(url)
        try:
            all_cookies = await context.cookies()
            cookies = [
                cookie for cookie in all_cookies
                if self.cookie_belongs_to_base_domain(cookie, base_domain)
            ]
            with open(cookie_file, "w", encoding="utf-8") as f:
                json.dump(cookies, f, ensure_ascii=False, indent=2)
            print(f"  已保存 Cookies：{cookie_file} ({len(cookies)} 条)")
            self.update_session_profile(
                url,
                {
                    "cookie_file": cookie_file,
                    "cookie_count": len(cookies),
                    "login_cached_at": int(time.time())
                }
            )
        except Exception as e:
            print(f"  警告：保存 Cookies 失败，已忽略。原因：{e}")

    async def is_login_required(self, page):
        login_keywords = ("login", "signin", "sign-in", "auth", "oauth", "sso", "passport")
        current_url = page.url.lower()
        if any(keyword in current_url for keyword in login_keywords):
            return True
        try:
            password_inputs = await page.locator('input[type="password"]').count()
            return password_inputs > 0
        except Exception:
            return False

    async def prompt_manual_login(self, page, timeout_seconds=300):
        await page.bring_to_front()
        await page.evaluate(
            """
            (() => {
              const existed = document.getElementById('__web_fetch_login_tip__');
              if (existed) return;
              const tip = document.createElement('div');
              tip.id = '__web_fetch_login_tip__';
              tip.style.position = 'fixed';
              tip.style.top = '0';
              tip.style.left = '0';
              tip.style.right = '0';
              tip.style.zIndex = '2147483647';
              tip.style.background = '#d9480f';
              tip.style.color = '#fff';
              tip.style.padding = '10px 14px';
              tip.style.fontSize = '14px';
              tip.style.fontFamily = 'sans-serif';
              tip.style.boxShadow = '0 2px 6px rgba(0,0,0,0.2)';
              tip.textContent = 'Web Fetch 提示：检测到需要登录，请在此页面手工登录。登录完成后脚本会自动继续。';
              document.body.appendChild(tip);
            })();
            """
        )

        print("  检测到需要登录，请在浏览器窗口手工登录。")
        deadline = asyncio.get_event_loop().time() + timeout_seconds
        while asyncio.get_event_loop().time() < deadline:
            if not await self.is_login_required(page):
                print("  登录状态已检测通过，继续抓取...")
                return
            await asyncio.sleep(2)
        raise TimeoutError(f"等待登录超时（{timeout_seconds} 秒）")

    async def wait_for_page_ready(self, page, min_wait_seconds=10, max_wait_seconds=30):
        """等待页面渲染稳定，避免过早保存。"""
        max_wait_seconds = max(max_wait_seconds, min_wait_seconds)
        # 并发越高，页面资源竞争越明显，等待策略需要更保守。
        concurrency_penalty = min(10, max(0, self.concurrency - 1) * 2)
        effective_min_wait = min_wait_seconds + concurrency_penalty
        effective_min_wait = min(effective_min_wait, max_wait_seconds)

        dom_ready = False
        load_ready = False
        network_idle_ready = False
        try:
            await page.wait_for_load_state("domcontentloaded", timeout=10000)
            dom_ready = True
        except Exception:
            pass
        try:
            await page.wait_for_load_state("load", timeout=10000)
            load_ready = True
        except Exception:
            pass
        try:
            await page.wait_for_load_state("networkidle", timeout=10000)
            network_idle_ready = True
        except Exception:
            pass

        if not network_idle_ready:
            # 未达到 networkidle 时加一段缓冲，避免在高并发下保存过早。
            effective_min_wait = min(max_wait_seconds, effective_min_wait + 5)

        print(
            f"  页面等待策略：min={effective_min_wait}s, max={max_wait_seconds}s, "
            f"dom={dom_ready}, load={load_ready}, networkidle={network_idle_ready}"
        )

        start = asyncio.get_event_loop().time()
        last_snapshot = None
        stable_rounds = 0
        required_stable_rounds = 4 if network_idle_ready else 6
        while True:
            now = asyncio.get_event_loop().time()
            elapsed = now - start
            if elapsed >= max_wait_seconds:
                print(f"  页面等待达到上限 {max_wait_seconds} 秒，继续保存。")
                return

            try:
                metrics = await page.evaluate(
                    """
                    () => {
                      const body = document.body;
                      return {
                        readyState: document.readyState || '',
                        elementCount: document.getElementsByTagName('*').length || 0,
                        textLength: body && body.innerText ? body.innerText.length : 0,
                        resourceCount: (performance.getEntriesByType('resource') || []).length || 0
                      };
                    }
                    """
                )
            except Exception:
                await asyncio.sleep(1)
                continue

            snapshot = (
                metrics.get("readyState"),
                metrics.get("elementCount"),
                metrics.get("textLength"),
                metrics.get("resourceCount"),
            )
            if snapshot == last_snapshot and metrics.get("readyState") == "complete":
                stable_rounds += 1
            else:
                stable_rounds = 0
            last_snapshot = snapshot

            if elapsed >= effective_min_wait and stable_rounds >= required_stable_rounds:
                print(f"  页面已稳定，实际等待 {int(elapsed)} 秒。")
                return

            await asyncio.sleep(1)

    async def cleanup_popups(self, page, rounds=2, aggressive=False, allow_click=False, url=None):
        """尽量关闭常见弹窗和浮层，减少保存噪音。"""
        collected_selectors = set()
        for _ in range(max(1, rounds)):
            try:
                try:
                    await page.keyboard.press("Escape")
                except Exception:
                    pass
                result = await page.evaluate(
                    """
                    ({ aggressive, allowClick }) => {
                      let clicked = 0;
                      let hidden = 0;
                      let iframeHidden = 0;

                      const closeSelectors = [
                        'button[aria-label*="close" i]',
                        '[role="button"][aria-label*="close" i]',
                        '[aria-label*="关闭"]',
                        '[title*="关闭"]',
                        '.close',
                        '.close-btn',
                        '.close-button',
                        '.modal-close',
                        '.ant-modal-close',
                        '.el-dialog__headerbtn',
                        '.layui-layer-close',
                        '.van-popup__close-icon',
                        '.weui-dialog__btn',
                        '[class*="close" i]',
                        '[id*="close" i]'
                      ];

                      if (allowClick) {
                        for (const selector of closeSelectors) {
                          for (const node of document.querySelectorAll(selector)) {
                            const style = window.getComputedStyle(node);
                            const rect = node.getBoundingClientRect();
                            const visible = style.display !== 'none' && style.visibility !== 'hidden'
                              && rect.width > 0 && rect.height > 0;
                            if (!visible) continue;
                            try {
                              node.click();
                              clicked += 1;
                            } catch (_) {}
                          }
                        }

                        const closeTexts = ['关闭', '取消', '知道了', '我知道了', '稍后', '以后再说', 'close', 'dismiss', 'not now', 'got it'];
                        const textNodes = document.querySelectorAll('button, [role="button"], input[type="button"], input[type="submit"]');
                        for (const node of textNodes) {
                          const text = (node.innerText || node.textContent || '').trim().toLowerCase();
                          if (!text) continue;
                          if (text.length > 12) continue;
                          if (!closeTexts.some(t => text === t || text.includes(t))) continue;
                          if (node.hasAttribute('href') || node.getAttribute('data-href') || node.getAttribute('data-url')) continue;
                          const cls = `${node.className || ''}`.toLowerCase();
                          const id = `${node.id || ''}`.toLowerCase();
                          const hasCloseHint = cls.includes('close') || cls.includes('cancel') || cls.includes('dismiss')
                            || id.includes('close') || id.includes('cancel') || id.includes('dismiss');
                          const inDialog = !!node.closest('[role="dialog"], [aria-modal="true"], .modal, .dialog, .popup, .overlay');
                          if (!hasCloseHint && !inDialog) continue;
                          const style = window.getComputedStyle(node);
                          const rect = node.getBoundingClientRect();
                          const visible = style.display !== 'none' && style.visibility !== 'hidden'
                            && rect.width > 0 && rect.height > 0;
                          if (!visible) continue;
                          try {
                            node.click();
                            clicked += 1;
                          } catch (_) {}
                        }
                      }

                      const overlaySelectors = [
                        '[role="dialog"]',
                        '[aria-modal="true"]',
                        '.modal',
                        '.dialog',
                        '.popup',
                        '.overlay',
                        '.backdrop',
                        '.ant-modal-wrap',
                        '.MuiDialog-root',
                        '.el-overlay',
                        '.el-dialog__wrapper',
                        '[class*="cookie" i]',
                        '[id*="cookie" i]',
                        '[class*="consent" i]',
                        '[id*="consent" i]'
                      ];

                      const matchedSelectors = [];
                      for (const selector of overlaySelectors) {
                        let matched = false;
                        for (const node of document.querySelectorAll(selector)) {
                          const style = window.getComputedStyle(node);
                          const rect = node.getBoundingClientRect();
                          const visible = style.display !== 'none' && style.visibility !== 'hidden'
                            && rect.width > 40 && rect.height > 40;
                          if (!visible) continue;
                          node.style.setProperty('display', 'none', 'important');
                          hidden += 1;
                          matched = true;
                        }
                        if (matched) matchedSelectors.push(selector);
                      }

                      const vw = window.innerWidth || 1;
                      const vh = window.innerHeight || 1;
                      const minAreaRatio = aggressive ? 0.03 : 0.12;
                      const minZIndex = aggressive ? 100 : 1000;
                      const fixedNodes = document.querySelectorAll('body *');
                      for (const node of fixedNodes) {
                        const style = window.getComputedStyle(node);
                        if (!(style.position === 'fixed' || style.position === 'sticky')) continue;
                        const z = parseInt(style.zIndex || '0', 10);
                        if (Number.isNaN(z) || z < minZIndex) continue;
                        const rect = node.getBoundingClientRect();
                        if (rect.width <= 0 || rect.height <= 0) continue;
                        const areaRatio = (rect.width * rect.height) / (vw * vh);
                        if (areaRatio < minAreaRatio) continue;
                        const tag = (node.tagName || '').toLowerCase();
                        if (!aggressive && (tag === 'header' || tag === 'footer' || tag === 'nav')) continue;
                        node.style.setProperty('display', 'none', 'important');
                        hidden += 1;
                      }

                      const popupIframes = document.querySelectorAll(
                        'iframe[class*="popup" i], iframe[id*="popup" i], iframe[class*="modal" i], iframe[id*="modal" i], iframe[style*="z-index"]'
                      );
                      for (const frame of popupIframes) {
                        const rect = frame.getBoundingClientRect();
                        if (rect.width <= 100 || rect.height <= 100) continue;
                        frame.style.setProperty('display', 'none', 'important');
                        iframeHidden += 1;
                      }

                      document.body.style.setProperty('overflow', 'auto', 'important');
                      document.documentElement.style.setProperty('overflow', 'auto', 'important');

                      return { clicked, hidden, iframeHidden, matchedSelectors };
                    }
                    """
                    ,
                    {"aggressive": aggressive, "allowClick": allow_click}
                )
                total = (
                    result.get("clicked", 0)
                    + result.get("hidden", 0)
                    + result.get("iframeHidden", 0)
                )
                for selector in result.get("matchedSelectors", []):
                    collected_selectors.add(selector)
                if total <= 0:
                    break
            except Exception:
                break
            await asyncio.sleep(0.8)

        if url and collected_selectors:
            self.update_session_profile(
                url,
                {
                    "popup_hide_selectors": sorted(list(collected_selectors)),
                    "popup_updated_at": int(time.time())
                }
            )

    async def protect_from_unwanted_popups(self, page):
        """拦截脚本触发的新窗口，降低误点击影响。"""
        page.on("popup", lambda popup: asyncio.create_task(popup.close()))
        await page.add_init_script(
            """
            (() => {
              const originalOpen = window.open;
              window.open = function(...args) {
                try { return null; } catch (_) { return null; }
              };
              window.__webFetchOriginalOpen = originalOpen;
              document.addEventListener('click', (e) => {
                const target = e.target && e.target.closest ? e.target.closest('a[target="_blank"]') : null;
                if (target) {
                  target.setAttribute('target', '_self');
                  target.removeAttribute('rel');
                }
              }, true);
            })();
            """
        )
    
    async def fetch_and_save(
        self,
        url,
        formats,
        output_dir,
        wait_seconds=10,
        timeout=90000,
        force=False,
        login_timeout=300,
        max_wait_seconds=30,
        cleanup_popups_enabled=True,
        aggressive_popup_cleanup=False,
        popup_click_cleanup=False,
        headless_profile_dir=None,
        goto_retries=2,
        network_recovery_timeout=120
    ):
        """抓取单个网页并保存"""
        from playwright.async_api import async_playwright
        
        results = {}
        output_paths = {}
        pending_formats = []

        for fmt in formats:
            output_path = generate_output_filename(url, fmt, output_dir)
            output_paths[fmt] = output_path
            if os.path.exists(output_path) and not force:
                size = os.path.getsize(output_path)
                results[fmt] = {
                    'path': output_path,
                    'size': size,
                    'success': True,
                    'skipped': True
                }
            else:
                pending_formats.append(fmt)

        if not pending_formats:
            return {
                'url': url,
                'skipped': True,
                'results': results
            }
        
        os.makedirs(self.user_data_dir, exist_ok=True)
        os.makedirs(self.cookies_dir, exist_ok=True)
        os.makedirs(self.session_dir, exist_ok=True)

        async def run_once(headless, user_data_dir):
            async with async_playwright() as p:
                # 并发场景下避免多个实例争抢同一个 profile 锁：
                # headless 抓取使用独立临时 profile，登录态通过 cookies/session 文件复用。
                isolated_profile_dir = None
                launch_user_data_dir = user_data_dir
                if headless:
                    if headless_profile_dir:
                        launch_user_data_dir = headless_profile_dir
                    else:
                        isolated_profile_dir = tempfile.mkdtemp(prefix="web-fetch-worker-")
                        launch_user_data_dir = isolated_profile_dir
                        self.seed_profile_from_host(launch_user_data_dir)

                launch_options = self.build_launch_options(launch_user_data_dir, headless)
                try:
                    context = await p.chromium.launch_persistent_context(**launch_options)
                except Exception as e:
                    if launch_options.get("channel") == "chrome" or launch_options.get("executable_path"):
                        print(f"  警告：系统 Chrome 启动失败，回退到 Playwright 内置 Chromium。原因：{e}")
                        launch_options.pop("channel", None)
                        launch_options.pop("executable_path", None)
                        context = await p.chromium.launch_persistent_context(**launch_options)
                    else:
                        raise
                try:
                    page = await context.new_page()
                    await self.protect_from_unwanted_popups(page)
                    await self.load_cookies(context, url)
                    await self.goto_with_retries(
                        page,
                        url,
                        timeout=timeout,
                        max_retries=max(0, goto_retries),
                        network_recovery_timeout=max(10, network_recovery_timeout)
                    )
                    await self.apply_session_profile(page, url)

                    if await self.is_login_required(page):
                        if headless:
                            raise RuntimeError("LOGIN_REQUIRED")
                        await self.prompt_manual_login(page, timeout_seconds=login_timeout)

                    await self.save_cookies(context, url)
                    await self.wait_for_page_ready(
                        page,
                        min_wait_seconds=wait_seconds,
                        max_wait_seconds=max_wait_seconds
                    )
                    if cleanup_popups_enabled:
                        await self.cleanup_popups(
                            page,
                            aggressive=aggressive_popup_cleanup,
                            allow_click=popup_click_cleanup,
                            url=url
                        )

                    title = await page.title()
                    for fmt in pending_formats:
                        try:
                            output_path = output_paths[fmt]

                            if fmt == 'mhtml':
                                cdp = await context.new_cdp_session(page)
                                result = await cdp.send('Page.captureSnapshot', {'format': 'mhtml'})
                                with open(output_path, 'w', encoding='utf-8') as f:
                                    f.write(result['data'])
                            elif fmt == 'pdf':
                                await page.emulate_media(media='screen')
                                await page.pdf(
                                    path=output_path,
                                    format='A4',
                                    print_background=True
                                )
                            elif fmt == 'png':
                                await page.screenshot(path=output_path, full_page=True)
                            elif fmt == 'html':
                                html = await page.content()
                                with open(output_path, 'w', encoding='utf-8') as f:
                                    f.write(html)

                            size = os.path.getsize(output_path)
                            results[fmt] = {
                                'path': output_path,
                                'size': size,
                                'success': True,
                                'skipped': False
                            }
                        except Exception as e:
                            results[fmt] = {'error': str(e), 'success': False}

                    return {
                        'url': url,
                        'title': title,
                        'results': results
                    }
                finally:
                    try:
                        await context.close()
                    except Exception:
                        pass
                    if isolated_profile_dir:
                        shutil.rmtree(isolated_profile_dir, ignore_errors=True)

        try:
            return await run_once(headless=True, user_data_dir=self.user_data_dir)
        except RuntimeError as e:
            if str(e) != "LOGIN_REQUIRED":
                return {'url': url, 'error': str(e)}
            async with self.manual_login_lock:
                # 其他任务可能刚完成登录，这里先重试一次无头流程
                try:
                    return await run_once(headless=True, user_data_dir=self.user_data_dir)
                except RuntimeError as retry_error:
                    if str(retry_error) != "LOGIN_REQUIRED":
                        return {'url': url, 'error': str(retry_error)}

                print("  检测到登录页面，自动切换到可见浏览器进行手工登录...")
                interactive_tmp_user_data = tempfile.mkdtemp(prefix="web-fetch-login-")
                try:
                    return await run_once(headless=False, user_data_dir=interactive_tmp_user_data)
                except Exception as visible_error:
                    return {'url': url, 'error': str(visible_error)}
                finally:
                    shutil.rmtree(interactive_tmp_user_data, ignore_errors=True)


async def main():
    parser = argparse.ArgumentParser(description='批量网页抓取工具')
    parser.add_argument('--urls', '-u', required=True, help='URL 列表文件（每行一个 URL）')
    parser.add_argument('--format', '-f', nargs='+', default=['mhtml'],
                        choices=['mhtml', 'pdf', 'png', 'html'],
                        help='保存格式')
    parser.add_argument('--output', '-o', default='./output',
                        help='输出目录')
    parser.add_argument('--wait', '-w', type=int, default=10,
                        help='最短等待渲染时间（秒）')
    parser.add_argument('--max-wait', type=int, default=30,
                        help='最长等待渲染时间（秒）')
    parser.add_argument('--concurrency', '-c', type=int, default=1,
                        help='并发数量')
    parser.add_argument('--user-data', help='Chrome 用户数据目录')
    parser.add_argument('--profile-directory', default='Default',
                        help='Chrome Profile 目录名（默认：Default）')
    parser.add_argument('--chrome-path', help='系统 Chrome 可执行文件路径（可选）')
    parser.add_argument('--no-system-chrome', action='store_true',
                        help='不使用系统 Chrome，强制使用 Playwright 内置 Chromium')
    parser.add_argument('--login-timeout', type=int, default=300,
                        help='手工登录最大等待时间（秒）')
    parser.add_argument('--no-popup-cleanup', action='store_true',
                        help='禁用弹窗/浮层自动清理')
    parser.add_argument('--aggressive-popup-cleanup', action='store_true',
                        help='启用更激进的浮窗清理策略')
    parser.add_argument('--popup-click-cleanup', action='store_true',
                        help='允许通过点击方式清理弹窗（默认关闭，避免误点击）')
    parser.add_argument('--task-timeout', type=int, default=300,
                        help='单个 URL 任务总超时（秒，默认：300）')
    parser.add_argument('--goto-retries', type=int, default=2,
                        help='页面导航网络错误重试次数（默认：2）')
    parser.add_argument('--network-recovery-timeout', type=int, default=120,
                        help='断网后等待恢复时间（秒，默认：120）')
    parser.add_argument('--force', action='store_true',
                        help='强制覆盖已存在文件')
    
    args = parser.parse_args()
    
    # 读取 URL 列表
    with open(args.urls, 'r', encoding='utf-8') as f:
        urls = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
    print("=" * 80)
    print("Web Fetch - 批量抓取工具")
    print("=" * 80)
    print(f"\nURL 文件：{args.urls}")
    print(f"URL 数量：{len(urls)}")
    print(f"保存格式：{', '.join(args.format)}")
    print(f"输出目录：{args.output}")
    print(f"并发数量：{args.concurrency}")
    print(f"最短等待：{args.wait}秒")
    print(f"最大等待：{args.max_wait}秒")
    print(f"强制覆盖：{'是' if args.force else '否'}")
    print()
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    requested_system_chrome = not args.no_system_chrome
    effective_use_system_chrome = requested_system_chrome
    if requested_system_chrome and args.concurrency > 1:
        print("检测到并发模式：为避免占用 host Chrome，自动切换为内置 Chromium 执行抓取（仍复用 host cookies/session）。")
        effective_use_system_chrome = False

    fetcher = BatchFetcher(
        user_data_dir=args.user_data,
        concurrency=args.concurrency,
        chrome_path=args.chrome_path,
        use_system_chrome=effective_use_system_chrome,
        profile_directory=args.profile_directory
    )
    
    # 抓取（并发）
    concurrency = max(1, args.concurrency)
    results = [None] * len(urls)
    next_index = 0
    index_lock = asyncio.Lock()

    async def worker():
        nonlocal next_index
        worker_profile_dir = tempfile.mkdtemp(prefix="web-fetch-worker-")
        fetcher.seed_profile_from_host(worker_profile_dir)
        processed_count = 0
        try:
            while True:
                async with index_lock:
                    if next_index >= len(urls):
                        return
                    index = next_index
                    next_index += 1
                    url = urls[index]

                print(f"\n[{index + 1}/{len(urls)}] {url}")
                try:
                    result = await asyncio.wait_for(
                        fetcher.fetch_and_save(
                            url,
                            args.format,
                            args.output,
                            wait_seconds=args.wait,
                            force=args.force,
                            login_timeout=args.login_timeout,
                            max_wait_seconds=args.max_wait,
                            cleanup_popups_enabled=not args.no_popup_cleanup,
                            aggressive_popup_cleanup=args.aggressive_popup_cleanup,
                            popup_click_cleanup=args.popup_click_cleanup,
                            headless_profile_dir=worker_profile_dir,
                            goto_retries=args.goto_retries,
                            network_recovery_timeout=args.network_recovery_timeout
                        ),
                        timeout=max(30, args.task_timeout)
                    )
                except asyncio.TimeoutError:
                    result = {
                        'url': url,
                        'error': f'任务超时（{max(30, args.task_timeout)} 秒）'
                    }
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    result = {
                        'url': url,
                        'error': f'任务异常：{e}'
                    }

                results[index] = result

                if 'title' in result:
                    print(f"  标题：{result['title']}")
                if result.get('skipped'):
                    print("  已存在同名文件，跳过下载")
                if 'error' in result:
                    print(f"  ✗ 错误：{result['error']}")

                for fmt, res in result.get('results', {}).items():
                    if res.get('success'):
                        size_mb = res['size'] / 1024 / 1024
                        if res.get('skipped'):
                            print(f"  - {fmt.upper()}: {res['path']} (已存在，跳过)")
                        else:
                            print(f"  ✓ {fmt.upper()}: {res['path']} ({size_mb:.2f} MB)")
                    else:
                        print(f"  ✗ {fmt.upper()}: {res.get('error', '未知错误')}")
                processed_count += 1
                # 周期性把 host 状态合并进 worker profile，降低长跑后登录态漂移
                if processed_count % 50 == 0:
                    fetcher.seed_profile_from_host(worker_profile_dir)
        finally:
            shutil.rmtree(worker_profile_dir, ignore_errors=True)

    workers = [asyncio.create_task(worker()) for _ in range(concurrency)]
    try:
        await asyncio.gather(*workers)
    except asyncio.CancelledError:
        for w in workers:
            w.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
        raise
    
    # 统计
    print("\n" + "=" * 80)
    print("抓取完成！")
    print("=" * 80)
    
    success_count = sum(1 for r in results if r and ('title' in r or r.get('skipped')))
    print(f"\n成功：{success_count}/{len(urls)}")
    print(f"输出目录：{args.output}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n检测到手工中断（Ctrl+C），已安全停止。")
