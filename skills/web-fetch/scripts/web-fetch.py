#!/usr/bin/env python3
"""
Web Fetch - 网页抓取工具

使用 Playwright + Chrome 持久化上下文抓取网页，支持保存为 MHTML、PDF、PNG、HTML 等格式。
特别适用于需要登录认证的网页抓取。

Usage:
    python3 web-fetch.py <url> [--format mhtml|pdf|png|html] [--output DIR] [--wait SECONDS]
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

class LoginRequiredError(Exception):
    """需要手工登录后才能继续抓取"""


class WebFetcher:
    """网页抓取器"""

    def __init__(
        self,
        user_data_dir=None,
        headless=True,
        chrome_path=None,
        use_system_chrome=True,
        profile_directory="Default"
    ):
        """
        初始化抓取器
        
        Args:
            user_data_dir: Chrome 用户数据目录
            headless: 是否无头模式
        """
        if user_data_dir is None:
            user_data_dir = default_chrome_user_data_dir()
        
        self.user_data_dir = user_data_dir
        self.headless = headless
        self.chrome_path = chrome_path
        self.use_system_chrome = use_system_chrome
        self.profile_directory = profile_directory
        self.browser = None
        self.context = None
        self.cookies_dir = os.path.join(self.user_data_dir, "cookies")
        self.session_dir = os.path.join(self.user_data_dir, "session")
        self.playwright = None

    def build_launch_options(self, user_data_dir=None, headless=None):
        """构建浏览器启动参数，优先使用系统 Chrome。"""
        if user_data_dir is None:
            user_data_dir = self.user_data_dir
        if headless is None:
            headless = self.headless

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
    
    async def start(self):
        """启动浏览器"""
        os.makedirs(self.user_data_dir, exist_ok=True)
        os.makedirs(self.cookies_dir, exist_ok=True)
        os.makedirs(self.session_dir, exist_ok=True)
        self.playwright = await async_playwright().start()
        launch_options = self.build_launch_options()
        try:
            self.context = await self.playwright.chromium.launch_persistent_context(**launch_options)
        except Exception as e:
            if launch_options.get("channel") == "chrome" or launch_options.get("executable_path"):
                print(f"警告：系统 Chrome 启动失败，回退到 Playwright 内置 Chromium。原因：{e}")
                launch_options.pop("channel", None)
                launch_options.pop("executable_path", None)
                self.context = await self.playwright.chromium.launch_persistent_context(**launch_options)
            else:
                raise
    
    async def close(self):
        """关闭浏览器"""
        if self.context:
            await self.context.close()
            self.context = None
        if self.playwright:
            await self.playwright.stop()
            self.playwright = None
    
    async def wait_for_page_ready(self, page, min_wait_seconds=10, max_wait_seconds=30):
        """
        等待页面渲染稳定，避免过早保存。

        - 至少等待 min_wait_seconds
        - 页面指标连续稳定后提前结束
        - 超过 max_wait_seconds 则兜底继续
        """
        max_wait_seconds = max(max_wait_seconds, min_wait_seconds)
        try:
            await page.wait_for_load_state("domcontentloaded", timeout=10000)
        except Exception:
            pass
        try:
            await page.wait_for_load_state("load", timeout=10000)
        except Exception:
            pass
        try:
            await page.wait_for_load_state("networkidle", timeout=10000)
        except Exception:
            pass

        start = asyncio.get_event_loop().time()
        last_snapshot = None
        stable_rounds = 0

        while True:
            now = asyncio.get_event_loop().time()
            elapsed = now - start
            if elapsed >= max_wait_seconds:
                print(f"页面等待达到上限 {max_wait_seconds} 秒，继续保存。")
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

            if elapsed >= min_wait_seconds and stable_rounds >= 2:
                print(f"页面已稳定，实际等待 {int(elapsed)} 秒。")
                return

            await asyncio.sleep(1)

    async def cleanup_popups(self, page, rounds=2, aggressive=False, url=None):
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
                    ({ aggressive }) => {
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
                    {"aggressive": aggressive}
                )
                for selector in result.get("matchedSelectors", []):
                    collected_selectors.add(selector)
                total = (
                    result.get("clicked", 0)
                    + result.get("hidden", 0)
                    + result.get("iframeHidden", 0)
                )
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

    async def fetch(
        self,
        url,
        wait_seconds=10,
        timeout=90000,
        login_timeout=300,
        max_wait_seconds=30,
        cleanup_popups_enabled=True,
        aggressive_popup_cleanup=False
    ):
        """
        抓取网页
        
        Args:
            url: 目标 URL
            wait_seconds: 等待渲染时间（秒）
            timeout: 超时时间（毫秒）
        
        Returns:
            page: Playwright page 对象
        """
        page = await self.context.new_page()
        await self.protect_from_unwanted_popups(page)
        await self.load_cookies(url)
        await page.goto(url, wait_until="networkidle", timeout=timeout)
        await self.apply_session_profile(page, url)

        if await self.is_login_required(page):
            if self.headless:
                raise LoginRequiredError("检测到登录页面，需切换到可见浏览器手工登录")
            await self.prompt_manual_login(page, timeout_seconds=login_timeout)

        await self.save_cookies(url)
        await self.wait_for_page_ready(
            page,
            min_wait_seconds=wait_seconds,
            max_wait_seconds=max_wait_seconds
        )
        if cleanup_popups_enabled:
            await self.cleanup_popups(page, aggressive=aggressive_popup_cleanup, url=url)
        return page

    def cookie_file_for_url(self, url):
        """按主域名生成 cookies 文件路径（如 gitee.com）"""
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
            print(f"已应用会话档案：{len(selectors)} 条弹窗规则")
        except Exception:
            pass

    @staticmethod
    def base_domain_for_url(url):
        """提取主域名，便于跨子域复用登录态"""
        hostname = (urlparse(url).hostname or "default").lower()
        parts = hostname.split(".")
        if len(parts) >= 2:
            return ".".join(parts[-2:])
        return hostname

    @staticmethod
    def cookie_belongs_to_base_domain(cookie, base_domain):
        """判断 cookie 是否属于目标主域名"""
        cookie_domain = (cookie.get("domain", "") or "").lstrip(".").lower()
        return cookie_domain == base_domain or cookie_domain.endswith(f".{base_domain}")

    async def load_cookies(self, url):
        """从本地文件加载 cookies"""
        cookie_file = self.cookie_file_for_url(url)
        if not os.path.exists(cookie_file):
            return

        try:
            with open(cookie_file, "r", encoding="utf-8") as f:
                cookies = json.load(f)
            if cookies:
                await self.context.add_cookies(cookies)
                print(f"已加载 Cookies：{cookie_file} ({len(cookies)} 条)")
        except Exception as e:
            print(f"警告：读取 Cookies 失败，已忽略。原因：{e}")

    async def save_cookies(self, url):
        """将目标主域名相关 cookies 保存到本地"""
        cookie_file = self.cookie_file_for_url(url)
        base_domain = self.base_domain_for_url(url)

        try:
            all_cookies = await self.context.cookies()
            cookies = [
                cookie for cookie in all_cookies
                if self.cookie_belongs_to_base_domain(cookie, base_domain)
            ]
            with open(cookie_file, "w", encoding="utf-8") as f:
                json.dump(cookies, f, ensure_ascii=False, indent=2)
            print(f"已保存 Cookies：{cookie_file} ({len(cookies)} 条)")
            self.update_session_profile(
                url,
                {
                    "cookie_file": cookie_file,
                    "cookie_count": len(cookies),
                    "login_cached_at": int(time.time())
                }
            )
        except Exception as e:
            print(f"警告：保存 Cookies 失败，已忽略。原因：{e}")

    async def is_login_required(self, page):
        """根据 URL 和页面元素判断是否进入登录流程"""
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
        """在页面提示用户手工登录并等待登录完成"""
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

        print("\n检测到需要登录。")
        print("请在浏览器窗口中手工完成登录（含验证码/二次验证）。")
        print(f"最长等待 {timeout_seconds} 秒，登录完成后将自动继续。\n")

        deadline = asyncio.get_event_loop().time() + timeout_seconds
        while asyncio.get_event_loop().time() < deadline:
            if not await self.is_login_required(page):
                print("登录状态已检测通过，继续抓取...")
                return
            await asyncio.sleep(2)

        raise TimeoutError(f"等待登录超时（{timeout_seconds} 秒）")
    
    async def save_mhtml(self, page, output_path):
        """保存为 MHTML"""
        cdp = await self.context.new_cdp_session(page)
        result = await cdp.send('Page.captureSnapshot', {'format': 'mhtml'})
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(result['data'])
        return os.path.getsize(output_path)
    
    async def save_pdf(self, page, output_path):
        """保存为 PDF"""
        await page.emulate_media(media='screen')
        await page.pdf(
            path=output_path,
            format='A4',
            print_background=True,
            margin={'top': '1cm', 'right': '1cm', 'bottom': '1cm', 'left': '1cm'}
        )
        return os.path.getsize(output_path)
    
    async def save_png(self, page, output_path):
        """保存为 PNG 截图"""
        await page.screenshot(path=output_path, full_page=True)
        return os.path.getsize(output_path)
    
    async def save_html(self, page, output_path):
        """保存为 HTML"""
        html = await page.content()
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        return os.path.getsize(output_path)
    
    @staticmethod
    def generate_filename(url, extension, output_dir=None):
        """
        生成输出文件名
        
        Args:
            url: 目标 URL
            extension: 文件扩展名
            output_dir: 输出目录
        
        Returns:
            str: 输出文件路径
        """
        return generate_output_filename(url, extension, output_dir)


async def main():
    parser = argparse.ArgumentParser(
        description='网页抓取工具 - 支持 MHTML/PDF/PNG/HTML 格式',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python3 web-fetch.py "https://example.com" --format mhtml
  python3 web-fetch.py "https://example.com" --format mhtml pdf png
  python3 web-fetch.py "https://example.com" --output /path/to/dir --wait 15
        """
    )
    
    parser.add_argument('url', help='目标网页 URL')
    parser.add_argument('--format', nargs='+', default=['mhtml'],
                        choices=['mhtml', 'pdf', 'png', 'html'],
                        help='保存格式 (默认：mhtml)')
    parser.add_argument('--output', '-o', help='输出目录')
    parser.add_argument('--wait', '-w', type=int, default=10,
                        help='最短等待渲染时间（秒，默认：10）')
    parser.add_argument('--max-wait', type=int, default=30,
                        help='最长等待渲染时间（秒，默认：30）')
    parser.add_argument('--no-popup-cleanup', action='store_true',
                        help='禁用弹窗/浮层自动清理')
    parser.add_argument('--aggressive-popup-cleanup', action='store_true',
                        help='启用更激进的浮窗清理策略')
    parser.add_argument('--timeout', '-t', type=int, default=90,
                        help='页面加载超时（秒，默认：90）')
    parser.add_argument('--user-data', help='Chrome 用户数据目录')
    parser.add_argument('--profile-directory', default='Default',
                        help='Chrome Profile 目录名（默认：Default）')
    parser.add_argument('--chrome-path', help='系统 Chrome 可执行文件路径（可选）')
    parser.add_argument('--no-system-chrome', action='store_true',
                        help='不使用系统 Chrome，强制使用 Playwright 内置 Chromium')
    parser.add_argument('--visible', action='store_true',
                        help='使用可见浏览器模式')
    parser.add_argument('--login-timeout', type=int, default=300,
                        help='手工登录最大等待时间（秒，默认：300）')
    parser.add_argument('--force', action='store_true',
                        help='强制覆盖已存在文件')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Web Fetch - 网页抓取工具")
    print("=" * 80)
    print(f"\n目标 URL: {args.url}")
    print(f"保存格式：{', '.join(args.format)}")
    print(f"等待时间：{args.wait}秒")
    print(f"最大等待：{args.max_wait}秒")
    print(f"超时时间：{args.timeout}秒")
    print(f"强制覆盖：{'是' if args.force else '否'}")

    output_paths = {}
    pending_formats = []
    for fmt in args.format:
        output_path = WebFetcher.generate_filename(args.url, fmt, args.output)
        output_paths[fmt] = output_path
        if os.path.exists(output_path) and not args.force:
            print(f"跳过 {fmt.upper()}：文件已存在 {output_path}")
        else:
            if os.path.exists(output_path) and args.force:
                print(f"覆盖 {fmt.upper()}：文件已存在，将重新下载 {output_path}")
            pending_formats.append(fmt)

    if not pending_formats:
        print("\n所有目标文件均已存在，跳过下载。")
        return
    
    runtime_user_data = args.user_data or default_chrome_user_data_dir()
    fetcher = WebFetcher(
        user_data_dir=runtime_user_data,
        headless=not args.visible,
        chrome_path=args.chrome_path,
        use_system_chrome=not args.no_system_chrome,
        profile_directory=args.profile_directory
    )
    interactive_tmp_user_data = None
    
    try:
        print("\n启动浏览器...")
        await fetcher.start()

        print("访问页面...")
        try:
            page = await fetcher.fetch(
                args.url,
                wait_seconds=args.wait,
                timeout=args.timeout * 1000,
                login_timeout=args.login_timeout,
                max_wait_seconds=args.max_wait,
                cleanup_popups_enabled=not args.no_popup_cleanup,
                aggressive_popup_cleanup=args.aggressive_popup_cleanup
            )
        except LoginRequiredError:
            print("检测到登录页面，自动切换到可见浏览器进行手工登录...")
            await fetcher.close()
            # 可见登录阶段使用临时 profile，避免与原 profile 锁冲突
            interactive_tmp_user_data = tempfile.mkdtemp(prefix="web-fetch-login-")
            fetcher = WebFetcher(
                user_data_dir=interactive_tmp_user_data,
                headless=False,
                chrome_path=args.chrome_path,
                use_system_chrome=not args.no_system_chrome,
                profile_directory=args.profile_directory
            )
            await fetcher.start()
            page = await fetcher.fetch(
                args.url,
                wait_seconds=args.wait,
                timeout=max(args.timeout * 1000, args.login_timeout * 1000),
                login_timeout=args.login_timeout,
                max_wait_seconds=args.max_wait,
                cleanup_popups_enabled=not args.no_popup_cleanup,
                aggressive_popup_cleanup=args.aggressive_popup_cleanup
            )
        
        title = await page.title()
        print(f"页面标题：{title}")
        
        # 保存文件
        print("\n保存文件...")
        for fmt in pending_formats:
            try:
                output_path = output_paths[fmt]
                
                print(f"\n保存 {fmt.upper()}...")
                
                if fmt == 'mhtml':
                    size = await fetcher.save_mhtml(page, output_path)
                elif fmt == 'pdf':
                    size = await fetcher.save_pdf(page, output_path)
                elif fmt == 'png':
                    size = await fetcher.save_png(page, output_path)
                elif fmt == 'html':
                    size = await fetcher.save_html(page, output_path)
                
                size_mb = size / 1024 / 1024
                print(f"✓ 已保存：{output_path}")
                print(f"  大小：{size_mb:.2f} MB ({size:,} 字节)")
                
            except Exception as e:
                print(f"✗ {fmt.upper()} 保存失败：{e}")
        
        print("\n" + "=" * 80)
        print("抓取完成！")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n错误：{e}")
        import traceback
        traceback.print_exc()
        exit(1)
    finally:
        await fetcher.close()
        if interactive_tmp_user_data and os.path.exists(interactive_tmp_user_data):
            shutil.rmtree(interactive_tmp_user_data, ignore_errors=True)


if __name__ == "__main__":
    asyncio.run(main())
