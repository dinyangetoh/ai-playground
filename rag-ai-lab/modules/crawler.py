import os
import asyncio
import hashlib
import re
import urllib.robotparser
from datetime import datetime
from typing import AsyncGenerator
from urllib.parse import urljoin, urlparse

import lxml
import nest_asyncio
from bs4 import BeautifulSoup, NavigableString
from dotenv import load_dotenv
from pathlib import Path
from playwright.async_api import async_playwright
from pydantic import BaseModel


_HEADING_RE = re.compile(r"^h[1-6]$")
_CONTAINER_TAGS = frozenset(
    {"div", "section", "article", "main", "aside", "ul", "ol", "dl", "blockquote", "form", "fieldset"}
)
_PLACEHOLDER_TABLE_CELLS = frozenset({"#", "Course Code", "Course Title", "Unit", ""})

USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

load_dotenv(override=True)

if not os.environ.get("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY is not set")


class CrawlOptions(BaseModel):
    bs_parser: str = "lxml"
    timeout: int = 30000
    max_depth: int = 3
    max_pages: int = 100
    max_retries: int = 3


class ConcurrencyOptions(BaseModel):
    limit: int = 5
    delay: float = 0.5


class RetryOptions(BaseModel):
    max_retries: int = 3
    backoff: float = 2


class Options(BaseModel):
    crawl_options: CrawlOptions
    concurrency_options: ConcurrencyOptions
    retry_options: RetryOptions


DEFAULT_OPTIONS: Options = Options(
    crawl_options=CrawlOptions(
        timeout=30000,
        max_depth=3,
        max_pages=100,
        max_retries=3,
    ),
    concurrency_options=ConcurrencyOptions(
        limit=5,
        delay=0.5,
    ),
    retry_options=RetryOptions(
        max_retries=3,
        backoff=2,
    ),
)


class Crawler:
    def __init__(self, user_agent: str = USER_AGENT):
        self.user_agent = user_agent
        self.crawl_options = DEFAULT_OPTIONS.crawl_options
        self.concurrency_options = DEFAULT_OPTIONS.concurrency_options
        self.retry_options = DEFAULT_OPTIONS.retry_options
        self.last_crawl: dict | None = None

    async def run(self, url: str, options: CrawlOptions = DEFAULT_OPTIONS.crawl_options) -> dict:
        """Collect all pages into memory and return at once. For large sites use run_generator()."""
        pages: list[dict] = []
        async for page in self.run_generator(url, options):
            pages.append(page)
        assert self.last_crawl is not None
        return {
            "data": {"pages": pages, **self.last_crawl["data"]},
            "summary": self.last_crawl["summary"],
        }

    async def run_generator(
        self,
        url: str,
        options: CrawlOptions = DEFAULT_OPTIONS.crawl_options,
    ) -> AsyncGenerator[dict, None]:
        """Yield each page the moment it is crawled — no buffering."""
        page_queue: asyncio.Queue[dict | None] = asyncio.Queue()

        async def _run() -> None:
            self.last_crawl = await self._crawl_site(url, options, page_queue)
            await page_queue.put(None)  # sentinel

        task = asyncio.create_task(_run())
        try:
            while True:
                page = await page_queue.get()
                if page is None:
                    break
                yield page
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            else:
                await task

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def fetch_page(self, context, url: str, timeout_ms: int) -> dict:
        """Fetch a single page with Playwright; returns html + title. Skips non-HTML assets."""
        page = await context.new_page()
        try:
            response = await page.goto(url, wait_until="networkidle", timeout=timeout_ms)
            status = response.status if response else None
            content_type = response.headers.get(
                "content-type", "") if response else ""

            if not content_type.startswith("text/html"):
                return {"html": "", "title": "", "status": status, "skipped": True}

            html = await page.content()
            title = await page.title()
            return {"html": html, "title": title, "status": status, "skipped": False}
        finally:
            await page.close()

    def normalize_url(self, url: str) -> str:
        """Strip trailing slash (except root), drop fragment."""
        p = urlparse(url)
        path = p.path.rstrip("/") or "/"
        return p._replace(path=path, fragment="").geturl()

    def extract_links(self, html: str, current_url: str, base_url: str) -> set[str]:
        """Parse all <a href> links; return only same-domain absolute URLs."""
        base_domain = urlparse(base_url).netloc
        soup = BeautifulSoup(html, self.crawl_options.bs_parser)
        links = set()
        for tag in soup.find_all("a", href=True):
            href = tag["href"].strip()
            if not href or href.startswith("mailto:") or href.startswith("tel:"):
                continue
            full_url = urljoin(current_url, href)
            parsed = urlparse(full_url)
            if parsed.netloc == base_domain and parsed.scheme in ("http", "https"):
                links.add(self.normalize_url(full_url))
        return links

    def extract_content(self, html: str, url: str, title: str) -> dict:
        """Extract readable text content from a page's HTML."""
        soup = BeautifulSoup(html, self.crawl_options.bs_parser)
        for tag in soup(["script", "style", "noscript", "nav", "footer", "head", "header"]):
            tag.decompose()

        # URL path metadata
        parsed_url = urlparse(url)
        path_segments = [s for s in parsed_url.path.strip("/").split("/") if s]
        breadcrumb = (
            " > ".join(s.replace("-", " ").title() for s in path_segments)
            if path_segments
            else parsed_url.netloc
        )

        headings = [
            re.sub(r"\s+", " ", h.get_text(strip=True))
            for h in soup.find_all(["h1", "h2", "h3"])
        ]
        body = soup.find("main") or soup.find("article") or soup.find("body")
        raw_text = body.get_text(separator=" ", strip=True) if body else ""
        text = re.sub(r"\s+", " ", raw_text).strip()
        sections = self._extract_sections(body) if body else []

        return {
            "url": url,
            "title": title,
            "path_segments": path_segments,
            "breadcrumb": breadcrumb,
            "headings": headings,
            "sections": sections,
            "content": text,
            "crawled_at": datetime.now().isoformat(),
        }

    def _extract_sections(self, body) -> list[dict]:
        """Walk the parsed body element, building content sections keyed by heading."""
        sections: list[dict] = []
        current_heading: str | None = None
        current_level: int = 0
        current_parts: list[str] = []

        def flush() -> None:
            nonlocal current_heading, current_level, current_parts
            content = re.sub(r"\s+", " ", " ".join(current_parts)).strip()
            content = re.sub(
                r"(#?\s*Course Code\s+Course Title\s+Unit\s*)+",
                "[Course details unavailable]",
                content,
            ).strip()
            if content or current_heading:
                sections.append(
                    {"heading": current_heading, "level": current_level, "content": content}
                )
            current_heading = None
            current_level = 0
            current_parts = []

        def walk(node) -> None:
            nonlocal current_heading, current_level, current_parts
            if isinstance(node, NavigableString):
                text = str(node).strip()
                if text:
                    current_parts.append(text)
                return
            if not getattr(node, "name", None):
                return
            name: str = node.name
            if _HEADING_RE.match(name):
                flush()
                current_heading = re.sub(r"\s+", " ", node.get_text(strip=True)).strip()
                current_level = int(name[1])
                current_parts = []
            elif name == "table":
                cells = [c.get_text(strip=True) for c in node.find_all("td")]
                meaningful = [c for c in cells if c not in _PLACEHOLDER_TABLE_CELLS]
                if meaningful:
                    current_parts.append(" | ".join(meaningful[:30]))
            elif name in ("p", "li", "dt", "dd"):
                text = re.sub(r"\s+", " ", node.get_text(separator=" ", strip=True)).strip()
                if text:
                    current_parts.append(text)
            elif name in _CONTAINER_TAGS:
                for child in node.children:
                    walk(child)

        for child in body.children:
            walk(child)
        flush()
        return sections

    def content_fingerprint(self, content: str) -> str:
        """MD5 of first 500 chars — used to detect soft 404s."""
        return hashlib.md5(content[:500].encode()).hexdigest()

    def load_robots(self, start_url: str) -> urllib.robotparser.RobotFileParser | None:
        """Fetch and parse robots.txt for the given site. Returns None on failure."""
        parsed = urlparse(start_url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        rp = urllib.robotparser.RobotFileParser()
        rp.set_url(robots_url)
        try:
            rp.read()
            return rp
        except Exception:
            return None

    async def fetch_with_retry(self, context, url: str) -> dict:
        """Fetch a page with exponential backoff retry on failure."""
        last_exc = None
        for attempt in range(self.retry_options.max_retries):
            try:
                return await self.fetch_page(context, url, self.crawl_options.timeout)
            except Exception as e:
                last_exc = e
                if attempt < self.retry_options.max_retries - 1:
                    await asyncio.sleep(self.retry_options.backoff ** attempt)
        raise last_exc

    async def _crawl_site(
        self,
        start_url: str,
        options: CrawlOptions,
        page_queue: asyncio.Queue,
    ) -> dict:
        """Concurrent BFS crawler. Puts each extracted page onto page_queue."""
        start_url = self.normalize_url(start_url)
        max_pages = options.max_pages
        concurrency = self.concurrency_options.limit

        robots = self.load_robots(start_url)

        def is_allowed(url: str) -> bool:
            return robots is None or robots.can_fetch(USER_AGENT, url)

        visited: set[str] = set()
        queued_urls: set[str] = {start_url}
        queue: asyncio.Queue[str] = asyncio.Queue()
        page_count = 0
        invalid_urls: list[dict] = []
        valid_urls: list[dict] = []
        skipped_urls: list[str] = []
        soft_404s: list[str] = []
        lock = asyncio.Lock()
        home_fingerprint: list[str] = []

        await queue.put(start_url)
        crawl_start = datetime.now()

        async def worker(context):
            nonlocal page_count
            while True:
                url = await queue.get()
                try:
                    async with lock:
                        if url in visited:
                            continue
                        if page_count >= max_pages:
                            continue
                        visited.add(url)

                    try:
                        result = await self.fetch_with_retry(context, url)
                        html, title, status = result["html"], result["title"], result["status"]

                        if result.get("skipped"):
                            async with lock:
                                skipped_urls.append(url)
                            print(
                                f"[{len(visited):>3}/{max_pages}] SKIP (non-HTML)  {url}")
                            continue

                        if status and status >= 400:
                            async with lock:
                                invalid_urls.append(
                                    {"url": url, "status": status})
                            print(
                                f"[{len(visited):>3}/{max_pages}] {status}  {url}")
                            continue

                        content = self.extract_content(html, url, title)
                        new_links = self.extract_links(html, url, start_url)
                        fp = self.content_fingerprint(content["content"])

                        if url == start_url:
                            home_fingerprint.append(fp)
                        elif home_fingerprint and fp == home_fingerprint[0]:
                            async with lock:
                                soft_404s.append(url)
                            print(
                                f"[{len(visited):>3}/{max_pages}] SOFT 404  {url}")
                            continue

                        async with lock:
                            page_count += 1
                            valid_urls.append({"url": url, "status": status})
                            if page_count < max_pages:
                                to_enqueue = {
                                    link for link in new_links - visited - queued_urls
                                    if is_allowed(link)
                                }
                                queued_urls.update(to_enqueue)
                            else:
                                to_enqueue = set()

                        for link in to_enqueue:
                            await queue.put(link)

                        await page_queue.put(content)

                        elapsed = (datetime.now() - crawl_start).seconds
                        print(
                            f"[{page_count:>3}/{max_pages}] {title!r}"
                            f"  +{len(to_enqueue)} links  q={queue.qsize()}  {elapsed}s"
                        )

                    except Exception as e:
                        print(
                            f"[{len(visited):>3}/{max_pages}] ERROR  {url}  {e}")
                        async with lock:
                            invalid_urls.append({"url": url, "error": str(e)})

                    finally:
                        await asyncio.sleep(self.concurrency_options.delay)

                finally:
                    queue.task_done()

        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=True,
                args=["--no-sandbox", "--disable-dev-shm-usage"],
            )
            context = await browser.new_context(user_agent=USER_AGENT)
            workers = [asyncio.create_task(worker(context))
                       for _ in range(concurrency)]

            await queue.join()

            for w in workers:
                w.cancel()
            await asyncio.gather(*workers, return_exceptions=True)
            await browser.close()

        total_time = (datetime.now() - crawl_start).total_seconds()
        summary = {
            "crawled_pages": page_count,
            "valid_urls": len(valid_urls),
            "invalid_urls": len(invalid_urls),
            "soft_404s": len(soft_404s),
            "skipped_assets": len(skipped_urls),
            "total_time_s": round(total_time, 1),
            "pages_per_second": round(page_count / total_time, 2) if total_time > 0 else 0,
        }
        print(f"\nDone. {summary}")
        return {
            "data": {
                "valid_urls": valid_urls,
                "invalid_urls": invalid_urls,
                "soft_404s": soft_404s,
                "skipped_urls": skipped_urls,
            },
            "summary": summary,
        }


# ---------------------------------------------------------------------------
# Markdown serialisation helpers (module-level, usable outside the class)
# ---------------------------------------------------------------------------

def _fmt_yaml_str(s: str) -> str:
    """Wrap a string in double quotes if it contains YAML-special characters."""
    specials = (":", "#", "[", "]", "{", "}", "&", "*", "?", "|", "<", ">", "=", "!", "%", "@", "`")
    if any(c in s for c in specials) or s.startswith((" ", "-")):
        return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'
    return s


def page_to_markdown(page: dict) -> str:
    """Convert a crawled page dict to structured markdown with YAML frontmatter."""
    lines = ["---"]
    lines.append(f"url: {page.get('url', '')}")
    lines.append(f"title: {_fmt_yaml_str(page.get('title', ''))}")
    segments = page.get("path_segments", [])
    if segments:
        lines.append("path_segments:")
        for seg in segments:
            lines.append(f"  - {seg}")
    else:
        lines.append("path_segments: []")
    lines.append(f"breadcrumb: {_fmt_yaml_str(page.get('breadcrumb', ''))}")
    lines.append(f"crawled_at: {page.get('crawled_at', '')}")
    lines.append("---")
    lines.append("")

    title = page.get("title", "").strip()
    if title:
        lines.append(f"# {title}")
        lines.append("")

    for section in page.get("sections", []):
        heading = section.get("heading")
        level = section.get("level", 2)
        content = section.get("content", "").strip()
        if heading:
            lines.append(f"{'#' * max(2, level)} {heading}")
            lines.append("")
        if content:
            lines.append(content)
            lines.append("")

    # Fallback: flat content if section extraction produced nothing
    if not page.get("sections"):
        content = page.get("content", "").strip()
        if content:
            lines.append(content)
            lines.append("")

    return "\n".join(lines)


def url_to_filename(url: str) -> str:
    """Derive a safe filename (no extension) from a URL path."""
    path = urlparse(url).path.strip("/")
    if not path:
        return "index"
    return re.sub(r"[^a-z0-9_-]", "_", path.replace("/", "_").lower()).strip("_")


def save_pages(pages: list[dict], output_dir: str) -> int:
    """Save crawled pages as structured markdown files. Returns count saved."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    saved = 0
    for page in pages:
        filepath = out / (url_to_filename(page["url"]) + ".md")
        filepath.write_text(page_to_markdown(page), encoding="utf-8")
        saved += 1
    return saved
