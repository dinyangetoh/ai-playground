import os
import asyncio
import hashlib
from pyclbr import Class
import urllib.robotparser
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import nest_asyncio
import lxml
from pydantic import BaseModel
import re
from datetime import datetime
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright


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
    def __init__(
            self,
            user_agent: str = USER_AGENT
    ):
        self.user_agent = user_agent
        self.crawl_options = DEFAULT_OPTIONS.crawl_options
        self.concurrency_options = DEFAULT_OPTIONS.concurrency_options
        self.retry_options = DEFAULT_OPTIONS.retry_options

    async def run(self, url: str, options: CrawlOptions = DEFAULT_OPTIONS.crawl_options):
        """Run the crawler."""
        print(f"Crawling {url} starting...")
        # loop = asyncio.get_event_loop()
        # crawl_results = loop.run_until_complete(self.crawl_site(
        #     url, self.crawl_options.max_pages, self.concurrency_options.limit))
        crawl_results = await self.crawl_site(
            url, self.crawl_options.max_pages, self.concurrency_options.limit)
        print(f"Crawling {url} completed.")
        return crawl_results

    async def fetch_page(self, context, url: str, timeout_ms: int) -> dict:
        """Fetch a single page with Playwright; returns html + title. Skips non-HTML assets."""

        page = await context.new_page()
        try:
            response = await page.goto(url, wait_until="networkidle", timeout=timeout_ms)
            status = response.status if response else None
            content_type = response.headers.get(
                "content-type", "") if response else ""

            # Skip binary assets (PDFs, images, etc.) — no point rendering them
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
        """
        Parse all <a href> links from html, return only same-domain absolute URLs.
        Normalizes trailing slashes to prevent duplicate crawls.
        """
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
        for tag in soup(["script", "style", "noscript", "nav", "footer", "head"]):
            tag.decompose()
        headings = [re.sub(r"\s+", " ", h.get_text(strip=True))
                    for h in soup.find_all(["h1", "h2", "h3"])]
        body = soup.find("main") or soup.find("article") or soup.find("body")
        raw_text = body.get_text(separator=" ", strip=True) if body else ""
        text = re.sub(r"\s+", " ", raw_text).strip()
        return {
            "url": url,
            "title": title,
            "headings": headings,
            "content": text,
            "crawled_at": datetime.now().isoformat(),
        }

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
            return None  # can't read robots.txt — proceed without blocking

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

    async def crawl_site(
        self,
        start_url: str,
        max_pages: int,
        concurrency: int
    ) -> dict:
        """
        Concurrent BFS crawler using N worker coroutines pulling from a shared asyncio.Queue.
        Respects robots.txt, detects soft 404s, skips binary assets, retries on failure.
        """
        start_url = self.normalize_url(start_url)

        # --- robots.txt ---
        robots = self.load_robots(start_url)

        def is_allowed(url: str) -> bool:
            return robots is None or robots.can_fetch(USER_AGENT, url)

        visited: set[str] = set()
        queued_urls: set[str] = {start_url}
        queue: asyncio.Queue[str] = asyncio.Queue()
        pages: list[dict] = []
        invalid_urls: list[dict] = []
        valid_urls: list[dict] = []
        skipped_urls: list[str] = []
        soft_404s: list[str] = []
        lock = asyncio.Lock()

        # Soft 404 fingerprint — set after the start page is successfully fetched
        # single-element list so closure can mutate it
        home_fingerprint: list[str] = []

        await queue.put(start_url)
        crawl_start = datetime.now()

        async def worker(context):
            while True:
                url = await queue.get()
                try:
                    async with lock:
                        if url in visited:
                            continue
                        # Budget based on successfully extracted pages, not all visited URLs
                        if len(pages) >= max_pages:
                            continue
                        visited.add(url)

                    print(f"[{len(visited):>3}/{max_pages}] Crawling: {url}")

                    try:
                        result = await self.fetch_with_retry(context, url)
                        html, title, status = result["html"], result["title"], result["status"]

                        if result.get("skipped"):
                            async with lock:
                                skipped_urls.append(url)
                            continue

                        if status and status >= 400:
                            async with lock:
                                invalid_urls.append(
                                    {"url": url, "status": status})
                            continue

                        # CPU-bound parsing — outside lock
                        content = self.extract_content(html, url, title)
                        new_links = self.extract_links(html, url, start_url)
                        fp = self.content_fingerprint(content["content"])

                        # Soft 404 detection: pages that silently serve the home page
                        if url == start_url:
                            home_fingerprint.append(fp)
                        elif home_fingerprint and fp == home_fingerprint[0]:
                            async with lock:
                                soft_404s.append(url)
                            print(f"        SOFT 404 (matches home): {url}")
                            continue

                        async with lock:
                            valid_urls.append({"url": url, "status": status})
                            pages.append(content)
                            if len(pages) < max_pages:
                                # Filter robots.txt before queuing
                                to_enqueue = {
                                    link for link in new_links - visited - queued_urls
                                    if is_allowed(link)
                                }
                                queued_urls.update(to_enqueue)
                            else:
                                to_enqueue = set()

                        for link in to_enqueue:
                            await queue.put(link)

                        elapsed = (datetime.now() - crawl_start).seconds
                        print(
                            f"        title={title!r}  +{len(to_enqueue)} links  queue={queue.qsize()}  {elapsed}s")

                    except Exception as e:
                        print(f"        ERROR {url}: {e}")
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
            "crawled_pages": len(pages),
            "valid_urls": len(valid_urls),
            "invalid_urls": len(invalid_urls),
            "soft_404s": len(soft_404s),
            "skipped_assets": len(skipped_urls),
            "total_time_s": round(total_time, 1),
            "pages_per_second": round(len(pages) / total_time, 2) if total_time > 0 else 0,
        }
        print(f"\nDone. {summary}")
        return {
            "data": {
                "pages": pages,
                "valid_urls": valid_urls,
                "invalid_urls": invalid_urls,
                "soft_404s": soft_404s,
                "skipped_urls": skipped_urls,
            },
            "summary": summary,
        }
