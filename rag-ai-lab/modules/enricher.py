"""
File-based document enricher — decoupled from the crawler.

Reads saved .md files, calls an LLM to generate a summary and key_points,
then writes the results back into YAML frontmatter in-place.

Key design decisions:
- Operates on files, not in-memory page dicts → crawler is never blocked
- Atomic writes (tmp → rename) → no corruption on crash
- Exponential backoff → survives transient rate limits
- skip_if_exists → idempotent, safe to re-run at any time
"""

import asyncio
import json
import logging
import os
from pathlib import Path

import yaml
from openai import APIConnectionError, APITimeoutError, AsyncOpenAI, RateLimitError
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class EnricherOptions(BaseModel):
    model: str = "gpt-4o-mini"
    max_summary_words: int = 120
    batch_size: int = 5
    """Max concurrent LLM calls (semaphore limit)."""
    skip_if_exists: bool = True
    """Skip files that already have a non-empty summary in frontmatter."""
    max_retries: int = 3
    retry_base_delay: float = 1.0
    """Base delay in seconds for exponential backoff (delay = base * 2^attempt)."""


class Enricher:
    def __init__(self, options: EnricherOptions | None = None) -> None:
        self.options = options or EnricherOptions()
        self._client = AsyncOpenAI()
        self._semaphore = asyncio.Semaphore(self.options.batch_size)

    async def enrich_directory(self, path: str | Path) -> int:
        """
        Enrich all .md files under *path*. Returns count of files enriched.

        *path* must be a resolved directory — not a glob string.
        """
        base = Path(path)
        if not base.is_dir():
            raise ValueError(
                f"enrich_directory() requires a directory path, got: {path!r}"
            )

        files = list(base.rglob("*.md"))
        if not files:
            logger.info("No .md files found under %s", base)
            return 0

        logger.info("Found %d .md files under %s", len(files), base)
        results = await asyncio.gather(
            *[self.enrich_file(f) for f in files],
            return_exceptions=True,
        )

        enriched = sum(1 for r in results if r is True)
        errors = [r for r in results if isinstance(r, Exception)]
        if errors:
            logger.warning("%d file(s) failed enrichment: %s", len(errors), errors[0])
        logger.info("Enriched %d / %d files", enriched, len(files))
        return enriched

    async def enrich_file(self, file_path: str | Path) -> bool:
        """
        Enrich a single .md file.

        Returns True if the file was enriched, False if skipped.
        Raises on unrecoverable LLM errors after all retries are exhausted.
        """
        async with self._semaphore:
            path = Path(file_path)
            text = path.read_text(encoding="utf-8")
            frontmatter, body = self._parse_frontmatter(text)

            if frontmatter is None:
                frontmatter = {}

            if self.options.skip_if_exists and frontmatter.get("summary"):
                logger.debug("Skipping (already enriched): %s", path.name)
                return False

            url_or_source = (
                frontmatter.get("url")
                or frontmatter.get("file_name")
                or path.name
                or "uploaded document"
            )
            title = frontmatter.get("title") or path.stem
            content_preview = body[:3000]

            summary, key_points = await asyncio.gather(
                self._generate_summary(title, url_or_source, content_preview),
                self._generate_key_points(title, body[:2000]),
            )

            frontmatter["summary"] = summary
            frontmatter["key_points"] = key_points

            self._atomic_write(path, self._build_markdown(frontmatter, body))
            logger.info("Enriched: %s", path.name)
            return True

    # ------------------------------------------------------------------
    # LLM helpers
    # ------------------------------------------------------------------

    async def _generate_summary(self, title: str, url_or_source: str, content: str) -> str:
        prompt = (
            f"Summarize this web page in 2-3 sentences. Focus on the main topic and key facts. "
            f"Be concise and stay under {self.options.max_summary_words} words.\n\n"
            f"Title: {title}\nURL: {url_or_source}\n\nContent:\n{content}"
        )
        return await self._call_llm_with_retry(prompt)

    async def _generate_key_points(self, title: str, content: str) -> list[str]:
        prompt = (
            "Extract 3-5 key topics or entities from this page as a JSON list of strings. "
            "Return ONLY valid JSON with no explanation or markdown fences. "
            'Example: ["Machine Learning", "Python 3.11", "API Reference"]\n\n'
            f"Title: {title}\nContent:\n{content}"
        )
        raw = await self._call_llm_with_retry(prompt)
        try:
            result = json.loads(raw)
            if isinstance(result, list):
                return [str(item) for item in result]
        except (json.JSONDecodeError, ValueError):
            logger.warning("Could not parse key_points JSON, got: %r", raw[:120])
        return []

    async def _call_llm_with_retry(self, prompt: str) -> str:
        last_exc: Exception | None = None
        for attempt in range(self.options.max_retries):
            try:
                response = await self._client.chat.completions.create(
                    model=self.options.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=300,
                    temperature=0,
                )
                return response.choices[0].message.content.strip()
            except (RateLimitError, APIConnectionError, APITimeoutError) as exc:
                last_exc = exc
                if attempt < self.options.max_retries - 1:
                    delay = self.options.retry_base_delay * (2**attempt)
                    logger.warning(
                        "LLM call failed (attempt %d/%d): %s. Retrying in %.1fs",
                        attempt + 1,
                        self.options.max_retries,
                        exc,
                        delay,
                    )
                    await asyncio.sleep(delay)
        raise last_exc  # type: ignore[misc]

    # ------------------------------------------------------------------
    # File I/O helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_frontmatter(text: str) -> tuple[dict | None, str]:
        """Parse YAML frontmatter from markdown text. Returns (frontmatter_dict, body)."""
        if not text.startswith("---\n"):
            return None, text
        end = text.find("\n---", 4)
        if end == -1:
            return None, text
        try:
            parsed = yaml.safe_load(text[4:end]) or {}
        except yaml.YAMLError:
            parsed = {}
        body = text[end + 4 :].lstrip("\n")
        return (parsed if isinstance(parsed, dict) else {}), body

    @staticmethod
    def _build_markdown(frontmatter: dict, body: str) -> str:
        yaml_block = yaml.dump(
            frontmatter,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        ).rstrip()
        return f"---\n{yaml_block}\n---\n\n{body}"

    @staticmethod
    def _atomic_write(path: Path, content: str) -> None:
        """Write content to a temp file then rename — prevents corruption on crash."""
        tmp = path.with_suffix(".md.tmp")
        try:
            tmp.write_text(content, encoding="utf-8")
            os.rename(tmp, path)
        except Exception:
            if tmp.exists():
                tmp.unlink(missing_ok=True)
            raise
