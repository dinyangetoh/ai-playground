"""
Knowledge base storage handler.

Provides a clean interface for saving and locating crawled pages and
uploaded files on the local filesystem. All path-returning methods return
pathlib.Path objects — callers use .rglob("*.md") rather than glob strings.

Designed for local-first use with an S3-extensible interface later.
"""

import shutil
from pathlib import Path

from pydantic import BaseModel

from modules.crawler import page_to_markdown, url_to_filename


class StorageOptions(BaseModel):
    base_path: str = "knowledge_base"
    crawled_subdir: str = "crawled"
    uploads_subdir: str = "uploads"


class KnowledgeBaseStorage:
    def __init__(self, options: StorageOptions | None = None) -> None:
        self.options = options or StorageOptions()
        self._base = Path(self.options.base_path)
        self._crawled = self._base / self.options.crawled_subdir
        self._uploads = self._base / self.options.uploads_subdir

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def save_crawled_page(self, page: dict, domain: str) -> Path:
        """
        Serialize a crawled page dict to markdown and save it under
        knowledge_base/crawled/{domain}/{filename}.md.

        Returns the path of the written file.
        """
        out_dir = self._crawled / domain
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / (url_to_filename(page["url"]) + ".md")
        path.write_text(page_to_markdown(page), encoding="utf-8")
        return path

    def save_uploaded_file(self, source_path: str | Path, markdown_content: str) -> Path:
        """
        Save an uploaded file's converted markdown to knowledge_base/uploads/{stem}.md
        and copy the original to knowledge_base/uploads/originals/.

        Returns the path of the written markdown file.
        """
        source = Path(source_path)
        originals_dir = self._uploads / "originals"
        originals_dir.mkdir(parents=True, exist_ok=True)
        self._uploads.mkdir(parents=True, exist_ok=True)

        shutil.copy2(source, originals_dir / source.name)
        md_path = self._uploads / (source.stem + ".md")
        md_path.write_text(markdown_content, encoding="utf-8")
        return md_path

    # ------------------------------------------------------------------
    # Path accessors — always return Path objects, never glob strings
    # ------------------------------------------------------------------

    def get_crawled_path(self, domain: str | None = None) -> Path:
        """
        Return the crawled directory for a specific domain, or the top-level
        crawled/ directory if no domain is given.

        Use path.rglob("*.md") to iterate files.
        """
        if domain:
            return self._crawled / domain
        return self._crawled

    def get_uploads_path(self) -> Path:
        """Return the uploads/ directory path."""
        return self._uploads

    # ------------------------------------------------------------------
    # Listing helpers
    # ------------------------------------------------------------------

    def list_crawled_domains(self) -> list[str]:
        """Return names of all domain subdirectories under crawled/."""
        if not self._crawled.exists():
            return []
        return sorted(d.name for d in self._crawled.iterdir() if d.is_dir())

    def list_uploaded_files(self) -> list[str]:
        """Return filenames of all .md files directly under uploads/."""
        if not self._uploads.exists():
            return []
        return sorted(f.name for f in self._uploads.glob("*.md"))
