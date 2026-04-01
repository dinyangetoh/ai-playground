import glob
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import MarkdownTextSplitter
from pydantic import BaseModel, Field

from modules.embeddings import get_embeddings

load_dotenv(override=True)

logger = logging.getLogger(__name__)


class ChunkingOptions(BaseModel):
    chunk_size: int = 1200
    chunk_overlap: int = 150
    min_chunk_chars: int = 60


class VectorStoreOptions(BaseModel):
    persist_directory: str = str(Path("db/vector_db"))
    delete_existing_collection: bool = False
    """When False (default), performs an incremental upsert keyed on URL.
    Set to True only for a full rebuild — this will wipe the existing collection."""


class EmbeddingOptions(BaseModel):
    provider: str = "openai"
    """Embedding provider. Supported: 'openai', 'ollama'."""
    model: str = "text-embedding-3-small"
    embedding_batch_size: int = 500
    """Max number of chunks sent to the embedding API in a single batch.
    Lower this if you hit rate limits on large corpora."""


class MarkdownFrontmatter(BaseModel):
    url: str = ""
    title: str = ""
    breadcrumb: str = ""
    path_segments: list[str] = Field(default_factory=list)
    crawled_at: datetime | str | None = ""


class IngestOptions(BaseModel):
    knowledge_base_path: str
    markdown_glob: str = "**/*.md"
    chunking: ChunkingOptions = Field(default_factory=ChunkingOptions)
    vector_store: VectorStoreOptions = Field(default_factory=VectorStoreOptions)
    embedding: EmbeddingOptions = Field(default_factory=EmbeddingOptions)


class IngestResult(BaseModel):
    vector_count: int
    embedding_dimensions: int


def _chroma_friendly_scalar(value: Any) -> str | int | float | bool | list[Any] | None:
    if value is None:
        return ""
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return value


def _parse_frontmatter(text: str) -> tuple[MarkdownFrontmatter | None, str]:
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
    if not isinstance(parsed, dict):
        return None, body

    return MarkdownFrontmatter.model_validate(parsed), body


def _fix_dangling_headings(chunks: list[Document]) -> list[Document]:
    """Merge heading-only chunks into the following content chunk so that
    headings always appear alongside their content in the vector store."""
    result: list[Document] = []
    pending_prefix = ""

    for chunk in chunks:
        text = chunk.page_content
        non_blank = [line.strip() for line in text.splitlines() if line.strip()]

        if non_blank and all(line.startswith("#") for line in non_blank):
            pending_prefix += text.strip() + "\n\n"
            continue

        if pending_prefix:
            chunk.page_content = pending_prefix + text
            text = chunk.page_content
            pending_prefix = ""

        lines = text.split("\n")
        i = len(lines) - 1
        while i >= 0 and (not lines[i].strip() or lines[i].strip().startswith("#")):
            i -= 1

        if i < len(lines) - 1:
            has_trailing_heading = any(
                line.strip().startswith("#") for line in lines[i + 1 :] if line.strip()
            )
            if has_trailing_heading:
                chunk.page_content = "\n".join(lines[: i + 1]).rstrip()

        if chunk.page_content.strip():
            result.append(chunk)

    return result


class Ingester:
    def __init__(self, options: IngestOptions):
        self.options = options
        self.embeddings: Embeddings = get_embeddings(
            provider=options.embedding.provider,
            model=options.embedding.model,
        )
        self._vectorstore: Chroma | None = None

    def fetch_documents(self) -> list[Document]:
        folders = glob.glob(str(Path(self.options.knowledge_base_path)))
        documents: list[Document] = []

        for folder in folders:
            doc_type = Path(folder).name
            loader = DirectoryLoader(
                folder,
                glob=self.options.markdown_glob,
                loader_cls=TextLoader,
                loader_kwargs={"encoding": "utf-8"},
            )
            folder_docs = loader.load()
            for doc in folder_docs:
                doc.metadata["doc_type"] = doc_type
                documents.append(doc)

        return documents

    def preprocess_documents(self, documents: list[Document]) -> list[Document]:
        for doc in documents:
            frontmatter, body = _parse_frontmatter(doc.page_content)
            if frontmatter:
                doc.metadata.update(
                    {
                        "url": frontmatter.url,
                        "page_title": frontmatter.title,
                        "breadcrumb": frontmatter.breadcrumb,
                        "path_segments": ",".join(frontmatter.path_segments),
                        "crawled_at": _chroma_friendly_scalar(frontmatter.crawled_at),
                    }
                )
            doc.page_content = body or doc.page_content
        return documents

    def create_chunks(self, documents: list[Document]) -> list[Document]:
        splitter = MarkdownTextSplitter(
            chunk_size=self.options.chunking.chunk_size,
            chunk_overlap=self.options.chunking.chunk_overlap,
        )
        chunks = splitter.split_documents(documents)
        chunks = _fix_dangling_headings(chunks)

        enriched: list[Document] = []
        for chunk in chunks:
            if len(chunk.page_content.strip()) < self.options.chunking.min_chunk_chars:
                continue

            breadcrumb = (
                chunk.metadata.get("breadcrumb")
                or chunk.metadata.get("page_title")
                or chunk.metadata.get("doc_type", "")
            )
            if breadcrumb:
                chunk.page_content = f"[{breadcrumb}]\n\n{chunk.page_content}"

            enriched.append(chunk)

        return enriched

    def _add_in_batches(self, vectorstore: Chroma, chunks: list[Document]) -> None:
        """Add *chunks* to *vectorstore* in batches to avoid API rate limits."""
        batch_size = self.options.embedding.embedding_batch_size
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            vectorstore.add_documents(batch)
            logger.info(
                "Embedded batch %d-%d of %d chunks",
                start, start + len(batch) - 1, len(chunks),
            )

    def vector_store_has_documents(self) -> bool:
        persist_directory = self.options.vector_store.persist_directory
        path = Path(persist_directory)
        if not path.exists():
            return False
        try:
            store = Chroma(
                persist_directory=persist_directory,
                embedding_function=self.embeddings,
            )
            got = store.get(limit=1)
            return bool(got.get("ids"))
        except Exception:
            logger.warning(
                "Could not read vector store at %s; will run ingest",
                persist_directory,
                exc_info=True,
            )
            return False

    def ingest(self, documents: list[Document] | None = None) -> tuple[Chroma, IngestResult]:
        docs = documents if documents is not None else self.fetch_documents()
        processed = self.preprocess_documents(docs)
        chunks = self.create_chunks(processed)

        persist_directory = self.options.vector_store.persist_directory
        batch_size = self.options.embedding.embedding_batch_size

        # ---- Full rebuild ----
        if self.options.vector_store.delete_existing_collection and Path(
            persist_directory
        ).exists():
            logger.warning(
                "delete_existing_collection=True — wiping vector store at %s",
                persist_directory,
            )
            Chroma(
                persist_directory=persist_directory,
                embedding_function=self.embeddings,
            ).delete_collection()

        if (
            not self.options.vector_store.delete_existing_collection
            and Path(persist_directory).exists()
        ):
            # ---- Incremental upsert keyed on URL or source path ----
            vectorstore = Chroma(
                persist_directory=persist_directory,
                embedding_function=self.embeddings,
            )
            # Group incoming chunks by URL (preferred) or source file path (fallback).
            # Without this fallback, PDF-sourced files (no url field) accumulate
            # duplicate chunks on every startup since the url-only key is empty.
            by_key: dict[str, list[Document]] = {}
            for chunk in chunks:
                key = chunk.metadata.get("url") or chunk.metadata.get("source", "")
                by_key.setdefault(key, []).append(chunk)

            for key, key_chunks in by_key.items():
                if key:
                    # Use whichever field was used as the key for filtering
                    field = "url" if key_chunks[0].metadata.get("url") else "source"
                    # Remove stale vectors before adding updated ones
                    existing = vectorstore.get(where={field: key})
                    if existing.get("ids"):
                        vectorstore.delete(ids=existing["ids"])
                        logger.info(
                            "Deleted %d stale vectors for %s: %s",
                            len(existing["ids"]), field, key,
                        )
                self._add_in_batches(vectorstore, key_chunks)
        else:
            # ---- First-time ingestion (collection does not yet exist) ----
            first_batch = chunks[:batch_size]
            rest = chunks[batch_size:]
            vectorstore = Chroma.from_documents(
                documents=first_batch,
                embedding=self.embeddings,
                persist_directory=persist_directory,
            )
            if rest:
                self._add_in_batches(vectorstore, rest)

        # Cache for subsequent similarity_search calls
        self._vectorstore = vectorstore

        # Gather stats via public API
        all_ids = vectorstore.get()
        count = len(all_ids["ids"])
        sample = vectorstore.get(limit=1, include=["embeddings"])
        emb = sample.get("embeddings")
        if emb is None or len(emb) == 0:
            dimensions = 0
        else:
            dimensions = len(emb[0])

        logger.info(
            "Ingestion complete: %d vectors, %d dimensions", count, dimensions
        )
        return vectorstore, IngestResult(
            vector_count=count,
            embedding_dimensions=dimensions,
        )

    def similarity_search(self, query: str, k: int = 4) -> list[Document]:
        """Search the vector store. Reuses cached client from the last ingest() call
        rather than constructing a new Chroma instance on every query."""
        if self._vectorstore is None:
            self._vectorstore = Chroma(
                embedding_function=self.embeddings,
                persist_directory=self.options.vector_store.persist_directory,
            )
        return self._vectorstore.similarity_search(query, k=k)
