import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import MarkdownTextSplitter
from pydantic import BaseModel, Field

from modules.embeddings import get_embeddings
from modules.vector_store import VectorStoreAdapter, create_vector_store, ChromaAdapterOptions

load_dotenv(override=True)

logger = logging.getLogger(__name__)


class ChunkingOptions(BaseModel):
    chunk_size: int = 1200
    chunk_overlap: int = 150
    min_chunk_chars: int = 60


class EmbeddingOptions(BaseModel):
    provider: str = "openai"
    """Embedding provider. Supported: 'openai', 'ollama'."""
    model: str = "text-embedding-3-small"


class MarkdownFrontmatter(BaseModel):
    url: str = ""
    title: str = ""
    breadcrumb: str = ""
    path_segments: list[str] = Field(default_factory=list)
    crawled_at: datetime | str | None = ""
    summary: str = ""
    key_points: list[str] = Field(default_factory=list)


class IngestOptions(BaseModel):
    knowledge_base_path: str
    """Directory (or glob) to scan for .md files."""
    markdown_glob: str = "**/*.md"
    enrich: bool = False
    """If True, run the enricher on enrich_path before ingesting."""
    enrich_path: str | None = None
    """Directory to enrich. Defaults to knowledge_base_path if not set.
    Must be a resolved directory — not a glob string."""
    chunking: ChunkingOptions = Field(default_factory=ChunkingOptions)
    embedding: EmbeddingOptions = Field(default_factory=EmbeddingOptions)


class IngestResult(BaseModel):
    vector_count: int
    embedding_dimensions: int


def load_pdf(file_path: str | Path) -> list[Document]:
    """Load a PDF file into LangChain Documents (one per page)."""
    from langchain_community.document_loaders import PyPDFLoader

    loader = PyPDFLoader(str(file_path))
    pages = loader.load()
    for doc in pages:
        doc.metadata["source_type"] = "pdf"
        doc.metadata["file_name"] = Path(file_path).name
    return pages


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
    def __init__(self, vector_store: VectorStoreAdapter, options: IngestOptions):
        self.options = options
        self.vector_store = vector_store
        self.embeddings: Embeddings = get_embeddings(
            provider=options.embedding.provider,
            model=options.embedding.model,
        )

    def fetch_documents(self) -> list[Document]:
        """Load all .md files from knowledge_base_path using pathlib."""
        base = Path(self.options.knowledge_base_path)
        documents: list[Document] = []

        # Support both a plain directory and a glob pattern
        if base.is_dir():
            paths = list(base.rglob(self.options.markdown_glob.lstrip("*/")))
        else:
            # Treat as a glob string (e.g. "knowledge_base/**")
            import glob as _glob
            paths = [Path(p) for p in _glob.glob(str(base), recursive=True) if Path(p).is_file()]

        for path in paths:
            try:
                text = path.read_text(encoding="utf-8")
                doc = Document(
                    page_content=text,
                    metadata={"source": str(path), "doc_type": path.parent.name},
                )
                documents.append(doc)
            except Exception as exc:
                logger.warning("Could not read %s: %s", path, exc)

        logger.info("Loaded %d documents from %s", len(documents), self.options.knowledge_base_path)
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
                        "summary": frontmatter.summary,
                        "topics": ", ".join(frontmatter.key_points),
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

            summary = chunk.metadata.get("summary", "")
            breadcrumb = (
                chunk.metadata.get("breadcrumb")
                or chunk.metadata.get("page_title")
                or chunk.metadata.get("doc_type", "")
            )

            prefix = ""
            if summary:
                prefix += f"[SUMMARY: {summary}]\n\n"
            if breadcrumb:
                prefix += f"[{breadcrumb}]\n\n"

            if prefix:
                chunk.page_content = prefix + chunk.page_content

            enriched.append(chunk)

        return enriched

    async def ingest(self, documents: list[Document] | None = None) -> IngestResult:
        # Optional enrichment preprocessing
        if self.options.enrich:
            from modules.enricher import Enricher

            enrich_dir = Path(self.options.enrich_path or self.options.knowledge_base_path)
            if not enrich_dir.is_dir():
                raise ValueError(
                    f"enrich_path must be a directory, got: {enrich_dir!r}. "
                    "Set enrich_path explicitly when knowledge_base_path is a glob."
                )
            logger.info("Running enricher on %s", enrich_dir)
            await Enricher().enrich_directory(enrich_dir)

        docs = documents if documents is not None else self.fetch_documents()
        processed = self.preprocess_documents(docs)
        chunks = self.create_chunks(processed)

        self.vector_store.upsert(chunks)

        # Gather stats
        sample_docs = self.vector_store.query("dimension probe", k=1)
        if sample_docs:
            sample_embedding = self.embeddings.embed_query(sample_docs[0].page_content[:100])
            dimensions = len(sample_embedding)
        else:
            dimensions = 0

        # Count via has_documents guard
        all_chunks_count = len(chunks)
        logger.info("Ingestion complete: ~%d chunks upserted", all_chunks_count)

        return IngestResult(vector_count=all_chunks_count, embedding_dimensions=dimensions)

    def similarity_search(self, query: str, k: int = 4) -> list[Document]:
        return self.vector_store.query(query, k=k)


# ---------------------------------------------------------------------------
# Convenience factory — mirrors old notebook usage with sensible defaults
# ---------------------------------------------------------------------------

def create_ingester(
    knowledge_base_path: str,
    backend: str = "chroma",
    persist_directory: str = "db/vector_db",
    collection_name: str = "rag_collection",
    **ingest_kwargs,
) -> "Ingester":
    """
    One-liner ingester construction for notebooks.

    Usage:
        ingester = create_ingester("data/topfaith.edu.ng", backend="chroma")
        result = await ingester.ingest()
    """
    embedding_opts = EmbeddingOptions()
    embeddings = get_embeddings(
        provider=embedding_opts.provider,
        model=embedding_opts.model,
    )
    store = create_vector_store(
        backend,
        embeddings,
        collection_name=collection_name,
        persist_directory=persist_directory,
    )
    options = IngestOptions(knowledge_base_path=knowledge_base_path, **ingest_kwargs)
    return Ingester(vector_store=store, options=options)
