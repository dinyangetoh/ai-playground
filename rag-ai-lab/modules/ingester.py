import glob
from datetime import date, datetime
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import MarkdownTextSplitter
from pydantic import BaseModel, Field


load_dotenv(override=True)


class ChunkingOptions(BaseModel):
    chunk_size: int = 1200
    chunk_overlap: int = 150
    min_chunk_chars: int = 60


class VectorStoreOptions(BaseModel):
    persist_directory: str = str(Path("db/vector_db"))
    delete_existing_collection: bool = True


class EmbeddingOptions(BaseModel):
    model: str = "text-embedding-3-small"


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
        self.embeddings = OpenAIEmbeddings(model=options.embedding.model)

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

    def ingest(self, documents: list[Document] | None = None) -> tuple[Chroma, IngestResult]:
        docs = documents if documents is not None else self.fetch_documents()
        processed = self.preprocess_documents(docs)
        chunks = self.create_chunks(processed)

        persist_directory = self.options.vector_store.persist_directory
        if self.options.vector_store.delete_existing_collection and Path(
            persist_directory
        ).exists():
            Chroma(
                persist_directory=persist_directory,
                embedding_function=self.embeddings,
            ).delete_collection()

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=persist_directory,
        )

        collection = vectorstore._collection
        count = collection.count()
        sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
        dimensions = len(sample_embedding)

        return vectorstore, IngestResult(
            vector_count=count,
            embedding_dimensions=dimensions,
        )

    def similarity_search(self, query: str, k: int = 4) -> list[Document]:
        vectorstore = Chroma(
            embedding_function=self.embeddings,
            persist_directory=self.options.vector_store.persist_directory,
        )
        return vectorstore.similarity_search(query, k=k)
