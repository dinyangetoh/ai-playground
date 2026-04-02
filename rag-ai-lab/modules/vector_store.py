"""
Vector store adapter — Adapter + Factory pattern.

The ingester depends only on VectorStoreAdapter (a Protocol).
Use create_vector_store() to instantiate the right backend.

Supported backends: 'chroma', 'qdrant'
"""

import logging
from typing import Protocol, runtime_checkable

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from pydantic import BaseModel

logger = logging.getLogger(__name__)


@runtime_checkable
class VectorStoreAdapter(Protocol):
    """API contract every vector store backend must satisfy."""

    def upsert(self, chunks: list[Document]) -> None:
        """Insert or replace chunks, keyed on url/source metadata."""
        ...

    def query(self, query: str, k: int = 4, filter: dict | None = None) -> list[Document]:
        """Semantic similarity search. Returns top-k documents."""
        ...

    def has_documents(self) -> bool:
        """Return True if the collection contains at least one vector."""
        ...

    def delete_collection(self) -> None:
        """Permanently delete the entire collection."""
        ...

    def get_collection_name(self) -> str:
        """Return the collection/index name."""
        ...


# ---------------------------------------------------------------------------
# Chroma adapter
# ---------------------------------------------------------------------------

class ChromaAdapterOptions(BaseModel):
    collection_name: str = "rag_collection"
    persist_directory: str = "db/vector_db"
    batch_size: int = 500


class ChromaAdapter:
    def __init__(self, options: ChromaAdapterOptions, embeddings: Embeddings) -> None:
        from langchain_chroma import Chroma

        self._options = options
        self._embeddings = embeddings
        self._store = Chroma(
            collection_name=options.collection_name,
            persist_directory=options.persist_directory,
            embedding_function=embeddings,
        )

    def upsert(self, chunks: list[Document]) -> None:
        """Group chunks by url/source, delete stale vectors, then add fresh ones."""
        by_key: dict[str, list[Document]] = {}
        for chunk in chunks:
            key = chunk.metadata.get("url") or chunk.metadata.get("source", "")
            by_key.setdefault(key, []).append(chunk)

        for key, key_chunks in by_key.items():
            if key:
                existing = self._store.get(where={"url": key})
                if existing.get("ids"):
                    self._store.delete(ids=existing["ids"])
                    logger.info(
                        "Deleted %d stale vectors for: %s",
                        len(existing["ids"]), key,
                    )
            self._add_in_batches(key_chunks)

    def _add_in_batches(self, chunks: list[Document]) -> None:
        batch_size = self._options.batch_size
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            self._store.add_documents(batch)
            logger.info(
                "Embedded batch %d–%d of %d chunks",
                start, start + len(batch) - 1, len(chunks),
            )

    def query(self, query: str, k: int = 4, filter: dict | None = None) -> list[Document]:
        return self._store.similarity_search(query, k=k, filter=filter)

    def has_documents(self) -> bool:
        return len(self._store.get()["ids"]) > 0

    def delete_collection(self) -> None:
        self._store.delete_collection()
        logger.warning("Deleted Chroma collection: %s", self._options.collection_name)

    def get_collection_name(self) -> str:
        return self._options.collection_name


# ---------------------------------------------------------------------------
# Qdrant adapter
# ---------------------------------------------------------------------------

class QdrantAdapterOptions(BaseModel):
    collection_name: str = "rag_collection"
    url: str = "http://localhost:6333"
    api_key: str | None = None
    batch_size: int = 500


class QdrantAdapter:
    def __init__(self, options: QdrantAdapterOptions, embeddings: Embeddings) -> None:
        try:
            from langchain_qdrant import QdrantVectorStore
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
        except ImportError as exc:
            raise ImportError(
                "QdrantAdapter requires qdrant-client and langchain-qdrant. "
                "Install with: uv add qdrant-client langchain-qdrant"
            ) from exc

        self._options = options
        self._embeddings = embeddings
        self._client = QdrantClient(url=options.url, api_key=options.api_key)

        # Probe embedding dimensions via a single test call
        dim = len(embeddings.embed_query("dim-probe"))

        existing_names = [c.name for c in self._client.get_collections().collections]
        if options.collection_name not in existing_names:
            self._client.create_collection(
                collection_name=options.collection_name,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
            logger.info(
                "Created Qdrant collection '%s' (%d dims)", options.collection_name, dim
            )

        self._store = QdrantVectorStore(
            client=self._client,
            collection_name=options.collection_name,
            embedding=embeddings,
        )

    def upsert(self, chunks: list[Document]) -> None:
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        by_key: dict[str, list[Document]] = {}
        for chunk in chunks:
            key = chunk.metadata.get("url") or chunk.metadata.get("source", "")
            by_key.setdefault(key, []).append(chunk)

        for key, key_chunks in by_key.items():
            if key:
                self._client.delete(
                    collection_name=self._options.collection_name,
                    points_selector=Filter(
                        must=[
                            FieldCondition(
                                key="metadata.url", match=MatchValue(value=key)
                            )
                        ]
                    ),
                )
                logger.info("Deleted stale Qdrant points for: %s", key)
            self._add_in_batches(key_chunks)

    def _add_in_batches(self, chunks: list[Document]) -> None:
        batch_size = self._options.batch_size
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            self._store.add_documents(batch)
            logger.info(
                "Embedded batch %d–%d of %d chunks",
                start, start + len(batch) - 1, len(chunks),
            )

    def query(self, query: str, k: int = 4, filter: dict | None = None) -> list[Document]:
        return self._store.similarity_search(query, k=k)

    def has_documents(self) -> bool:
        return self._client.count(collection_name=self._options.collection_name).count > 0

    def delete_collection(self) -> None:
        self._client.delete_collection(self._options.collection_name)
        logger.warning("Deleted Qdrant collection: %s", self._options.collection_name)

    def get_collection_name(self) -> str:
        return self._options.collection_name


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_vector_store(
    backend: str,
    embeddings: Embeddings,
    **kwargs,
) -> VectorStoreAdapter:
    """
    Instantiate the right adapter for the given backend.

    Usage:
        store = create_vector_store("chroma", embeddings, persist_directory=".chroma")
        store = create_vector_store("qdrant", embeddings, url="http://localhost:6333")
    """
    if backend == "chroma":
        return ChromaAdapter(ChromaAdapterOptions(**kwargs), embeddings)
    if backend == "qdrant":
        return QdrantAdapter(QdrantAdapterOptions(**kwargs), embeddings)
    raise ValueError(
        f"Unknown vector store backend: {backend!r}. Supported: 'chroma', 'qdrant'."
    )
