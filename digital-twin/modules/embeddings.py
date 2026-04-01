"""
Provider-agnostic embedding factory.

Usage:
    from modules.embeddings import get_embeddings
    embeddings = get_embeddings(provider="openai", model="text-embedding-3-small")
    embeddings = get_embeddings(provider="ollama", model="nomic-embed-text")
"""

from langchain_core.embeddings import Embeddings


def get_embeddings(provider: str, model: str) -> Embeddings:
    """
    Return a LangChain Embeddings instance for the given provider and model.

    Supported providers:
        - "openai"   : OpenAIEmbeddings (requires OPENAI_API_KEY env var)
        - "ollama"   : OllamaEmbeddings (requires a running Ollama server)

    Raises:
        ValueError : if an unknown provider is requested.
        ImportError: if the required LangChain integration package is not installed.
    """
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings  # type: ignore[import]
        return OpenAIEmbeddings(model=model)

    if provider == "ollama":
        from langchain_ollama import OllamaEmbeddings  # type: ignore[import]
        return OllamaEmbeddings(model=model)

    raise ValueError(
        f"Unknown embedding provider: {provider!r}. "
        "Supported values: 'openai', 'ollama'."
    )
