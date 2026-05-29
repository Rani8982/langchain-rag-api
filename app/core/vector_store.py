"""
VectorStoreManager — owns the ChromaDB instance and HuggingFace embeddings.

Design decisions
----------------
- Singleton via module-level instance (safe for single-process FastAPI).
- HuggingFace sentence-transformers run **locally** (no API call for embeddings).
- LLM calls go to HuggingFace Inference API (serverless, free tier).
- MMR retrieval by default for diverse, non-redundant results.
- All config driven by Settings — no magic strings in this file.
"""

import threading
from typing import List, Optional

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)

_lock = threading.Lock()


class VectorStoreManager:
    """Thread-safe wrapper around ChromaDB + HuggingFace embeddings."""

    _instance: Optional["VectorStoreManager"] = None
    _vectorstore: Optional[Chroma] = None
    _embeddings: Optional[HuggingFaceEmbeddings] = None

    def __new__(cls) -> "VectorStoreManager":
        with _lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
        return cls._instance

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def initialize(self) -> None:
        """Load (or create) the ChromaDB collection and embeddings model."""
        if self._vectorstore is not None:
            logger.debug("Vector store already initialized — skipping.")
            return

        logger.info(
            "Loading HuggingFace embedding model",
            model=settings.EMBEDDING_MODEL,
        )
        self._embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},          # change to "cuda" if GPU available
            encode_kwargs={"normalize_embeddings": True},
        )

        logger.info(
            "Connecting to ChromaDB",
            persist_dir=settings.CHROMA_PERSIST_DIR,
            collection=settings.CHROMA_COLLECTION_NAME,
        )
        self._vectorstore = Chroma(
            collection_name=settings.CHROMA_COLLECTION_NAME,
            embedding_function=self._embeddings,
            persist_directory=settings.CHROMA_PERSIST_DIR,
        )
        count = self._vectorstore._collection.count()
        logger.info("Vector store ready", document_count=count)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _ensure_initialized(self) -> None:
        if self._vectorstore is None:
            raise RuntimeError("VectorStoreManager not initialized. Call initialize() first.")

    # ── Ingest ────────────────────────────────────────────────────────────────

    def add_documents(self, documents: List[Document]) -> List[str]:
        """Embed and store documents. Returns list of assigned IDs."""
        self._ensure_initialized()
        ids = self._vectorstore.add_documents(documents)
        logger.info("Documents added", count=len(ids))
        return ids

    def delete_documents(self, ids: List[str]) -> None:
        """Remove documents by ID."""
        self._ensure_initialized()
        self._vectorstore.delete(ids=ids)
        logger.info("Documents deleted", count=len(ids))

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def similarity_search(
        self,
        query: str,
        k: int = settings.RETRIEVAL_K,
        filter: Optional[dict] = None,
    ) -> List[Document]:
        self._ensure_initialized()
        return self._vectorstore.similarity_search(query, k=k, filter=filter)

    def similarity_search_with_scores(
        self,
        query: str,
        k: int = settings.RETRIEVAL_K,
        filter: Optional[dict] = None,
    ) -> List[tuple[Document, float]]:
        self._ensure_initialized()
        raw = self._vectorstore.similarity_search_with_score(query, k=k, filter=filter)
        # Convert L2 distance → cosine-style similarity score [0, 1]
        return [(doc, round(1 / (1 + score), 4)) for doc, score in raw]

    def get_retriever(
        self,
        search_type: str = settings.RETRIEVAL_SEARCH_TYPE,
        k: int = settings.RETRIEVAL_K,
        fetch_k: int = settings.RETRIEVAL_FETCH_K,
        filter: Optional[dict] = None,
    ) -> BaseRetriever:
        self._ensure_initialized()
        search_kwargs: dict = {"k": k}
        if search_type == "mmr":
            search_kwargs["fetch_k"] = fetch_k
        if filter:
            search_kwargs["filter"] = filter
        return self._vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs,
        )

    # ── Stats ─────────────────────────────────────────────────────────────────

    def count(self) -> int:
        self._ensure_initialized()
        return self._vectorstore._collection.count()

    def collection_name(self) -> str:
        return settings.CHROMA_COLLECTION_NAME
