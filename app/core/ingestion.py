"""
Ingestion service — handles text splitting and document preparation
before handing off to the vector store.
"""

import hashlib
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import settings
from app.core.logger import get_logger
from app.core.vector_store import VectorStoreManager

logger = get_logger(__name__)


def _make_doc_id(content: str, metadata: dict) -> str:
    """Deterministic ID so re-ingesting the same doc is idempotent."""
    payload = content + str(sorted(metadata.items()))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


class IngestionService:
    def __init__(self) -> None:
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        self.vs_manager = VectorStoreManager()

    # ── Public interface ──────────────────────────────────────────────────────

    def ingest_texts(
        self,
        texts: List[str],
        metadatas: List[dict] | None = None,
    ) -> dict:
        """
        Split raw texts → chunks → embed → store.
        Returns summary dict with chunk count and assigned IDs.
        """
        if metadatas is None:
            metadatas = [{} for _ in texts]

        if len(texts) != len(metadatas):
            raise ValueError("texts and metadatas must have the same length.")

        # Build raw documents
        raw_docs = [
            Document(page_content=text, metadata=meta)
            for text, meta in zip(texts, metadatas)
        ]

        # Split
        chunks = self.splitter.split_documents(raw_docs)
        logger.info("Text split into chunks", raw_docs=len(raw_docs), chunks=len(chunks))

        # Assign deterministic IDs
        ids = [_make_doc_id(c.page_content, c.metadata) for c in chunks]

        # Store
        stored_ids = self.vs_manager.add_documents(chunks)

        return {
            "raw_document_count": len(raw_docs),
            "chunk_count": len(chunks),
            "stored_ids": stored_ids,
        }

    def ingest_file(self, file_path: str | Path, extra_metadata: dict | None = None) -> dict:
        """
        Read a plain-text file and ingest it.
        For PDFs / Word docs, swap in the relevant LangChain loader.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        text = path.read_text(encoding="utf-8")
        metadata = {"source": str(path), "filename": path.name}
        if extra_metadata:
            metadata.update(extra_metadata)

        return self.ingest_texts([text], [metadata])
