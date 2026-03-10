"""
src/retriever/vector_store.py
==============================
ChromaDB-backed vector store for persistent document storage and retrieval.

Wraps ChromaDB with a clean interface so the rest of the pipeline is
not coupled to any specific vector DB implementation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb
import numpy as np
from chromadb.config import Settings

from src.retriever.document_loader import Document
from src.utils import get_logger, load_config

logger = get_logger(__name__)


class VectorStore:
    """
    Persistent vector store backed by ChromaDB.

    Stores document embeddings and metadata, and supports similarity
    search by cosine distance.

    Parameters
    ----------
    collection_name : str | None
        Name for the ChromaDB collection. Defaults to config value.
    persist_dir : str | Path | None
        Directory for persistent storage. Defaults to env/config value.

    Examples
    --------
    >>> store = VectorStore()
    >>> store.add_documents(docs, embeddings)
    >>> results = store.query(query_embedding, top_k=5)
    """

    def __init__(
        self,
        collection_name: str | None = None,
        persist_dir: str | Path | None = None,
    ) -> None:
        cfg = load_config()
        self.collection_name = collection_name or cfg["retriever"]["collection_name"]

        import os
        persist_path = str(
            persist_dir
            or os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
        )
        Path(persist_path).mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=persist_path,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            f"VectorStore ready. Collection '{self.collection_name}' "
            f"has {self._collection.count()} documents."
        )

    # ── Write operations ──────────────────────────────────────────────────────

    def add_documents(
        self,
        documents: list[Document],
        embeddings: np.ndarray,
        batch_size: int = 256,
    ) -> None:
        """
        Add a list of documents with their embeddings to the store.

        Skips documents that are already present (by doc_id) to allow
        safe re-runs without duplicating data.

        Parameters
        ----------
        documents : list[Document]
            Documents to index.
        embeddings : np.ndarray
            Float32 array of shape (len(documents), embedding_dim).
        batch_size : int
            Number of documents to upsert per batch.
        """
        if len(documents) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(documents)} documents but {len(embeddings)} embeddings."
            )

        ids = [doc.doc_id for doc in documents]
        texts = [doc.content for doc in documents]
        metadatas = [
            {
                **doc.metadata,
                "source": doc.source,
                "is_adversarial": str(doc.is_adversarial),
            }
            for doc in documents
        ]
        embedding_list = embeddings.tolist()

        # Batch upserts to avoid memory issues with large collections
        for start in range(0, len(ids), batch_size):
            end = start + batch_size
            self._collection.upsert(
                ids=ids[start:end],
                documents=texts[start:end],
                metadatas=metadatas[start:end],
                embeddings=embedding_list[start:end],
            )

        logger.info(f"Upserted {len(documents)} documents into '{self.collection_name}'.")

    def add_adversarial_documents(
        self,
        documents: list[Document],
        embeddings: np.ndarray,
    ) -> None:
        """
        Inject adversarial documents into the store with is_adversarial=True.

        Parameters
        ----------
        documents : list[Document]
            Adversarial documents to inject.
        embeddings : np.ndarray
            Corresponding embeddings.
        """
        for doc in documents:
            doc.is_adversarial = True
        self.add_documents(documents, embeddings)
        logger.info(f"Injected {len(documents)} adversarial documents.")

    def clear(self) -> None:
        """Delete all documents from the collection."""
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"Cleared collection '{self.collection_name}'.")

    # ── Read operations ───────────────────────────────────────────────────────

    def query(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Retrieve the top-k most similar documents by cosine similarity.

        Parameters
        ----------
        query_embedding : np.ndarray
            1-D float32 embedding for the query.
        top_k : int
            Number of documents to return.

        Returns
        -------
        list[dict]
            Each dict contains ``doc_id``, ``content``, ``source``,
            ``metadata``, and ``distance`` keys.
        """
        results = self._collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(top_k, self._collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        for doc_id, content, metadata, distance in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            hits.append({
                "doc_id": doc_id,
                "content": content,
                "source": metadata.get("source", ""),
                "metadata": metadata,
                "distance": distance,
                "similarity": 1 - distance,
            })

        return hits

    def count(self) -> int:
        """Return the number of documents currently in the store."""
        return self._collection.count()
