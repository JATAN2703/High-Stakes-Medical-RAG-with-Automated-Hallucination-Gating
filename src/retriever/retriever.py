"""
src/retriever/retriever.py
===========================
Main Retriever class implementing hybrid BM25 + dense retrieval.

The hybrid approach scores each candidate document using a weighted
combination of BM25 (lexical) and cosine similarity (semantic) scores,
then returns the top-k by combined score.
"""

from __future__ import annotations

import textwrap
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from src.retriever.document_loader import Document
from src.retriever.embedder import Embedder
from src.retriever.vector_store import VectorStore
from src.utils import get_logger, load_config

logger = get_logger(__name__)


class Retriever:
    """
    Hybrid retriever combining dense vector search with BM25 lexical search.

    Dense retrieval captures semantic meaning while BM25 excels at exact
    drug name and terminology matching — both are critical in pharmacology.

    Parameters
    ----------
    embedder : Embedder | None
        Pre-constructed Embedder. If None, one is created from config.
    vector_store : VectorStore | None
        Pre-constructed VectorStore. If None, one is created from config.
    strategy : str | None
        One of ``"dense"``, ``"bm25"``, or ``"hybrid"``. Defaults to config.

    Examples
    --------
    >>> retriever = Retriever()
    >>> retriever.index(documents)
    >>> results = retriever.retrieve("What are warfarin drug interactions?")
    """

    def __init__(
        self,
        embedder: Embedder | None = None,
        vector_store: VectorStore | None = None,
        strategy: str | None = None,
    ) -> None:
        cfg = load_config()
        self._cfg = cfg["retriever"]

        self.embedder = embedder or Embedder()
        self.vector_store = vector_store or VectorStore()
        self.strategy = strategy or self._cfg["strategy"]
        self.top_k = self._cfg["top_k"]
        self.alpha = self._cfg["hybrid_alpha"]

        # BM25 index is built in-memory after indexing
        self._bm25: BM25Okapi | None = None
        self._indexed_docs: list[Document] = []

        logger.info(f"Retriever initialised (strategy={self.strategy}, top_k={self.top_k})")

    # ── Indexing ──────────────────────────────────────────────────────────────

    def index(self, documents: list[Document], show_progress: bool = True) -> None:
        """
        Embed and index a list of documents.

        This method (a) builds the BM25 index in memory and (b) upserts
        embeddings into ChromaDB. Call this once before any retrieve() calls.

        Parameters
        ----------
        documents : list[Document]
            Documents to index.
        show_progress : bool
            Whether to display embedding progress bar.
        """
        if not documents:
            logger.warning("index() called with empty document list.")
            return

        logger.info(f"Indexing {len(documents)} documents...")
        texts = [doc.content for doc in documents]

        # Build BM25 index
        tokenised = [text.lower().split() for text in texts]
        self._bm25 = BM25Okapi(tokenised)
        self._indexed_docs = list(documents)

        # Embed and persist to ChromaDB
        embeddings = self.embedder.embed(texts, show_progress=show_progress)
        self.vector_store.add_documents(documents, embeddings)

        logger.info(f"Indexing complete. {len(documents)} documents available for retrieval.")

    def inject_adversarial(self, documents: list[Document]) -> None:
        """
        Inject adversarial documents into the existing index.

        Adversarial documents are flagged with ``is_adversarial=True``
        in their metadata for post-hoc analysis.

        Parameters
        ----------
        documents : list[Document]
            Adversarial documents to inject.
        """
        if not documents:
            return
        texts = [doc.content for doc in documents]
        embeddings = self.embedder.embed(texts)
        self.vector_store.add_adversarial_documents(documents, embeddings)

        # Extend BM25 index
        if self._bm25 is not None:
            extra_tokenised = [t.lower().split() for t in texts]
            all_tokenised = [d.content.lower().split() for d in self._indexed_docs] + extra_tokenised
            self._bm25 = BM25Okapi(all_tokenised)
            self._indexed_docs.extend(documents)

        logger.info(f"Injected {len(documents)} adversarial documents.")

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        max_context_tokens: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Retrieve the most relevant documents for a query.

        Parameters
        ----------
        query : str
            The user's question or search string.
        top_k : int | None
            Number of documents to return. Defaults to config value.
        max_context_tokens : int | None
            If set, truncate results so total token count stays under limit.
            Used in context-window sensitivity experiments.

        Returns
        -------
        list[dict]
            Ranked list of result dicts, each with ``doc_id``, ``content``,
            ``source``, ``metadata``, and ``score`` keys.
        """
        k = top_k or self.top_k

        if self.strategy == "dense":
            results = self._dense_retrieve(query, k)
        elif self.strategy == "bm25":
            results = self._bm25_retrieve(query, k)
        else:
            results = self._hybrid_retrieve(query, k)

        if max_context_tokens:
            results = self._truncate_to_token_budget(results, max_context_tokens)

        return results

    def _dense_retrieve(self, query: str, top_k: int) -> list[dict[str, Any]]:
        """Dense-only retrieval via ChromaDB cosine similarity."""
        query_embedding = self.embedder.embed_single(query)
        return self.vector_store.query(query_embedding, top_k=top_k)

    def _bm25_retrieve(self, query: str, top_k: int) -> list[dict[str, Any]]:
        """BM25-only lexical retrieval."""
        if self._bm25 is None or not self._indexed_docs:
            logger.warning("BM25 index not built. Call index() first.")
            return []

        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] == 0:
                continue
            doc = self._indexed_docs[idx]
            results.append({
                "doc_id": doc.doc_id,
                "content": doc.content,
                "source": doc.source,
                "metadata": doc.metadata,
                "score": float(scores[idx]),
            })
        return results

    def _hybrid_retrieve(self, query: str, top_k: int) -> list[dict[str, Any]]:
        """
        Hybrid retrieval: weighted sum of normalised BM25 and dense scores.

        Score = alpha * dense_similarity + (1 - alpha) * bm25_normalised
        """
        dense_results = self._dense_retrieve(query, top_k * 2)
        bm25_results = self._bm25_retrieve(query, top_k * 2)

        # Build score maps keyed by doc_id
        dense_map: dict[str, float] = {r["doc_id"]: r.get("similarity", 0.0) for r in dense_results}
        bm25_map: dict[str, float] = {r["doc_id"]: r["score"] for r in bm25_results}

        # Normalise BM25 scores to [0, 1]
        if bm25_map:
            max_bm25 = max(bm25_map.values()) or 1.0
            bm25_map = {k: v / max_bm25 for k, v in bm25_map.items()}

        # Combine all unique doc_ids
        all_ids = set(dense_map) | set(bm25_map)
        combined: list[tuple[str, float]] = []
        for doc_id in all_ids:
            score = (
                self.alpha * dense_map.get(doc_id, 0.0)
                + (1 - self.alpha) * bm25_map.get(doc_id, 0.0)
            )
            combined.append((doc_id, score))

        combined.sort(key=lambda x: x[1], reverse=True)
        top_ids = [doc_id for doc_id, _ in combined[:top_k]]
        score_map = dict(combined)

        # Fetch full document objects for top results
        # Prefer dense results (already have content), fall back to BM25
        content_map = {r["doc_id"]: r for r in dense_results + bm25_results}

        results = []
        for doc_id in top_ids:
            if doc_id not in content_map:
                continue
            r = content_map[doc_id]
            results.append({
                **r,
                "score": score_map[doc_id],
            })
        return results

    @staticmethod
    def _truncate_to_token_budget(
        results: list[dict[str, Any]],
        max_tokens: int,
    ) -> list[dict[str, Any]]:
        """
        Truncate result list so total character count is within token budget.

        Uses a rough approximation of 4 characters per token.

        Parameters
        ----------
        results : list[dict]
            Ranked retrieval results.
        max_tokens : int
            Maximum token budget.

        Returns
        -------
        list[dict]
            Truncated result list.
        """
        char_budget = max_tokens * 4
        total = 0
        truncated = []
        for r in results:
            length = len(r.get("content", ""))
            if total + length > char_budget:
                break
            truncated.append(r)
            total += length
        return truncated

    # ── Formatting ────────────────────────────────────────────────────────────

    @staticmethod
    def format_context(results: list[dict[str, Any]]) -> str:
        """
        Format retrieval results into a numbered context string for the LLM.

        Parameters
        ----------
        results : list[dict]
            Retrieved documents.

        Returns
        -------
        str
            Formatted context block with source labels.
        """
        parts = []
        for i, r in enumerate(results, start=1):
            source = r.get("source", "Unknown")
            content = textwrap.fill(r.get("content", ""), width=120)
            parts.append(f"[Source {i}] {source}\n{content}")
        return "\n\n".join(parts)
