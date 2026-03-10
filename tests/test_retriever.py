"""
tests/test_retriever.py
========================
Unit tests for the Retriever, Embedder, VectorStore, and DocumentLoader modules.
"""

from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from src.retriever.document_loader import Document, DailyMedLoader
from src.retriever.embedder import Embedder
from src.retriever.retriever import Retriever
from src.retriever.vector_store import VectorStore


# ── Document tests ────────────────────────────────────────────────────────────

class TestDocument:
    def test_valid_document_creation(self):
        doc = Document(doc_id="doc_001", content="Drug A causes drowsiness.", source="Drug A")
        assert doc.doc_id == "doc_001"
        assert doc.is_adversarial is False

    def test_empty_content_raises_error(self):
        with pytest.raises(ValueError, match="empty content"):
            Document(doc_id="doc_001", content="   ", source="Drug A")

    def test_adversarial_flag(self):
        doc = Document(doc_id="adv_001", content="Some content.", source="Drug A", is_adversarial=True)
        assert doc.is_adversarial is True

    def test_metadata_defaults_to_empty_dict(self):
        doc = Document(doc_id="doc_001", content="Content.", source="Drug A")
        assert doc.metadata == {}


# ── Embedder tests ────────────────────────────────────────────────────────────

class TestEmbedder:
    @pytest.fixture
    def mock_embedder(self):
        """Create an Embedder with a mocked SentenceTransformer."""
        with patch("src.retriever.embedder.SentenceTransformer") as MockST:
            mock_model = MagicMock()
            mock_model.encode.return_value = np.random.rand(3, 384).astype(np.float32)
            MockST.return_value = mock_model
            embedder = Embedder(model_name="mock-model")
            embedder.model = mock_model
            return embedder

    def test_embed_returns_correct_shape(self, mock_embedder):
        texts = ["Text one.", "Text two.", "Text three."]
        mock_embedder.model.encode.return_value = np.random.rand(3, 384).astype(np.float32)
        vectors = mock_embedder.embed(texts)
        assert vectors.shape == (3, 384)

    def test_embed_empty_list_raises_error(self, mock_embedder):
        with pytest.raises(ValueError, match="empty list"):
            mock_embedder.embed([])

    def test_embed_single_returns_1d(self, mock_embedder):
        mock_embedder.model.encode.return_value = np.random.rand(1, 384).astype(np.float32)
        vector = mock_embedder.embed_single("Single text.")
        assert vector.ndim == 1
        assert vector.shape == (384,)

    def test_output_is_float32(self, mock_embedder):
        mock_embedder.model.encode.return_value = np.random.rand(2, 384).astype(np.float64)
        vectors = mock_embedder.embed(["Text A", "Text B"])
        assert vectors.dtype == np.float32


# ── VectorStore tests ─────────────────────────────────────────────────────────

class TestVectorStore:
    @pytest.fixture
    def store(self, tmp_path):
        """Create a temporary VectorStore for testing."""
        return VectorStore(
            collection_name="test_collection",
            persist_dir=str(tmp_path / "chroma"),
        )

    @pytest.fixture
    def sample_docs(self):
        return [
            Document(doc_id=f"doc_{i}", content=f"Drug {i} content.", source=f"Drug {i}")
            for i in range(5)
        ]

    @pytest.fixture
    def sample_embeddings(self):
        return np.random.rand(5, 384).astype(np.float32)

    def test_initial_count_is_zero(self, store):
        assert store.count() == 0

    def test_add_documents_increases_count(self, store, sample_docs, sample_embeddings):
        store.add_documents(sample_docs, sample_embeddings)
        assert store.count() == 5

    def test_mismatched_docs_embeddings_raises_error(self, store, sample_docs):
        wrong_embeddings = np.random.rand(3, 384).astype(np.float32)
        with pytest.raises(ValueError, match="Mismatch"):
            store.add_documents(sample_docs, wrong_embeddings)

    def test_query_returns_results(self, store, sample_docs, sample_embeddings):
        store.add_documents(sample_docs, sample_embeddings)
        query_embedding = np.random.rand(384).astype(np.float32)
        results = store.query(query_embedding, top_k=3)
        assert len(results) == 3

    def test_query_results_have_required_keys(self, store, sample_docs, sample_embeddings):
        store.add_documents(sample_docs, sample_embeddings)
        query_embedding = np.random.rand(384).astype(np.float32)
        results = store.query(query_embedding, top_k=1)
        assert "doc_id" in results[0]
        assert "content" in results[0]
        assert "distance" in results[0]

    def test_clear_resets_count(self, store, sample_docs, sample_embeddings):
        store.add_documents(sample_docs, sample_embeddings)
        store.clear()
        assert store.count() == 0

    def test_upsert_does_not_duplicate(self, store, sample_docs, sample_embeddings):
        store.add_documents(sample_docs, sample_embeddings)
        store.add_documents(sample_docs, sample_embeddings)  # same docs again
        assert store.count() == 5  # not 10


# ── Retriever tests ───────────────────────────────────────────────────────────

class TestRetriever:
    @pytest.fixture
    def docs(self):
        return [
            Document(
                doc_id=f"doc_{i}",
                content=f"Warfarin drug interaction {i}: avoid concomitant use with NSAIDs.",
                source="Warfarin Label",
                metadata={"drug_name": "warfarin", "section": "drug_interactions"},
            )
            for i in range(10)
        ]

    @pytest.fixture
    def mock_retriever(self, tmp_path, docs):
        """Create a Retriever with mocked embedder."""
        with patch("src.retriever.retriever.Embedder") as MockEmb, \
             patch("src.retriever.retriever.VectorStore") as MockVS:

            mock_emb = MagicMock()
            mock_emb.embed.return_value = np.random.rand(len(docs), 384).astype(np.float32)
            mock_emb.embed_single.return_value = np.random.rand(384).astype(np.float32)
            MockEmb.return_value = mock_emb

            mock_vs = MagicMock()
            mock_vs.count.return_value = len(docs)
            mock_vs.query.return_value = [
                {"doc_id": f"doc_{i}", "content": docs[i].content,
                 "source": docs[i].source, "metadata": {}, "distance": 0.1, "similarity": 0.9}
                for i in range(5)
            ]
            MockVS.return_value = mock_vs

            retriever = Retriever(strategy="dense")
            retriever.index(docs)
            return retriever, docs

    def test_index_builds_bm25(self, mock_retriever):
        retriever, docs = mock_retriever
        assert retriever._bm25 is not None
        assert len(retriever._indexed_docs) == len(docs)

    def test_dense_retrieve_returns_results(self, mock_retriever):
        retriever, _ = mock_retriever
        results = retriever.retrieve("warfarin drug interactions")
        assert len(results) > 0

    def test_format_context_includes_source_labels(self, mock_retriever):
        retriever, _ = mock_retriever
        results = retriever.retrieve("drug interactions")
        context = retriever.format_context(results)
        assert "[Source 1]" in context

    def test_token_budget_truncation(self, mock_retriever):
        retriever, _ = mock_retriever
        results = [
            {"doc_id": "doc_1", "content": "A" * 1000, "source": "Drug A"},
            {"doc_id": "doc_2", "content": "B" * 1000, "source": "Drug B"},
            {"doc_id": "doc_3", "content": "C" * 1000, "source": "Drug C"},
        ]
        truncated = retriever._truncate_to_token_budget(results, max_tokens=300)
        # 300 tokens * 4 chars = 1200 chars budget, so only 1 doc fits
        assert len(truncated) == 1

    def test_inject_adversarial_extends_index(self, mock_retriever):
        retriever, _ = mock_retriever
        initial_count = len(retriever._indexed_docs)
        adv_docs = [
            Document(
                doc_id="adv_001",
                content="Adversarial content about drug interactions.",
                source="[ADVERSARIAL]",
                is_adversarial=True,
            )
        ]
        retriever.inject_adversarial(adv_docs)
        assert len(retriever._indexed_docs) == initial_count + 1
