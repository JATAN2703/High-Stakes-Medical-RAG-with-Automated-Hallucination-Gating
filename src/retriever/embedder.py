"""
src/retriever/embedder.py
==========================
Text embedding using sentence-transformers.

Defaults to BioBERT for biomedical domain accuracy, with an automatic
fallback to all-MiniLM-L6-v2 if BioBERT cannot be loaded.
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from src.utils import get_logger, load_config

logger = get_logger(__name__)


class Embedder:
    """
    Wraps a sentence-transformers model to produce dense vector embeddings.

    The model is loaded once and reused for all encode calls, avoiding
    repeated disk I/O.

    Parameters
    ----------
    model_name : str | None
        HuggingFace model name or local path. If None, reads from config.

    Examples
    --------
    >>> embedder = Embedder()
    >>> vectors = embedder.embed(["Drug A causes drowsiness.", "Avoid alcohol."])
    >>> vectors.shape
    (2, 768)
    """

    def __init__(self, model_name: str | None = None) -> None:
        cfg = load_config()
        primary = model_name or cfg["retriever"]["embedding_model"]
        fallback = cfg["retriever"]["embedding_fallback"]
        self.model = self._load_model(primary, fallback)
        self.model_name = primary

    def embed(self, texts: list[str], batch_size: int = 64, show_progress: bool = False) -> np.ndarray:
        """
        Encode a list of strings into dense embedding vectors.

        Parameters
        ----------
        texts : list[str]
            Input texts to embed.
        batch_size : int
            Number of texts to process per forward pass.
        show_progress : bool
            Whether to display a tqdm progress bar.

        Returns
        -------
        np.ndarray
            Float32 array of shape (len(texts), embedding_dim).
        """
        if not texts:
            raise ValueError("Cannot embed an empty list of texts.")

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    def embed_single(self, text: str) -> np.ndarray:
        """
        Convenience wrapper to embed a single string.

        Parameters
        ----------
        text : str
            Input text.

        Returns
        -------
        np.ndarray
            1-D float32 array of shape (embedding_dim,).
        """
        return self.embed([text])[0]

    @staticmethod
    def _load_model(primary: str, fallback: str) -> SentenceTransformer:
        """Attempt to load primary model, fall back if unavailable."""
        for model_name in [primary, fallback]:
            try:
                logger.info(f"Loading embedding model: {model_name}")
                model = SentenceTransformer(model_name)
                logger.info(f"Embedding model loaded: {model_name}")
                return model
            except Exception as e:
                logger.warning(f"Could not load {model_name}: {e}. Trying fallback...")
        raise RuntimeError("Failed to load any embedding model.")
