"""
src/evaluator/methods/hhem.py
==============================
Hallucination detection using Vectara's HHEM model.

HHEM (Hughes Hallucination Evaluation Model) is a fine-tuned
cross-encoder that classifies (source, summary) pairs as
factually consistent or not. It runs locally via HuggingFace.
"""

from __future__ import annotations

import time

from src.evaluator.methods.base import BaseDetector, DetectionResult
from src.utils import get_logger, load_config

logger = get_logger(__name__)


class HHEMScorer(BaseDetector):
    """
    Hallucination detector using Vectara's HHEM cross-encoder model.

    HHEM scores (context, answer) pairs on a 0–1 scale where values
    closer to 1 indicate higher factual consistency. Unlike LLM-based
    methods, HHEM is a small specialised model (runs on CPU) and is
    significantly faster.

    The model is loaded lazily on first use to avoid startup overhead
    when HHEM is not included in the active method set.

    Parameters
    ----------
    model_name : str | None
        HuggingFace model ID. Defaults to config value.
    threshold : float | None
        Minimum consistency score to pass. Defaults to config value.

    Examples
    --------
    >>> scorer = HHEMScorer()
    >>> result = scorer.detect(question, context, answer)
    >>> print(result.confidence)  # HHEM consistency score
    """

    @property
    def name(self) -> str:
        return "hhem"

    def __init__(
        self,
        model_name: str | None = None,
        threshold: float | None = None,
    ) -> None:
        cfg = load_config()
        hhem_cfg = cfg["evaluator"]["hhem"]
        self.model_name = model_name or hhem_cfg["model_name"]
        self.threshold = threshold or hhem_cfg["threshold"]
        self._pipeline = None  # loaded lazily

    def detect(self, question: str, context: str, answer: str) -> DetectionResult:
        """
        Score the factual consistency of an answer against its context.

        Parameters
        ----------
        question : str
            Original question (not used directly by HHEM, included for interface).
        context : str
            Retrieved source documents.
        answer : str
            Generated answer to evaluate.

        Returns
        -------
        DetectionResult
        """
        start = time.perf_counter()
        pipeline = self._get_pipeline()

        # HHEM takes (premise=context, hypothesis=answer) pairs
        # Truncate to avoid exceeding model max length
        truncated_context = context[:2000]
        truncated_answer = answer[:500]

        try:
            results = pipeline(
                [{"text": truncated_context, "text_pair": truncated_answer}],
                top_k=None,
            )
            score = self._extract_consistency_score(results)
        except Exception as e:
            logger.warning(f"HHEM inference failed: {e}. Returning neutral score.")
            score = 0.5

        latency_ms = (time.perf_counter() - start) * 1000
        verdict = "PASS" if score >= self.threshold else "FAIL"

        reasoning = (
            f"HHEM consistency score: {score:.3f} "
            f"(threshold: {self.threshold}). "
            f"Higher scores indicate stronger factual grounding."
        )

        return DetectionResult(
            method=self.name,
            verdict=verdict,
            confidence=score,
            reasoning=reasoning,
            hallucinated_claims=[],
            latency_ms=latency_ms,
            raw={"hhem_score": score, "model": self.model_name},
        )

    def _get_pipeline(self):
        """Lazily load the HHEM pipeline on first use."""
        if self._pipeline is None:
            try:
                from transformers import pipeline
                logger.info(f"Loading HHEM model: {self.model_name}")
                self._pipeline = pipeline(
                    "text-classification",
                    model=self.model_name,
                    device=-1,  # CPU
                )
                logger.info("HHEM model loaded.")
            except Exception as e:
                logger.error(f"Failed to load HHEM model: {e}")
                raise
        return self._pipeline

    @staticmethod
    def _extract_consistency_score(results: list) -> float:
        """
        Extract the factual consistency score from HHEM output.

        HHEM returns labels like ``"consistent"`` / ``"inconsistent"``
        with a confidence score. We normalise to [0, 1] where 1 = consistent.

        Parameters
        ----------
        results : list
            Raw pipeline output.

        Returns
        -------
        float
            Consistency score in [0, 1].
        """
        if not results or not results[0]:
            return 0.5

        # results[0] is a list of {label, score} dicts
        label_scores = results[0] if isinstance(results[0], list) else results

        for item in label_scores:
            label = item.get("label", "").lower()
            score = item.get("score", 0.5)
            if "consistent" in label and "in" not in label:
                return float(score)
            if "factual" in label:
                return float(score)

        # Fallback: return score of first result
        return float(label_scores[0].get("score", 0.5))
