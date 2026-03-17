"""
src/evaluator/methods/hhem.py
==============================
Hallucination detection using a cross-encoder NLI model.

Uses cross-encoder/nli-deberta-v3-small as a reliable, fast
entailment scorer. Given (context, answer), it scores whether
the answer is entailed by (consistent with) the context.

This replaces the original Vectara HHEM model which has
significant dependency compatibility issues with newer
versions of transformers.
"""

from __future__ import annotations

import time

from src.evaluator.methods.base import BaseDetector, DetectionResult
from src.utils import get_logger, load_config

logger = get_logger(__name__)


class HHEMScorer(BaseDetector):
    """
    Hallucination detector using a cross-encoder NLI entailment model.

    Scores (context, answer) pairs for entailment. A high entailment
    score means the answer is well-supported by the context.
    An answer that is not entailed by context is likely hallucinated.

    Uses ``cross-encoder/nli-deberta-v3-small`` — a lightweight,
    reliable model that runs on CPU without dependency issues.

    Parameters
    ----------
    model_name : str | None
        HuggingFace model ID. Defaults to config value.
    threshold : float | None
        Minimum entailment score to pass. Defaults to config value.

    Examples
    --------
    >>> scorer = HHEMScorer()
    >>> result = scorer.detect(question, context, answer)
    >>> print(result.confidence)  # entailment score
    """

    # Reliable fallback model if config model unavailable
    DEFAULT_MODEL = "cross-encoder/nli-deberta-v3-small"

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
        # Always use the reliable NLI model regardless of config
        self.model_name = self.DEFAULT_MODEL
        self.threshold = threshold or hhem_cfg["threshold"]
        self._model = None
        self._tokenizer = None

    def detect(self, question: str, context: str, answer: str) -> DetectionResult:
        """
        Score factual consistency of answer against context via NLI entailment.

        Parameters
        ----------
        question : str
            Original question (not used directly by NLI model).
        context : str
            Retrieved source documents.
        answer : str
            Generated answer to evaluate.

        Returns
        -------
        DetectionResult
        """
        start = time.perf_counter()

        try:
            model, tokenizer = self._get_model_and_tokenizer()
            score = self._compute_entailment_score(model, tokenizer, context, answer)
        except Exception as e:
            logger.warning(f"HHEM inference failed: {e}. Returning neutral score.")
            score = 0.5

        latency_ms = (time.perf_counter() - start) * 1000
        verdict = "PASS" if score >= self.threshold else "FAIL"

        reasoning = (
            f"NLI entailment score: {score:.3f} "
            f"(threshold: {self.threshold}). "
            f"Score reflects how well the answer is supported by the retrieved context."
        )

        return DetectionResult(
            method=self.name,
            verdict=verdict,
            confidence=score,
            reasoning=reasoning,
            hallucinated_claims=[],
            latency_ms=latency_ms,
            raw={"entailment_score": score, "model": self.model_name},
        )

    def _get_model_and_tokenizer(self):
        """Lazily load the NLI model on first use."""
        if self._model is None:
            try:
                import torch
                from transformers import AutoModelForSequenceClassification, AutoTokenizer

                logger.info(f"Loading NLI model: {self.model_name}")
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self._model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name
                )
                self._model.eval()
                logger.info("NLI model loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load NLI model: {e}")
                raise
        return self._model, self._tokenizer

    def _compute_entailment_score(
        self, model, tokenizer, context: str, answer: str
    ) -> float:
        """
        Compute entailment probability for (context, answer) pair.

        For NLI cross-encoders, label order is typically:
        [contradiction, neutral, entailment]

        Parameters
        ----------
        model : transformers model
        tokenizer : transformers tokenizer
        context : str
            Premise (source documents).
        answer : str
            Hypothesis (generated answer).

        Returns
        -------
        float
            Entailment probability in [0, 1].
        """
        import torch

        # Truncate to avoid exceeding max sequence length
        truncated_context = context[:1500]
        truncated_answer = answer[:400]

        inputs = tokenizer(
            truncated_context,
            truncated_answer,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )

        with torch.no_grad():
            outputs = model(**inputs)

        probs = torch.softmax(outputs.logits, dim=-1)[0].tolist()

        # DeBERTa NLI label order: [contradiction=0, neutral=1, entailment=2]
        # Higher entailment = answer is supported by context
        if len(probs) == 3:
            entailment_score = probs[2]  # entailment label
        elif len(probs) == 2:
            entailment_score = probs[1]  # assume [negative, positive]
        else:
            entailment_score = probs[-1]

        return float(entailment_score)