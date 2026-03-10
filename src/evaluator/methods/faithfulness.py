"""
src/evaluator/methods/faithfulness.py
=======================================
Hallucination detection via retrieval faithfulness scoring.

Measures what fraction of factual claims in the answer can be
traced back to a passage in the retrieved context. Unlike LLM-based
methods, this is fully transparent and interpretable.
"""

from __future__ import annotations

import re
import time

import nltk
from rouge_score import rouge_scorer

from src.evaluator.methods.base import BaseDetector, DetectionResult
from src.utils import get_logger, load_config

logger = get_logger(__name__)

# Download NLTK sentence tokenizer if not available
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)


class FaithfulnessScorer(BaseDetector):
    """
    Scores answer faithfulness by checking claim-level grounding in context.

    Algorithm:
    1. Split the answer into individual sentences (claims).
    2. For each claim, compute max ROUGE-L against all context passages.
    3. A claim is "grounded" if its max ROUGE-L exceeds a threshold.
    4. Faithfulness = grounded_claims / total_claims.
    5. FAIL if faithfulness < config threshold.

    This method is fast (no LLM call), deterministic, and produces
    interpretable per-claim scores. It serves as the baseline in
    our benchmark.

    Parameters
    ----------
    threshold : float | None
        Minimum faithfulness fraction to pass. Defaults to config value.
    claim_grounding_threshold : float
        Minimum ROUGE-L for a claim to be considered grounded.

    Examples
    --------
    >>> scorer = FaithfulnessScorer()
    >>> result = scorer.detect(question, context, answer)
    >>> print(result.confidence)  # faithfulness score
    """

    @property
    def name(self) -> str:
        return "faithfulness"

    def __init__(
        self,
        threshold: float | None = None,
        claim_grounding_threshold: float = 0.3,
    ) -> None:
        cfg = load_config()
        self.threshold = threshold or cfg["evaluator"]["faithfulness"]["threshold"]
        self.claim_grounding_threshold = claim_grounding_threshold
        self._scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    def detect(self, question: str, context: str, answer: str) -> DetectionResult:
        """
        Score answer faithfulness against the retrieved context.

        Parameters
        ----------
        question : str
            Original question (unused directly, included for interface consistency).
        context : str
            Retrieved context block.
        answer : str
            Generated answer to evaluate.

        Returns
        -------
        DetectionResult
        """
        start = time.perf_counter()

        claims = self._extract_claims(answer)
        context_passages = self._split_context_passages(context)

        if not claims:
            return DetectionResult(
                method=self.name,
                verdict="PASS",
                confidence=1.0,
                reasoning="No factual claims found in answer.",
                hallucinated_claims=[],
                latency_ms=(time.perf_counter() - start) * 1000,
                raw={},
            )

        grounded = []
        ungrounded = []
        claim_scores = {}

        for claim in claims:
            max_score = self._max_rouge_against_context(claim, context_passages)
            claim_scores[claim] = max_score
            if max_score >= self.claim_grounding_threshold:
                grounded.append(claim)
            else:
                ungrounded.append(claim)

        faithfulness = len(grounded) / len(claims)
        verdict = "PASS" if faithfulness >= self.threshold else "FAIL"
        latency_ms = (time.perf_counter() - start) * 1000

        reasoning = (
            f"Faithfulness score: {faithfulness:.2%} "
            f"({len(grounded)}/{len(claims)} claims grounded). "
            f"Threshold: {self.threshold:.0%}."
        )

        return DetectionResult(
            method=self.name,
            verdict=verdict,
            confidence=faithfulness,
            reasoning=reasoning,
            hallucinated_claims=ungrounded,
            latency_ms=latency_ms,
            raw={
                "faithfulness_score": faithfulness,
                "grounded_claims": grounded,
                "ungrounded_claims": ungrounded,
                "claim_scores": claim_scores,
                "n_claims": len(claims),
            },
        )

    def _extract_claims(self, answer: str) -> list[str]:
        """
        Split answer into individual sentence-level claims.

        Filters out very short sentences and uncertainty phrases.

        Parameters
        ----------
        answer : str
            Generated answer text.

        Returns
        -------
        list[str]
            List of claim strings.
        """
        # Tokenize into sentences
        try:
            sentences = nltk.sent_tokenize(answer)
        except Exception:
            sentences = [s.strip() for s in re.split(r"[.!?]", answer) if s.strip()]

        # Filter out uncertainty disclaimers and very short sentences
        skip_patterns = [
            "do not contain",
            "insufficient information",
            "cannot answer",
            "note:",
            "source [",
        ]
        claims = []
        for sentence in sentences:
            s = sentence.strip()
            if len(s) < 20:
                continue
            if any(pattern in s.lower() for pattern in skip_patterns):
                continue
            claims.append(s)

        return claims

    def _split_context_passages(self, context: str) -> list[str]:
        """
        Split the formatted context block into individual passages.

        Parameters
        ----------
        context : str
            Context string formatted as ``[Source N] text\n\n[Source N+1] ...``

        Returns
        -------
        list[str]
            Individual passage strings.
        """
        # Split on [Source N] markers
        passages = re.split(r"\[Source \d+\]", context)
        return [p.strip() for p in passages if p.strip()]

    def _max_rouge_against_context(
        self,
        claim: str,
        context_passages: list[str],
    ) -> float:
        """
        Compute the maximum ROUGE-L F1 between a claim and all context passages.

        Parameters
        ----------
        claim : str
            A single claim sentence from the answer.
        context_passages : list[str]
            List of context passage strings.

        Returns
        -------
        float
            Maximum ROUGE-L F1 score across all passages.
        """
        if not context_passages:
            return 0.0

        max_score = 0.0
        for passage in context_passages:
            result = self._scorer.score(claim, passage)
            score = result["rougeL"].fmeasure
            if score > max_score:
                max_score = score

        return max_score
