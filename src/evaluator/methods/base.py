"""
src/evaluator/methods/base.py
==============================
Abstract base class that every hallucination detection method must implement.

Enforcing a common interface means the Evaluator can run any method
without knowing its internal implementation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class DetectionResult:
    """
    Standardised output from a hallucination detection method.

    Attributes
    ----------
    method : str
        Name of the detection method.
    verdict : str
        ``"PASS"`` (no hallucination detected) or ``"FAIL"`` (hallucination detected).
    confidence : float
        Confidence in the verdict, in [0.0, 1.0].
    reasoning : str
        Human-readable explanation of the verdict.
    hallucinated_claims : list[str]
        Specific claims identified as hallucinated (may be empty).
    latency_ms : float
        Wall-clock time taken by this detection method, in milliseconds.
    raw : dict
        Raw output from the underlying model or scorer for debugging.
    """
    method: str
    verdict: str           # "PASS" | "FAIL"
    confidence: float
    reasoning: str
    hallucinated_claims: list[str]
    latency_ms: float
    raw: dict

    @property
    def is_hallucination(self) -> bool:
        """True if this result indicates a hallucination."""
        return self.verdict == "FAIL"


class BaseDetector(ABC):
    """
    Abstract base class for all hallucination detection methods.

    Subclasses must implement ``detect()``. They should be stateless
    (no mutable state between calls) to ensure thread safety and
    reproducible experiment results.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this method (e.g. ``"llm_judge"``)."""
        ...

    @abstractmethod
    def detect(
        self,
        question: str,
        context: str,
        answer: str,
    ) -> DetectionResult:
        """
        Evaluate whether an answer contains hallucinations.

        Parameters
        ----------
        question : str
            The original user question.
        context : str
            The source documents passed to the generator.
        answer : str
            The generated answer to evaluate.

        Returns
        -------
        DetectionResult
            Structured evaluation result.
        """
        ...
