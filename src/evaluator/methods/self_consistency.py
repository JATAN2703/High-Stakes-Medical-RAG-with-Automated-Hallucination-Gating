"""
src/evaluator/methods/self_consistency.py
==========================================
Hallucination detection via self-consistency sampling.

Generates N responses to the same question/context and measures
agreement. Low agreement suggests the model is uncertain or hallucinating.
"""

from __future__ import annotations

import time
from collections import Counter

from rouge_score import rouge_scorer

from src.evaluator.methods.base import BaseDetector, DetectionResult
from src.utils import call_llm, get_logger, load_config, load_prompts

logger = get_logger(__name__)


class SelfConsistencyChecker(BaseDetector):
    """
    Detects hallucinations by measuring agreement across N sampled responses.

    If a model is confidently grounded in the context, its answers
    should be consistent across samples. High variance suggests either
    the question is unanswerable from context (good) or the model is
    confabulating (bad). Agreement is measured using ROUGE-L F1.

    Parameters
    ----------
    n_samples : int | None
        Number of responses to sample. Defaults to config value.
    temperature : float | None
        Sampling temperature. Must be > 0 for meaningful variance.
    agreement_threshold : float | None
        Minimum mean pairwise ROUGE-L to consider responses consistent.

    Examples
    --------
    >>> checker = SelfConsistencyChecker()
    >>> result = checker.detect(question, context, answer)
    """

    @property
    def name(self) -> str:
        return "self_consistency"

    def __init__(
        self,
        n_samples: int | None = None,
        temperature: float | None = None,
        agreement_threshold: float | None = None,
    ) -> None:
        cfg = load_config()
        sc_cfg = cfg["evaluator"]["self_consistency"]
        gen_cfg = cfg["generator"]

        self.n_samples = n_samples or sc_cfg["n_samples"]
        self.temperature = temperature or sc_cfg["temperature"]
        self.threshold = agreement_threshold or sc_cfg["agreement_threshold"]
        self.model = gen_cfg["model"]

        prompts = load_prompts()
        version = gen_cfg["prompt_version"]
        self._system = prompts[version]["consistency_system"]
        self._user_template = prompts[version]["consistency_user"]

        self._scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    def detect(self, question: str, context: str, answer: str) -> DetectionResult:
        """
        Sample N responses and measure pairwise agreement.

        The ``answer`` parameter is used as the reference response for
        comparison against the N additional samples.

        Parameters
        ----------
        question : str
            Original question.
        context : str
            Retrieved context.
        answer : str
            The primary generated answer (counted as sample 0).

        Returns
        -------
        DetectionResult
        """
        start = time.perf_counter()

        # Generate N-1 additional samples (answer is sample 0)
        samples = [answer] + self._sample_responses(question, context)

        agreement_score = self._mean_pairwise_rouge(samples)
        latency_ms = (time.perf_counter() - start) * 1000

        is_consistent = agreement_score >= self.threshold
        verdict = "PASS" if is_consistent else "FAIL"
        confidence = min(1.0, agreement_score / self.threshold) if is_consistent else \
                     max(0.0, agreement_score / self.threshold)

        reasoning = (
            f"Mean pairwise ROUGE-L across {len(samples)} samples: {agreement_score:.3f}. "
            f"Threshold: {self.threshold}. "
            f"{'Consistent' if is_consistent else 'Inconsistent'} responses suggest "
            f"{'reliable grounding' if is_consistent else 'possible hallucination or insufficient context'}."
        )

        return DetectionResult(
            method=self.name,
            verdict=verdict,
            confidence=confidence,
            reasoning=reasoning,
            hallucinated_claims=[],
            latency_ms=latency_ms,
            raw={
                "agreement_score": agreement_score,
                "n_samples": len(samples),
                "samples": samples,
            },
        )

    def _sample_responses(self, question: str, context: str) -> list[str]:
        """Generate N-1 additional responses for consistency comparison."""
        user_prompt = self._user_template.format(context=context, question=question)
        responses = []
        for _ in range(self.n_samples - 1):
            response = call_llm(
                system_prompt=self._system,
                user_prompt=user_prompt,
                model=self.model,
                temperature=self.temperature,
                max_tokens=256,
            )
            responses.append(response)
        return responses

    def _mean_pairwise_rouge(self, samples: list[str]) -> float:
        """
        Compute mean pairwise ROUGE-L F1 across all sample pairs.

        Parameters
        ----------
        samples : list[str]
            List of generated responses.

        Returns
        -------
        float
            Mean pairwise ROUGE-L F1 score in [0, 1].
        """
        if len(samples) < 2:
            return 1.0

        scores = []
        for i in range(len(samples)):
            for j in range(i + 1, len(samples)):
                result = self._scorer.score(samples[i], samples[j])
                scores.append(result["rougeL"].fmeasure)

        return sum(scores) / len(scores) if scores else 0.0
