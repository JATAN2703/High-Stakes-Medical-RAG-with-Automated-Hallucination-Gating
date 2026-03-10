"""
src/evaluator/evaluator.py
===========================
Main Evaluator class: orchestrates all hallucination detection methods
and produces structured benchmark results.

This is the research engine of the project. It runs each detection
method against every generated answer and produces the metrics table
that forms the core of the final report.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.evaluator.methods.base import BaseDetector, DetectionResult
from src.evaluator.methods.faithfulness import FaithfulnessScorer
from src.evaluator.methods.hhem import HHEMScorer
from src.evaluator.methods.llm_judge import LLMJudge
from src.evaluator.methods.self_consistency import SelfConsistencyChecker
from src.utils import get_logger, load_config

logger = get_logger(__name__)


# ── Result models ─────────────────────────────────────────────────────────────

@dataclass
class EvaluationResult:
    """
    Complete evaluation of a single (question, context, answer) triple.

    Attributes
    ----------
    question : str
        The original question.
    context : str
        Retrieved context passed to the generator.
    answer : str
        Generated answer being evaluated.
    ground_truth_label : str | None
        ``"hallucination"`` or ``"grounded"`` — from adversarial ground truth.
    detections : dict[str, DetectionResult]
        Method name → DetectionResult for each active method.
    context_token_count : int
        Approximate token count of the context (for window analysis).
    metadata : dict
        Additional experiment metadata.
    """
    question: str
    context: str
    answer: str
    ground_truth_label: str | None
    detections: dict[str, DetectionResult] = field(default_factory=dict)
    context_token_count: int = 0
    metadata: dict = field(default_factory=dict)

    def method_verdict(self, method: str) -> str | None:
        """Return the verdict of a specific method, or None if not run."""
        r = self.detections.get(method)
        return r.verdict if r else None

    def is_true_positive(self, method: str) -> bool:
        """Correctly identified hallucination (ground truth=hallucination, verdict=FAIL)."""
        return (
            self.ground_truth_label == "hallucination"
            and self.method_verdict(method) == "FAIL"
        )

    def is_false_positive(self, method: str) -> bool:
        """Incorrectly flagged clean answer (ground truth=grounded, verdict=FAIL)."""
        return (
            self.ground_truth_label == "grounded"
            and self.method_verdict(method) == "FAIL"
        )

    def is_true_negative(self, method: str) -> bool:
        """Correctly passed clean answer (ground truth=grounded, verdict=PASS)."""
        return (
            self.ground_truth_label == "grounded"
            and self.method_verdict(method) == "PASS"
        )

    def is_false_negative(self, method: str) -> bool:
        """Missed hallucination (ground truth=hallucination, verdict=PASS)."""
        return (
            self.ground_truth_label == "hallucination"
            and self.method_verdict(method) == "PASS"
        )


@dataclass
class BenchmarkReport:
    """
    Aggregated metrics across all evaluated examples.

    Attributes
    ----------
    method_metrics : dict[str, dict]
        Per-method precision, recall, F1, FPR, mean latency.
    results : list[EvaluationResult]
        All individual evaluation results.
    condition : str
        Experimental condition (``"clean"``, ``"adversarial"``, ``"long_context"``).
    context_window : int | None
        Token limit used, if applicable.
    """
    method_metrics: dict[str, dict]
    results: list[EvaluationResult]
    condition: str
    context_window: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise report to a JSON-compatible dictionary."""
        return {
            "condition": self.condition,
            "context_window": self.context_window,
            "n_examples": len(self.results),
            "method_metrics": self.method_metrics,
        }

    def print_summary(self) -> None:
        """Print a formatted metrics table to stdout."""
        print(f"\n{'='*70}")
        print(f"  Benchmark Report — Condition: {self.condition}")
        if self.context_window:
            print(f"  Context window: {self.context_window} tokens")
        print(f"  Examples: {len(self.results)}")
        print(f"{'='*70}")
        header = f"{'Method':<22} {'Recall':>8} {'Precision':>10} {'F1':>8} {'FPR':>8} {'Latency(ms)':>12}"
        print(header)
        print("-" * 70)
        for method, m in self.method_metrics.items():
            print(
                f"{method:<22} "
                f"{m['recall']:>8.2%} "
                f"{m['precision']:>10.2%} "
                f"{m['f1']:>8.3f} "
                f"{m['false_positive_rate']:>8.2%} "
                f"{m['mean_latency_ms']:>12.1f}"
            )
        print(f"{'='*70}\n")


# ── Evaluator ─────────────────────────────────────────────────────────────────

class Evaluator:
    """
    Orchestrates all hallucination detection methods and produces benchmark reports.

    The Evaluator is the core research component of this project. Given a list
    of (question, context, answer, label) tuples, it runs every configured
    detection method and computes precision, recall, F1, false-positive rate,
    and latency for each.

    Parameters
    ----------
    methods : list[str] | None
        Which methods to run. Defaults to config list.
        Options: ``"llm_judge"``, ``"self_consistency"``, ``"faithfulness"``, ``"hhem"``

    Examples
    --------
    >>> evaluator = Evaluator(methods=["llm_judge", "faithfulness"])
    >>> report = evaluator.benchmark(samples, condition="adversarial")
    >>> report.print_summary()
    """

    METHOD_REGISTRY: dict[str, type[BaseDetector]] = {
        "llm_judge": LLMJudge,
        "self_consistency": SelfConsistencyChecker,
        "faithfulness": FaithfulnessScorer,
        "hhem": HHEMScorer,
    }

    def __init__(self, methods: list[str] | None = None) -> None:
        cfg = load_config()
        active_methods = methods or cfg["evaluator"]["methods"]

        self._detectors: dict[str, BaseDetector] = {}
        for method_name in active_methods:
            if method_name not in self.METHOD_REGISTRY:
                raise ValueError(
                    f"Unknown method '{method_name}'. "
                    f"Valid options: {list(self.METHOD_REGISTRY.keys())}"
                )
            self._detectors[method_name] = self.METHOD_REGISTRY[method_name]()
            logger.info(f"Registered detector: {method_name}")

        logger.info(f"Evaluator ready with {len(self._detectors)} methods.")

    def evaluate_single(
        self,
        question: str,
        context: str,
        answer: str,
        ground_truth_label: str | None = None,
        metadata: dict | None = None,
    ) -> EvaluationResult:
        """
        Run all detectors on a single (question, context, answer) triple.

        Parameters
        ----------
        question : str
            The user question.
        context : str
            Retrieved context.
        answer : str
            Generated answer to evaluate.
        ground_truth_label : str | None
            ``"hallucination"`` or ``"grounded"`` if known.
        metadata : dict | None
            Optional extra metadata to attach.

        Returns
        -------
        EvaluationResult
        """
        token_count = len(context.split()) * 4 // 3  # rough approximation

        result = EvaluationResult(
            question=question,
            context=context,
            answer=answer,
            ground_truth_label=ground_truth_label,
            context_token_count=token_count,
            metadata=metadata or {},
        )

        for method_name, detector in self._detectors.items():
            try:
                detection = detector.detect(question=question, context=context, answer=answer)
                result.detections[method_name] = detection
                logger.debug(
                    f"{method_name}: {detection.verdict} "
                    f"(confidence={detection.confidence:.2f}, "
                    f"latency={detection.latency_ms:.0f}ms)"
                )
            except Exception as e:
                logger.error(f"Detector '{method_name}' failed: {e}")

        return result

    def benchmark(
        self,
        samples: list[dict[str, Any]],
        condition: str = "clean",
        context_window: int | None = None,
    ) -> BenchmarkReport:
        """
        Run all detectors over a dataset and compute aggregate metrics.

        Parameters
        ----------
        samples : list[dict]
            Each dict must have keys: ``question``, ``context``, ``answer``,
            and optionally ``ground_truth_label`` and ``metadata``.
        condition : str
            Experiment condition label for the report.
        context_window : int | None
            Token limit applied to context (for window sensitivity experiments).

        Returns
        -------
        BenchmarkReport
        """
        logger.info(
            f"Starting benchmark: {len(samples)} samples, "
            f"condition='{condition}', methods={list(self._detectors.keys())}"
        )

        results: list[EvaluationResult] = []
        for i, sample in enumerate(samples):
            logger.info(f"Evaluating sample {i+1}/{len(samples)}...")
            result = self.evaluate_single(
                question=sample["question"],
                context=sample["context"],
                answer=sample["answer"],
                ground_truth_label=sample.get("ground_truth_label"),
                metadata=sample.get("metadata"),
            )
            results.append(result)

        metrics = self._compute_metrics(results)
        report = BenchmarkReport(
            method_metrics=metrics,
            results=results,
            condition=condition,
            context_window=context_window,
        )
        logger.info("Benchmark complete.")
        return report

    def _compute_metrics(
        self,
        results: list[EvaluationResult],
    ) -> dict[str, dict[str, float]]:
        """
        Compute precision, recall, F1, FPR, and mean latency per method.

        Only examples with a non-None ground_truth_label are included
        in precision/recall/F1 calculations.

        Parameters
        ----------
        results : list[EvaluationResult]

        Returns
        -------
        dict[str, dict[str, float]]
            Nested dict: method_name → metric_name → value.
        """
        metrics = {}
        for method_name in self._detectors:
            labelled = [r for r in results if r.ground_truth_label is not None]

            tp = sum(1 for r in labelled if r.is_true_positive(method_name))
            fp = sum(1 for r in labelled if r.is_false_positive(method_name))
            tn = sum(1 for r in labelled if r.is_true_negative(method_name))
            fn = sum(1 for r in labelled if r.is_false_negative(method_name))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)
                  if (precision + recall) > 0 else 0.0)
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

            latencies = [
                r.detections[method_name].latency_ms
                for r in results
                if method_name in r.detections
            ]
            mean_latency = sum(latencies) / len(latencies) if latencies else 0.0

            metrics[method_name] = {
                "true_positives": tp,
                "false_positives": fp,
                "true_negatives": tn,
                "false_negatives": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "false_positive_rate": fpr,
                "mean_latency_ms": mean_latency,
            }

        return metrics

    def save_results(
        self,
        report: BenchmarkReport,
        output_path: str | Path,
    ) -> None:
        """
        Persist a benchmark report to a JSON file.

        Parameters
        ----------
        report : BenchmarkReport
            The report to save.
        output_path : str | Path
            Destination file path.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        serialisable = {
            **report.to_dict(),
            "results": [
                {
                    "question": r.question,
                    "answer": r.answer[:200],
                    "ground_truth_label": r.ground_truth_label,
                    "context_token_count": r.context_token_count,
                    "detections": {
                        method: {
                            "verdict": d.verdict,
                            "confidence": d.confidence,
                            "reasoning": d.reasoning,
                            "latency_ms": d.latency_ms,
                            "hallucinated_claims": d.hallucinated_claims,
                        }
                        for method, d in r.detections.items()
                    },
                    "metadata": r.metadata,
                }
                for r in report.results
            ],
        }

        with open(output_path, "w") as f:
            json.dump(serialisable, f, indent=2)

        logger.info(f"Results saved to {output_path}")
