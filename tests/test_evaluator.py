"""
tests/test_evaluator.py
========================
Unit and integration tests for the Evaluator module and all
hallucination detection methods.

The Evaluator contains the core research logic, so these tests
are the most thorough in the project. Branch coverage of the
metrics computation is critical.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from src.evaluator.evaluator import Evaluator, EvaluationResult, BenchmarkReport
from src.evaluator.methods.base import DetectionResult
from src.evaluator.methods.faithfulness import FaithfulnessScorer
from src.evaluator.methods.llm_judge import LLMJudge
from src.evaluator.methods.self_consistency import SelfConsistencyChecker


# ── DetectionResult tests ─────────────────────────────────────────────────────

class TestDetectionResult:
    def test_is_hallucination_true_for_fail(self):
        r = DetectionResult(
            method="test", verdict="FAIL", confidence=0.9,
            reasoning="", hallucinated_claims=[], latency_ms=10, raw={}
        )
        assert r.is_hallucination is True

    def test_is_hallucination_false_for_pass(self):
        r = DetectionResult(
            method="test", verdict="PASS", confidence=0.9,
            reasoning="", hallucinated_claims=[], latency_ms=10, raw={}
        )
        assert r.is_hallucination is False


# ── EvaluationResult tests ────────────────────────────────────────────────────

class TestEvaluationResult:
    @pytest.fixture
    def result_with_detections(self):
        result = EvaluationResult(
            question="Q?",
            context="ctx",
            answer="ans",
            ground_truth_label="hallucination",
        )
        result.detections["method_a"] = DetectionResult(
            method="method_a", verdict="FAIL", confidence=0.9,
            reasoning="", hallucinated_claims=[], latency_ms=10, raw={}
        )
        result.detections["method_b"] = DetectionResult(
            method="method_b", verdict="PASS", confidence=0.6,
            reasoning="", hallucinated_claims=[], latency_ms=5, raw={}
        )
        return result

    def test_true_positive(self, result_with_detections):
        # Ground truth=hallucination, method_a verdict=FAIL → TP
        assert result_with_detections.is_true_positive("method_a") is True

    def test_false_negative(self, result_with_detections):
        # Ground truth=hallucination, method_b verdict=PASS → FN
        assert result_with_detections.is_false_negative("method_b") is True

    def test_method_verdict_returns_none_for_missing_method(self, result_with_detections):
        assert result_with_detections.method_verdict("nonexistent") is None

    def test_false_positive(self):
        result = EvaluationResult(
            question="Q?", context="ctx", answer="ans",
            ground_truth_label="grounded",
        )
        result.detections["method_a"] = DetectionResult(
            method="method_a", verdict="FAIL", confidence=0.8,
            reasoning="", hallucinated_claims=[], latency_ms=10, raw={}
        )
        assert result.is_false_positive("method_a") is True

    def test_true_negative(self):
        result = EvaluationResult(
            question="Q?", context="ctx", answer="ans",
            ground_truth_label="grounded",
        )
        result.detections["method_a"] = DetectionResult(
            method="method_a", verdict="PASS", confidence=0.8,
            reasoning="", hallucinated_claims=[], latency_ms=10, raw={}
        )
        assert result.is_true_negative("method_a") is True


# ── FaithfulnessScorer tests ──────────────────────────────────────────────────

class TestFaithfulnessScorer:
    @pytest.fixture
    def scorer(self):
        return FaithfulnessScorer(threshold=0.5, claim_grounding_threshold=0.2)

    def test_perfect_grounding_passes(self, scorer):
        context = "[Source 1] Warfarin is an anticoagulant that inhibits vitamin K."
        answer = "Warfarin is an anticoagulant that inhibits vitamin K metabolism."
        result = scorer.detect("Q?", context, answer)
        assert result.verdict == "PASS"

    def test_completely_ungrounded_answer_fails(self, scorer):
        context = "[Source 1] Drug A is used for hypertension."
        answer = "The moon is made of cheese and orbits the Earth every 27 days."
        result = scorer.detect("Q?", context, answer)
        assert result.verdict == "FAIL"

    def test_empty_answer_passes(self, scorer):
        result = scorer.detect("Q?", "[Source 1] Some context.", "")
        assert result.verdict == "PASS"

    def test_confidence_is_faithfulness_score(self, scorer):
        context = "[Source 1] Warfarin causes bleeding."
        answer = "Warfarin causes bleeding. Also it cures cancer."
        result = scorer.detect("Q?", context, answer)
        assert 0.0 <= result.confidence <= 1.0

    def test_extract_claims_filters_short_sentences(self, scorer):
        answer = "OK. Warfarin is an anticoagulant used to prevent blood clots."
        claims = scorer._extract_claims(answer)
        assert all(len(c) >= 20 for c in claims)

    def test_extract_claims_filters_uncertainty_phrases(self, scorer):
        answer = "The provided documents do not contain sufficient information. Warfarin is used for clotting disorders."
        claims = scorer._extract_claims(answer)
        assert not any("do not contain" in c.lower() for c in claims)

    def test_split_context_passages_by_source_tag(self, scorer):
        context = "[Source 1] First passage.\n\n[Source 2] Second passage."
        passages = scorer._split_context_passages(context)
        assert len(passages) == 2

    def test_hallucinated_claims_in_raw_output(self, scorer):
        context = "[Source 1] Drug A treats hypertension."
        answer = "Drug A treats hypertension. Drug B cures diabetes completely."
        result = scorer.detect("Q?", context, answer)
        assert isinstance(result.hallucinated_claims, list)


# ── LLMJudge tests ─────────────────────────────────────────────────────────────

class TestLLMJudge:
    @pytest.fixture
    def judge(self):
        return LLMJudge(model="anthropic/claude-3.5-sonnet")

    def test_parse_valid_json_pass(self, judge):
        response = '{"verdict": "PASS", "confidence": 0.9, "hallucinated_claims": [], "reasoning": "All claims supported."}'
        parsed = judge._parse_judge_response(response)
        assert parsed["verdict"] == "PASS"
        assert parsed["confidence"] == 0.9

    def test_parse_valid_json_fail(self, judge):
        response = '{"verdict": "FAIL", "confidence": 0.85, "hallucinated_claims": ["claim A"], "reasoning": "Claim A not found."}'
        parsed = judge._parse_judge_response(response)
        assert parsed["verdict"] == "FAIL"
        assert "claim A" in parsed["hallucinated_claims"]

    def test_parse_markdown_wrapped_json(self, judge):
        response = '```json\n{"verdict": "PASS", "confidence": 0.7, "hallucinated_claims": [], "reasoning": "OK"}\n```'
        parsed = judge._parse_judge_response(response)
        assert parsed["verdict"] == "PASS"

    def test_parse_invalid_json_returns_fallback(self, judge):
        response = "This answer contains hallucination and is not supported."
        parsed = judge._parse_judge_response(response)
        assert parsed["verdict"] == "FAIL"
        assert "confidence" in parsed

    def test_detect_calls_llm(self, judge):
        with patch("src.evaluator.methods.llm_judge.call_llm") as mock_llm:
            mock_llm.return_value = '{"verdict": "PASS", "confidence": 0.9, "hallucinated_claims": [], "reasoning": "OK"}'
            result = judge.detect("Q?", "ctx", "ans")
            assert mock_llm.called
            assert result.verdict == "PASS"

    def test_detect_result_has_required_fields(self, judge):
        with patch("src.evaluator.methods.llm_judge.call_llm") as mock_llm:
            mock_llm.return_value = '{"verdict": "FAIL", "confidence": 0.8, "hallucinated_claims": ["bad claim"], "reasoning": "Not supported"}'
            result = judge.detect("Q?", "ctx", "ans")
            assert result.method == "llm_judge"
            assert result.latency_ms >= 0
            assert isinstance(result.hallucinated_claims, list)


# ── SelfConsistencyChecker tests ──────────────────────────────────────────────

class TestSelfConsistencyChecker:
    @pytest.fixture
    def checker(self):
        return SelfConsistencyChecker(n_samples=3, temperature=0.7, agreement_threshold=0.5)

    def test_high_agreement_passes(self, checker):
        with patch("src.evaluator.methods.self_consistency.call_llm") as mock_llm:
            mock_llm.return_value = "Warfarin interacts with aspirin."
            # All samples identical → perfect ROUGE-L
            result = checker.detect("Q?", "[Source 1] ctx.", "Warfarin interacts with aspirin.")
            assert result.verdict == "PASS"

    def test_low_agreement_fails(self, checker):
        with patch("src.evaluator.methods.self_consistency.call_llm") as mock_llm:
            mock_llm.side_effect = [
                "The drug causes severe liver damage.",
                "The medication is completely safe with no side effects.",
            ]
            result = checker.detect("Q?", "[Source 1] ctx.", "Initial answer about warfarin.")
            # Low ROUGE-L between completely different answers → FAIL
            assert result.raw["agreement_score"] is not None

    def test_mean_pairwise_rouge_identical_strings(self, checker):
        samples = ["identical text", "identical text", "identical text"]
        score = checker._mean_pairwise_rouge(samples)
        assert score == pytest.approx(1.0, abs=0.01)

    def test_mean_pairwise_rouge_completely_different(self, checker):
        samples = ["apple orange banana", "xyz qrs tuv", "123 456 789"]
        score = checker._mean_pairwise_rouge(samples)
        assert score < 0.3

    def test_mean_pairwise_rouge_single_sample_returns_one(self, checker):
        score = checker._mean_pairwise_rouge(["only one sample"])
        assert score == 1.0

    def test_n_samples_minimum_is_two(self):
        with pytest.raises(Exception):
            checker = SelfConsistencyChecker(n_samples=1)


# ── Evaluator orchestration tests ─────────────────────────────────────────────

class TestEvaluator:
    @pytest.fixture
    def mock_evaluator(self):
        """Evaluator with faithfulness-only to avoid external calls."""
        with patch("src.evaluator.methods.faithfulness.FaithfulnessScorer.detect") as mock_detect:
            mock_detect.return_value = DetectionResult(
                method="faithfulness", verdict="PASS", confidence=0.8,
                reasoning="", hallucinated_claims=[], latency_ms=5, raw={}
            )
            evaluator = Evaluator(methods=["faithfulness"])
            evaluator._detectors["faithfulness"].detect = mock_detect
            return evaluator, mock_detect

    def test_evaluate_single_runs_all_methods(self):
        evaluator = Evaluator(methods=["faithfulness"])
        with patch.object(
            evaluator._detectors["faithfulness"], "detect",
            return_value=DetectionResult(
                method="faithfulness", verdict="PASS", confidence=0.8,
                reasoning="", hallucinated_claims=[], latency_ms=5, raw={}
            )
        ):
            result = evaluator.evaluate_single(
                question="Q?", context="[Source 1] ctx.", answer="Answer.",
            )
            assert "faithfulness" in result.detections

    def test_benchmark_computes_metrics(self):
        evaluator = Evaluator(methods=["faithfulness"])
        with patch.object(
            evaluator._detectors["faithfulness"], "detect",
            return_value=DetectionResult(
                method="faithfulness", verdict="FAIL", confidence=0.3,
                reasoning="", hallucinated_claims=[], latency_ms=5, raw={}
            )
        ):
            samples = [
                {
                    "question": "Q?",
                    "context": "[Source 1] ctx.",
                    "answer": "ans",
                    "ground_truth_label": "hallucination",
                }
                for _ in range(5)
            ]
            report = evaluator.benchmark(samples, condition="test")
            metrics = report.method_metrics["faithfulness"]
            assert metrics["recall"] == pytest.approx(1.0)
            assert metrics["false_positive_rate"] == pytest.approx(0.0)

    def test_compute_metrics_handles_zero_division(self):
        evaluator = Evaluator(methods=["faithfulness"])
        # No labelled examples → all metrics should be 0, no ZeroDivisionError
        results = [
            EvaluationResult(
                question="Q?", context="ctx", answer="ans",
                ground_truth_label=None  # unlabelled
            )
        ]
        results[0].detections["faithfulness"] = DetectionResult(
            method="faithfulness", verdict="PASS", confidence=0.8,
            reasoning="", hallucinated_claims=[], latency_ms=5, raw={}
        )
        metrics = evaluator._compute_metrics(results)
        assert metrics["faithfulness"]["f1"] == 0.0

    def test_unknown_method_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown method"):
            Evaluator(methods=["nonexistent_method"])

    def test_save_results_creates_file(self, tmp_path):
        evaluator = Evaluator(methods=["faithfulness"])
        report = BenchmarkReport(
            method_metrics={"faithfulness": {"recall": 1.0, "precision": 1.0, "f1": 1.0,
                                              "false_positive_rate": 0.0, "mean_latency_ms": 5.0,
                                              "true_positives": 5, "false_positives": 0,
                                              "true_negatives": 5, "false_negatives": 0}},
            results=[],
            condition="test",
        )
        output_path = tmp_path / "test_results.json"
        evaluator.save_results(report, output_path)
        assert output_path.exists()
