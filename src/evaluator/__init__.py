"""
Evaluator module: hallucination detection benchmarking suite.
Four independent detection methods with a unified interface.
"""
from .methods.llm_judge import LLMJudge
from .methods.self_consistency import SelfConsistencyChecker
from .methods.faithfulness import FaithfulnessScorer
from .methods.hhem import HHEMScorer
from .evaluator import Evaluator, EvaluationResult

__all__ = [
    "LLMJudge",
    "SelfConsistencyChecker",
    "FaithfulnessScorer",
    "HHEMScorer",
    "Evaluator",
    "EvaluationResult",
]
