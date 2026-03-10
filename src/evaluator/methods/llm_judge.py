"""
src/evaluator/methods/llm_judge.py
====================================
Hallucination detection via LLM-as-Judge.

A strong LLM (Claude 3.5 Sonnet via OpenRouter) is prompted to act as
an auditor and evaluate whether a generated answer is faithfully grounded
in the provided context documents.
"""

from __future__ import annotations

import json
import time

from src.evaluator.methods.base import BaseDetector, DetectionResult
from src.utils import call_llm, get_logger, load_config, load_prompts

logger = get_logger(__name__)


class LLMJudge(BaseDetector):
    """
    Uses a strong LLM as an external judge to detect hallucinations.

    The judge is given the question, source context, and generated answer
    and asked to return a structured JSON verdict with reasoning.

    This method is expensive (requires an extra LLM call per evaluation)
    but tends to be the most nuanced, especially for subtle medical errors.

    Parameters
    ----------
    model : str | None
        OpenRouter model to use as judge. Defaults to config value.

    Examples
    --------
    >>> judge = LLMJudge()
    >>> result = judge.detect(question, context, answer)
    >>> print(result.verdict, result.confidence)
    """

    @property
    def name(self) -> str:
        return "llm_judge"

    def __init__(self, model: str | None = None) -> None:
        cfg = load_config()
        self.model = model or cfg["evaluator"]["llm_judge"]["model"]
        self.temperature = cfg["evaluator"]["llm_judge"]["temperature"]
        self.max_tokens = cfg["evaluator"]["llm_judge"]["max_tokens"]
        prompts = load_prompts()
        version = cfg["generator"]["prompt_version"]
        self._system = prompts[version]["judge_system"]
        self._user_template = prompts[version]["judge_user"]

    def detect(self, question: str, context: str, answer: str) -> DetectionResult:
        """
        Ask a judge LLM to evaluate the answer for hallucinations.

        Parameters
        ----------
        question : str
            Original user question.
        context : str
            Retrieved source documents.
        answer : str
            Generated answer to evaluate.

        Returns
        -------
        DetectionResult
        """
        user_prompt = self._user_template.format(
            question=question,
            context=context,
            answer=answer,
        )

        start = time.perf_counter()
        raw_response = call_llm(
            system_prompt=self._system,
            user_prompt=user_prompt,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        latency_ms = (time.perf_counter() - start) * 1000

        parsed = self._parse_judge_response(raw_response)

        return DetectionResult(
            method=self.name,
            verdict=parsed.get("verdict", "FAIL"),
            confidence=float(parsed.get("confidence", 0.5)),
            reasoning=parsed.get("reasoning", ""),
            hallucinated_claims=parsed.get("hallucinated_claims", []),
            latency_ms=latency_ms,
            raw={"response": raw_response, "parsed": parsed},
        )

    @staticmethod
    def _parse_judge_response(response: str) -> dict:
        """
        Extract the JSON payload from the judge's response.

        The judge is prompted to respond only in JSON, but LLMs sometimes
        wrap their JSON in markdown code fences. This method handles both.

        Parameters
        ----------
        response : str
            Raw LLM response string.

        Returns
        -------
        dict
            Parsed JSON, or a safe fallback dict if parsing fails.
        """
        # Strip markdown code fences if present
        clean = response.strip()
        if clean.startswith("```"):
            clean = clean.split("```")[1]
            if clean.startswith("json"):
                clean = clean[4:]
            clean = clean.strip()

        try:
            return json.loads(clean)
        except json.JSONDecodeError:
            logger.warning(f"LLM judge returned non-JSON response: {response[:200]}")
            # Fallback: try to determine verdict from text
            verdict = "FAIL" if any(
                word in response.lower()
                for word in ["hallucination", "not supported", "incorrect", "unsupported"]
            ) else "PASS"
            return {
                "verdict": verdict,
                "confidence": 0.4,
                "hallucinated_claims": [],
                "reasoning": f"JSON parse failed. Raw: {response[:200]}",
            }
