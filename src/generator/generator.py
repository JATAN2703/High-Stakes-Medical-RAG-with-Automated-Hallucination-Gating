"""
src/generator/generator.py
===========================
Grounded answer generation using OpenRouter-hosted LLMs.

The Generator takes a user question and a formatted context block
(from the Retriever) and produces a cited, grounded answer. It is
deliberately strict: answers must cite sources and must not speculate
beyond the provided context.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.utils import call_llm, get_logger, load_config, load_prompts

logger = get_logger(__name__)


# ── Response model ────────────────────────────────────────────────────────────

@dataclass
class GeneratorResponse:
    """
    Structured output from the Generator.

    Attributes
    ----------
    answer : str
        The model's grounded answer text.
    question : str
        The original question.
    context : str
        The context block passed to the model.
    model : str
        OpenRouter model string used.
    retrieved_docs : list[dict]
        Raw retrieval results for downstream analysis.
    metadata : dict
        Additional metadata (token counts, latency, etc.)
    """
    answer: str
    question: str
    context: str
    model: str
    retrieved_docs: list[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def is_uncertain(self) -> bool:
        """True if the model flagged insufficient context."""
        return "do not contain sufficient information" in self.answer.lower()

    @property
    def flags_contradiction(self) -> bool:
        """True if the model detected conflicting source documents."""
        return "conflicting information" in self.answer.lower()


# ── Generator ─────────────────────────────────────────────────────────────────

class Generator:
    """
    Generates grounded, cited answers from retrieved medical context.

    Uses a system prompt that enforces strict grounding rules:
    - Every factual claim must be supported by a cited source.
    - Contradictions between sources must be explicitly flagged.
    - No speculation or outside knowledge is permitted.

    Parameters
    ----------
    model : str | None
        OpenRouter model string. Defaults to config value.
    prompt_version : str
        Which version of prompts.yaml to use.

    Examples
    --------
    >>> gen = Generator()
    >>> response = gen.generate(question="What is warfarin's half-life?", context=ctx)
    >>> print(response.answer)
    """

    def __init__(
        self,
        model: str | None = None,
        prompt_version: str | None = None,
    ) -> None:
        cfg = load_config()
        gen_cfg = cfg["generator"]

        self.model = model or gen_cfg["model"]
        self.temperature = gen_cfg["temperature"]
        self.max_tokens = gen_cfg["max_tokens"]

        version = prompt_version or gen_cfg["prompt_version"]
        prompts = load_prompts()
        if version not in prompts:
            raise ValueError(f"Prompt version '{version}' not found in prompts.yaml.")
        self._prompts = prompts[version]

        logger.info(f"Generator ready (model={self.model}, prompt_version={version})")

    def generate(
        self,
        question: str,
        context: str,
        retrieved_docs: list[dict] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> GeneratorResponse:
        """
        Generate a grounded answer for a question given context.

        Parameters
        ----------
        question : str
            The user's clinical or drug-safety question.
        context : str
            Formatted context from the Retriever (numbered sources).
        retrieved_docs : list[dict] | None
            Raw retrieval results to attach to the response.
        metadata : dict | None
            Additional metadata to attach (e.g. context_token_count).

        Returns
        -------
        GeneratorResponse
            Structured response with answer and provenance.
        """
        if not question.strip():
            raise ValueError("Question cannot be empty.")
        if not context.strip():
            raise ValueError("Context cannot be empty. Retrieval must run before generation.")

        system_prompt = self._prompts["generator_system"]
        user_prompt = self._prompts["generator_user"].format(
            context=context,
            question=question,
        )

        logger.debug(f"Generating answer for: {question[:80]}...")
        answer = call_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        response = GeneratorResponse(
            answer=answer,
            question=question,
            context=context,
            model=self.model,
            retrieved_docs=retrieved_docs or [],
            metadata=metadata or {},
        )

        if response.flags_contradiction:
            logger.info("Generator flagged a contradiction in source documents.")
        if response.is_uncertain:
            logger.info("Generator indicated insufficient context.")

        return response

    def generate_n(
        self,
        question: str,
        context: str,
        n: int = 5,
        temperature: float = 0.7,
    ) -> list[str]:
        """
        Generate N different responses for the same question/context.

        Used by the self-consistency hallucination detector, which
        requires varied outputs to measure agreement.

        Parameters
        ----------
        question : str
            The question to answer.
        context : str
            Context block from retriever.
        n : int
            Number of responses to generate.
        temperature : float
            Sampling temperature (must be > 0 for variance).

        Returns
        -------
        list[str]
            N independent response strings.
        """
        if n < 2:
            raise ValueError("n must be at least 2 for consistency checking.")

        prompts = load_prompts()
        system = prompts[load_config()["generator"]["prompt_version"]]["consistency_system"]
        user = prompts[load_config()["generator"]["prompt_version"]]["consistency_user"].format(
            context=context,
            question=question,
        )

        responses = []
        for i in range(n):
            answer = call_llm(
                system_prompt=system,
                user_prompt=user,
                model=self.model,
                temperature=temperature,
                max_tokens=256,
            )
            responses.append(answer)
            logger.debug(f"Consistency sample {i+1}/{n} complete.")

        return responses
