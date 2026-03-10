"""
tests/test_generator.py
========================
Unit tests for the Generator module.
"""

from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock

from src.generator.generator import Generator, GeneratorResponse


class TestGeneratorResponse:
    def test_is_uncertain_detects_disclaimer(self):
        response = GeneratorResponse(
            answer="The provided documents do not contain sufficient information.",
            question="Q?",
            context="ctx",
            model="test",
        )
        assert response.is_uncertain is True

    def test_is_uncertain_false_for_confident_answer(self):
        response = GeneratorResponse(
            answer="Warfarin interacts with NSAIDs [Source 1].",
            question="Q?",
            context="ctx",
            model="test",
        )
        assert response.is_uncertain is False

    def test_flags_contradiction_detects_conflict(self):
        response = GeneratorResponse(
            answer="Note: Sources [1] and [2] present conflicting information on this point.",
            question="Q?",
            context="ctx",
            model="test",
        )
        assert response.flags_contradiction is True

    def test_flags_contradiction_false_for_clean_answer(self):
        response = GeneratorResponse(
            answer="Drug A causes drowsiness in 10% of patients [Source 1].",
            question="Q?",
            context="ctx",
            model="test",
        )
        assert response.flags_contradiction is False


class TestGenerator:
    @pytest.fixture
    def mock_generator(self):
        """Generator with mocked LLM call."""
        with patch("src.generator.generator.call_llm") as mock_call:
            mock_call.return_value = "Warfarin interacts with aspirin [Source 1]."
            gen = Generator(model="openai/gpt-4o-mini")
            return gen, mock_call

    def test_generate_returns_response_object(self, mock_generator):
        gen, _ = mock_generator
        response = gen.generate(
            question="What interacts with warfarin?",
            context="[Source 1] Drug A content.",
        )
        assert isinstance(response, GeneratorResponse)

    def test_generate_attaches_question_and_context(self, mock_generator):
        gen, _ = mock_generator
        q = "What interacts with warfarin?"
        ctx = "[Source 1] Drug A content."
        response = gen.generate(question=q, context=ctx)
        assert response.question == q
        assert response.context == ctx

    def test_generate_empty_question_raises_error(self, mock_generator):
        gen, _ = mock_generator
        with pytest.raises(ValueError, match="Question cannot be empty"):
            gen.generate(question="  ", context="[Source 1] Some context.")

    def test_generate_empty_context_raises_error(self, mock_generator):
        gen, _ = mock_generator
        with pytest.raises(ValueError, match="Context cannot be empty"):
            gen.generate(question="What is warfarin?", context="   ")

    def test_generate_n_returns_n_strings(self, mock_generator):
        gen, mock_call = mock_generator
        mock_call.return_value = "Some answer."
        responses = gen.generate_n(question="What is warfarin?", context="[Source 1] ctx.", n=5)
        assert isinstance(responses, list)
        assert len(responses) == 4  # n-1 additional samples (answer is sample 0)

    def test_generate_n_requires_at_least_2(self, mock_generator):
        gen, _ = mock_generator
        with pytest.raises(ValueError, match="at least 2"):
            gen.generate_n("Q?", "ctx", n=1)

    def test_model_name_stored(self, mock_generator):
        gen, _ = mock_generator
        assert gen.model == "openai/gpt-4o-mini"
