"""
Tests for HallucinationEvaluator.

All tests mock the Anthropic client — the evaluator logic is what's under test,
not the API. Tests can run in CI without credentials.

Mocking pattern: assign directly to evaluator._client because `client` is a
property and patch.object cannot override a read-only property on an instance.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from src.evals.hallucination import (
    HallucinationCase,
    HallucinationEvaluator,
    HallucinationResult,
    Strategy,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

GROUNDED_CASE = HallucinationCase(
    id="grounded-001",
    prompt="What is the capital of France?",
    context="France is a country in Western Europe. Its capital and largest city is Paris.",
    response="The capital of France is Paris.",
    expected_facts=["The capital of France is Paris."],
)

HALLUCINATED_CASE = HallucinationCase(
    id="hallucinated-001",
    prompt="What is the capital of France?",
    context="France is a country in Western Europe. Its capital and largest city is Paris.",
    response="The capital of France is Lyon, which has held that status since 1850.",
    expected_facts=["The capital of France is Paris."],
)

NO_FACTS_CASE = HallucinationCase(
    id="no-facts-001",
    prompt="Summarize the context.",
    context="The sky is blue during the day because of Rayleigh scattering.",
    response="The sky appears blue due to how sunlight scatters through the atmosphere.",
)


def _make_judge_response(
    score: float,
    is_hallucination: bool,
    flagged: list[str],
    explanation: str,
) -> MagicMock:
    payload = json.dumps(
        {
            "score": score,
            "is_hallucination": is_hallucination,
            "flagged_claims": flagged,
            "explanation": explanation,
        }
    )
    content_block = MagicMock()
    content_block.text = payload
    response = MagicMock()
    response.content = [content_block]
    return response


# ---------------------------------------------------------------------------
# LLM judge strategy
# ---------------------------------------------------------------------------


class TestLLMJudgeStrategy:
    def test_grounded_response_scores_high(self):
        evaluator = HallucinationEvaluator(strategy=Strategy.LLM_JUDGE)
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _make_judge_response(
            score=0.95,
            is_hallucination=False,
            flagged=[],
            explanation="Response is fully supported by context.",
        )
        evaluator._client = mock_client
        result = evaluator.evaluate(GROUNDED_CASE)

        assert result.score == pytest.approx(0.95)
        assert not result.is_hallucination
        assert result.case_id == "grounded-001"
        assert result.strategy == "llm_judge"
        assert result.flagged_claims == []

    def test_hallucinated_response_scores_low(self):
        evaluator = HallucinationEvaluator(strategy=Strategy.LLM_JUDGE)
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _make_judge_response(
            score=0.05,
            is_hallucination=True,
            flagged=["Lyon is described as capital since 1850 — contradicts context"],
            explanation="Response contradicts context which states Paris is the capital.",
        )
        evaluator._client = mock_client
        result = evaluator.evaluate(HALLUCINATED_CASE)

        assert result.score < 0.5
        assert result.is_hallucination
        assert len(result.flagged_claims) == 1

    def test_threshold_controls_binary_label(self):
        mock_response = _make_judge_response(score=0.6, is_hallucination=False, flagged=[], explanation="")

        # Score 0.6 is above default threshold 0.5 → not a hallucination
        evaluator = HallucinationEvaluator(strategy=Strategy.LLM_JUDGE, threshold=0.5)
        evaluator._client = MagicMock()
        evaluator._client.messages.create.return_value = mock_response
        assert not evaluator.evaluate(GROUNDED_CASE).is_hallucination

        # Same score 0.6 is below threshold 0.7 → hallucination
        evaluator_strict = HallucinationEvaluator(strategy=Strategy.LLM_JUDGE, threshold=0.7)
        evaluator_strict._client = MagicMock()
        evaluator_strict._client.messages.create.return_value = mock_response
        assert evaluator_strict.evaluate(GROUNDED_CASE).is_hallucination

    def test_malformed_json_returns_hallucination_without_raising(self):
        content_block = MagicMock()
        content_block.text = "I cannot determine this."
        bad_response = MagicMock()
        bad_response.content = [content_block]

        evaluator = HallucinationEvaluator(strategy=Strategy.LLM_JUDGE)
        evaluator._client = MagicMock()
        evaluator._client.messages.create.return_value = bad_response
        result = evaluator.evaluate(GROUNDED_CASE)

        assert isinstance(result, HallucinationResult)
        assert result.score == 0.0
        assert result.is_hallucination
        assert result.strategy == "llm_judge"

    def test_raw_response_is_stored(self):
        evaluator = HallucinationEvaluator(strategy=Strategy.LLM_JUDGE)
        evaluator._client = MagicMock()
        evaluator._client.messages.create.return_value = _make_judge_response(
            score=0.9, is_hallucination=False, flagged=[], explanation="ok"
        )
        result = evaluator.evaluate(GROUNDED_CASE)
        assert result.raw_response != ""

    def test_judge_uses_prompt_caching(self):
        evaluator = HallucinationEvaluator(strategy=Strategy.LLM_JUDGE)
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _make_judge_response(
            score=0.9, is_hallucination=False, flagged=[], explanation="ok"
        )
        evaluator._client = mock_client
        evaluator.evaluate(GROUNDED_CASE)

        call_kwargs = mock_client.messages.create.call_args.kwargs
        system_blocks = call_kwargs["system"]
        assert any(
            block.get("cache_control", {}).get("type") == "ephemeral"
            for block in system_blocks
        )


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------


class TestBatchEvaluation:
    def test_batch_returns_one_result_per_case(self):
        evaluator = HallucinationEvaluator(strategy=Strategy.LLM_JUDGE)
        evaluator._client = MagicMock()
        evaluator._client.messages.create.return_value = _make_judge_response(
            score=0.9, is_hallucination=False, flagged=[], explanation=""
        )
        results = evaluator.evaluate_batch([GROUNDED_CASE, HALLUCINATED_CASE, NO_FACTS_CASE])

        assert len(results) == 3
        assert all(isinstance(r, HallucinationResult) for r in results)

    def test_batch_preserves_case_ids(self):
        evaluator = HallucinationEvaluator(strategy=Strategy.LLM_JUDGE)
        evaluator._client = MagicMock()
        evaluator._client.messages.create.return_value = _make_judge_response(
            score=0.9, is_hallucination=False, flagged=[], explanation=""
        )
        results = evaluator.evaluate_batch([GROUNDED_CASE, HALLUCINATED_CASE])

        assert results[0].case_id == "grounded-001"
        assert results[1].case_id == "hallucinated-001"


# ---------------------------------------------------------------------------
# Self-consistency strategy
# ---------------------------------------------------------------------------


class TestSelfConsistencyStrategy:
    def _make_client_with_side_effect(self, n_samples: int):
        sample_block = MagicMock()
        sample_block.text = "Paris is the capital."
        sample_response = MagicMock()
        sample_response.content = [sample_block]
        judge_response = _make_judge_response(score=0.9, is_hallucination=False, flagged=[], explanation="")

        call_count = 0

        def side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            return sample_response if call_count <= n_samples else judge_response

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = side_effect
        return mock_client

    def test_strategy_tag_is_set_correctly(self):
        evaluator = HallucinationEvaluator(strategy=Strategy.SELF_CONSISTENCY)
        evaluator._client = self._make_client_with_side_effect(n_samples=5)
        result = evaluator.evaluate(GROUNDED_CASE)

        assert result.strategy == "self_consistency"
        assert result.metadata.get("n_samples") == 5

    def test_makes_n_plus_one_api_calls(self):
        n = 3
        evaluator = HallucinationEvaluator(strategy=Strategy.SELF_CONSISTENCY)
        mock_client = self._make_client_with_side_effect(n_samples=n)
        evaluator._client = mock_client
        evaluator._evaluate_self_consistency(GROUNDED_CASE, n_samples=n)

        assert mock_client.messages.create.call_count == n + 1


# ---------------------------------------------------------------------------
# Keyword overlap fallback
# ---------------------------------------------------------------------------


class TestKeywordOverlapFallback:
    def test_full_overlap_returns_one(self):
        score = HallucinationEvaluator._keyword_overlap(
            "the capital is paris", "the capital is paris"
        )
        assert score == pytest.approx(1.0)

    def test_no_overlap_returns_zero(self):
        score = HallucinationEvaluator._keyword_overlap(
            "france located europe", "bananas grow tropics"
        )
        assert score == pytest.approx(0.0)

    def test_partial_overlap_is_between_zero_and_one(self):
        score = HallucinationEvaluator._keyword_overlap(
            "france is in europe", "france is in asia"
        )
        assert 0.0 < score < 1.0

    def test_empty_response_returns_zero(self):
        score = HallucinationEvaluator._keyword_overlap("some context here", "")
        assert score == pytest.approx(0.0)

    def test_empty_context_still_returns_value(self):
        score = HallucinationEvaluator._keyword_overlap("", "some response here")
        assert score == pytest.approx(0.0)
