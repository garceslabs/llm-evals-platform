"""
Tests for FactualityEvaluator.

All tests mock the Anthropic client — no credentials required.
Mocking pattern: assign directly to evaluator._client (client is a property).
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, call

import pytest

from src.evals.factuality import (
    FactualityCase,
    FactualityEvaluator,
    FactualityResult,
    Strategy,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

ACCURATE_CASE = FactualityCase(
    id="accurate-001",
    prompt="Tell me about the Eiffel Tower.",
    response="The Eiffel Tower is located in Paris, France. It was built in 1889 for the World's Fair.",
    reference="The Eiffel Tower is an iron lattice tower in Paris, France, constructed in 1889 as the entrance arch for the 1889 World's Fair.",
)

INACCURATE_CASE = FactualityCase(
    id="inaccurate-001",
    prompt="Tell me about the Eiffel Tower.",
    response="The Eiffel Tower is in Berlin, Germany. It was built in 1920.",
    reference="The Eiffel Tower is an iron lattice tower in Paris, France, constructed in 1889.",
)

NO_REFERENCE_CASE = FactualityCase(
    id="no-ref-001",
    prompt="What is the capital of Japan?",
    response="The capital of Japan is Tokyo.",
)


def _make_api_response(text: str) -> MagicMock:
    block = MagicMock()
    block.text = text
    response = MagicMock()
    response.content = [block]
    return response


def _make_judge_response(
    score: float,
    verified: list[str],
    failed: list[str],
    unverifiable: list[str],
    explanation: str = "",
) -> MagicMock:
    return _make_api_response(
        json.dumps(
            {
                "score": score,
                "verified_claims": verified,
                "failed_claims": failed,
                "unverifiable_claims": unverifiable,
                "explanation": explanation,
            }
        )
    )


# ---------------------------------------------------------------------------
# LLM judge strategy
# ---------------------------------------------------------------------------


class TestLLMJudgeStrategy:
    def test_accurate_response_scores_high(self):
        evaluator = FactualityEvaluator(strategy=Strategy.LLM_JUDGE)
        evaluator._client = MagicMock()
        evaluator._client.messages.create.return_value = _make_judge_response(
            score=0.95,
            verified=["Eiffel Tower is in Paris", "Built in 1889"],
            failed=[],
            unverifiable=[],
            explanation="Both claims are supported by the reference.",
        )
        result = evaluator.evaluate(ACCURATE_CASE)

        assert result.score == pytest.approx(0.95)
        assert len(result.verified_claims) == 2
        assert result.failed_claims == []
        assert result.case_id == "accurate-001"
        assert result.strategy == "llm_judge"

    def test_inaccurate_response_has_failed_claims(self):
        evaluator = FactualityEvaluator(strategy=Strategy.LLM_JUDGE)
        evaluator._client = MagicMock()
        evaluator._client.messages.create.return_value = _make_judge_response(
            score=0.0,
            verified=[],
            failed=["Located in Berlin — reference says Paris", "Built in 1920 — reference says 1889"],
            unverifiable=[],
            explanation="Both claims contradict the reference.",
        )
        result = evaluator.evaluate(INACCURATE_CASE)

        assert result.score == pytest.approx(0.0)
        assert len(result.failed_claims) == 2
        assert result.verified_claims == []

    def test_unverifiable_claims_do_not_affect_score(self):
        evaluator = FactualityEvaluator(strategy=Strategy.LLM_JUDGE)
        evaluator._client = MagicMock()
        evaluator._client.messages.create.return_value = _make_judge_response(
            score=1.0,
            verified=["Tokyo is the capital of Japan"],
            failed=[],
            unverifiable=["Tokyo has a population of 14 million"],
        )
        result = evaluator.evaluate(NO_REFERENCE_CASE)

        assert result.score == pytest.approx(1.0)
        assert len(result.unverifiable_claims) == 1

    def test_malformed_json_returns_zero_score_without_raising(self):
        evaluator = FactualityEvaluator(strategy=Strategy.LLM_JUDGE)
        evaluator._client = MagicMock()
        evaluator._client.messages.create.return_value = _make_api_response("not valid json")
        result = evaluator.evaluate(ACCURATE_CASE)

        assert isinstance(result, FactualityResult)
        assert result.score == 0.0
        assert result.strategy == "llm_judge"

    def test_judge_uses_prompt_caching(self):
        evaluator = FactualityEvaluator(strategy=Strategy.LLM_JUDGE)
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _make_judge_response(
            score=0.9, verified=[], failed=[], unverifiable=[]
        )
        evaluator._client = mock_client
        evaluator.evaluate(ACCURATE_CASE)

        call_kwargs = mock_client.messages.create.call_args.kwargs
        system_blocks = call_kwargs["system"]
        assert any(
            block.get("cache_control", {}).get("type") == "ephemeral"
            for block in system_blocks
        )

    def test_missing_reference_still_evaluates(self):
        evaluator = FactualityEvaluator(strategy=Strategy.LLM_JUDGE)
        evaluator._client = MagicMock()
        evaluator._client.messages.create.return_value = _make_judge_response(
            score=0.9, verified=["Tokyo is the capital of Japan"], failed=[], unverifiable=[]
        )
        result = evaluator.evaluate(NO_REFERENCE_CASE)
        assert isinstance(result, FactualityResult)

        # The judge call should include the no-reference placeholder
        call_kwargs = evaluator._client.messages.create.call_args.kwargs
        user_content = call_kwargs["messages"][0]["content"]
        assert "no reference provided" in user_content


# ---------------------------------------------------------------------------
# Claim decomposition strategy
# ---------------------------------------------------------------------------


class TestClaimDecompositionStrategy:
    def _make_client(self, claims: list[str], verdicts: dict[str, str]) -> MagicMock:
        extract_response = _make_api_response(json.dumps(claims))
        verify_response = _make_api_response(json.dumps(verdicts))
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = [extract_response, verify_response]
        return mock_client

    def test_all_correct_claims_score_one(self):
        claims = ["Eiffel Tower is in Paris", "Built in 1889"]
        verdicts = {c: "correct" for c in claims}
        evaluator = FactualityEvaluator(strategy=Strategy.CLAIM_DECOMPOSITION)
        evaluator._client = self._make_client(claims, verdicts)
        result = evaluator.evaluate(ACCURATE_CASE)

        assert result.score == pytest.approx(1.0)
        assert result.verified_claims == claims
        assert result.failed_claims == []

    def test_all_incorrect_claims_score_zero(self):
        claims = ["Eiffel Tower is in Berlin", "Built in 1920"]
        verdicts = {c: "incorrect" for c in claims}
        evaluator = FactualityEvaluator(strategy=Strategy.CLAIM_DECOMPOSITION)
        evaluator._client = self._make_client(claims, verdicts)
        result = evaluator.evaluate(INACCURATE_CASE)

        assert result.score == pytest.approx(0.0)
        assert result.failed_claims == claims

    def test_unverifiable_claims_excluded_from_score(self):
        claims = ["Tokyo is the capital", "Population is 14 million"]
        verdicts = {"Tokyo is the capital": "correct", "Population is 14 million": "unverifiable"}
        evaluator = FactualityEvaluator(strategy=Strategy.CLAIM_DECOMPOSITION)
        evaluator._client = self._make_client(claims, verdicts)
        result = evaluator.evaluate(NO_REFERENCE_CASE)

        # 1 correct / (1 correct + 0 incorrect) = 1.0
        assert result.score == pytest.approx(1.0)
        assert len(result.unverifiable_claims) == 1

    def test_makes_exactly_two_api_calls(self):
        claims = ["claim one", "claim two"]
        verdicts = {c: "correct" for c in claims}
        evaluator = FactualityEvaluator(strategy=Strategy.CLAIM_DECOMPOSITION)
        mock_client = self._make_client(claims, verdicts)
        evaluator._client = mock_client
        evaluator.evaluate(ACCURATE_CASE)

        assert mock_client.messages.create.call_count == 2

    def test_empty_claim_extraction_returns_perfect_score(self):
        evaluator = FactualityEvaluator(strategy=Strategy.CLAIM_DECOMPOSITION)
        evaluator._client = MagicMock()
        evaluator._client.messages.create.return_value = _make_api_response("[]")
        result = evaluator.evaluate(ACCURATE_CASE)

        assert result.score == pytest.approx(1.0)
        assert result.metadata.get("claim_count") == 0

    def test_verify_parse_failure_marks_all_unverifiable(self):
        claims = ["claim one"]
        evaluator = FactualityEvaluator(strategy=Strategy.CLAIM_DECOMPOSITION)
        evaluator._client = MagicMock()
        extract_response = _make_api_response(json.dumps(claims))
        bad_verify = _make_api_response("not json")
        evaluator._client.messages.create.side_effect = [extract_response, bad_verify]
        result = evaluator.evaluate(ACCURATE_CASE)

        assert result.unverifiable_claims == claims
        assert result.score == pytest.approx(1.0)  # no scored claims → defaults to 1.0


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------


class TestBatchEvaluation:
    def test_batch_returns_one_result_per_case(self):
        evaluator = FactualityEvaluator(strategy=Strategy.LLM_JUDGE)
        evaluator._client = MagicMock()
        evaluator._client.messages.create.return_value = _make_judge_response(
            score=0.9, verified=[], failed=[], unverifiable=[]
        )
        cases = [ACCURATE_CASE, INACCURATE_CASE, NO_REFERENCE_CASE]
        results = evaluator.evaluate_batch(cases)

        assert len(results) == 3
        assert all(isinstance(r, FactualityResult) for r in results)
