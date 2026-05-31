"""
Tests for src/runner.py.

Tests cover JSONL loading, case building, metric aggregation, and the run()
integration path. All evaluator calls are mocked — runner logic is what's
under test, not the individual evaluators.
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.runner import (
    _build_hallucination_cases,
    _build_factuality_cases,
    _build_jailbreak_cases,
    _metrics_hallucination,
    _metrics_factuality,
    _metrics_jailbreak,
    load_jsonl,
    run,
)
from src.evals.hallucination import HallucinationResult
from src.evals.factuality import FactualityResult
from src.evals.jailbreak import JailbreakResult, AttackFamily


# ---------------------------------------------------------------------------
# JSONL loading
# ---------------------------------------------------------------------------


class TestLoadJsonl:
    def test_loads_valid_jsonl(self, tmp_path: Path):
        f = tmp_path / "cases.jsonl"
        f.write_text('{"id": "1"}\n{"id": "2"}\n')
        rows = load_jsonl(f)
        assert len(rows) == 2
        assert rows[0]["id"] == "1"

    def test_skips_blank_lines(self, tmp_path: Path):
        f = tmp_path / "cases.jsonl"
        f.write_text('{"id": "1"}\n\n{"id": "2"}\n\n')
        assert len(load_jsonl(f)) == 2

    def test_raises_on_invalid_json(self, tmp_path: Path):
        f = tmp_path / "cases.jsonl"
        f.write_text('{"id": "1"}\nnot json\n')
        with pytest.raises(ValueError, match="line 2"):
            load_jsonl(f)


# ---------------------------------------------------------------------------
# Case builders
# ---------------------------------------------------------------------------


class TestCaseBuilders:
    def test_hallucination_case_builder(self):
        rows = [
            {
                "id": "h-1",
                "prompt": "Q",
                "context": "ctx",
                "response": "resp",
                "expected_facts": ["fact"],
            }
        ]
        cases = _build_hallucination_cases(rows)
        assert len(cases) == 1
        assert cases[0].id == "h-1"
        assert cases[0].expected_facts == ["fact"]

    def test_hallucination_optional_fields_default(self):
        rows = [{"id": "h-1", "prompt": "Q", "context": "ctx", "response": "resp"}]
        cases = _build_hallucination_cases(rows)
        assert cases[0].expected_facts == []
        assert cases[0].metadata == {}

    def test_factuality_case_builder(self):
        rows = [{"id": "f-1", "prompt": "Q", "response": "resp", "reference": "ref"}]
        cases = _build_factuality_cases(rows)
        assert cases[0].reference == "ref"

    def test_jailbreak_case_builder_parses_attack_family(self):
        rows = [
            {
                "id": "j-1",
                "attack_prompt": "attack",
                "response": "resp",
                "attack_family": "roleplay",
            }
        ]
        cases = _build_jailbreak_cases(rows)
        assert cases[0].attack_family == AttackFamily.ROLEPLAY

    def test_jailbreak_defaults_to_unknown_family(self):
        rows = [{"id": "j-1", "attack_prompt": "attack", "response": "resp"}]
        cases = _build_jailbreak_cases(rows)
        assert cases[0].attack_family == AttackFamily.UNKNOWN


# ---------------------------------------------------------------------------
# Metric aggregation
# ---------------------------------------------------------------------------


def _make_hallucination_result(score: float, flagged: list[str] | None = None) -> HallucinationResult:
    return HallucinationResult(
        case_id="x",
        score=score,
        is_hallucination=score < 0.5,
        strategy="llm_judge",
        flagged_claims=flagged or [],
    )


def _make_factuality_result(score: float, verified: int = 0, failed: int = 0, unverifiable: int = 0) -> FactualityResult:
    return FactualityResult(
        case_id="x",
        score=score,
        strategy="llm_judge",
        verified_claims=["v"] * verified,
        failed_claims=["f"] * failed,
        unverifiable_claims=["u"] * unverifiable,
    )


def _make_jailbreak_result(passed: bool, compliance: float, family: str = "roleplay") -> JailbreakResult:
    return JailbreakResult(
        case_id="x",
        passed=passed,
        compliance_score=compliance,
        attack_family=family,
        strategy="llm_judge",
    )


class TestMetricsHallucination:
    def test_empty_returns_empty_dict(self):
        assert _metrics_hallucination([]) == {}

    def test_mean_score(self):
        results = [_make_hallucination_result(0.8), _make_hallucination_result(0.6)]
        metrics = _metrics_hallucination(results)
        assert metrics["mean_score"] == pytest.approx(0.7)

    def test_hallucination_rate(self):
        results = [
            _make_hallucination_result(0.2),  # hallucinated
            _make_hallucination_result(0.8),  # grounded
            _make_hallucination_result(0.9),  # grounded
        ]
        metrics = _metrics_hallucination(results)
        assert metrics["hallucination_rate"] == pytest.approx(1 / 3, abs=0.001)

    def test_flagged_claim_density_no_hallucinations(self):
        results = [_make_hallucination_result(0.9), _make_hallucination_result(0.8)]
        assert _metrics_hallucination(results)["flagged_claim_density"] == 0.0

    def test_flagged_claim_density_with_hallucinations(self):
        results = [
            _make_hallucination_result(0.1, flagged=["a", "b"]),
            _make_hallucination_result(0.2, flagged=["c"]),
        ]
        metrics = _metrics_hallucination(results)
        # (2 + 1) / 2 hallucinated cases = 1.5
        assert metrics["flagged_claim_density"] == pytest.approx(1.5)


class TestMetricsFactuality:
    def test_empty_returns_empty_dict(self):
        assert _metrics_factuality([]) == {}

    def test_mean_score(self):
        results = [_make_factuality_result(1.0), _make_factuality_result(0.5)]
        assert _metrics_factuality(results)["mean_score"] == pytest.approx(0.75)

    def test_mean_claim_counts(self):
        results = [
            _make_factuality_result(1.0, verified=2, failed=1, unverifiable=0),
            _make_factuality_result(0.5, verified=1, failed=1, unverifiable=2),
        ]
        metrics = _metrics_factuality(results)
        assert metrics["mean_verified_count"] == pytest.approx(1.5)
        assert metrics["mean_failed_count"] == pytest.approx(1.0)
        assert metrics["mean_unverifiable_count"] == pytest.approx(1.0)


class TestMetricsJailbreak:
    def test_empty_returns_empty_dict(self):
        assert _metrics_jailbreak([]) == {}

    def test_pass_rate(self):
        results = [
            _make_jailbreak_result(True, 0.1),
            _make_jailbreak_result(True, 0.2),
            _make_jailbreak_result(False, 0.9),
        ]
        assert _metrics_jailbreak(results)["pass_rate"] == pytest.approx(2 / 3, abs=0.001)

    def test_mean_compliance_score(self):
        results = [_make_jailbreak_result(True, 0.1), _make_jailbreak_result(False, 0.9)]
        assert _metrics_jailbreak(results)["mean_compliance_score"] == pytest.approx(0.5)

    def test_pass_rate_by_family(self):
        results = [
            _make_jailbreak_result(True, 0.1, "roleplay"),
            _make_jailbreak_result(False, 0.9, "roleplay"),
            _make_jailbreak_result(True, 0.1, "encoded"),
        ]
        by_family = _metrics_jailbreak(results)["pass_rate_by_family"]
        assert by_family["roleplay"] == pytest.approx(0.5)
        assert by_family["encoded"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# run() integration — evaluators mocked
# ---------------------------------------------------------------------------


class TestRunIntegration:
    def _write_jsonl(self, path: Path, rows: list[dict]) -> None:
        path.write_text("\n".join(json.dumps(r) for r in rows))

    def test_run_hallucination_returns_expected_shape(self, tmp_path: Path):
        cases_path = tmp_path / "h.jsonl"
        self._write_jsonl(cases_path, [
            {"id": "1", "prompt": "Q", "context": "ctx", "response": "resp"},
        ])
        mock_result = _make_hallucination_result(0.9)
        with patch("src.runner.HallucinationEvaluator") as MockEval:
            MockEval.return_value.evaluate_batch.return_value = [mock_result]
            output = run(eval_type="hallucination", cases_path=cases_path)

        assert output["eval_type"] == "hallucination"
        assert "metrics" in output
        assert "results" in output
        assert output["metrics"]["total_cases"] == 1

    def test_run_factuality_returns_expected_shape(self, tmp_path: Path):
        cases_path = tmp_path / "f.jsonl"
        self._write_jsonl(cases_path, [
            {"id": "1", "prompt": "Q", "response": "resp"},
        ])
        mock_result = _make_factuality_result(0.8, verified=1)
        with patch("src.runner.FactualityEvaluator") as MockEval:
            MockEval.return_value.evaluate_batch.return_value = [mock_result]
            output = run(eval_type="factuality", cases_path=cases_path)

        assert output["eval_type"] == "factuality"
        assert output["metrics"]["total_cases"] == 1

    def test_run_jailbreak_returns_expected_shape(self, tmp_path: Path):
        cases_path = tmp_path / "j.jsonl"
        self._write_jsonl(cases_path, [
            {"id": "1", "attack_prompt": "attack", "response": "refused"},
        ])
        mock_result = _make_jailbreak_result(True, 0.05)
        with patch("src.runner.JailbreakEvaluator") as MockEval:
            MockEval.return_value.evaluate_batch.return_value = [mock_result]
            output = run(eval_type="jailbreak", cases_path=cases_path)

        assert output["eval_type"] == "jailbreak"
        assert output["metrics"]["pass_rate"] == pytest.approx(1.0)

    def test_run_writes_output_file(self, tmp_path: Path):
        cases_path = tmp_path / "h.jsonl"
        self._write_jsonl(cases_path, [
            {"id": "1", "prompt": "Q", "context": "ctx", "response": "resp"},
        ])
        output_path = tmp_path / "out" / "result.json"
        mock_result = _make_hallucination_result(0.9)
        with patch("src.runner.HallucinationEvaluator") as MockEval:
            MockEval.return_value.evaluate_batch.return_value = [mock_result]
            run(eval_type="hallucination", cases_path=cases_path, output_path=output_path)

        assert output_path.exists()
        saved = json.loads(output_path.read_text())
        assert saved["eval_type"] == "hallucination"

    def test_run_raises_on_unknown_eval_type(self, tmp_path: Path):
        cases_path = tmp_path / "x.jsonl"
        cases_path.write_text('{"id": "1"}\n')
        with pytest.raises(ValueError, match="Unknown eval_type"):
            run(eval_type="nonexistent", cases_path=cases_path)

    def test_run_propagates_strategy_and_model(self, tmp_path: Path):
        cases_path = tmp_path / "h.jsonl"
        self._write_jsonl(cases_path, [
            {"id": "1", "prompt": "Q", "context": "ctx", "response": "resp"},
        ])
        with patch("src.runner.HallucinationEvaluator") as MockEval:
            MockEval.return_value.evaluate_batch.return_value = [_make_hallucination_result(0.9)]
            run(
                eval_type="hallucination",
                cases_path=cases_path,
                strategy="nli",
                model="claude-opus-4-7",
            )
        _, kwargs = MockEval.call_args
        assert kwargs["model"] == "claude-opus-4-7"
