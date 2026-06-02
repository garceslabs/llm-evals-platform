"""
Tests for src/metrics.py — classification metrics over labeled eval datasets.

All tests are self-contained: no file I/O, no API calls.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from src.metrics import (
    ClassificationReport,
    MetricsReport,
    accuracy,
    classification_report,
    compute_hallucination_metrics,
    compute_jailbreak_metrics,
    compute_metrics,
    compute_factuality_metrics,
    f1,
    precision,
    recall,
)


# ---------------------------------------------------------------------------
# Core math
# ---------------------------------------------------------------------------


class TestPrecision:
    def test_perfect(self):
        assert precision(10, 0) == 1.0

    def test_zero_tp(self):
        assert precision(0, 5) == 0.0

    def test_zero_denominator(self):
        assert precision(0, 0) == 0.0

    def test_partial(self):
        assert precision(3, 1) == pytest.approx(0.75)


class TestRecall:
    def test_perfect(self):
        assert recall(10, 0) == 1.0

    def test_zero_tp(self):
        assert recall(0, 5) == 0.0

    def test_zero_denominator(self):
        assert recall(0, 0) == 0.0

    def test_partial(self):
        assert recall(3, 1) == pytest.approx(0.75)


class TestF1:
    def test_perfect(self):
        assert f1(1.0, 1.0) == pytest.approx(1.0)

    def test_zero_both(self):
        assert f1(0.0, 0.0) == 0.0

    def test_harmonic_mean(self):
        assert f1(0.5, 0.5) == pytest.approx(0.5)

    def test_imbalanced(self):
        # harmonic mean penalises low recall more than arithmetic mean would
        assert f1(1.0, 0.5) == pytest.approx(2 / 3, abs=1e-6)


class TestAccuracy:
    def test_all_correct(self):
        assert accuracy(10, 10) == 1.0

    def test_none_correct(self):
        assert accuracy(0, 10) == 0.0

    def test_zero_total(self):
        assert accuracy(0, 0) == 0.0

    def test_partial(self):
        assert accuracy(7, 10) == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# classification_report
# ---------------------------------------------------------------------------


class TestClassificationReport:
    def _report(self, preds, labels):
        return classification_report(preds, labels)

    def test_perfect_classifier(self):
        r = self._report([True, False, True, False], [True, False, True, False])
        assert r.tp == 2
        assert r.tn == 2
        assert r.fp == 0
        assert r.fn == 0
        assert r.precision == 1.0
        assert r.recall == 1.0
        assert r.f1 == 1.0
        assert r.accuracy == 1.0
        assert r.support == 4

    def test_all_wrong(self):
        r = self._report([True, True], [False, False])
        assert r.tp == 0
        assert r.fp == 2
        assert r.fn == 0
        assert r.tn == 0
        assert r.precision == 0.0
        assert r.f1 == 0.0

    def test_high_recall_low_precision(self):
        # predicts positive for everything, half are actually positive
        preds = [True, True, True, True]
        labels = [True, True, False, False]
        r = self._report(preds, labels)
        assert r.recall == 1.0
        assert r.precision == 0.5
        assert r.f1 == pytest.approx(2 / 3, abs=1e-3)
        assert r.precision == pytest.approx(0.5, abs=1e-3)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            classification_report([True], [True, False])

    def test_empty_inputs(self):
        r = classification_report([], [])
        assert r.support == 0
        assert r.precision == 0.0
        assert r.recall == 0.0
        assert r.f1 == 0.0
        assert r.accuracy == 0.0

    def test_all_positive_predictions_all_negative_labels(self):
        r = self._report([True, True, True], [False, False, False])
        assert r.tp == 0
        assert r.tn == 0
        assert r.fp == 3
        assert r.fn == 0
        assert r.precision == 0.0
        assert r.recall == 0.0

    def test_single_case_correct(self):
        r = self._report([True], [True])
        assert r.tp == 1
        assert r.f1 == 1.0


# ---------------------------------------------------------------------------
# compute_hallucination_metrics
# ---------------------------------------------------------------------------


def _make_hallucination_result(case_id: str, score: float, is_hallucination: bool) -> dict:
    return {
        "case_id": case_id,
        "score": score,
        "is_hallucination": is_hallucination,
        "strategy": "llm_judge",
        "flagged_claims": [],
        "explanation": "",
    }


def _make_hallucination_case(case_id: str, expected_label: bool | None = None) -> dict:
    case: dict = {
        "id": case_id,
        "prompt": "test",
        "context": "test context",
        "response": "test response",
    }
    if expected_label is not None:
        case["expected_label"] = expected_label
    return case


class TestComputeHallucinationMetrics:
    def test_perfect_classifier(self):
        results = [
            _make_hallucination_result("c1", 0.9, False),
            _make_hallucination_result("c2", 0.1, True),
        ]
        cases = {
            "c1": _make_hallucination_case("c1", expected_label=False),
            "c2": _make_hallucination_case("c2", expected_label=True),
        }
        report = compute_hallucination_metrics(results, cases)
        assert report.classification is not None
        assert report.classification.f1 == 1.0
        assert report.labeled == 2
        assert report.unlabeled == 0

    def test_unlabeled_cases_excluded(self):
        results = [
            _make_hallucination_result("c1", 0.9, False),
            _make_hallucination_result("c2", 0.1, True),
        ]
        cases = {
            "c1": _make_hallucination_case("c1", expected_label=False),
            "c2": _make_hallucination_case("c2"),  # no expected_label
        }
        report = compute_hallucination_metrics(results, cases)
        assert report.labeled == 1
        assert report.unlabeled == 1

    def test_all_unlabeled_returns_none_classification(self):
        results = [_make_hallucination_result("c1", 0.5, True)]
        cases = {"c1": _make_hallucination_case("c1")}
        report = compute_hallucination_metrics(results, cases)
        assert report.classification is None

    def test_aggregate_metrics_present(self):
        results = [
            _make_hallucination_result("c1", 0.8, False),
            _make_hallucination_result("c2", 0.2, True),
        ]
        cases = {
            "c1": _make_hallucination_case("c1", False),
            "c2": _make_hallucination_case("c2", True),
        }
        report = compute_hallucination_metrics(results, cases)
        assert "mean_score" in report.aggregate
        assert "hallucination_rate" in report.aggregate
        assert report.aggregate["mean_score"] == pytest.approx(0.5)
        assert report.aggregate["hallucination_rate"] == pytest.approx(0.5)

    def test_missing_case_id_treated_as_unlabeled(self):
        results = [_make_hallucination_result("c1", 0.5, True)]
        cases = {}  # no matching case
        report = compute_hallucination_metrics(results, cases)
        assert report.unlabeled == 1
        assert report.labeled == 0

    def test_eval_type(self):
        report = compute_hallucination_metrics([], {})
        assert report.eval_type == "hallucination"


# ---------------------------------------------------------------------------
# compute_factuality_metrics
# ---------------------------------------------------------------------------


def _make_factuality_result(case_id: str, score: float) -> dict:
    return {
        "case_id": case_id,
        "score": score,
        "verified_claims": [],
        "failed_claims": [],
        "unverifiable_claims": [],
        "strategy": "llm_judge",
    }


def _make_factuality_case(case_id: str, expected_factual: bool | None = None) -> dict:
    case: dict = {
        "id": case_id,
        "prompt": "test",
        "response": "test",
        "reference": "test",
    }
    if expected_factual is not None:
        case["expected_factual"] = expected_factual
    return case


class TestComputeFactualityMetrics:
    def test_threshold_separates_classes(self):
        results = [
            _make_factuality_result("c1", 0.8),  # predicted factual
            _make_factuality_result("c2", 0.2),  # predicted not factual
        ]
        cases = {
            "c1": _make_factuality_case("c1", expected_factual=True),
            "c2": _make_factuality_case("c2", expected_factual=False),
        }
        report = compute_factuality_metrics(results, cases, threshold=0.5)
        assert report.classification is not None
        assert report.classification.f1 == 1.0

    def test_unlabeled_excluded(self):
        results = [_make_factuality_result("c1", 0.7)]
        cases = {"c1": _make_factuality_case("c1")}
        report = compute_factuality_metrics(results, cases)
        assert report.labeled == 0
        assert report.unlabeled == 1
        assert report.classification is None

    def test_custom_threshold(self):
        results = [_make_factuality_result("c1", 0.6)]
        cases = {"c1": _make_factuality_case("c1", expected_factual=False)}
        # with threshold=0.7, score 0.6 is predicted not-factual → correct
        report = compute_factuality_metrics(results, cases, threshold=0.7)
        assert report.classification is not None
        assert report.classification.tp == 0
        assert report.classification.tn == 1

    def test_aggregate_mean_score(self):
        results = [
            _make_factuality_result("c1", 0.8),
            _make_factuality_result("c2", 0.4),
        ]
        cases = {
            "c1": _make_factuality_case("c1", True),
            "c2": _make_factuality_case("c2", False),
        }
        report = compute_factuality_metrics(results, cases)
        assert report.aggregate["mean_score"] == pytest.approx(0.6)

    def test_eval_type(self):
        report = compute_factuality_metrics([], {})
        assert report.eval_type == "factuality"


# ---------------------------------------------------------------------------
# compute_jailbreak_metrics
# ---------------------------------------------------------------------------


def _make_jailbreak_result(case_id: str, passed: bool, compliance: float = 0.0) -> dict:
    return {
        "case_id": case_id,
        "passed": passed,
        "compliance_score": compliance,
        "attack_family": "roleplay",
        "strategy": "heuristic",
    }


def _make_jailbreak_case(case_id: str, expected_passed: bool | None = None) -> dict:
    case: dict = {
        "id": case_id,
        "attack_prompt": "test",
        "response": "test",
        "attack_family": "roleplay",
    }
    if expected_passed is not None:
        case["expected_passed"] = expected_passed
    return case


class TestComputeJailbreakMetrics:
    def test_all_passed_correctly(self):
        results = [
            _make_jailbreak_result("j1", True),
            _make_jailbreak_result("j2", True),
        ]
        cases = {
            "j1": _make_jailbreak_case("j1", expected_passed=True),
            "j2": _make_jailbreak_case("j2", expected_passed=True),
        }
        report = compute_jailbreak_metrics(results, cases)
        assert report.classification is not None
        assert report.classification.accuracy == 1.0
        assert report.aggregate["pass_rate"] == 1.0

    def test_false_positive(self):
        # model said passed=True but expected passed=False (model should have refused but didn't)
        results = [_make_jailbreak_result("j1", True, 0.8)]
        cases = {"j1": _make_jailbreak_case("j1", expected_passed=False)}
        report = compute_jailbreak_metrics(results, cases)
        assert report.classification is not None
        assert report.classification.fp == 1
        assert report.classification.tp == 0

    def test_unlabeled_excluded(self):
        results = [_make_jailbreak_result("j1", True)]
        cases = {"j1": _make_jailbreak_case("j1")}
        report = compute_jailbreak_metrics(results, cases)
        assert report.unlabeled == 1
        assert report.labeled == 0
        assert report.classification is None

    def test_pass_rate_aggregate(self):
        results = [
            _make_jailbreak_result("j1", True),
            _make_jailbreak_result("j2", False),
            _make_jailbreak_result("j3", True),
            _make_jailbreak_result("j4", True),
        ]
        cases = {
            "j1": _make_jailbreak_case("j1", True),
            "j2": _make_jailbreak_case("j2", False),
            "j3": _make_jailbreak_case("j3", True),
            "j4": _make_jailbreak_case("j4", True),
        }
        report = compute_jailbreak_metrics(results, cases)
        assert report.aggregate["pass_rate"] == pytest.approx(0.75)

    def test_eval_type(self):
        report = compute_jailbreak_metrics([], {})
        assert report.eval_type == "jailbreak"


# ---------------------------------------------------------------------------
# compute_metrics (end-to-end with temp files)
# ---------------------------------------------------------------------------


class TestComputeMetrics:
    def _write_result_json(self, tmp_path: Path, eval_type: str, results: list[dict]) -> Path:
        p = tmp_path / "result.json"
        p.write_text(json.dumps({
            "eval_type": eval_type,
            "strategy": "heuristic",
            "model": "test",
            "timestamp": "2026-01-01T00:00:00+00:00",
            "cases_path": "datasets/test.jsonl",
            "metrics": {},
            "results": results,
        }))
        return p

    def _write_cases_jsonl(self, tmp_path: Path, cases: list[dict]) -> Path:
        p = tmp_path / "cases.jsonl"
        p.write_text("\n".join(json.dumps(c) for c in cases))
        return p

    def test_hallucination_end_to_end(self, tmp_path):
        result_path = self._write_result_json(tmp_path, "hallucination", [
            {"case_id": "h1", "score": 0.9, "is_hallucination": False, "strategy": "heuristic", "flagged_claims": []},
            {"case_id": "h2", "score": 0.1, "is_hallucination": True, "strategy": "heuristic", "flagged_claims": []},
        ])
        cases_path = self._write_cases_jsonl(tmp_path, [
            {"id": "h1", "expected_label": False, "prompt": "", "context": "", "response": ""},
            {"id": "h2", "expected_label": True, "prompt": "", "context": "", "response": ""},
        ])
        report = compute_metrics(result_path, cases_path)
        assert report.eval_type == "hallucination"
        assert report.labeled == 2
        assert report.classification is not None
        assert report.classification.f1 == 1.0

    def test_jailbreak_end_to_end(self, tmp_path):
        result_path = self._write_result_json(tmp_path, "jailbreak", [
            {"case_id": "j1", "passed": True, "compliance_score": 0.0, "attack_family": "roleplay", "strategy": "heuristic"},
        ])
        cases_path = self._write_cases_jsonl(tmp_path, [
            {"id": "j1", "expected_passed": True, "attack_prompt": "", "response": "", "attack_family": "roleplay"},
        ])
        report = compute_metrics(result_path, cases_path)
        assert report.eval_type == "jailbreak"
        assert report.classification.tp == 1

    def test_factuality_end_to_end(self, tmp_path):
        result_path = self._write_result_json(tmp_path, "factuality", [
            {"case_id": "f1", "score": 0.85, "verified_claims": [], "failed_claims": [], "unverifiable_claims": [], "strategy": "llm_judge"},
        ])
        cases_path = self._write_cases_jsonl(tmp_path, [
            {"id": "f1", "expected_factual": True, "prompt": "", "response": "", "reference": ""},
        ])
        report = compute_metrics(result_path, cases_path, threshold=0.5)
        assert report.eval_type == "factuality"
        assert report.classification.tp == 1

    def test_unknown_eval_type_raises(self, tmp_path):
        result_path = self._write_result_json(tmp_path, "unknown_type", [])
        cases_path = self._write_cases_jsonl(tmp_path, [])
        with pytest.raises(ValueError, match="Unknown eval_type"):
            compute_metrics(result_path, cases_path)

    def test_empty_jsonl_blank_lines_skipped(self, tmp_path):
        result_path = self._write_result_json(tmp_path, "hallucination", [])
        cases_path = tmp_path / "cases.jsonl"
        cases_path.write_text("\n\n\n")
        report = compute_metrics(result_path, cases_path)
        assert report.total_results == 0
        assert report.labeled == 0
