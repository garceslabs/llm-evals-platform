"""
Tests for src/regression.py.

All tests are self-contained: no file I/O beyond tmp_path, no API calls.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from src.regression import (
    CaseDiff,
    MetricDiff,
    RegressionReport,
    compare,
    _case_score,
    _key_metric,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_result(
    tmp_path: Path,
    name: str,
    eval_type: str,
    metrics: dict,
    results: list[dict],
    timestamp: str = "2026-01-01T00:00:00+00:00",
) -> Path:
    p = tmp_path / name
    p.write_text(json.dumps({
        "eval_type": eval_type,
        "strategy": "heuristic",
        "model": "test-model",
        "timestamp": timestamp,
        "cases_path": "datasets/test.jsonl",
        "metrics": metrics,
        "results": results,
    }))
    return p


def _hal_result(case_id: str, score: float) -> dict:
    return {
        "case_id": case_id,
        "score": score,
        "is_hallucination": score < 0.5,
        "strategy": "llm_judge",
        "flagged_claims": [],
    }


def _fac_result(case_id: str, score: float) -> dict:
    return {
        "case_id": case_id,
        "score": score,
        "verified_claims": [],
        "failed_claims": [],
        "unverifiable_claims": [],
        "strategy": "llm_judge",
    }


def _jb_result(case_id: str, passed: bool, compliance: float) -> dict:
    return {
        "case_id": case_id,
        "passed": passed,
        "compliance_score": compliance,
        "attack_family": "roleplay",
        "strategy": "heuristic",
    }


# ---------------------------------------------------------------------------
# _key_metric
# ---------------------------------------------------------------------------


class TestKeyMetric:
    def test_hallucination(self):
        assert _key_metric("hallucination") == "mean_score"

    def test_factuality(self):
        assert _key_metric("factuality") == "mean_score"

    def test_jailbreak(self):
        assert _key_metric("jailbreak") == "pass_rate"

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown eval_type"):
            _key_metric("unknown")


# ---------------------------------------------------------------------------
# _case_score
# ---------------------------------------------------------------------------


class TestCaseScore:
    def test_hallucination_uses_score_field(self):
        result = {"score": 0.8, "is_hallucination": False}
        assert _case_score(result, "hallucination") == pytest.approx(0.8)

    def test_factuality_uses_score_field(self):
        result = {"score": 0.7}
        assert _case_score(result, "factuality") == pytest.approx(0.7)

    def test_jailbreak_inverts_compliance(self):
        result = {"compliance_score": 0.2, "passed": True}
        assert _case_score(result, "jailbreak") == pytest.approx(0.8)

    def test_jailbreak_full_compliance_scores_zero(self):
        result = {"compliance_score": 1.0, "passed": False}
        assert _case_score(result, "jailbreak") == pytest.approx(0.0)

    def test_jailbreak_full_refusal_scores_one(self):
        result = {"compliance_score": 0.0, "passed": True}
        assert _case_score(result, "jailbreak") == pytest.approx(1.0)

    def test_unknown_eval_type_raises(self):
        with pytest.raises(ValueError, match="Unknown eval_type"):
            _case_score({}, "unknown")


# ---------------------------------------------------------------------------
# compare — hallucination
# ---------------------------------------------------------------------------


class TestCompareHallucination:
    def test_no_regressions_when_scores_equal(self, tmp_path):
        results = [_hal_result("h1", 0.9), _hal_result("h2", 0.8)]
        metrics = {"mean_score": 0.85, "hallucination_rate": 0.0}
        baseline = _write_result(tmp_path, "base.json", "hallucination", metrics, results)
        current = _write_result(tmp_path, "curr.json", "hallucination", metrics, results)

        report = compare(baseline, current)

        assert not report.has_regression
        assert report.regression_rate == 0.0
        assert len(report.regressions) == 0

    def test_case_regression_flagged(self, tmp_path):
        baseline_results = [_hal_result("h1", 0.9)]
        current_results = [_hal_result("h1", 0.7)]  # dropped 0.2 > threshold 0.05
        metrics = {"mean_score": 0.9}

        b = _write_result(tmp_path, "b.json", "hallucination", metrics, baseline_results)
        c = _write_result(tmp_path, "c.json", "hallucination", {"mean_score": 0.7}, current_results)

        report = compare(b, c)

        assert report.has_regression
        assert len(report.regressions) == 1
        assert report.regressions[0].case_id == "h1"
        assert report.regressions[0].delta == pytest.approx(-0.2)

    def test_score_improvement_not_flagged(self, tmp_path):
        b = _write_result(tmp_path, "b.json", "hallucination", {"mean_score": 0.7},
                          [_hal_result("h1", 0.7)])
        c = _write_result(tmp_path, "c.json", "hallucination", {"mean_score": 0.9},
                          [_hal_result("h1", 0.9)])

        report = compare(b, c)

        assert not report.has_regression
        assert len(report.case_diffs) == 1
        assert report.case_diffs[0].delta > 0

    def test_suite_metric_regression_flagged(self, tmp_path):
        b = _write_result(tmp_path, "b.json", "hallucination", {"mean_score": 0.90},
                          [_hal_result("h1", 0.9)])
        c = _write_result(tmp_path, "c.json", "hallucination", {"mean_score": 0.80},
                          [_hal_result("h1", 0.8)])  # 0.10 drop > threshold 0.05

        report = compare(b, c, threshold=0.05)

        assert report.has_regression
        suite_reg = [m for m in report.metric_diffs if m.is_regression]
        assert len(suite_reg) == 1
        assert suite_reg[0].metric == "mean_score"

    def test_within_threshold_not_flagged(self, tmp_path):
        b = _write_result(tmp_path, "b.json", "hallucination", {"mean_score": 0.90},
                          [_hal_result("h1", 0.9)])
        c = _write_result(tmp_path, "c.json", "hallucination", {"mean_score": 0.87},
                          [_hal_result("h1", 0.87)])  # 0.03 drop < threshold 0.05

        report = compare(b, c, threshold=0.05)

        assert not report.has_regression

    def test_missing_case_flagged(self, tmp_path):
        b = _write_result(tmp_path, "b.json", "hallucination", {"mean_score": 0.9},
                          [_hal_result("h1", 0.9), _hal_result("h2", 0.8)])
        c = _write_result(tmp_path, "c.json", "hallucination", {"mean_score": 0.9},
                          [_hal_result("h1", 0.9)])  # h2 missing

        report = compare(b, c)

        assert report.has_regression
        assert "h2" in report.missing_cases

    def test_eval_type_populated(self, tmp_path):
        b = _write_result(tmp_path, "b.json", "hallucination", {}, [])
        c = _write_result(tmp_path, "c.json", "hallucination", {}, [])
        report = compare(b, c)
        assert report.eval_type == "hallucination"

    def test_timestamps_captured(self, tmp_path):
        b = _write_result(tmp_path, "b.json", "hallucination", {}, [],
                          timestamp="2026-01-01T00:00:00+00:00")
        c = _write_result(tmp_path, "c.json", "hallucination", {}, [],
                          timestamp="2026-02-01T00:00:00+00:00")
        report = compare(b, c)
        assert report.baseline_timestamp == "2026-01-01T00:00:00+00:00"
        assert report.current_timestamp == "2026-02-01T00:00:00+00:00"

    def test_custom_threshold(self, tmp_path):
        b = _write_result(tmp_path, "b.json", "hallucination", {"mean_score": 0.9},
                          [_hal_result("h1", 0.9)])
        c = _write_result(tmp_path, "c.json", "hallucination", {"mean_score": 0.85},
                          [_hal_result("h1", 0.85)])  # 0.05 drop

        # With threshold=0.10 the drop is within tolerance
        report_lenient = compare(b, c, threshold=0.10)
        assert not report_lenient.has_regression

        # With threshold=0.03 the drop exceeds tolerance
        report_strict = compare(b, c, threshold=0.03)
        assert report_strict.has_regression


# ---------------------------------------------------------------------------
# compare — jailbreak
# ---------------------------------------------------------------------------


class TestCompareJailbreak:
    def test_jailbreak_pass_rate_regression(self, tmp_path):
        b = _write_result(tmp_path, "b.json", "jailbreak", {"pass_rate": 1.0},
                          [_jb_result("j1", True, 0.0), _jb_result("j2", True, 0.0)])
        # j2 now partially complies — compliance 0.8 → case score 0.2 (was 1.0)
        c = _write_result(tmp_path, "c.json", "jailbreak", {"pass_rate": 0.5},
                          [_jb_result("j1", True, 0.0), _jb_result("j2", False, 0.8)])

        report = compare(b, c, threshold=0.05)

        assert report.has_regression
        reg_ids = [r.case_id for r in report.regressions]
        assert "j2" in reg_ids

    def test_jailbreak_no_regression_when_stable(self, tmp_path):
        results = [_jb_result("j1", True, 0.0)]
        b = _write_result(tmp_path, "b.json", "jailbreak", {"pass_rate": 1.0}, results)
        c = _write_result(tmp_path, "c.json", "jailbreak", {"pass_rate": 1.0}, results)

        report = compare(b, c)
        assert not report.has_regression


# ---------------------------------------------------------------------------
# compare — factuality
# ---------------------------------------------------------------------------


class TestCompareFactuality:
    def test_factuality_regression_flagged(self, tmp_path):
        b = _write_result(tmp_path, "b.json", "factuality", {"mean_score": 0.85},
                          [_fac_result("f1", 0.85)])
        c = _write_result(tmp_path, "c.json", "factuality", {"mean_score": 0.70},
                          [_fac_result("f1", 0.70)])

        report = compare(b, c, threshold=0.05)
        assert report.has_regression


# ---------------------------------------------------------------------------
# compare — error cases
# ---------------------------------------------------------------------------


class TestCompareErrors:
    def test_mismatched_eval_types_raises(self, tmp_path):
        b = _write_result(tmp_path, "b.json", "hallucination", {}, [])
        c = _write_result(tmp_path, "c.json", "jailbreak", {}, [])

        with pytest.raises(ValueError, match="eval_type mismatch"):
            compare(b, c)

    def test_missing_baseline_file_raises(self, tmp_path):
        c = _write_result(tmp_path, "c.json", "hallucination", {}, [])
        with pytest.raises(FileNotFoundError):
            compare(tmp_path / "nonexistent.json", c)


# ---------------------------------------------------------------------------
# RegressionReport.summary and to_dict
# ---------------------------------------------------------------------------


class TestRegressionReportOutput:
    def _minimal_report(self, has_regression: bool = False) -> RegressionReport:
        cd = CaseDiff(
            case_id="h1",
            baseline_score=0.9,
            current_score=0.7,
            delta=-0.2,
            is_regression=True,
        )
        md = MetricDiff(
            metric="mean_score",
            baseline=0.9,
            current=0.7,
            delta=-0.2,
            is_regression=True,
        )
        return RegressionReport(
            eval_type="hallucination",
            baseline_path="outputs/base.json",
            current_path="outputs/curr.json",
            baseline_timestamp="2026-01-01T00:00:00+00:00",
            current_timestamp="2026-02-01T00:00:00+00:00",
            threshold=0.05,
            metric_diffs=[md],
            case_diffs=[cd],
            missing_cases=[],
            regressions=[cd],
            regression_rate=1.0,
            has_regression=has_regression,
        )

    def test_summary_contains_eval_type(self):
        report = self._minimal_report()
        assert "hallucination" in report.summary()

    def test_summary_contains_regression_label_when_found(self):
        report = self._minimal_report(has_regression=True)
        assert "REGRESSION" in report.summary()

    def test_summary_ok_when_clean(self):
        report = self._minimal_report(has_regression=False)
        assert "OK" in report.summary()

    def test_to_dict_is_json_serialisable(self):
        report = self._minimal_report(has_regression=True)
        d = report.to_dict()
        serialised = json.dumps(d)
        assert "has_regression" in serialised

    def test_to_dict_includes_regression_ids(self):
        report = self._minimal_report(has_regression=True)
        d = report.to_dict()
        assert "h1" in d["regressions"]

    def test_to_dict_has_expected_top_level_keys(self):
        report = self._minimal_report()
        d = report.to_dict()
        for key in ("eval_type", "baseline_path", "current_path", "threshold",
                    "has_regression", "regression_rate", "metric_diffs",
                    "case_diffs", "regressions", "missing_cases"):
            assert key in d, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# Multiple regressions / regression_rate
# ---------------------------------------------------------------------------


class TestRegressionRate:
    def test_regression_rate_correct(self, tmp_path):
        b = _write_result(tmp_path, "b.json", "hallucination", {"mean_score": 0.9},
                          [_hal_result("h1", 0.9), _hal_result("h2", 0.9),
                           _hal_result("h3", 0.9), _hal_result("h4", 0.9)])
        c = _write_result(tmp_path, "c.json", "hallucination", {"mean_score": 0.75},
                          [_hal_result("h1", 0.9),   # no change
                           _hal_result("h2", 0.9),   # no change
                           _hal_result("h3", 0.5),   # regressed
                           _hal_result("h4", 0.5)])  # regressed

        report = compare(b, c, threshold=0.05)

        assert len(report.regressions) == 2
        assert report.regression_rate == pytest.approx(0.5)

    def test_no_matched_cases_rate_is_zero(self, tmp_path):
        b = _write_result(tmp_path, "b.json", "hallucination", {}, [])
        c = _write_result(tmp_path, "c.json", "hallucination", {}, [])
        report = compare(b, c)
        assert report.regression_rate == 0.0
