"""
Regression tracking: compare two runner result files and flag score drops.

Each result file is a JSON object produced by src/runner.py. The comparison
matches cases by ID and computes per-case deltas. Cases present in the baseline
but missing in the current run are counted as regressions.

Key metric (higher-is-better) per eval type
-------------------------------------------
  hallucination  →  mean_score   (1.0 = fully grounded)
  factuality     →  mean_score   (1.0 = fully factual)
  jailbreak      →  pass_rate    (1.0 = all attacks refused)

Per-case score (also higher-is-better)
---------------------------------------
  hallucination  →  result["score"]
  factuality     →  result["score"]
  jailbreak      →  1.0 − result["compliance_score"]

Usage (library)
---------------
    from src.regression import compare
    report = compare("outputs/baseline.json", "outputs/current.json")
    if report.has_regression:
        print(report.summary())

Usage (CLI)
-----------
    python scripts/compare_runs.py outputs/baseline.json outputs/current.json
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class CaseDiff:
    case_id: str
    baseline_score: float
    current_score: float
    delta: float           # current - baseline; negative = regression
    is_regression: bool    # delta < -threshold


@dataclass
class MetricDiff:
    metric: str
    baseline: float
    current: float
    delta: float
    is_regression: bool


@dataclass
class RegressionReport:
    eval_type: str
    baseline_path: str
    current_path: str
    baseline_timestamp: str
    current_timestamp: str
    threshold: float
    metric_diffs: list[MetricDiff]
    case_diffs: list[CaseDiff]
    missing_cases: list[str]          # in baseline but absent from current run
    regressions: list[CaseDiff]       # cases where delta < -threshold
    regression_rate: float            # regressions / matched_cases
    has_regression: bool              # any metric_diff OR case regression

    def summary(self) -> str:
        lines: list[str] = [
            f"Regression report: {self.eval_type}",
            f"  Baseline : {self.baseline_path} ({self.baseline_timestamp})",
            f"  Current  : {self.current_path} ({self.current_timestamp})",
            f"  Threshold: {self.threshold}",
            "",
        ]

        lines.append("Suite metrics:")
        for md in self.metric_diffs:
            flag = " [REGRESSION]" if md.is_regression else ""
            lines.append(
                f"  {md.metric:20s}  baseline={md.baseline:.4f}  "
                f"current={md.current:.4f}  delta={md.delta:+.4f}{flag}"
            )

        lines.append("")
        matched = len(self.case_diffs)
        lines.append(
            f"Per-case: {matched} matched, "
            f"{len(self.regressions)} regressed "
            f"({self.regression_rate:.1%}), "
            f"{len(self.missing_cases)} missing from current run"
        )

        if self.regressions:
            lines.append("")
            lines.append("Regressed cases:")
            for cd in self.regressions:
                lines.append(
                    f"  {cd.case_id:30s}  "
                    f"{cd.baseline_score:.4f} → {cd.current_score:.4f}  "
                    f"({cd.delta:+.4f})"
                )

        if self.missing_cases:
            lines.append("")
            lines.append("Missing cases (in baseline, not in current run):")
            for cid in self.missing_cases:
                lines.append(f"  {cid}")

        lines.append("")
        lines.append("Result: " + ("REGRESSION DETECTED" if self.has_regression else "OK"))
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "eval_type": self.eval_type,
            "baseline_path": self.baseline_path,
            "current_path": self.current_path,
            "baseline_timestamp": self.baseline_timestamp,
            "current_timestamp": self.current_timestamp,
            "threshold": self.threshold,
            "has_regression": self.has_regression,
            "regression_rate": self.regression_rate,
            "metric_diffs": [
                {
                    "metric": md.metric,
                    "baseline": md.baseline,
                    "current": md.current,
                    "delta": md.delta,
                    "is_regression": md.is_regression,
                }
                for md in self.metric_diffs
            ],
            "case_diffs": [
                {
                    "case_id": cd.case_id,
                    "baseline_score": cd.baseline_score,
                    "current_score": cd.current_score,
                    "delta": cd.delta,
                    "is_regression": cd.is_regression,
                }
                for cd in self.case_diffs
            ],
            "regressions": [cd.case_id for cd in self.regressions],
            "missing_cases": self.missing_cases,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _key_metric(eval_type: str) -> str:
    if eval_type in ("hallucination", "factuality"):
        return "mean_score"
    if eval_type == "jailbreak":
        return "pass_rate"
    raise ValueError(f"Unknown eval_type: {eval_type!r}")


def _case_score(result: dict, eval_type: str) -> float:
    """Extract a normalised higher-is-better score from a result dict."""
    if eval_type in ("hallucination", "factuality"):
        return float(result.get("score", 0.0))
    if eval_type == "jailbreak":
        return 1.0 - float(result.get("compliance_score", 1.0))
    raise ValueError(f"Unknown eval_type: {eval_type!r}")


def _index_results(results: list[dict], eval_type: str) -> dict[str, float]:
    return {r["case_id"]: _case_score(r, eval_type) for r in results}


# ---------------------------------------------------------------------------
# Core comparison
# ---------------------------------------------------------------------------


def compare(
    baseline_path: str | Path,
    current_path: str | Path,
    threshold: float = 0.05,
) -> RegressionReport:
    """
    Compare two runner result files and return a RegressionReport.

    A case is flagged as a regression if its score dropped by more than
    `threshold` (e.g. 0.05 = 5 percentage points).
    A suite regression is flagged if the key metric dropped by more than
    `threshold` between the two runs.
    """
    b_path = Path(baseline_path)
    c_path = Path(current_path)
    baseline = _load(b_path)
    current = _load(c_path)

    eval_type = baseline.get("eval_type", "")
    if current.get("eval_type", "") != eval_type:
        raise ValueError(
            f"eval_type mismatch: baseline={eval_type!r}, "
            f"current={current.get('eval_type')!r}"
        )

    key = _key_metric(eval_type)
    b_metrics = baseline.get("metrics", {})
    c_metrics = current.get("metrics", {})

    metric_diffs: list[MetricDiff] = []
    for metric in set(b_metrics) | set(c_metrics):
        b_val = float(b_metrics.get(metric, 0.0))
        c_val = float(c_metrics.get(metric, 0.0))
        # Only flag the primary key metric as a suite regression
        is_reg = (metric == key) and (c_val - b_val < -threshold)
        metric_diffs.append(
            MetricDiff(
                metric=metric,
                baseline=round(b_val, 4),
                current=round(c_val, 4),
                delta=round(c_val - b_val, 4),
                is_regression=is_reg,
            )
        )
    metric_diffs.sort(key=lambda m: m.metric)

    b_index = _index_results(baseline.get("results", []), eval_type)
    c_index = _index_results(current.get("results", []), eval_type)

    case_diffs: list[CaseDiff] = []
    missing: list[str] = []

    for case_id, b_score in b_index.items():
        if case_id not in c_index:
            missing.append(case_id)
            continue
        c_score = c_index[case_id]
        delta = round(c_score - b_score, 4)
        case_diffs.append(
            CaseDiff(
                case_id=case_id,
                baseline_score=round(b_score, 4),
                current_score=round(c_score, 4),
                delta=delta,
                is_regression=delta < -threshold,
            )
        )

    regressions = [cd for cd in case_diffs if cd.is_regression]
    regression_rate = len(regressions) / len(case_diffs) if case_diffs else 0.0
    has_suite_regression = any(md.is_regression for md in metric_diffs)
    has_regression = has_suite_regression or bool(regressions) or bool(missing)

    return RegressionReport(
        eval_type=eval_type,
        baseline_path=str(b_path),
        current_path=str(c_path),
        baseline_timestamp=baseline.get("timestamp", ""),
        current_timestamp=current.get("timestamp", ""),
        threshold=threshold,
        metric_diffs=metric_diffs,
        case_diffs=case_diffs,
        missing_cases=missing,
        regressions=regressions,
        regression_rate=round(regression_rate, 4),
        has_regression=has_regression,
    )
