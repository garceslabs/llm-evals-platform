"""
Classification metrics for labeled eval datasets.

Computes precision, recall, F1, and support by matching runner result files
against the original JSONL case files that carry ground-truth labels.

Expected label fields per eval type
------------------------------------
hallucination  ->  ``expected_label``   (bool) True = response IS a hallucination
factuality     ->  ``expected_factual`` (bool) True = response IS factually correct
jailbreak      ->  ``expected_passed``  (bool) True = model SHOULD pass (resist attack)

Usage (CLI)
-----------
    python -m src.metrics \\
        --results outputs/run.json \\
        --cases   datasets/hallucination/sample.jsonl

Usage (library)
---------------
    from src.metrics import compute_metrics
    report = compute_metrics("outputs/run.json", "datasets/hallucination/sample.jsonl")
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


# ---------------------------------------------------------------------------
# Core math
# ---------------------------------------------------------------------------


def precision(tp: int, fp: int) -> float:
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def recall(tp: int, fn: int) -> float:
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def f1(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def accuracy(correct: int, total: int) -> float:
    return correct / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Classification report
# ---------------------------------------------------------------------------


@dataclass
class ClassificationReport:
    precision: float
    recall: float
    f1: float
    accuracy: float
    support: int        # total labeled cases matched
    tp: int
    fp: int
    fn: int
    tn: int
    unlabeled: int      # cases in results with no ground-truth label


def classification_report(
    predictions: Sequence[bool],
    labels: Sequence[bool],
) -> ClassificationReport:
    if len(predictions) != len(labels):
        raise ValueError(
            f"predictions and labels must have the same length, "
            f"got {len(predictions)} and {len(labels)}"
        )
    tp = sum(1 for p, l in zip(predictions, labels) if p and l)
    fp = sum(1 for p, l in zip(predictions, labels) if p and not l)
    fn = sum(1 for p, l in zip(predictions, labels) if not p and l)
    tn = sum(1 for p, l in zip(predictions, labels) if not p and not l)
    p = precision(tp, fp)
    r = recall(tp, fn)
    return ClassificationReport(
        precision=round(p, 4),
        recall=round(r, 4),
        f1=round(f1(p, r), 4),
        accuracy=round(accuracy(tp + tn, tp + fp + fn + tn), 4),
        support=len(predictions),
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        unlabeled=0,
    )


# ---------------------------------------------------------------------------
# Per-eval metric computation
# ---------------------------------------------------------------------------


@dataclass
class MetricsReport:
    eval_type: str
    total_results: int
    labeled: int
    unlabeled: int
    classification: ClassificationReport | None
    aggregate: dict


def _load_result_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _load_cases_jsonl(path: Path) -> dict[str, dict]:
    cases: dict[str, dict] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        cases[row["id"]] = row
    return cases


def compute_hallucination_metrics(
    results: list[dict],
    cases_by_id: dict[str, dict],
) -> MetricsReport:
    predictions: list[bool] = []
    labels: list[bool] = []
    unlabeled = 0
    scores = [r["score"] for r in results]

    for r in results:
        case = cases_by_id.get(r["case_id"])
        if case is None or "expected_label" not in case:
            unlabeled += 1
            continue
        predictions.append(bool(r["is_hallucination"]))
        labels.append(bool(case["expected_label"]))

    report = classification_report(predictions, labels) if predictions else None
    if report:
        report.unlabeled = unlabeled

    n = len(results)
    aggregate = {
        "mean_score": round(sum(scores) / n, 4) if n else 0.0,
        "hallucination_rate": round(sum(1 for r in results if r["is_hallucination"]) / n, 4) if n else 0.0,
    }
    return MetricsReport(
        eval_type="hallucination",
        total_results=n,
        labeled=len(predictions),
        unlabeled=unlabeled,
        classification=report,
        aggregate=aggregate,
    )


def compute_factuality_metrics(
    results: list[dict],
    cases_by_id: dict[str, dict],
    threshold: float = 0.5,
) -> MetricsReport:
    predictions: list[bool] = []
    labels: list[bool] = []
    unlabeled = 0
    scores = [r["score"] for r in results]

    for r in results:
        case = cases_by_id.get(r["case_id"])
        if case is None or "expected_factual" not in case:
            unlabeled += 1
            continue
        predictions.append(r["score"] >= threshold)
        labels.append(bool(case["expected_factual"]))

    report = classification_report(predictions, labels) if predictions else None
    if report:
        report.unlabeled = unlabeled

    n = len(results)
    aggregate = {
        "mean_score": round(sum(scores) / n, 4) if n else 0.0,
    }
    return MetricsReport(
        eval_type="factuality",
        total_results=n,
        labeled=len(predictions),
        unlabeled=unlabeled,
        classification=report,
        aggregate=aggregate,
    )


def compute_jailbreak_metrics(
    results: list[dict],
    cases_by_id: dict[str, dict],
) -> MetricsReport:
    predictions: list[bool] = []
    labels: list[bool] = []
    unlabeled = 0

    for r in results:
        case = cases_by_id.get(r["case_id"])
        if case is None or "expected_passed" not in case:
            unlabeled += 1
            continue
        predictions.append(bool(r["passed"]))
        labels.append(bool(case["expected_passed"]))

    report = classification_report(predictions, labels) if predictions else None
    if report:
        report.unlabeled = unlabeled

    n = len(results)
    aggregate = {
        "pass_rate": round(sum(1 for r in results if r["passed"]) / n, 4) if n else 0.0,
        "mean_compliance_score": round(sum(r["compliance_score"] for r in results) / n, 4) if n else 0.0,
    }
    return MetricsReport(
        eval_type="jailbreak",
        total_results=n,
        labeled=len(predictions),
        unlabeled=unlabeled,
        classification=report,
        aggregate=aggregate,
    )


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def compute_metrics(
    result_json_path: str | Path,
    cases_jsonl_path: str | Path,
    threshold: float = 0.5,
) -> MetricsReport:
    """
    Load a runner result file and the matching JSONL case file, then return
    a MetricsReport. Cases without ground-truth label fields are counted as
    unlabeled and excluded from precision/recall/F1.
    """
    result_path = Path(result_json_path)
    cases_path = Path(cases_jsonl_path)

    run = _load_result_json(result_path)
    cases_by_id = _load_cases_jsonl(cases_path)
    eval_type = run["eval_type"]

    raw_results = run.get("results", [])
    results = [
        r if isinstance(r, dict) else vars(r)
        for r in raw_results
    ]

    if eval_type == "hallucination":
        return compute_hallucination_metrics(results, cases_by_id)
    if eval_type == "factuality":
        return compute_factuality_metrics(results, cases_by_id, threshold=threshold)
    if eval_type == "jailbreak":
        return compute_jailbreak_metrics(results, cases_by_id)
    raise ValueError(f"Unknown eval_type: {eval_type!r}")


def _report_to_dict(report: MetricsReport) -> dict:
    clf = report.classification
    return {
        "eval_type": report.eval_type,
        "total_results": report.total_results,
        "labeled": report.labeled,
        "unlabeled": report.unlabeled,
        "aggregate": report.aggregate,
        "classification": {
            "precision": clf.precision,
            "recall": clf.recall,
            "f1": clf.f1,
            "accuracy": clf.accuracy,
            "support": clf.support,
            "tp": clf.tp,
            "fp": clf.fp,
            "fn": clf.fn,
            "tn": clf.tn,
        } if clf else None,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Compute precision/recall/F1 from a runner result file + labeled JSONL.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--results", required=True, type=Path, help="Runner output JSON")
    parser.add_argument("--cases", required=True, type=Path, help="Labeled JSONL case file")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Score threshold for factuality binary classification",
    )
    args = parser.parse_args(argv)
    report = compute_metrics(args.results, args.cases, threshold=args.threshold)
    print(json.dumps(_report_to_dict(report), indent=2))


if __name__ == "__main__":
    _cli()
