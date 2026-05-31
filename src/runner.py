"""
EvalSuite runner — loads JSONL cases, runs an evaluator, writes JSON results.

Each JSONL file contains one case per line. Field names must match the
corresponding Case dataclass for the chosen eval type.

Usage:
    python -m src.runner --eval hallucination --cases datasets/hallucination/sample.jsonl
    python -m src.runner --eval factuality   --cases datasets/factuality/sample.jsonl --output outputs/run.json
    python -m src.runner --eval jailbreak    --cases datasets/jailbreak/sample.jsonl  --strategy heuristic
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from src.evals.factuality import (
    FactualityCase,
    FactualityEvaluator,
    FactualityResult,
)
from src.evals.factuality import Strategy as FStrategy
from src.evals.hallucination import (
    HallucinationCase,
    HallucinationEvaluator,
    HallucinationResult,
)
from src.evals.hallucination import Strategy as HStrategy
from src.evals.jailbreak import (
    AttackFamily,
    JailbreakCase,
    JailbreakEvaluator,
    JailbreakResult,
)
from src.evals.jailbreak import Strategy as JStrategy

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "claude-haiku-4-5-20251001"


# ---------------------------------------------------------------------------
# JSON serialisation
# ---------------------------------------------------------------------------


class _Encoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return dataclasses.asdict(obj)
        if isinstance(obj, Enum):
            return obj.value
        return super().default(obj)


# ---------------------------------------------------------------------------
# JSONL loading
# ---------------------------------------------------------------------------


def load_jsonl(path: Path) -> list[dict]:
    cases = []
    with open(path) as fh:
        for lineno, raw in enumerate(fh, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                cases.append(json.loads(raw))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {lineno} of {path}: {exc}") from exc
    return cases


# ---------------------------------------------------------------------------
# Case builders
# ---------------------------------------------------------------------------


def _build_hallucination_cases(rows: list[dict]) -> list[HallucinationCase]:
    return [
        HallucinationCase(
            id=row["id"],
            prompt=row["prompt"],
            context=row["context"],
            response=row["response"],
            expected_facts=row.get("expected_facts", []),
            metadata=row.get("metadata", {}),
        )
        for row in rows
    ]


def _build_factuality_cases(rows: list[dict]) -> list[FactualityCase]:
    return [
        FactualityCase(
            id=row["id"],
            prompt=row["prompt"],
            response=row["response"],
            reference=row.get("reference", ""),
            metadata=row.get("metadata", {}),
        )
        for row in rows
    ]


def _build_jailbreak_cases(rows: list[dict]) -> list[JailbreakCase]:
    return [
        JailbreakCase(
            id=row["id"],
            attack_prompt=row["attack_prompt"],
            response=row["response"],
            attack_family=AttackFamily(row.get("attack_family", "unknown")),
            system_prompt=row.get("system_prompt", ""),
            metadata=row.get("metadata", {}),
        )
        for row in rows
    ]


# ---------------------------------------------------------------------------
# Metric aggregation
# ---------------------------------------------------------------------------


def _metrics_hallucination(results: list[HallucinationResult]) -> dict:
    if not results:
        return {}
    scores = [r.score for r in results]
    hallucinated = [r for r in results if r.is_hallucination]
    return {
        "total_cases": len(results),
        "mean_score": round(sum(scores) / len(scores), 4),
        "hallucination_rate": round(len(hallucinated) / len(results), 4),
        "flagged_claim_density": round(
            sum(len(r.flagged_claims) for r in hallucinated) / len(hallucinated)
            if hallucinated else 0.0,
            4,
        ),
    }


def _metrics_factuality(results: list[FactualityResult]) -> dict:
    if not results:
        return {}
    scores = [r.score for r in results]
    n = len(results)
    return {
        "total_cases": n,
        "mean_score": round(sum(scores) / n, 4),
        "mean_verified_count": round(sum(len(r.verified_claims) for r in results) / n, 2),
        "mean_failed_count": round(sum(len(r.failed_claims) for r in results) / n, 2),
        "mean_unverifiable_count": round(sum(len(r.unverifiable_claims) for r in results) / n, 2),
    }


def _metrics_jailbreak(results: list[JailbreakResult]) -> dict:
    if not results:
        return {}
    n = len(results)
    families: dict[str, list[JailbreakResult]] = {}
    for r in results:
        families.setdefault(r.attack_family, []).append(r)
    return {
        "total_cases": n,
        "pass_rate": round(sum(1 for r in results if r.passed) / n, 4),
        "mean_compliance_score": round(sum(r.compliance_score for r in results) / n, 4),
        "pass_rate_by_family": {
            family: round(sum(1 for r in fam if r.passed) / len(fam), 4)
            for family, fam in families.items()
        },
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run(
    eval_type: str,
    cases_path: Path,
    strategy: str | None = None,
    model: str = _DEFAULT_MODEL,
    threshold: float = 0.5,
    output_path: Path | None = None,
) -> dict:
    """
    Run an eval suite and return the result dict.

    If output_path is provided, the result is also written as JSON.
    The printed return value is always suitable for json.dumps(_Encoder).
    """
    rows = load_jsonl(cases_path)
    logger.info("Loaded %d cases from %s", len(rows), cases_path)

    if eval_type == "hallucination":
        cases = _build_hallucination_cases(rows)
        evaluator = HallucinationEvaluator(
            strategy=HStrategy(strategy or HStrategy.LLM_JUDGE),
            model=model,
            threshold=threshold,
        )
        results = evaluator.evaluate_batch(cases)
        metrics = _metrics_hallucination(results)

    elif eval_type == "factuality":
        cases = _build_factuality_cases(rows)
        evaluator = FactualityEvaluator(
            strategy=FStrategy(strategy or FStrategy.LLM_JUDGE),
            model=model,
        )
        results = evaluator.evaluate_batch(cases)
        metrics = _metrics_factuality(results)

    elif eval_type == "jailbreak":
        cases = _build_jailbreak_cases(rows)
        evaluator = JailbreakEvaluator(
            strategy=JStrategy(strategy or JStrategy.LLM_JUDGE),
            model=model,
            compliance_threshold=threshold,
        )
        results = evaluator.evaluate_batch(cases)
        metrics = _metrics_jailbreak(results)

    else:
        raise ValueError(
            f"Unknown eval_type: {eval_type!r}. Choose: hallucination, factuality, jailbreak"
        )

    output = {
        "eval_type": eval_type,
        "strategy": strategy or "llm_judge",
        "model": model,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cases_path": str(cases_path),
        "metrics": metrics,
        "results": results,
    }

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(output, cls=_Encoder, indent=2))
        logger.info("Results written to %s", output_path)

    return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Run an eval suite against a JSONL case file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--eval",
        required=True,
        choices=["hallucination", "factuality", "jailbreak"],
        dest="eval_type",
        help="Eval type to run",
    )
    parser.add_argument("--cases", required=True, type=Path, help="Path to JSONL case file")
    parser.add_argument("--strategy", default=None, help="Scoring strategy (default: llm_judge)")
    parser.add_argument("--model", default=_DEFAULT_MODEL, help="Model identifier")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Binary label threshold (hallucination score / jailbreak compliance)",
    )
    parser.add_argument("--output", type=Path, default=None, help="Write full JSON results here")
    args = parser.parse_args(argv)

    output = run(
        eval_type=args.eval_type,
        cases_path=args.cases,
        strategy=args.strategy,
        model=args.model,
        threshold=args.threshold,
        output_path=args.output,
    )
    print(json.dumps(output["metrics"], indent=2))


if __name__ == "__main__":
    _cli()
