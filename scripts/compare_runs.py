"""
Compare two runner result files and report score regressions.

Usage
-----
    python scripts/compare_runs.py outputs/baseline.json outputs/current.json
    python scripts/compare_runs.py outputs/run_a.json outputs/run_b.json --threshold 0.10
    python scripts/compare_runs.py outputs/run_a.json outputs/run_b.json --format json

Exit codes
----------
    0  No regressions detected.
    1  Regression detected (suite metric drop or per-case score drop).
    2  Usage error (mismatched eval types, missing files, etc.).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running as `python scripts/compare_runs.py` from the project root.
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.regression import compare


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare two eval result files and flag regressions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("baseline", type=Path, help="Baseline result JSON")
    parser.add_argument("current", type=Path, help="Current result JSON to compare against baseline")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="Score drop magnitude that counts as a regression",
    )
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format",
    )
    args = parser.parse_args(argv)

    try:
        report = compare(args.baseline, args.current, threshold=args.threshold)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    if args.format == "json":
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(report.summary())

    return 1 if report.has_regression else 0


if __name__ == "__main__":
    sys.exit(main())
