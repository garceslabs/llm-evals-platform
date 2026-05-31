# LLM Evals Platform

Production-style evaluation infrastructure for hallucination detection, factuality scoring, jailbreak resistance testing, and reliability benchmarking of LLM systems.

This project is built as a Staff AI Engineer portfolio artifact — prioritizing evaluation methodology, clean interfaces, and composable design over UI.

---

## Problem

Deploying LLMs in production requires answering specific, measurable questions:

- Does the model fabricate facts when given a grounding context?
- Does it maintain consistent answers under paraphrased prompts?
- Does it resist adversarial jailbreak attempts across prompt families?
- Does factual accuracy degrade across model versions or fine-tune iterations?

Off-the-shelf benchmarks (MMLU, TruthfulQA) answer questions about aggregate capability. They don't answer questions about *your* model, *your* data distribution, or *your* deployment constraints. This platform is infrastructure for that second category.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Eval Cases  (datasets/*)                                    │
│  HallucinationCase | FactualityCase | JailbreakCase          │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│  Evaluators  (src/evals/*)                                   │
│  BaseEvaluator → HallucinationEvaluator                      │
│                → FactualityEvaluator                         │
│                → JailbreakEvaluator                          │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│  Runner  (src/runner.py)                                     │
│  EvalSuite: load cases → run evaluators → collect results    │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│  Reports  (outputs/*)                                        │
│  JSON results + per-suite metric aggregation                 │
└─────────────────────────────────────────────────────────────┘
```

Each evaluator implements a single interface: `evaluate(case) → result`. The runner knows nothing about scoring strategy. Results are plain dataclasses serialized to JSON — no database required to get started.

---

## Evaluation Strategies

| Evaluator       | Primary Strategy                    | Fallback               | Notes                                  |
|-----------------|-------------------------------------|------------------------|----------------------------------------|
| Hallucination   | LLM-as-judge (Claude Haiku)         | NLI cross-encoder      | Structured JSON output; configurable threshold |
| Factuality      | Claim decomposition + verification  | Keyword overlap        | Entity and date extraction             |
| Jailbreak       | LLM compliance check                | Heuristic refusal detection | Attack family classification      |

See [`docs/eval_framework.md`](docs/eval_framework.md) for methodology detail.

---

## Quickstart

```bash
# Install
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Set credentials
cp .env.example .env
# edit .env — add ANTHROPIC_API_KEY

# Run tests (no API key required — mocked)
pytest tests/ -v

# Run a hallucination eval against a case file
python -m src.runner --eval hallucination --cases datasets/hallucination/sample.jsonl
```

---

## Project Layout

```
src/
  evals/
    base.py           # EvalCase, EvalResult, BaseEvaluator ABC
    hallucination.py  # HallucinationEvaluator (LLM judge + NLI)
    factuality.py     # FactualityEvaluator (claim decomposition)
    jailbreak.py      # JailbreakEvaluator (attack classification)
  runner.py           # EvalSuite orchestration
  metrics.py          # Aggregation: precision, recall, pass@k

tests/
  test_hallucination.py
  test_factuality.py
  test_jailbreak.py

datasets/             # JSONL eval case files (git-tracked samples)
docs/
  eval_framework.md   # Methodology
  project_plan.md     # Roadmap and decisions
outputs/              # Run results (gitignored)
```

---

## Roadmap

See [`docs/project_plan.md`](docs/project_plan.md).

**Phase 1 (current):** Core evaluators, runner, base test coverage  
**Phase 2:** Factuality + jailbreak evaluators, dataset management, CI integration  
**Phase 3:** Self-consistency eval, regression tracking across model versions  
**Phase 4:** Agent task evaluation, multi-turn reliability benchmarks
