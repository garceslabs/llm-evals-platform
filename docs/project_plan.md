# Project Plan

**Status:** Phase 2 complete — Runner, Datasets, CI  
**Last updated:** 2026-06-02

---

## Goal

Build evaluation infrastructure that a team could realistically adopt to measure LLM quality in production. The measure of success is not benchmark scores — it's whether the tooling changes what decisions engineers make about models in deployment.

---

## Scope

### In scope
- Hallucination detection (context-grounded response evaluation)
- Factuality scoring (claim-level verification against a knowledge source)
- Jailbreak resistance (adversarial prompt classification and compliance scoring)
- A composable runner that treats evaluators as interchangeable
- JSON-serializable results that can feed a CI gate or a Grafana dashboard
- Tests that mock the API layer so evals can run in CI without credentials

### Out of scope (intentionally)
- A UI or web dashboard
- Managed dataset hosting
- Human-in-the-loop annotation tooling
- Fine-tuning pipelines
- Multimodal inputs

These are the right next things *after* the evaluation methodology is solid. Building infra for them before the scoring logic is validated is waste.

---

## Phase 1 — Core Evaluators (current)

**Deliverables:**
- `src/evals/base.py` — `EvalCase`, `EvalResult`, `BaseEvaluator` abstract contract
- `src/evals/hallucination.py` — LLM-judge and NLI strategies
- `src/evals/factuality.py` — claim decomposition + verification
- `src/evals/jailbreak.py` — attack family classification + compliance check
- `tests/` — full mock-based coverage for all three evaluators
- `requirements.txt` / `pyproject.toml` — reproducible install
- `datasets/*/sample.jsonl` — 10–20 labeled cases per evaluator for smoke tests

**Key decisions:**

*LLM-as-judge as the primary strategy:* LLM judges correlate well with human ratings on hallucination and factuality tasks (Zheng et al. 2023, MT-Bench). Claude Haiku is cheap enough that judging 1000 cases costs under $1. The tradeoff is latency and non-determinism — which is why we also implement NLI as a fast, deterministic fallback.

*Structured JSON output from the judge:* Returning score + flagged claims + explanation gives more signal than a scalar. The explanation is useful for debugging; the flagged claims can feed finer-grained analysis without re-calling the API.

*Threshold as a constructor parameter:* The right cutoff between "hallucination" and "grounded" is deployment-specific. A RAG system for legal contracts needs a different threshold than a creative writing assistant. Encoding it as a default forces the caller to think about it.

*Dataclasses over Pydantic (for now):* Adds no dependencies. If we need validation at dataset ingestion boundaries or a REST API layer, we swap then — not before.

---

## Phase 2 — Runner, Datasets, CI

**Deliverables:**
- `src/runner.py` — `EvalSuite` that loads JSONL, runs evaluators, writes results
- `src/metrics.py` — precision/recall/F1 over hallucination labels; pass rate for jailbreak
- `datasets/` — 50+ labeled cases per evaluator, sourced from TruthfulQA, HaluEval, and manually authored adversarial prompts
- `.github/workflows/eval.yml` — CI job that runs evals on every PR, fails if pass rate drops > 5%
- `outputs/` — versioned result files, one per run

**Key decision — JSONL for datasets, not a database:**
Eval cases need to be version-controlled alongside code. JSONL is human-readable, diff-friendly, and directly loadable. A database adds an ops dependency that isn't justified until we're storing thousands of runs.

---

## Phase 3 — Regression Tracking

**Deliverables:**
- `src/regression.py` — compare two result files, flag per-case score drops
- `scripts/compare_runs.py` — `compare_runs outputs/run_a.json outputs/run_b.json`
- Self-consistency evaluator: sample N responses, measure claim variance

**Why self-consistency matters:** A model that gives different factual answers to the same question under paraphrasing is unreliable regardless of whether each individual answer is "correct." This matters for production systems where users rephrase queries naturally.

---

## Phase 4 — Agent and Multi-turn Evals

**Deliverables:**
- `src/evals/agent.py` — task completion scoring for multi-step agent trajectories
- Multi-turn conversation hallucination: does the model contradict earlier statements?
- Long-context faithfulness: does answer quality degrade as context window fills?

This phase is intentionally speculative. The right design depends on learnings from Phases 1–3.

---

## Known Risks

| Risk | Mitigation |
|------|-----------|
| LLM judge is non-deterministic | Pin model version; store `raw_response`; run with `temperature=0` where possible |
| Judge scores depend on prompt wording | Treat judge prompt as a versioned artifact; test prompt changes as regressions |
| Small labeled datasets make precision/recall noisy | Report confidence intervals; distinguish "not enough data" from "bad model" |
| API costs make CI eval expensive at scale | Use NLI strategy in CI; reserve LLM judge for nightly runs or PR gates on main |
