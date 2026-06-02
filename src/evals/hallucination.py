"""
Hallucination evaluator: detects when a model response contains claims not
supported by or contradicting the provided grounding context.

Three scoring strategies:
  - llm_judge:         LLM-as-judge with structured JSON output (default)
  - nli:               NLI cross-encoder entailment scoring (local, no API cost)
  - self_consistency:  Sample N responses, score variance as a hallucination signal
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

logger = logging.getLogger(__name__)


class Strategy(str, Enum):
    LLM_JUDGE = "llm_judge"
    NLI = "nli"
    SELF_CONSISTENCY = "self_consistency"


@dataclass
class HallucinationCase:
    id: str
    prompt: str
    context: str
    response: str
    expected_facts: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class HallucinationResult:
    case_id: str
    score: float            # 1.0 = fully grounded, 0.0 = fully hallucinated
    is_hallucination: bool
    strategy: str
    flagged_claims: list[str] = field(default_factory=list)
    explanation: str = ""
    raw_response: str = ""
    metadata: dict = field(default_factory=dict)


_CONSISTENCY_SYSTEM = """\
You are an evaluation judge assessing both factual grounding and self-consistency \
across multiple model responses to the same prompt.

Respond ONLY with valid JSON matching this exact schema:
{
  "consistency_score": <float 0.0–1.0>,
  "score": <float 0.0–1.0>,
  "is_hallucination": <bool>,
  "flagged_claims": [<string>, ...],
  "explanation": <string>
}

consistency_score: 1.0 = all sampled responses agree on every key fact; \
0.0 = responses frequently contradict each other.
score: how well-grounded the TARGET response is in the given context. \
1.0 = every claim is supported; 0.0 = entirely fabricated or contradicts context.
is_hallucination: true if the Target response contains unsupported or contradictory claims."""

_CONSISTENCY_USER = """\
## Context
{context}

## Prompt
{prompt}

## Sampled Responses
{samples_text}

## Target Response
{response}

Evaluate: (1) how consistent the sampled responses are with each other on key facts, \
and (2) how well-grounded the Target Response is in the Context."""


# System prompt is stable across calls — good candidate for prompt caching.
_JUDGE_SYSTEM = """\
You are an evaluation judge assessing factual grounding. Determine whether the \
model response contains claims that are unsupported by or contradict the provided context.

Respond ONLY with valid JSON matching this exact schema:
{
  "score": <float 0.0–1.0>,
  "is_hallucination": <bool>,
  "flagged_claims": [<string>, ...],
  "explanation": <string>
}

Where score 1.0 means every claim in the response is fully supported by the context, \
and 0.0 means the response is entirely fabricated or contradicts the context."""

_JUDGE_USER = """\
## Context
{context}

## Model Response
{response}

Evaluate whether the response is grounded in the context above."""


class HallucinationEvaluator:
    def __init__(
        self,
        strategy: Strategy = Strategy.LLM_JUDGE,
        model: str = "claude-haiku-4-5-20251001",
        threshold: float = 0.5,
    ) -> None:
        self.strategy = strategy
        self.model = model
        self.threshold = threshold
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic()
        return self._client

    def evaluate(self, case: HallucinationCase) -> HallucinationResult:
        if self.strategy == Strategy.LLM_JUDGE:
            return self._evaluate_llm_judge(case)
        if self.strategy == Strategy.NLI:
            return self._evaluate_nli(case)
        if self.strategy == Strategy.SELF_CONSISTENCY:
            return self._evaluate_self_consistency(case)
        raise ValueError(f"Unknown strategy: {self.strategy}")

    def evaluate_batch(self, cases: list[HallucinationCase]) -> list[HallucinationResult]:
        return [self.evaluate(case) for case in cases]

    # ------------------------------------------------------------------
    # Strategies
    # ------------------------------------------------------------------

    def _evaluate_llm_judge(self, case: HallucinationCase) -> HallucinationResult:
        message = self.client.messages.create(
            model=self.model,
            max_tokens=512,
            system=[
                {
                    "type": "text",
                    "text": _JUDGE_SYSTEM,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[
                {
                    "role": "user",
                    "content": _JUDGE_USER.format(
                        context=case.context,
                        response=case.response,
                    ),
                }
            ],
        )
        raw = message.content[0].text

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Judge returned non-JSON for case %s: %.200s", case.id, raw)
            return HallucinationResult(
                case_id=case.id,
                score=0.0,
                is_hallucination=True,
                strategy=self.strategy.value,
                explanation=raw,
                raw_response=raw,
            )

        score = float(parsed.get("score", 0.0))
        return HallucinationResult(
            case_id=case.id,
            score=score,
            is_hallucination=score < self.threshold,
            strategy=self.strategy.value,
            flagged_claims=parsed.get("flagged_claims", []),
            explanation=parsed.get("explanation", ""),
            raw_response=raw,
        )

    def _evaluate_nli(self, case: HallucinationCase) -> HallucinationResult:
        """
        Score via NLI entailment. Requires sentence-transformers; falls back to
        keyword overlap if the package is not installed.
        """
        claims = case.expected_facts if case.expected_facts else [case.response]

        try:
            from sentence_transformers import CrossEncoder

            model = CrossEncoder("cross-encoder/nli-deberta-v3-small")
            pairs = [(case.context, claim) for claim in claims]
            raw_scores = model.predict(pairs)
            # Each row is [contradiction, neutral, entailment]
            entailment_scores = [float(row[2]) for row in raw_scores]
            score = sum(entailment_scores) / len(entailment_scores)
        except ImportError:
            logger.warning(
                "sentence-transformers not installed; using keyword overlap fallback"
            )
            score = self._keyword_overlap(case.context, case.response)

        return HallucinationResult(
            case_id=case.id,
            score=score,
            is_hallucination=score < self.threshold,
            strategy=self.strategy.value,
            explanation="NLI entailment score across expected_facts",
        )

    def _evaluate_self_consistency(
        self, case: HallucinationCase, n_samples: int = 5
    ) -> HallucinationResult:
        """
        Sample N independent responses to the same prompt, then measure claim
        variance across them. High variance signals that the model is uncertain
        or confabulating; low variance signals reliable grounding.

        A single consistency-judge call evaluates both inter-sample agreement
        (consistency_score) and whether the TARGET response is grounded in the
        context (score / is_hallucination). This keeps the total API call count
        at n_samples + 1 while producing a richer signal than the single-judge
        approach of stuffing all samples into one context string.
        """
        samples = []
        for _ in range(n_samples):
            msg = self.client.messages.create(
                model=self.model,
                max_tokens=256,
                messages=[
                    {
                        "role": "user",
                        "content": f"Context: {case.context}\n\n{case.prompt}",
                    }
                ],
            )
            samples.append(msg.content[0].text)

        samples_text = "\n\n---\n\n".join(
            f"Sample {i + 1}: {s}" for i, s in enumerate(samples)
        )
        message = self.client.messages.create(
            model=self.model,
            max_tokens=512,
            system=[
                {
                    "type": "text",
                    "text": _CONSISTENCY_SYSTEM,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[
                {
                    "role": "user",
                    "content": _CONSISTENCY_USER.format(
                        context=case.context,
                        prompt=case.prompt,
                        samples_text=samples_text,
                        response=case.response,
                    ),
                }
            ],
        )
        raw = message.content[0].text

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning(
                "Consistency judge returned non-JSON for case %s: %.200s", case.id, raw
            )
            return HallucinationResult(
                case_id=case.id,
                score=0.0,
                is_hallucination=True,
                strategy=self.strategy.value,
                explanation=raw,
                raw_response=raw,
                metadata={"n_samples": n_samples},
            )

        score = float(parsed.get("score", 0.0))
        consistency_score = float(parsed.get("consistency_score", 0.0))
        return HallucinationResult(
            case_id=case.id,
            score=score,
            is_hallucination=score < self.threshold,
            strategy=self.strategy.value,
            flagged_claims=parsed.get("flagged_claims", []),
            explanation=parsed.get("explanation", ""),
            raw_response=raw,
            metadata={
                "n_samples": n_samples,
                "consistency_score": consistency_score,
            },
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _keyword_overlap(context: str, response: str) -> float:
        ctx_tokens = set(context.lower().split())
        resp_tokens = set(response.lower().split())
        if not resp_tokens:
            return 0.0
        return len(ctx_tokens & resp_tokens) / len(resp_tokens)
