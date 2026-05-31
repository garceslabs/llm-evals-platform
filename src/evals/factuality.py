"""
Factuality evaluator: measures whether a model response contains accurate claims
with respect to a reference source or world knowledge.

Two scoring strategies:
  - llm_judge:            Single call returning a scored claim breakdown (default)
  - claim_decomposition:  Two-step — extract atomic claims, then verify each one
                          against the reference. More granular; costs two API calls.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from enum import Enum

logger = logging.getLogger(__name__)


class Strategy(str, Enum):
    LLM_JUDGE = "llm_judge"
    CLAIM_DECOMPOSITION = "claim_decomposition"


@dataclass
class FactualityCase:
    id: str
    prompt: str
    response: str
    reference: str = ""          # ground-truth text to verify against; optional
    metadata: dict = field(default_factory=dict)


@dataclass
class FactualityResult:
    case_id: str
    score: float                 # fraction of claims verified correct (0.0–1.0)
    strategy: str
    verified_claims: list[str] = field(default_factory=list)
    failed_claims: list[str] = field(default_factory=list)
    unverifiable_claims: list[str] = field(default_factory=list)
    explanation: str = ""
    raw_response: str = ""
    metadata: dict = field(default_factory=dict)


_JUDGE_SYSTEM = """\
You are a factuality evaluation judge. Given a model response and an optional reference, \
assess the factual accuracy of every claim in the response.

Respond ONLY with valid JSON matching this exact schema:
{
  "score": <float 0.0–1.0>,
  "verified_claims": [<string>, ...],
  "failed_claims": [<string>, ...],
  "unverifiable_claims": [<string>, ...],
  "explanation": <string>
}

score = verified / (verified + failed). Unverifiable claims do not affect the score.
A claim is "unverifiable" only when the reference is absent and the claim cannot be \
assessed from widely available knowledge."""

_JUDGE_USER = """\
## Reference
{reference}

## Model Response
{response}

Assess the factual accuracy of every claim in the response."""

_EXTRACT_SYSTEM = """\
You are a claim extractor. Given a text, identify and return every distinct factual \
claim as a JSON array of short, atomic strings. Each element should be a single \
self-contained claim. Do not include opinions, hedges, or filler phrases."""

_VERIFY_SYSTEM = """\
You are a fact checker. Given a list of claims and an optional reference text, \
verify each claim.

Respond ONLY with valid JSON as an object mapping each claim (verbatim) to one of:
  "correct", "incorrect", or "unverifiable"

Example:
{
  "The Eiffel Tower is in Paris": "correct",
  "It was built in 1820": "incorrect",
  "It has 3 floors": "unverifiable"
}"""


class FactualityEvaluator:
    def __init__(
        self,
        strategy: Strategy = Strategy.LLM_JUDGE,
        model: str = "claude-haiku-4-5-20251001",
    ) -> None:
        self.strategy = strategy
        self.model = model
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic()
        return self._client

    def evaluate(self, case: FactualityCase) -> FactualityResult:
        if self.strategy == Strategy.LLM_JUDGE:
            return self._evaluate_llm_judge(case)
        if self.strategy == Strategy.CLAIM_DECOMPOSITION:
            return self._evaluate_claim_decomposition(case)
        raise ValueError(f"Unknown strategy: {self.strategy}")

    def evaluate_batch(self, cases: list[FactualityCase]) -> list[FactualityResult]:
        return [self.evaluate(case) for case in cases]

    # ------------------------------------------------------------------
    # Strategies
    # ------------------------------------------------------------------

    def _evaluate_llm_judge(self, case: FactualityCase) -> FactualityResult:
        reference_text = case.reference if case.reference else "(no reference provided)"
        message = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
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
                        reference=reference_text,
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
            return FactualityResult(
                case_id=case.id,
                score=0.0,
                strategy=self.strategy.value,
                explanation=raw,
                raw_response=raw,
            )

        return FactualityResult(
            case_id=case.id,
            score=float(parsed.get("score", 0.0)),
            strategy=self.strategy.value,
            verified_claims=parsed.get("verified_claims", []),
            failed_claims=parsed.get("failed_claims", []),
            unverifiable_claims=parsed.get("unverifiable_claims", []),
            explanation=parsed.get("explanation", ""),
            raw_response=raw,
        )

    def _evaluate_claim_decomposition(self, case: FactualityCase) -> FactualityResult:
        """
        Step 1: Extract atomic claims from the response.
        Step 2: Verify all claims against the reference in a single call.

        Two API calls total, regardless of claim count.
        """
        claims = self._extract_claims(case.response)
        if not claims:
            return FactualityResult(
                case_id=case.id,
                score=1.0,
                strategy=self.strategy.value,
                explanation="No factual claims found in response.",
                metadata={"claim_count": 0},
            )

        verdicts = self._verify_claims(claims, case.reference)

        verified = [c for c, v in verdicts.items() if v == "correct"]
        failed = [c for c, v in verdicts.items() if v == "incorrect"]
        unverifiable = [c for c, v in verdicts.items() if v == "unverifiable"]

        scored_total = len(verified) + len(failed)
        score = len(verified) / scored_total if scored_total > 0 else 1.0

        return FactualityResult(
            case_id=case.id,
            score=score,
            strategy=self.strategy.value,
            verified_claims=verified,
            failed_claims=failed,
            unverifiable_claims=unverifiable,
            metadata={"claim_count": len(claims)},
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_claims(self, text: str) -> list[str]:
        message = self.client.messages.create(
            model=self.model,
            max_tokens=512,
            system=[
                {
                    "type": "text",
                    "text": _EXTRACT_SYSTEM,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": text}],
        )
        raw = message.content[0].text
        try:
            claims = json.loads(raw)
            if isinstance(claims, list):
                return [str(c) for c in claims]
        except json.JSONDecodeError:
            logger.warning("Claim extractor returned non-JSON: %.200s", raw)
        return []

    def _verify_claims(self, claims: list[str], reference: str) -> dict[str, str]:
        reference_text = reference if reference else "(no reference provided)"
        user_content = (
            f"Reference:\n{reference_text}\n\nClaims to verify:\n"
            + json.dumps(claims, indent=2)
        )
        message = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=[
                {
                    "type": "text",
                    "text": _VERIFY_SYSTEM,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": user_content}],
        )
        raw = message.content[0].text
        try:
            verdicts = json.loads(raw)
            if isinstance(verdicts, dict):
                return verdicts
        except json.JSONDecodeError:
            logger.warning("Claim verifier returned non-JSON: %.200s", raw)
        # Fall back: mark everything unverifiable rather than fail
        return {claim: "unverifiable" for claim in claims}
