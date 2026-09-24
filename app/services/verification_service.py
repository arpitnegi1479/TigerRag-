from __future__ import annotations

import json
import re

from pydantic import ValidationError

from app.llm.gemini import GeminiProvider
from app.llm.prompts import VERIFY_CLAIM_PROMPT
from app.schemas.verification import ClaimVerification, VerificationVerdict


class VerificationService:
    def __init__(self, provider: GeminiProvider | None = None):
        self.provider = provider or GeminiProvider()

    def _schema(self) -> dict:
        return {
            "type": "OBJECT",
            "properties": {
                "verdict": {"type": "STRING", "enum": [item.value for item in VerificationVerdict]},
                "explanation": {"type": "STRING"},
                "supporting_spans": {"type": "ARRAY", "items": {"type": "STRING"}},
            },
            "required": ["verdict", "explanation", "supporting_spans"],
        }

    def verify_claim(self, claim: str, evidence: str) -> ClaimVerification:
        prompt = VERIFY_CLAIM_PROMPT.format(claim=claim, evidence=evidence)
        if self.provider.configured:
            try:
                raw = self.provider.generate_json(prompt, self._schema())
                return ClaimVerification.model_validate_json(raw)
            except (ValidationError, ValueError, KeyError, TypeError, json.JSONDecodeError):
                pass
            except Exception:
                pass
        return self._fallback(claim, evidence)

    def _fallback(self, claim: str, evidence: str) -> ClaimVerification:
        claim_terms = {term.lower() for term in re.findall(r"\b[a-zA-Z]{3,}\b", claim)}
        evidence_lower = evidence.lower()
        matched = sorted(term for term in claim_terms if term in evidence_lower)
        coverage = len(matched) / max(1, len(claim_terms))
        claim_is_negated = bool(re.search(r"\b(not|never|did not|does not|isn't|is not)\b", claim.lower()))
        evidence_is_negated = any(marker in evidence_lower for marker in ("not ", "never ", "did not ", "does not", "isn't", "is not"))
        contradiction = (claim_is_negated and not evidence_is_negated) or (evidence_is_negated and not claim_is_negated)

        if contradiction and coverage >= 0.35:
            verdict = VerificationVerdict.CONTRADICTS
        elif coverage >= 0.8:
            verdict = VerificationVerdict.SUPPORTS
        elif coverage >= 0.35:
            verdict = VerificationVerdict.PARTIALLY_SUPPORTS
        else:
            verdict = VerificationVerdict.DOES_NOT_SUPPORT
        return ClaimVerification(
            verdict=verdict,
            explanation=f"Fallback lexical verification matched {len(matched)} of {len(claim_terms)} claim terms.",
            supporting_spans=matched,
        )