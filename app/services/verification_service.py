from __future__ import annotations

import json
import re

from pydantic import ValidationError

from app.llm.gemini import GeminiProvider
from app.llm.prompts import VERIFY_CLAIM_PROMPT, escape_untrusted_content
from app.schemas.verification import ClaimVerification, VerificationVerdict
from app.services.evaluation_telemetry import evaluation_step, record_fallback

DEFAULT_FALLBACK_VERSION = "sentence-scoped-negation-v3"
LEGACY_FALLBACK_VERSION = "claim-local-negation-v2"


class VerificationService:
    def __init__(
        self,
        provider: GeminiProvider | None = None,
        *,
        fallback_version: str = DEFAULT_FALLBACK_VERSION,
    ):
        if fallback_version not in {DEFAULT_FALLBACK_VERSION, LEGACY_FALLBACK_VERSION}:
            raise ValueError(f"Unsupported verifier fallback version: {fallback_version}")
        self.provider = provider or GeminiProvider()
        self.fallback_version = fallback_version

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

    def verify_claim(self, claim: str, evidence: str, *, telemetry_step: str = "claim_verification") -> ClaimVerification:
        prompt = VERIFY_CLAIM_PROMPT.format(
            claim=escape_untrusted_content(claim),
            evidence=escape_untrusted_content(evidence),
        )
        if self.provider.configured:
            try:
                with evaluation_step(telemetry_step):
                    raw = self.provider.generate_json(prompt, self._schema())
                    return ClaimVerification.model_validate_json(raw)
            except (ValidationError, ValueError, KeyError, TypeError, json.JSONDecodeError):
                record_fallback("claim_verification", "invalid_gemini_verification")
            except Exception:
                record_fallback("claim_verification", "gemini_verification_failed")
        else:
            record_fallback("claim_verification", "gemini_not_configured")
        return self._fallback(claim, evidence)

    def _fallback(self, claim: str, evidence: str) -> ClaimVerification:
        if self.fallback_version == LEGACY_FALLBACK_VERSION:
            return self._fallback_single_claim(claim, evidence)
        claims = [
            part.strip()
            for part in re.split(r"(?<=[.!?])\s+|[\r\n]+", claim)
            if part.strip()
        ]
        if len(claims) <= 1:
            return self._fallback_single_claim(claim, evidence)

        results = [self._fallback_single_claim(item, evidence) for item in claims]
        verdicts = [item.verdict for item in results]
        if VerificationVerdict.CONTRADICTS in verdicts:
            verdict = VerificationVerdict.CONTRADICTS
        elif all(item == VerificationVerdict.SUPPORTS for item in verdicts):
            verdict = VerificationVerdict.SUPPORTS
        elif any(item in {VerificationVerdict.SUPPORTS, VerificationVerdict.PARTIALLY_SUPPORTS} for item in verdicts):
            verdict = VerificationVerdict.PARTIALLY_SUPPORTS
        else:
            verdict = VerificationVerdict.DOES_NOT_SUPPORT
        supporting_spans = list(dict.fromkeys(span for item in results for span in item.supporting_spans))
        return ClaimVerification(
            verdict=verdict,
            explanation=f"Fallback lexical verification evaluated {len(claims)} answer claims independently.",
            supporting_spans=supporting_spans,
        )

    def _fallback_single_claim(self, claim: str, evidence: str) -> ClaimVerification:
        claim_terms = {term.lower() for term in re.findall(r"\b[a-zA-Z]{3,}\b", claim)}
        evidence_sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+|[\r\n]+", evidence)
            if sentence.strip()
        ]
        sentence_matches = [
            (
                sum(term in sentence.lower() for term in claim_terms),
                sentence,
            )
            for sentence in evidence_sentences
        ]
        best_match_count, relevant_sentence = max(sentence_matches, default=(0, ""), key=lambda item: item[0])
        relevant_lower = relevant_sentence.lower()
        matched = sorted(term for term in claim_terms if term in relevant_lower)
        coverage = best_match_count / max(1, len(claim_terms))
        claim_is_negated = bool(re.search(r"\b(not|never|did not|does not|isn't|is not)\b", claim.lower()))
        evidence_is_negated = bool(
            re.search(r"\b(not|never|did not|does not|isn't|is not)\b", relevant_lower)
        )
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