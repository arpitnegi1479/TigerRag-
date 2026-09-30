from __future__ import annotations

import re
from typing import Any

from app.schemas.api import EvidenceStatus

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is", "it",
    "of", "on", "or", "that", "the", "their", "this", "to", "was", "were", "which", "with",
}


def normalize_answer(value: str) -> str:
    return " ".join(_TOKEN_PATTERN.findall(value.casefold()))


def answer_tokens(value: str) -> set[str]:
    return set(_TOKEN_PATTERN.findall(value.casefold()))


def split_claims(answer: str) -> list[str]:
    return [
        claim.strip()
        for claim in re.split(r"(?<=[.!?])\s+|\n+|;\s+", answer.strip())
        if claim.strip()
    ]


def score_answer_correctness(question: dict[str, Any], answer: str, evidence_status: str) -> dict[str, Any]:
    expected_status = question.get("expected_status")
    if expected_status is None and question.get("category") == "conflict":
        expected_status = EvidenceStatus.CONFLICTING_EVIDENCE.value
    if expected_status is not None:
        matched = evidence_status == expected_status
        return {
            "score": 1.0 if matched else 0.0,
            "method": "evidence_status_match",
            "expected_status": expected_status,
            "actual_status": evidence_status,
        }

    normalized_answer = normalize_answer(answer)
    expected_values = [question.get("expected_answer", ""), *question.get("acceptable_answer_variants", [])]
    normalized_values = [normalize_answer(value) for value in expected_values if value]
    if any(value and value in normalized_answer for value in normalized_values):
        return {"score": 1.0, "method": "normalized_containment"}

    expected = answer_tokens(question.get("expected_answer", ""))
    actual = answer_tokens(answer)
    if not expected or not actual:
        return {"score": 0.0, "method": "token_f1"}
    overlap = len(expected & actual)
    precision = overlap / len(actual)
    recall = overlap / len(expected)
    score = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return {"score": round(score, 4), "method": "token_f1"}


def score_retrieval_at_k(
    cited_document_ids: list[str], gold_sources: list[str], k: int
) -> dict[str, float | None]:
    if not gold_sources:
        return {"precision_at_k": None, "recall_at_k": None}
    retrieved = list(dict.fromkeys(cited_document_ids))[: max(0, k)]
    relevant = len(set(retrieved) & set(gold_sources))
    precision = relevant / k if k > 0 else 0.0
    recall = relevant / len(set(gold_sources))
    return {"precision_at_k": round(precision, 4), "recall_at_k": round(recall, 4)}


def score_citation_support(claims: list[str], citations: list[dict[str, Any]]) -> dict[str, Any]:
    pairs = []
    for claim in claims:
        claim_tokens = {token for token in answer_tokens(claim) if token not in _STOP_WORDS and len(token) > 2}
        if not claim_tokens:
            continue
        for citation in citations:
            evidence_tokens = {
                token
                for token in answer_tokens(str(citation.get("snippet") or citation.get("content") or ""))
                if token not in _STOP_WORDS and len(token) > 2
            }
            coverage = len(claim_tokens & evidence_tokens) / len(claim_tokens)
            pairs.append(
                {
                    "claim": claim,
                    "document_id": citation.get("document_id"),
                    "chunk_id": citation.get("chunk_id"),
                    "support_coverage": round(coverage, 4),
                    "supported": coverage >= 0.5,
                    "method": "normalized_claim_token_coverage",
                }
            )
    return {
        "score": round(sum(pair["supported"] for pair in pairs) / len(pairs), 4) if pairs else None,
        "method": "normalized_claim_token_coverage" if pairs else None,
        "checked_pairs": pairs,
    }