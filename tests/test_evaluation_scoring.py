import pytest

from app.services.evaluation_scoring import score_answer_correctness, score_retrieval_at_k, split_claims
from app.schemas.api import EvidenceStatus


def test_correct_answer_uses_normalized_containment():
    question = {
        "category": "factual",
        "expected_answer": "Helios Research Institute leads the Aster Program.",
        "acceptable_answer_variants": ["Helios leads Aster."],
    }

    score = score_answer_correctness(
        question,
        "  HELIOS RESEARCH INSTITUTE leads the Aster Program! ",
        EvidenceStatus.VERIFIED.value,
    )

    assert score == {"score": 1.0, "method": "normalized_containment"}


def test_plainly_wrong_answer_scores_zero():
    question = {"category": "factual", "expected_answer": "Helios leads Aster."}

    score = score_answer_correctness(question, "Nova owns the vessel.", EvidenceStatus.VERIFIED.value)

    assert score["score"] == 0.0


def test_partially_correct_answer_gets_intermediate_token_f1():
    question = {"category": "factual", "expected_answer": "Helios Research Institute leads the Aster Program."}

    score = score_answer_correctness(question, "Helios leads Aster.", EvidenceStatus.VERIFIED.value)

    assert score["method"] == "token_f1"
    assert 0.0 < score["score"] < 1.0


def test_insufficient_evidence_answer_requires_status_match():
    question = {
        "category": "factual",
        "expected_answer": "A supported answer is unavailable.",
        "expected_status": EvidenceStatus.INSUFFICIENT_EVIDENCE.value,
    }

    correct = score_answer_correctness(
        question,
        "I could not find enough evidence.",
        EvidenceStatus.INSUFFICIENT_EVIDENCE.value,
    )
    wrong = score_answer_correctness(question, "An invented answer.", EvidenceStatus.VERIFIED.value)

    assert correct["score"] == 1.0
    assert wrong["score"] == 0.0


def test_retrieval_scores_use_document_level_gold_sources():
    score = score_retrieval_at_k(["doc-a", "doc-b", "doc-a"], ["doc-a", "doc-c"], 3)

    assert score["precision_at_k"] == pytest.approx(1 / 3, abs=0.0001)
    assert score["recall_at_k"] == 0.5


def test_claim_splitter_separates_sentences():
    assert split_claims("First claim. Second claim! Third claim?") == [
        "First claim.", "Second claim!", "Third claim?",
    ]