from enum import Enum

from pydantic import BaseModel, Field


class VerificationVerdict(str, Enum):
    SUPPORTS = "SUPPORTS"
    PARTIALLY_SUPPORTS = "PARTIALLY_SUPPORTS"
    CONTRADICTS = "CONTRADICTS"
    DOES_NOT_SUPPORT = "DOES_NOT_SUPPORT"


class ClaimVerification(BaseModel):
    verdict: VerificationVerdict
    explanation: str = Field(min_length=1)
    supporting_spans: list[str] = Field(default_factory=list)