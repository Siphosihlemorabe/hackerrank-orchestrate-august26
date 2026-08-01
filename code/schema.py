from typing import Literal
from pydantic import BaseModel, Field, field_validator, model_validator

Action = Literal["notify", "digest", "mute"]

MessageType = Literal[
    "personal", "urgent", "event", "payment", "business_update",
    "promotion", "greeting", "forward", "spam", "scam", "unknown",
]

RISK_TYPES = {"scam", "spam"}

# Terms that mean the reason itself is asserting the message is a risk.
_RISK_REASON_TERMS = ("scam", "phish", "fraud", "spam")


class RoutingDecision(BaseModel):
    message_id: str = Field(min_length=1)
    action: Action
    message_type: MessageType
    reason: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_message_ids: list[str] = Field(default_factory=list)

    @field_validator("evidence_message_ids", mode="before")
    @classmethod
    def _coerce_evidence(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            s = v.strip()
            if not s or s.lower() == "none":
                return []
            return [x.strip() for x in s.split(";") if x.strip()]
        return list(v)

    @model_validator(mode="after")
    def _risk_mute_is_typed_as_risk(self):
        """A mute justified as a risk must carry a risk message_type."""
        if self.action == "mute" and self.message_type not in RISK_TYPES:
            reason = self.reason.lower()
            if any(term in reason for term in _RISK_REASON_TERMS):
                raise ValueError(
                    f"muted for a risk reason but message_type is "
                    f"{self.message_type!r}; expected one of {sorted(RISK_TYPES)}"
                )
        return self

    def to_output_row(self) -> dict:
        return {
            "message_id": self.message_id,
            "action": self.action,
            "message_type": self.message_type,
            "reason": self.reason,
            "confidence": f"{self.confidence:.2f}",
            "evidence_message_ids": ";".join(self.evidence_message_ids) or "none",
        }


OUTPUT_COLUMNS = [
    "message_id", "action", "message_type", "reason", "confidence",
    "evidence_message_ids",
]