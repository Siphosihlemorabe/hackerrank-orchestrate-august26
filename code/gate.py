"""Deterministic safety gate, applied after the model has decided.

Two narrow rules only. The gate exists for defensiveness, not for accuracy: it
catches cases where the evidence is unambiguous and a miss would be expensive.

It is monotonic. A rule may move a decision toward mute; it can never move one
away from mute, and it never touches a decision it does not fire on.

  notify  <  digest  <  mute
"""

from dataclasses import dataclass

import loaders
from decide import signal_confidence
from schema import RoutingDecision

STRICTNESS = {"notify": 0, "digest": 1, "mute": 2}

# A mismatched sender domain on its own is not evidence of anything: established
# brands routinely send marketing from a separate long-lived domain. What marks
# a scam is that the lookalike domain was registered recently. In this dataset
# the two populations are far apart - genuine scams send from domains 2-17 days
# old, while legitimate senders' domains run 390-3455 days - so the threshold
# sits in open space rather than on a boundary.
MAX_LOOKALIKE_DOMAIN_AGE_DAYS = 90


@dataclass(frozen=True)
class Override:
    """One rule firing on one message."""
    message_id: str
    rule: str
    from_action: str
    to_action: str
    from_type: str
    to_type: str
    detail: str

    def describe(self) -> str:
        change = f"{self.from_action}/{self.from_type} -> {self.to_action}/{self.to_type}"
        return f"{self.message_id}  [{self.rule}]  {change}  {self.detail}"


def _normalise(text: str) -> str:
    return " ".join((text or "").split()).casefold()


def _domain_mismatch(context: dict):
    """A recently-registered lookalike sender domain is treated as scam.

    Both conditions are required. Mismatch alone fires on established brands
    that send marketing from a separate domain, which suppresses mail the user
    asked for.
    """
    business = context["business"]
    if business is None:
        return None
    if not business.official_domain or not business.domain_used_by_sender:
        return None
    if business.official_domain == business.domain_used_by_sender:
        return None
    if business.domain_used_by_sender_age_days >= MAX_LOOKALIKE_DOMAIN_AGE_DAYS:
        return None
    return (
        "mute",
        "scam",
        f"Sender domain {business.domain_used_by_sender!r} is not the official "
        f"domain {business.official_domain!r} for {business.brand_name}, and "
        f"was registered only {business.domain_used_by_sender_age_days} days ago.",
    )


def _reported_duplicate(context: dict):
    """An exact repeat of something this user already reported.

    Scans the user's whole history rather than the truncated candidate list, so
    the rule cannot be defeated by a match falling outside the top few rows.
    """
    incoming = _normalise(context["message"].message_text)
    if not incoming:
        return None

    user_id = context["message"].user_id
    for prior in loaders.message_history().get(user_id, []):
        if _normalise(prior.message_text) != incoming:
            continue
        event = loaders.message_events().get((user_id, prior.message_id))
        if event is not None and event.message_reported:
            return (
                "mute",
                None,
                f"Identical to {prior.message_id}, which this user reported.",
            )
    return None


RULES = (
    ("domain_mismatch", _domain_mismatch),
    ("reported_duplicate", _reported_duplicate),
)


def apply_gate(context: dict, decision: RoutingDecision
               ) -> tuple[RoutingDecision, list[Override]]:
    """Return the gated decision plus every override that fired."""
    action = decision.action
    message_type = decision.message_type
    reason = decision.reason
    confidence = decision.confidence
    overrides: list[Override] = []

    for rule_name, rule in RULES:
        verdict = rule(context)
        if verdict is None:
            continue
        new_action, new_type, detail = verdict

        # Never loosen: a rule can only hold or increase strictness.
        if STRICTNESS[new_action] < STRICTNESS[action]:
            continue

        action_changes = new_action != action
        type_changes = new_type is not None and new_type != message_type
        if not (action_changes or type_changes):
            continue

        overrides.append(Override(
            message_id=decision.message_id,
            rule=rule_name,
            from_action=action,
            to_action=new_action,
            from_type=message_type,
            to_type=new_type or message_type,
            detail=detail,
        ))

        if action_changes:
            action = new_action
            # The confidence was computed for the action the model chose, so it
            # has to be recomputed for the one actually being emitted.
            confidence = signal_confidence(
                context, action, list(decision.evidence_message_ids))
        if new_type:
            message_type = new_type
        reason = detail

    if not overrides:
        return decision, []

    return RoutingDecision(
        message_id=decision.message_id,
        action=action,
        message_type=message_type,
        reason=reason,
        confidence=confidence,
        evidence_message_ids=list(decision.evidence_message_ids),
    ), overrides
