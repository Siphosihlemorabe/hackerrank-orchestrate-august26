"""Turn a raw model response into a validated RoutingDecision.

Two post-processing stages, each independently switchable so their effect can
be measured separately:

  trim_evidence()       the model over-cites; keep only the decisive prior
                        message(s) it named.
  signal_confidence()   the model's self-reported confidence does not separate
                        right from wrong; recompute it from the deterministic
                        signals instead.

Neither stage changes action, message_type, or reason. That is gate territory.
"""

from loaders import jaccard, tokens
from schema import RoutingDecision

# How decisive a recorded outcome is. A report or a mute is a deliberate act;
# opening without replying barely registers as a preference.
OUTCOME_WEIGHT = {
    "reported": 0.35,
    "muted": 0.30,
    "replied": 0.25,
    "dismissed": 0.20,
    "opened": 0.10,
    "ignored": 0.05,
    "none": 0.00,
}

SAME_COUNTERPARTY_BONUS = 0.10

# A second id is kept only when it is comparably decisive AND is itself a real
# textual match - that similarity floor is what does the work, since it stops a
# weak-but-similarly-scored id riding along. Chosen by sweeping both thresholds
# on the 30 samples; the similarity floor mattered, the ratio barely did.
SECOND_ID_SCORE_RATIO = 0.60
SECOND_ID_MIN_SIMILARITY = 0.30

CONFIDENCE_FLOOR = 0.70
CONFIDENCE_CEILING = 0.92
CONFIDENCE_BASE = 0.80


def _outcome_key(event) -> str:
    if event is None:
        return "none"
    if event.message_reported:
        return "reported"
    if event.muted_after_message:
        return "muted"
    if event.notification_dismissed:
        return "dismissed"
    if event.message_replied:
        return "replied"
    if event.message_opened:
        return "opened"
    return "ignored"


def _decisiveness(incoming_tokens: set[str], row: dict) -> tuple[float, float]:
    """Score how much a prior message could actually have driven the decision."""
    similarity = jaccard(incoming_tokens, tokens(row["message"].message_text))
    score = similarity + OUTCOME_WEIGHT[_outcome_key(row["event"])]
    if row["same_counterparty"]:
        score += SAME_COUNTERPARTY_BONUS
    return score, similarity


def trim_evidence(context: dict, cited_ids) -> list[str]:
    """Keep only the decisive prior message(s) from the ids the model named.

    Ids that are not real candidates are dropped outright, so a hallucinated id
    can never reach the output.
    """
    candidates = {row["message"].message_id: row for row in context["history"]}
    valid = [message_id for message_id in cited_ids if message_id in candidates]
    if not valid:
        return []

    incoming = tokens(context["message"].message_text)
    ranked = sorted(
        ((message_id,) + _decisiveness(incoming, candidates[message_id])
         for message_id in valid),
        key=lambda entry: -entry[1],
    )

    kept = [ranked[0][0]]
    if len(ranked) > 1:
        top_score = ranked[0][1]
        _, second_score, second_similarity = ranked[1]
        if (second_score >= SECOND_ID_SCORE_RATIO * top_score
                and second_similarity >= SECOND_ID_MIN_SIMILARITY):
            kept.append(ranked[1][0])
    return kept


def signal_confidence(context: dict, action: str, evidence: list[str]) -> float:
    """Confidence from how strongly the deterministic signals agree.

    High when independent signals point the same way (a reported near-duplicate
    backing a mute, a domain mismatch backing a scam call). Low when there is no
    history to lean on or the signals pull against the chosen action.
    """
    signals = context["signals"]
    near_duplicate = context["near_duplicate"]
    score = CONFIDENCE_BASE

    if near_duplicate and near_duplicate["is_near_duplicate"]:
        outcome = near_duplicate["outcome"]
        suppressed = any(word in outcome
                         for word in ("reported", "muted", "dismissed"))
        engaged = "replied" in outcome
        if action == "mute" and suppressed:
            score += 0.07
        elif action == "notify" and engaged:
            score += 0.07
        elif action == "digest":
            score += 0.03
        else:
            score += 0.02  # a match exists but points elsewhere
    elif near_duplicate is None or near_duplicate["similarity"] < 0.20:
        score -= 0.05  # nothing comparable in this user's history

    if action == "mute":
        if "does not match" in signals["domain_mismatch"]:
            score += 0.05
        if "NOT verified" in signals["account_standing"]:
            score += 0.03
        if "no prior relationship" in signals["prior_relationship"]:
            score += 0.02
        # Muting a business the user actively deals with is a bolder call.
        if "knows the account through" in signals["prior_relationship"]:
            score -= 0.04

    if action == "notify":
        if signals["mention"].startswith("The message @-mentions"):
            score += 0.05
        # Interrupting out of a group the user silenced cuts against them.
        if "has muted" in signals["group_mute"]:
            score -= 0.06

    if not evidence:
        score -= 0.04

    return round(min(CONFIDENCE_CEILING, max(CONFIDENCE_FLOOR, score)), 2)


def build_decision(context: dict, raw: dict, *, trim: bool = True,
                   recompute_confidence: bool = True) -> RoutingDecision:
    """Assemble the final decision. Raises if the result fails validation."""
    action = raw["action"]
    evidence = list(raw.get("evidence_message_ids") or [])

    if trim:
        evidence = trim_evidence(context, evidence)

    confidence = (signal_confidence(context, action, evidence)
                  if recompute_confidence else raw["confidence"])

    return RoutingDecision(
        message_id=context["message"].message_id,
        action=action,
        message_type=raw["message_type"],
        reason=raw["reason"],
        confidence=confidence,
        evidence_message_ids=evidence,
    )
