"""Dataset loading and per-message context gathering.

The CSVs under dataset/ are trusted local files, so rows are plain dataclasses
rather than validated models. Every file is read at most once per process and
cached; build_context() then assembles the rows relevant to a single message
plus a "signals" sub-dict of plain-English conclusions.

No routing decisions are made here.
"""

import csv
import re
from dataclasses import dataclass, fields
from datetime import datetime, time
from functools import lru_cache
from pathlib import Path

DATASET_DIR = Path(__file__).resolve().parent.parent / "dataset"

TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M"

# How many history rows build_context() carries per message.
HISTORY_LIMIT = 8
# Jaccard score at or above which two messages are called near-duplicates.
NEAR_DUPLICATE_THRESHOLD = 0.6

# build_context() always emits exactly these signal keys, in this order, so the
# decision layer can read any of them without guarding for absence.
ALWAYS_SIGNAL_KEYS = ("near_duplicate", "fatigue", "dnd_overlap", "staleness", "media")
GROUP_SIGNAL_KEYS = ("group_mute", "mention", "group_volume", "sender_role")
BUSINESS_SIGNAL_KEYS = (
    "domain_mismatch",
    "account_standing",
    "prior_relationship",
    "promotions",
)
SIGNAL_KEYS = ALWAYS_SIGNAL_KEYS + GROUP_SIGNAL_KEYS + BUSINESS_SIGNAL_KEYS


# --------------------------------------------------------------------------
# Row types
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Message:
    message_id: str
    user_id: str
    conversation_type: str
    group_id: str
    business_id: str
    sender_user_id: str
    created_at: str
    message_text: str
    media_type: str
    media_id: str
    forwarded_count: int

    @property
    def timestamp(self) -> datetime:
        return datetime.strptime(self.created_at, TIMESTAMP_FORMAT)


@dataclass(frozen=True)
class User:
    user_id: str
    do_not_disturb_window: str
    messages_opened_30d: int
    messages_replied_30d: int
    notifications_dismissed_30d: int
    messages_reported_30d: int


@dataclass(frozen=True)
class Group:
    group_id: str
    group_name: str
    group_type: str
    member_count: int
    admin_count: int
    created_at: str
    messages_30d: int


@dataclass(frozen=True)
class GroupMember:
    group_id: str
    user_id: str
    role: str
    joined_at: str
    messages_sent_30d: int
    messages_read_30d: int
    replies_sent_30d: int
    notifications_dismissed_30d: int
    group_muted_by_user: bool


@dataclass(frozen=True)
class BusinessAccount:
    business_id: str
    display_name: str
    brand_name: str
    category: str
    verified: bool
    official_domain: str
    domain_used_by_sender: str
    account_age_days: int
    messages_sent_30d: int
    user_reports_30d: int
    domain_used_by_sender_age_days: int


@dataclass(frozen=True)
class UserBusinessHistory:
    user_id: str
    business_id: str
    why_user_knows_account: str
    last_activity_at: str
    allows_promotions: bool
    promotions_opted_out_at: str
    activity_count_180d: int
    messages_opened_30d: int
    messages_dismissed_30d: int
    messages_replied_30d: int
    last_reply_at: str


@dataclass(frozen=True)
class MessageEvent:
    user_id: str
    message_id: str
    message_opened: bool
    message_replied: bool
    reaction_time_minutes: int
    notification_dismissed: bool
    muted_after_message: bool
    message_reported: bool


@dataclass(frozen=True)
class DailySummary:
    user_id: str
    date: str
    notifications_sent: int
    notifications_dismissed: int


# --------------------------------------------------------------------------
# CSV reading
# --------------------------------------------------------------------------


def _build(cls, row: dict):
    """Coerce a CSV row into `cls`, using each field's annotated type."""
    kwargs = {}
    for field in fields(cls):
        raw = (row.get(field.name) or "").strip()
        if field.type in (int, "int"):
            kwargs[field.name] = int(raw) if raw else 0
        elif field.type in (bool, "bool"):
            kwargs[field.name] = raw not in ("", "0", "false", "False")
        else:
            kwargs[field.name] = raw
    return cls(**kwargs)


def _read(filename: str, cls) -> list:
    path = DATASET_DIR / filename
    with path.open(newline="", encoding="utf-8") as handle:
        return [_build(cls, row) for row in csv.DictReader(handle)]


@lru_cache(maxsize=None)
def messages() -> list[Message]:
    return _read("messages.csv", Message)


@lru_cache(maxsize=None)
def sample_messages() -> list[Message]:
    """The 30 labelled examples.

    _build() only reads fields declared on Message, so the label columns
    (action, message_type, reason, confidence, evidence_message_ids) are
    dropped here and can never reach a prompt.
    """
    return _read("sample_messages.csv", Message)


@lru_cache(maxsize=None)
def users() -> dict[str, User]:
    return {row.user_id: row for row in _read("users.csv", User)}


@lru_cache(maxsize=None)
def groups() -> dict[str, Group]:
    return {row.group_id: row for row in _read("groups.csv", Group)}


@lru_cache(maxsize=None)
def group_members() -> dict[tuple[str, str], GroupMember]:
    return {
        (row.group_id, row.user_id): row
        for row in _read("group_members.csv", GroupMember)
    }


@lru_cache(maxsize=None)
def business_accounts() -> dict[str, BusinessAccount]:
    return {
        row.business_id: row
        for row in _read("business_accounts.csv", BusinessAccount)
    }


@lru_cache(maxsize=None)
def user_business_history() -> dict[tuple[str, str], UserBusinessHistory]:
    return {
        (row.user_id, row.business_id): row
        for row in _read("user_business_history.csv", UserBusinessHistory)
    }


@lru_cache(maxsize=None)
def message_history() -> dict[str, list[Message]]:
    """History rows grouped by the user who received them, newest first."""
    by_user: dict[str, list[Message]] = {}
    for row in _read("message_history.csv", Message):
        by_user.setdefault(row.user_id, []).append(row)
    for rows in by_user.values():
        rows.sort(key=lambda row: row.timestamp, reverse=True)
    return by_user


@lru_cache(maxsize=None)
def message_events() -> dict[tuple[str, str], MessageEvent]:
    return {
        (row.user_id, row.message_id): row
        for row in _read("message_events.csv", MessageEvent)
    }


@lru_cache(maxsize=None)
def daily_summaries() -> dict[tuple[str, str], DailySummary]:
    return {
        (row.user_id, row.date): row
        for row in _read("daily_notification_summary.csv", DailySummary)
    }


@lru_cache(maxsize=None)
def notification_baselines() -> dict[str, dict]:
    """Each user's habitual notification load, aggregated over the whole table.

    daily_notification_summary.csv ends the day before messages.csv begins, so a
    same-day lookup is always empty. The habit is the usable signal.
    """
    totals: dict[str, dict] = {}
    for row in daily_summaries().values():
        entry = totals.setdefault(
            row.user_id, {"days": 0, "sent": 0, "dismissed": 0}
        )
        entry["days"] += 1
        entry["sent"] += row.notifications_sent
        entry["dismissed"] += row.notifications_dismissed
    for entry in totals.values():
        entry["per_day"] = entry["sent"] / entry["days"] if entry["days"] else 0.0
        entry["dismiss_ratio"] = (
            entry["dismissed"] / entry["sent"] if entry["sent"] else 0.0
        )
    return totals


@lru_cache(maxsize=None)
def media_paths() -> dict[str, str]:
    """Both media tables keyed by media_id; the id namespaces do not overlap."""
    paths = {}
    for filename, id_column in (
        ("images.csv", "image_id"),
        ("voice_notes.csv", "voice_note_id"),
    ):
        with (DATASET_DIR / filename).open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                paths[row[id_column]] = row["file_path"]
    return paths


@lru_cache(maxsize=None)
def latest_timestamp() -> datetime:
    """Newest created_at anywhere in the dataset, used as 'now' for staleness."""
    stamps = [row.timestamp for row in messages()]
    for rows in message_history().values():
        stamps.extend(row.timestamp for row in rows)
    return max(stamps)


# --------------------------------------------------------------------------
# Signal helpers
# --------------------------------------------------------------------------


def tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    union = len(left | right)
    return len(left & right) / union if union else 0.0


def _counterparty(message: Message) -> str:
    """The id that identifies who this message is from, by conversation type."""
    if message.conversation_type == "group":
        return message.group_id
    if message.conversation_type == "business":
        return message.business_id
    return message.sender_user_id


def _relevant_history(message: Message) -> list[dict]:
    """This user's history, same-counterparty first then most recent."""
    counterparty = _counterparty(message)
    rows = message_history().get(message.user_id, [])
    ranked = sorted(
        rows,
        key=lambda row: (_counterparty(row) != counterparty, -row.timestamp.timestamp()),
    )
    return [
        {
            "message": row,
            "event": message_events().get((row.user_id, row.message_id)),
            "same_counterparty": _counterparty(row) == counterparty,
        }
        for row in ranked[:HISTORY_LIMIT]
    ]


def _describe_outcome(event: MessageEvent | None) -> str:
    if event is None:
        return "no recorded outcome"
    if event.message_reported:
        return "the user reported it"
    if event.muted_after_message:
        return "the user muted the conversation afterwards"
    if event.notification_dismissed:
        return "the user dismissed the notification"
    if event.message_replied:
        return f"the user replied in {event.reaction_time_minutes} minutes"
    if event.message_opened:
        return "the user opened it but did not reply"
    return "the user ignored it"


def _near_duplicate(message: Message) -> tuple[dict | None, str]:
    """Best Jaccard match against this user's history, with its outcome."""
    incoming = tokens(message.message_text)
    best_row, best_score = None, 0.0
    for row in message_history().get(message.user_id, []):
        score = jaccard(incoming, tokens(row.message_text))
        if score > best_score:
            best_row, best_score = row, score

    if best_row is None:
        return None, "This user has no message history to compare against."

    event = message_events().get((best_row.user_id, best_row.message_id))
    outcome = _describe_outcome(event)
    match = {
        "message_id": best_row.message_id,
        "similarity": round(best_score, 3),
        "outcome": outcome,
        "is_near_duplicate": best_score >= NEAR_DUPLICATE_THRESHOLD,
    }

    if best_score >= NEAR_DUPLICATE_THRESHOLD:
        sentence = (
            f"Near-duplicate of {best_row.message_id} (similarity "
            f"{best_score:.2f}), which this user saw before and {outcome}."
        )
    else:
        sentence = (
            f"Nothing close in this user's history; the best match is "
            f"{best_row.message_id} at similarity {best_score:.2f}."
        )
    return match, sentence


def _fatigue(message: Message) -> str:
    """This user's habitual notification load and dismissal rate."""
    baseline = notification_baselines().get(message.user_id)
    if baseline is None or not baseline["sent"]:
        return "No notification history on record for this user."
    return (
        f"This user averages {baseline['per_day']:.1f} notifications a day and "
        f"dismisses {baseline['dismiss_ratio']:.0%} of them "
        f"({baseline['dismissed']} of {baseline['sent']} over "
        f"{baseline['days']} days)."
    )


def _parse_window(window: str) -> tuple[time, time] | None:
    match = re.fullmatch(r"\s*(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})\s*", window or "")
    if not match:
        return None
    start_h, start_m, end_h, end_m = (int(part) for part in match.groups())
    return time(start_h, start_m), time(end_h, end_m)


def _dnd_overlap(message: Message, user: User) -> str:
    parsed = _parse_window(user.do_not_disturb_window)
    if parsed is None:
        return "This user has no do-not-disturb window set."

    start, end = parsed
    moment = message.timestamp.time()
    # Windows such as 22:00-07:00 wrap past midnight.
    inside = start <= moment < end if start <= end else (moment >= start or moment < end)
    if inside:
        return (
            f"Arrived at {moment.strftime('%H:%M')}, inside the user's "
            f"do-not-disturb window ({user.do_not_disturb_window})."
        )
    return (
        f"Arrived at {moment.strftime('%H:%M')}, outside the user's "
        f"do-not-disturb window ({user.do_not_disturb_window})."
    )


def _staleness(message: Message) -> str:
    age = latest_timestamp() - message.timestamp
    hours = age.total_seconds() / 3600
    if hours < 24:
        return f"Arrived {hours:.0f} hours before the end of the dataset; still fresh."
    return (
        f"Arrived {age.days} days before the end of the dataset; any deadline "
        "or time-sensitive ask in it may already have passed."
    )


def _group_signals(message: Message, context: dict) -> dict:
    """Always returns every key in GROUP_SIGNAL_KEYS."""
    group = context["group"]
    membership = context["group_membership"]
    sender = context["sender_membership"]
    signals = {}

    if membership is None:
        signals["group_mute"] = "No membership row for this user in this group."
    elif membership.group_muted_by_user:
        signals["group_mute"] = (
            f"This user has muted {group.group_name if group else message.group_id}, "
            f"and dismissed {membership.notifications_dismissed_30d} of its "
            "notifications in the last 30 days."
        )
    else:
        signals["group_mute"] = (
            f"This user has not muted the group; they sent "
            f"{membership.messages_sent_30d} and replied "
            f"{membership.replies_sent_30d} times in the last 30 days."
        )

    mentioned = f"@{message.user_id}" in message.message_text
    if mentioned:
        signals["mention"] = "The message @-mentions this user directly."
    else:
        signals["mention"] = "The message does not @-mention this user."

    if group is not None:
        signals["group_volume"] = (
            f"{group.group_name} is a {group.group_type} group of "
            f"{group.member_count} with {group.messages_30d} messages in the "
            "last 30 days."
        )
    else:
        signals["group_volume"] = "No group row found for this message."

    if sender is None:
        signals["sender_role"] = "The sender is not a recorded member of this group."
    else:
        signals["sender_role"] = f"The sender is a group {sender.role}."

    return signals


def _business_signals(message: Message, context: dict) -> dict:
    """Always returns every key in BUSINESS_SIGNAL_KEYS."""
    business = context["business"]
    history = context["business_history"]
    signals = {}

    if business is None:
        missing = "No business account row found for this message."
        return {key: missing for key in BUSINESS_SIGNAL_KEYS}

    # The domain's age is what makes a mismatch meaningful. Established brands
    # routinely send marketing from a separate long-lived domain; a scam sends
    # from a lookalike registered days ago. Reporting the mismatch without the
    # age invites the reader to treat both the same way.
    if business.official_domain != business.domain_used_by_sender:
        signals["domain_mismatch"] = (
            f"Sender domain {business.domain_used_by_sender!r} does not match "
            f"the brand's official domain {business.official_domain!r}, and was "
            f"registered {business.domain_used_by_sender_age_days} days ago."
        )
    else:
        signals["domain_mismatch"] = (
            f"Sender domain matches the brand's official domain "
            f"({business.official_domain}), registered "
            f"{business.domain_used_by_sender_age_days} days ago."
        )

    signals["account_standing"] = (
        f"{business.brand_name} is "
        f"{'verified' if business.verified else 'NOT verified'}, "
        f"{business.account_age_days} days old, with "
        f"{business.user_reports_30d} user reports in the last 30 days."
    )

    if history is None:
        signals["prior_relationship"] = (
            "This user has no prior relationship with this business on record."
        )
        signals["promotions"] = (
            "No promotion preference on record, because this user has no prior "
            "relationship with this business."
        )
    else:
        signals["prior_relationship"] = (
            f"This user knows the account through {history.why_user_knows_account!r}, "
            f"last active {history.last_activity_at}, with "
            f"{history.activity_count_180d} interactions in 180 days."
        )
        if history.allows_promotions:
            signals["promotions"] = "This user allows promotions from this business."
        else:
            opted_out = history.promotions_opted_out_at or "an unrecorded date"
            signals["promotions"] = (
                f"This user does not allow promotions from this business "
                f"(opted out {opted_out})."
            )

    return signals


def _media_signal(message: Message) -> tuple[dict | None, str]:
    if not message.media_type:
        return None, "Text-only message; no attached media."
    relative = media_paths().get(message.media_id)
    if relative is None:
        return None, (
            f"Message claims {message.media_type} {message.media_id!r}, but no "
            "matching media row exists."
        )
    media = {
        "media_type": message.media_type,
        "media_id": message.media_id,
        "path": relative,
        "full_path": str(DATASET_DIR / relative),
    }
    return media, (
        f"Carries a {message.media_type} at {relative}; its content is not read "
        "here and must be inspected downstream."
    )


# --------------------------------------------------------------------------
# Context assembly
# --------------------------------------------------------------------------


def build_context(message: Message) -> dict:
    """Gather the rows and derived signals relevant to a single message."""
    user = users().get(message.user_id)

    context: dict = {
        "message": message,
        "user": user,
        "group": None,
        "group_membership": None,
        "sender_membership": None,
        "business": None,
        "business_history": None,
        "media": None,
        "history": _relevant_history(message),
        "near_duplicate": None,
        "signals": {},
    }

    near_duplicate, duplicate_sentence = _near_duplicate(message)
    context["near_duplicate"] = near_duplicate

    # Seed the branch keys so a personal message still carries every key.
    not_applicable = (
        f"Not applicable; this is a {message.conversation_type} conversation."
    )
    signals = {key: not_applicable for key in GROUP_SIGNAL_KEYS + BUSINESS_SIGNAL_KEYS}
    signals.update(
        near_duplicate=duplicate_sentence,
        fatigue=_fatigue(message),
        dnd_overlap=(
            _dnd_overlap(message, user) if user else "No user row found for this message."
        ),
        staleness=_staleness(message),
    )

    if message.conversation_type == "group":
        context["group"] = groups().get(message.group_id)
        context["group_membership"] = group_members().get(
            (message.group_id, message.user_id)
        )
        if message.sender_user_id:
            context["sender_membership"] = group_members().get(
                (message.group_id, message.sender_user_id)
            )
        signals.update(_group_signals(message, context))

    elif message.conversation_type == "business":
        context["business"] = business_accounts().get(message.business_id)
        context["business_history"] = user_business_history().get(
            (message.user_id, message.business_id)
        )
        signals.update(_business_signals(message, context))

    media, media_sentence = _media_signal(message)
    context["media"] = media
    signals["media"] = media_sentence

    # Fixed shape, fixed order, no missing keys.
    assert set(signals) == set(SIGNAL_KEYS), sorted(set(signals) ^ set(SIGNAL_KEYS))
    context["signals"] = {key: signals[key] for key in SIGNAL_KEYS}
    return context


if __name__ == "__main__":
    def preview(label: str, message: Message) -> None:
        context = build_context(message)
        print(f"\n{'=' * 70}\n{label}: {message.message_id}\n{'=' * 70}")
        print(f"text: {message.message_text[:160]!r}")
        print(f"\nrows: user={context['user'].user_id if context['user'] else None} "
              f"group={context['group'].group_id if context['group'] else None} "
              f"business={context['business'].business_id if context['business'] else None} "
              f"history={len(context['history'])} rows "
              f"({sum(1 for row in context['history'] if row['same_counterparty'])} same counterparty)")
        print(f"near_duplicate: {context['near_duplicate']}")
        print("\nsignals:")
        for name, sentence in context["signals"].items():
            print(f"  {name}: {sentence}")

    first_group = next(m for m in messages() if m.conversation_type == "group")
    first_business = next(m for m in messages() if m.conversation_type == "business")
    preview("FIRST GROUP MESSAGE", first_group)
    preview("FIRST BUSINESS MESSAGE", first_business)
