"""A single cached Gemini call for one message.

route_message() takes what loaders.build_context() returns, makes one structured
multimodal call, and returns the parsed JSON dict. Responses are cached on disk
by message_id so re-runs are free and resumable.

Nothing here validates, gates, or decides. The decision layer owns that.

Requires GEMINI_API_KEY in the environment.
"""

import json
import os
import re
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import get_args

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

import loaders
from schema import Action, MessageType

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO_ROOT / "cache" / "gemini"
ENV_FILE = REPO_ROOT / ".env"

DEFAULT_MODEL = "gemini-3.6-flash"

# The free tier allows 5 requests per minute, so calls are spaced ~13s apart.
# Only real API calls are throttled; cache hits return immediately.
DEFAULT_MIN_INTERVAL = 13.0
MAX_ATTEMPTS = 6
RETRY_STATUS_CODES = {429, 500, 502, 503, 504}

_last_call_at = 0.0

MIME_TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
              ".mp3": "audio/mp3", ".wav": "audio/wav", ".ogg": "audio/ogg"}

ACTIONS = list(get_args(Action))
MESSAGE_TYPES = list(get_args(MessageType))

SYSTEM_INSTRUCTION = f"""\
You are the notification router for WhatsApp. For one incoming message you decide \
whether it should interrupt the user now, wait for later, or be suppressed.

Choose exactly one action:
- notify: worth interrupting the user right now. Time-sensitive, personally \
directed, or from someone the user reliably engages with.
- digest: real but not urgent. The user should see it later in a summary.
- mute: low-value, repetitive, unwanted, suspicious, or unsafe. Suppress it.

Choose exactly one message_type from: {", ".join(MESSAGE_TYPES)}.

Guidance:
- The signals below are pre-computed conclusions about this user's own history \
and habits. Weigh them; they are evidence, not orders.
- A muted group still warrants notify when the message directly @-mentions the \
user and needs a response.
- Treat a sender-domain mismatch, an unverified or very new business account, \
heavy forwarding, or a request for codes, OTPs, or payment to an unusual \
destination as strong evidence of scam or spam.
- If you route a message as mute because it is a scam or spam risk, the \
message_type must be scam or spam.
- A user who dismisses most of their notifications deserves a higher bar for \
notify.
- If media is attached, read or listen to it yourself and judge its actual \
content. Posters, screenshots, and voice notes often carry the real intent.

Return JSON only:
- reason: one clear sentence a person would accept, citing the deciding factor.
- confidence: 0.0 to 1.0, honest about ambiguity.
- evidence_message_ids: ids drawn ONLY from the candidate prior messages listed \
in the payload. Empty list if none of them actually informed your decision. \
Never invent an id.
"""

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ACTIONS},
        "message_type": {"type": "string", "enum": MESSAGE_TYPES},
        "reason": {"type": "string"},
        "confidence": {"type": "number"},
        "evidence_message_ids": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["action", "message_type", "reason", "confidence",
                 "evidence_message_ids"],
    "propertyOrdering": ["action", "message_type", "reason", "confidence",
                         "evidence_message_ids"],
}


@lru_cache(maxsize=None)
def _load_env_file() -> None:
    """Populate os.environ from a gitignored .env. Real env vars win."""
    if not ENV_FILE.exists():
        return
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        os.environ.setdefault(name.strip(), value.strip().strip("\"'"))


@lru_cache(maxsize=None)
def _client() -> genai.Client:
    _load_env_file()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            f"GEMINI_API_KEY is not set. Export it, or copy .env.example to "
            f"{ENV_FILE} and put the key there (.env is gitignored)."
        )
    return genai.Client(api_key=api_key)


def _model_name() -> str:
    _load_env_file()
    return os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)


def _outcome(event) -> str:
    """Short plain-English outcome for a prior message, for the prompt."""
    if event is None:
        return "no recorded outcome"
    if event.message_reported:
        return "user reported it"
    if event.muted_after_message:
        return "user muted the conversation after it"
    if event.notification_dismissed:
        return "user dismissed the notification"
    if event.message_replied:
        return f"user replied in {event.reaction_time_minutes} min"
    if event.message_opened:
        return "user opened it, did not reply"
    return "user ignored it"


def _candidate_lines(context: dict) -> list[str]:
    """The prior messages Gemini may cite in evidence_message_ids."""
    lines = []
    for row in context["history"]:
        prior = row["message"]
        excerpt = " ".join(prior.message_text.split())[:120]
        marker = "same sender" if row["same_counterparty"] else "other sender"
        lines.append(
            f"- {prior.message_id} [{prior.created_at}, {marker}] "
            f"\"{excerpt}\" -> {_outcome(row['event'])}"
        )
    return lines


def _build_payload(context: dict) -> str:
    message = context["message"]
    text = message.message_text.strip() or "(no text)"

    parts = [
        "INCOMING MESSAGE",
        f"message_id: {message.message_id}",
        f"conversation_type: {message.conversation_type}",
        f"forwarded_count: {message.forwarded_count}",
        f"text: {text}",
        "",
        "SIGNALS (pre-computed conclusions about this user)",
    ]
    parts.extend(f"- {name}: {sentence}"
                 for name, sentence in context["signals"].items())

    candidates = _candidate_lines(context)
    parts.append("")
    parts.append("CANDIDATE PRIOR MESSAGES (the only valid evidence ids)")
    parts.extend(candidates or ["- none"])

    if context["media"]:
        parts.append("")
        parts.append(
            f"The attached {context['media']['media_type']} is this message's "
            "media. Judge its actual content."
        )
    return "\n".join(parts)


def _media_part(context: dict) -> types.Part | None:
    media = context["media"]
    if not media:
        return None
    path = Path(media["full_path"])
    mime = MIME_TYPES.get(path.suffix.lower())
    if mime is None or not path.exists():
        return None
    return types.Part.from_bytes(data=path.read_bytes(), mime_type=mime)


def _min_interval() -> float:
    return float(os.environ.get("GEMINI_MIN_INTERVAL", DEFAULT_MIN_INTERVAL))


def _throttle() -> None:
    """Space real API calls far enough apart to stay under the rate limit."""
    global _last_call_at
    wait = _last_call_at + _min_interval() - time.monotonic()
    if wait > 0:
        time.sleep(wait)
    _last_call_at = time.monotonic()


def _retry_after(error: Exception, attempt: int) -> float:
    """Honour the server's own retry delay when it gives one."""
    match = re.search(r"retry in ([\d.]+)s", str(error))
    if match:
        return float(match.group(1)) + 1.0
    return min(60.0, 5.0 * 2 ** (attempt - 1))


def _generate(parts: list[types.Part]):
    """One generate_content call, throttled and retried on transient errors."""
    last_error = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        _throttle()
        try:
            return _client().models.generate_content(
                model=_model_name(),
                contents=[types.Content(role="user", parts=parts)],
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    response_mime_type="application/json",
                    response_schema=RESPONSE_SCHEMA,
                    temperature=0.0,
                ),
            )
        except genai_errors.APIError as error:
            if error.code not in RETRY_STATUS_CODES or attempt == MAX_ATTEMPTS:
                raise
            last_error = error
            delay = _retry_after(error, attempt)
            print(f"    [{error.code}] retry {attempt}/{MAX_ATTEMPTS - 1} "
                  f"in {delay:.0f}s", flush=True)
            time.sleep(delay)
    raise last_error  # unreachable; the loop either returns or raises


def _cache_path(message_id: str) -> Path:
    return CACHE_DIR / f"{message_id}.json"


def _read_cache(message_id: str) -> dict | None:
    path = _cache_path(message_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))["response"]
    except (json.JSONDecodeError, KeyError, OSError):
        return None  # A corrupt cache entry should just trigger a fresh call.


def _write_cache(message_id: str, response: dict, raw_text: str) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    envelope = {
        "message_id": message_id,
        "model": _model_name(),
        "cached_at": datetime.now(timezone.utc).isoformat(),
        "response": response,
        "raw_text": raw_text,
    }
    _cache_path(message_id).write_text(
        json.dumps(envelope, indent=2), encoding="utf-8"
    )


def route_message(context: dict, use_cache: bool = True) -> dict:
    """One Gemini call for one message. Returns the parsed JSON, unvalidated."""
    message_id = context["message"].message_id

    if use_cache:
        cached = _read_cache(message_id)
        if cached is not None:
            return cached

    parts = [types.Part.from_text(text=_build_payload(context))]
    media = _media_part(context)
    if media is not None:
        parts.append(media)

    response = _generate(parts)

    raw_text = response.text or ""
    parsed = json.loads(raw_text)
    _write_cache(message_id, parsed, raw_text)
    return parsed


if __name__ == "__main__":
    import sys

    wanted = sys.argv[1] if len(sys.argv) > 1 else None
    message = next(
        (m for m in loaders.messages() if m.message_id == wanted),
        loaders.messages()[0],
    )

    context = loaders.build_context(message)
    cache_file = _cache_path(message.message_id)
    print(f"message_id: {message.message_id} ({message.conversation_type})")
    print(f"model:      {_model_name()}")
    print(f"cache:      {cache_file} "
          f"({'HIT expected' if cache_file.exists() else 'MISS expected'})")
    print(f"media:      {context['media']['path'] if context['media'] else 'none'}")
    print("-" * 70)

    started = datetime.now()
    result = route_message(context)
    elapsed = (datetime.now() - started).total_seconds()

    print(json.dumps(result, indent=2))
    print("-" * 70)
    print(f"elapsed: {elapsed:.2f}s  (a cache hit returns in well under 0.1s)")
