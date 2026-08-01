"""Route the 30 labelled sample messages and write predictions to CSV.

This is the first place model output is validated. Anything that fails
validation is logged and replaced with a safe fallback row, so the output CSV
always has exactly one row per sample message.

No deterministic gate runs here - this is the raw model baseline.

Usage: python code/run_samples.py [output_csv]
"""

import csv
import json
import sys
from pathlib import Path

from google.genai import errors as genai_errors
from pydantic import ValidationError

import loaders
from gemini_client import route_message
from schema import OUTPUT_COLUMNS, RoutingDecision

DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "sample_output.csv"

# A rejected prediction becomes digest/unknown: it neither interrupts the user
# nor silently suppresses something that might matter.
FALLBACK_ACTION = "digest"
FALLBACK_TYPE = "unknown"
FALLBACK_CONFIDENCE = 0.30


def _fallback(message_id: str, problem: str) -> RoutingDecision:
    return RoutingDecision(
        message_id=message_id,
        action=FALLBACK_ACTION,
        message_type=FALLBACK_TYPE,
        reason=f"Fallback after a failed prediction: {problem}",
        confidence=FALLBACK_CONFIDENCE,
        evidence_message_ids=[],
    )


def _decide(message) -> tuple[RoutingDecision, str | None]:
    """Route one message. Returns the decision and a failure note, if any.

    A transport failure is not a prediction, so it is never papered over with a
    fallback row: it propagates and aborts the run. The cache makes the re-run
    resume where this one stopped.
    """
    try:
        raw = route_message(loaders.build_context(message))
    except json.JSONDecodeError as error:
        problem = f"model returned unparseable JSON: {error}"
        return _fallback(message.message_id, problem), problem

    try:
        return RoutingDecision(message_id=message.message_id, **raw), None
    except (ValidationError, TypeError) as error:
        problem = str(error).replace("\n", " ")[:200]
        return _fallback(message.message_id, problem), problem


def run(output_path: Path) -> list[RoutingDecision]:
    messages = loaders.sample_messages()
    decisions, failures = [], []

    for index, message in enumerate(messages, start=1):
        try:
            decision, problem = _decide(message)
        except genai_errors.APIError as error:
            sys.exit(
                f"\nAborted at {message.message_id} after {index - 1} of "
                f"{len(messages)} messages.\n"
                f"  API error {error.code}: {str(error)[:160]}\n"
                f"  No fallback rows were written for this. Successful calls "
                f"are cached, so re-running resumes from here."
            )
        decisions.append(decision)
        if problem:
            failures.append((message.message_id, problem))
            print(f"[{index:>2}/{len(messages)}] {message.message_id}  "
                  f"VALIDATION FAILED -> fallback  ({problem})")
        else:
            print(f"[{index:>2}/{len(messages)}] {message.message_id}  "
                  f"{decision.action:<6} {decision.message_type:<16} "
                  f"conf={decision.confidence:.2f}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for decision in decisions:
            writer.writerow(decision.to_output_row())

    print(f"\nwrote {len(decisions)} rows to {output_path}")
    if failures:
        print(f"{len(failures)} row(s) fell back to "
              f"{FALLBACK_ACTION}/{FALLBACK_TYPE}:")
        for message_id, problem in failures:
            print(f"  {message_id}: {problem}")
    else:
        print("all predictions passed RoutingDecision validation")
    return decisions


if __name__ == "__main__":
    destination = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUTPUT
    run(destination)
