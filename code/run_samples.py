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
from decide import build_decision
from gate import apply_gate
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


def _decide(message, trim: bool, recompute_confidence: bool, gate: bool
            ) -> tuple[RoutingDecision, str | None, list]:
    """Route one message. Returns the decision and a failure note, if any.

    A transport failure is not a prediction, so it is never papered over with a
    fallback row: it propagates and aborts the run. The cache makes the re-run
    resume where this one stopped.
    """
    context = loaders.build_context(message)
    try:
        raw = route_message(context)
    except json.JSONDecodeError as error:
        problem = f"model returned unparseable JSON: {error}"
        return _fallback(message.message_id, problem), problem, []

    try:
        decision = build_decision(
            context, raw, trim=trim,
            recompute_confidence=recompute_confidence,
        )
        overrides = []
        if gate:
            decision, overrides = apply_gate(context, decision)
        return decision, None, overrides
    except (ValidationError, TypeError, KeyError) as error:
        problem = str(error).replace("\n", " ")[:200]
        return _fallback(message.message_id, problem), problem, []


def run(output_path: Path, trim: bool = True,
        recompute_confidence: bool = True,
        gate: bool = True) -> list[RoutingDecision]:
    messages = loaders.sample_messages()
    decisions, failures, all_overrides = [], [], []
    print(f"evidence trimming: {'on' if trim else 'off'}   "
          f"signal confidence: {'on' if recompute_confidence else 'off'}   "
          f"gate: {'on' if gate else 'off'}")

    for index, message in enumerate(messages, start=1):
        try:
            decision, problem, overrides = _decide(
                message, trim, recompute_confidence, gate)
            all_overrides.extend(overrides)
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
            mark = "  <- GATED" if overrides else ""
            print(f"[{index:>2}/{len(messages)}] {message.message_id}  "
                  f"{decision.action:<6} {decision.message_type:<16} "
                  f"conf={decision.confidence:.2f}{mark}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for decision in decisions:
            writer.writerow(decision.to_output_row())

    print(f"\nwrote {len(decisions)} rows to {output_path}")
    if gate:
        print(f"\ngate overrides: {len(all_overrides)}")
        for override in all_overrides:
            print(f"  {override.describe()}")
    if failures:
        print(f"{len(failures)} row(s) fell back to "
              f"{FALLBACK_ACTION}/{FALLBACK_TYPE}:")
        for message_id, problem in failures:
            print(f"  {message_id}: {problem}")
    else:
        print("all predictions passed RoutingDecision validation")
    return decisions


if __name__ == "__main__":
    # --raw disables both stages, --trim-only disables the confidence stage.
    # Every variant reads the same cached responses, so switching costs nothing.
    flags = {arg for arg in sys.argv[1:] if arg.startswith("--")}
    positional = [arg for arg in sys.argv[1:] if not arg.startswith("--")]

    destination = Path(positional[0]) if positional else DEFAULT_OUTPUT
    run(
        destination,
        trim="--raw" not in flags,
        recompute_confidence=not flags & {"--raw", "--trim-only"},
        gate=not flags & {"--raw", "--trim-only", "--no-gate"},
    )
