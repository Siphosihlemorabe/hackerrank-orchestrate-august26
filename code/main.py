"""Route every message in dataset/messages.csv and write output.csv.

Pipeline per message:
    loaders.build_context  ->  gemini_client.route_message  ->
    decide.build_decision  ->  gate.apply_gate  ->  RoutingDecision

Responses are cached by message_id, so an interrupted run resumes for free.

Usage:
    python code/main.py [output_csv] [--no-gate] [--raw] [--trim-only]
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
from run_samples import _fallback
from schema import OUTPUT_COLUMNS, RoutingDecision

DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "output.csv"


def run(output_path: Path, trim: bool = True,
        recompute_confidence: bool = True,
        gate: bool = True) -> list[RoutingDecision]:
    messages = loaders.messages()
    decisions, failures, all_overrides = [], [], []
    print(f"routing {len(messages)} messages   "
          f"trim={'on' if trim else 'off'}   "
          f"signal-confidence={'on' if recompute_confidence else 'off'}   "
          f"gate={'on' if gate else 'off'}", flush=True)

    for index, message in enumerate(messages, start=1):
        context = loaders.build_context(message)
        try:
            raw = route_message(context)
        except genai_errors.APIError as error:
            sys.exit(
                f"\nAborted at {message.message_id} after {index - 1} of "
                f"{len(messages)}.\n  API error {error.code}: "
                f"{str(error)[:160]}\n  Successful calls are cached; "
                f"re-running resumes from here."
            )
        except json.JSONDecodeError as error:
            problem = f"model returned unparseable JSON: {error}"
            decisions.append(_fallback(message.message_id, problem))
            failures.append((message.message_id, problem))
            print(f"[{index:>3}/{len(messages)}] {message.message_id}  "
                  f"FALLBACK ({problem})", flush=True)
            continue

        try:
            decision = build_decision(
                context, raw, trim=trim,
                recompute_confidence=recompute_confidence)
            overrides = []
            if gate:
                decision, overrides = apply_gate(context, decision)
        except (ValidationError, TypeError, KeyError) as error:
            problem = str(error).replace("\n", " ")[:200]
            decisions.append(_fallback(message.message_id, problem))
            failures.append((message.message_id, problem))
            print(f"[{index:>3}/{len(messages)}] {message.message_id}  "
                  f"FALLBACK ({problem})", flush=True)
            continue

        all_overrides.extend(overrides)
        decisions.append(decision)
        print(f"[{index:>3}/{len(messages)}] {message.message_id}  "
              f"{decision.action:<6} {decision.message_type:<16} "
              f"conf={decision.confidence:.2f}"
              f"{'  <- GATED' if overrides else ''}", flush=True)

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for decision in decisions:
            writer.writerow(decision.to_output_row())

    print(f"\nwrote {len(decisions)} rows to {output_path}")
    counts: dict[str, int] = {}
    for decision in decisions:
        counts[decision.action] = counts.get(decision.action, 0) + 1
    print("action distribution:", dict(sorted(counts.items())))

    if gate:
        print(f"\ngate overrides: {len(all_overrides)}")
        for override in all_overrides:
            print(f"  {override.describe()}")
    if failures:
        print(f"\n{len(failures)} fallback row(s):")
        for message_id, problem in failures:
            print(f"  {message_id}: {problem}")
    else:
        print("\nall predictions passed RoutingDecision validation")
    return decisions


if __name__ == "__main__":
    flags = {arg for arg in sys.argv[1:] if arg.startswith("--")}
    positional = [arg for arg in sys.argv[1:] if not arg.startswith("--")]
    run(
        Path(positional[0]) if positional else DEFAULT_OUTPUT,
        trim="--raw" not in flags,
        recompute_confidence=not flags & {"--raw", "--trim-only"},
        gate=not flags & {"--raw", "--trim-only", "--no-gate"},
    )
