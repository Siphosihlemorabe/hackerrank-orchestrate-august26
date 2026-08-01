"""Score predictions against the labelled rows in dataset/sample_messages.csv.

Reports action accuracy, message_type accuracy, evidence precision/recall, and
confidence calibration (average confidence when right versus wrong).

Usage: python code/evaluate.py [predictions_csv]
"""

import csv
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
GOLD_PATH = REPO_ROOT / "dataset" / "sample_messages.csv"
DEFAULT_PREDICTIONS = REPO_ROOT / "sample_output.csv"

ACTIONS = ["notify", "digest", "mute"]


def _evidence(value: str) -> set[str]:
    value = (value or "").strip()
    if not value or value.lower() == "none":
        return set()
    return {part.strip() for part in value.split(";") if part.strip()}


def _read(path: Path) -> dict[str, dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["message_id"]: row for row in csv.DictReader(handle)}


def _bar(value: float, width: int = 28) -> str:
    filled = round(value * width)
    return "#" * filled + "." * (width - filled)


def evaluate(predictions_path: Path) -> None:
    gold = _read(GOLD_PATH)
    predicted = _read(predictions_path)

    missing = sorted(set(gold) - set(predicted))
    extra = sorted(set(predicted) - set(gold))
    scored = [message_id for message_id in gold if message_id in predicted]

    print("=" * 68)
    print(f"EVALUATION  {predictions_path.name} vs {GOLD_PATH.name}")
    print("=" * 68)
    print(f"gold rows: {len(gold)}   predicted rows: {len(predicted)}   "
          f"scored: {len(scored)}")
    if missing:
        print(f"MISSING predictions ({len(missing)}): {', '.join(missing)}")
    if extra:
        print(f"EXTRA predictions ({len(extra)}): {', '.join(extra)}")
    if not scored:
        print("nothing to score")
        return

    action_hits = type_hits = both_hits = 0
    confusion: Counter = Counter()
    type_errors: Counter = Counter()
    evidence_tp = evidence_predicted = evidence_gold = 0
    evidence_exact = 0
    confidence_right: list[float] = []
    confidence_wrong: list[float] = []
    per_action_total: Counter = Counter()
    per_action_hit: Counter = Counter()

    for message_id in scored:
        want, got = gold[message_id], predicted[message_id]

        action_ok = want["action"] == got["action"]
        type_ok = want["message_type"] == got["message_type"]
        action_hits += action_ok
        type_hits += type_ok
        both_hits += action_ok and type_ok

        confusion[(want["action"], got["action"])] += 1
        per_action_total[want["action"]] += 1
        per_action_hit[want["action"]] += action_ok
        if not type_ok:
            type_errors[(want["message_type"], got["message_type"])] += 1

        want_ids, got_ids = _evidence(want["evidence_message_ids"]), _evidence(
            got["evidence_message_ids"])
        evidence_tp += len(want_ids & got_ids)
        evidence_predicted += len(got_ids)
        evidence_gold += len(want_ids)
        evidence_exact += want_ids == got_ids

        try:
            confidence = float(got["confidence"])
        except (TypeError, ValueError):
            confidence = 0.0
        (confidence_right if action_ok else confidence_wrong).append(confidence)

    total = len(scored)
    print()
    print("-" * 68)
    print("ACCURACY")
    print("-" * 68)
    for label, hits in (("action", action_hits), ("message_type", type_hits),
                        ("both correct", both_hits)):
        rate = hits / total
        print(f"  {label:<14} {hits:>3}/{total:<3} {rate:>7.1%}  {_bar(rate)}")

    print()
    print("  per gold action:")
    for action in ACTIONS:
        seen = per_action_total[action]
        if not seen:
            continue
        rate = per_action_hit[action] / seen
        print(f"    {action:<8} {per_action_hit[action]:>2}/{seen:<2} "
              f"{rate:>7.1%}  {_bar(rate, 20)}")

    print()
    print("  confusion (rows = gold, cols = predicted):")
    header = "".join(f"{action:>9}" for action in ACTIONS)
    print(f"    {'':<8}{header}")
    for want in ACTIONS:
        cells = "".join(f"{confusion[(want, got)]:>9}" for got in ACTIONS)
        print(f"    {want:<8}{cells}")

    if type_errors:
        print()
        print("  message_type misses (gold -> predicted):")
        for (want, got), count in type_errors.most_common():
            print(f"    {want:<16} -> {got:<16} x{count}")

    print()
    print("-" * 68)
    print("EVIDENCE")
    print("-" * 68)
    precision = evidence_tp / evidence_predicted if evidence_predicted else 0.0
    recall = evidence_tp / evidence_gold if evidence_gold else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if precision + recall else 0.0)
    print(f"  gold ids: {evidence_gold}   predicted ids: {evidence_predicted}"
          f"   correct: {evidence_tp}")
    print(f"  precision  {precision:>7.1%}  {_bar(precision)}")
    print(f"  recall     {recall:>7.1%}  {_bar(recall)}")
    print(f"  f1         {f1:>7.1%}  {_bar(f1)}")
    print(f"  exact set match: {evidence_exact}/{total} "
          f"({evidence_exact / total:.1%})")

    print()
    print("-" * 68)
    print("CONFIDENCE CALIBRATION (by action correctness)")
    print("-" * 68)
    mean_right = (sum(confidence_right) / len(confidence_right)
                  if confidence_right else 0.0)
    mean_wrong = (sum(confidence_wrong) / len(confidence_wrong)
                  if confidence_wrong else 0.0)
    print(f"  when RIGHT  n={len(confidence_right):<3} "
          f"avg confidence {mean_right:.3f}")
    print(f"  when WRONG  n={len(confidence_wrong):<3} "
          f"avg confidence {mean_wrong:.3f}")
    gap = mean_right - mean_wrong
    print(f"  separation  {gap:+.3f}", end="  ")
    if not confidence_wrong:
        print("(no errors to compare against)")
    elif gap > 0.05:
        print("(useful - the model is less sure when it is wrong)")
    elif gap > 0:
        print("(weak - barely distinguishes right from wrong)")
    else:
        print("(inverted - it is at least as confident when wrong)")

    if confidence_right or confidence_wrong:
        every = confidence_right + confidence_wrong
        print(f"  overall range {min(every):.2f} to {max(every):.2f}, "
              f"mean {sum(every) / len(every):.3f}")
    print("=" * 68)


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PREDICTIONS
    if not path.exists():
        sys.exit(f"predictions file not found: {path}")
    evaluate(path)
