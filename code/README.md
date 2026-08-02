# Message Notification Router

Routes every message in `dataset/messages.csv` to `notify`, `digest`, or `mute`,
and writes `output.csv`.

Each message is decided by one multimodal Gemini call. Images and voice notes
are attached to that same call, so their content is judged directly - there is
no separate OCR or transcription stage. Deterministic code surrounds the model:
it assembles the evidence beforehand, and trims, re-scores, and safety-checks
the result afterwards.

---

## Setup

Requires **Python 3.10 or newer** (tested on 3.13.7) and a Gemini API key.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

If you are packaging only this folder, the two dependencies are
`google-genai>=2.16.0` and `pydantic>=2.13.4`.

Then provide the API key. Copy `.env.example` to `.env` in the repository root
and fill it in:

```text
GEMINI_API_KEY=your-key-here
# GEMINI_MODEL=gemini-3.6-flash    # optional, this is the default
```

`.env` is gitignored and never committed. A real `GEMINI_API_KEY` environment
variable takes precedence over the file, so exporting it works too:

```bash
export GEMINI_API_KEY=your-key-here     # Windows: $env:GEMINI_API_KEY="..."
```

---

## Run

From the repository root:

```bash
python code/main.py
```

That writes `output.csv` - one row per message, with the required columns:

```text
message_id,action,message_type,reason,confidence,evidence_message_ids
```

Paths resolve relative to the source files, so the command works from any
working directory. Pass a different destination if you want one:

```bash
python code/main.py predictions.csv
```

### Score against the labelled examples

```bash
python code/run_samples.py      # routes dataset/sample_messages.csv -> sample_output.csv
python code/evaluate.py         # scores it against the gold labels
```

`evaluate.py` reports action accuracy, message_type accuracy, an action
confusion matrix, evidence precision/recall/F1, and confidence calibration
(average confidence when right versus wrong). Point it at any predictions file:

```bash
python code/evaluate.py path/to/predictions.csv
```

### Ablation flags

Both runners accept flags that switch off post-processing stages, which is how
each stage's contribution was measured. All variants reuse the cache, so
switching between them costs no API calls.

| Flag | Effect |
|---|---|
| `--raw` | Raw model output: no trimming, no re-scored confidence, no gate |
| `--trim-only` | Evidence trimming only |
| `--no-gate` | Everything except the safety gate |
| *(none)* | Full pipeline |

---

## Caching

Every model response is written to `cache/gemini/<message_id>.json`. A cached
message is never re-sent, so:

- an interrupted run resumes where it stopped, at no cost
- re-running to change post-processing is free and instant

`cache/` is gitignored. Delete a message's file to force it to be re-routed;
delete the directory to start clean. Note that editing a prompt or a signal in
`loaders.py` does **not** invalidate the cache automatically - clear the
affected entries if you change what the model sees.

Calls are paced by `GEMINI_MIN_INTERVAL` (default 1 second) and retried twice
with backoff on 429 and 5xx. If you are on a rate-limited key, raise it:

```bash
GEMINI_MIN_INTERVAL=16 python code/main.py
```

A transport failure aborts the run rather than writing a placeholder row, so a
rate limit can never quietly become a wrong prediction. Only genuine model
failures - unparseable JSON, or output that fails validation - produce a
fallback row (`digest`/`unknown`), which neither interrupts the user nor
silently suppresses something that might matter.

---

## How it works

| File | Responsibility |
|---|---|
| `loaders.py` | Reads every CSV once (cached), then `build_context(message)` gathers the rows for one message and derives 13 plain-English signals |
| `gemini_client.py` | One structured, cached, multimodal call per message; returns parsed JSON |
| `decide.py` | Trims over-cited evidence; recomputes confidence from signal agreement |
| `gate.py` | Deterministic safety rules; can only make a decision stricter |
| `schema.py` | `RoutingDecision` - the validation boundary and output formatting |
| `main.py` | Full dataset runner |
| `run_samples.py` | Labelled-sample runner |
| `evaluate.py` | Scoring |

**Signals.** `build_context` always emits the same 13 keys in the same order, so
downstream code never guards for missing keys. Each is a sentence with the
numbers inside it - "this user averages 8.5 notifications a day and dismisses
15% of them" - rather than a raw table. Signals cover near-duplicate history and
its recorded outcome, notification fatigue, do-not-disturb overlap, staleness,
media, and the group or business branch.

**Validation.** Model output only becomes a `RoutingDecision` after passing
Pydantic validation, including a soft rule that a message muted for a stated
risk reason must be typed `scam` or `spam`.

**Evidence.** The model tends to cite every similar prior message. `decide.py`
ranks the ids it named by textual similarity, how decisive the recorded outcome
was, and whether the sender matches, then keeps the top one - a second only when
it is comparably decisive and a genuine textual match. Ids that are not real
candidates are dropped, so a hallucinated id cannot reach the output.

**Confidence.** The model's self-reported confidence did not separate its
correct answers from its wrong ones, so it is replaced by a value computed from
how strongly the deterministic signals agree with the chosen action.

**Gate.** Two narrow rules, applied after the model, that can only move a
decision toward `mute`:

1. A sender domain that differs from the brand's official domain **and** was
   registered under 90 days ago - the lookalike-domain pattern. Both conditions
   are required: mismatch alone fires on established brands that send marketing
   from a separate long-lived domain.
2. An exact repeat of a message this user previously reported.

Every override is logged with its reason.

---

## Results on the 30 labelled examples

| Metric | Score |
|---|---|
| Action accuracy | 96.7% (29/30) |
| `message_type` accuracy | 86.7% (26/30) |
| Evidence precision / recall / F1 | 56.7% / 54.8% / 55.7% |
| Confidence, correct vs incorrect | 0.808 vs 0.710 |
| Validation failures | 0 |

`notify` and `mute` are both perfect (9/9 and 10/10); the single action error is
one `digest` predicted as `notify`.

Two caveats worth knowing. The evidence-trimming thresholds were tuned on these
30 rows, so the rule's shape is principled but the exact constants may not
generalise. And confidence separation is computed against a single incorrect
row, which makes it directionally right but statistically thin - the sturdier
evidence is that mean absolute error against the gold confidences halved.
