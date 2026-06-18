"""Phase 2 — THE WEDGE. Opinionated eval-quality guardrails.

Before an eval can gate a model, the tool AUDITS the eval itself and BLOCKS or WARNS — because a
technically-valid-but-meaningless eval that lets you feel rigorous while being fooled is WORSE than no
eval. Every finding explains WHY in plain language (teach the judgment, don't just flag).

BLOCK (cannot gate until fixed):
  - too_small        n below a hard floor (20): the pass/fail is noise.
  - contamination    an eval example is also in the training data: the cardinal sin.
  - missing_expected the template needs a label/answer per example and some are missing.
WARN (allowed, but argued):
  - bar_below_baseline  the bar is at/below what the BASE already scores — rubber-stamping.
  - small_n             20-50: wide error bars; a few-point change may be noise.
  - class_imbalance     one label dominates; accuracy is misleading.
  - single_dimension    only one thing measured; training may have broken something else.
  - round_bar           a suspiciously round bar — justified, or a guess?
"""
from __future__ import annotations

import json
from pathlib import Path

from . import eval as evalmod
from . import projects
from .common import read_json

HARD_FLOOR = 20          # below this, pass/fail is statistical noise
SMALL_N = 50             # 20..50 -> wide error bars
IMBALANCE = 0.85         # one label > 85% -> misleading accuracy


def _norm(s) -> str:
    return evalmod._norm(s)


def training_inputs(project: str) -> set[str]:
    """Normalized user inputs the model was/will be trained on (for contamination checks)."""
    out = set()
    for fn in ("train.jsonl", "valid.jsonl"):
        p = projects.project_dir(project) / "data" / fn
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if "messages" in r:
                txt = " ".join(m.get("content", "") for m in r["messages"] if m.get("role") == "user")
            elif "prompt" in r:
                txt = r["prompt"]
            else:
                txt = r.get("text", "")
            out.add(_norm(txt))
    return out


def _needs_expected(ev: dict) -> bool:
    t = ev.get("spec", {}).get("template", ev["kind"])
    return ev["kind"] == "template" and t in ("exact", "classification", "numeric", "refusal")


def audit(project: str, eval_version: str, proposed_criteria: list | None = None,
          baseline: dict | None = None) -> dict:
    ev = evalmod.load_eval(project, eval_version)
    ds = ev["dataset"]
    n = len(ds)
    blocks, warns = [], []

    def add(bucket, code, msg, why, detail=None):
        bucket.append({"code": code, "msg": msg, "why": why, **({"detail": detail} if detail else {})})

    # ---- BLOCKS ----
    if n < HARD_FLOOR:
        add(blocks, "too_small", f"eval has {n} examples (< {HARD_FLOOR})",
            f"With n={n}, one or two flips swing the pass rate by double digits — the result is noise, "
            f"not a measurement. Add examples until you have at least {HARD_FLOOR}.")

    train = training_inputs(project)
    overlap = [d for d in ds if _norm(d.get("input", "")) in train]
    if overlap:
        add(blocks, "contamination", f"{len(overlap)} eval example(s) are also in the training data",
            "If you evaluate on what you trained on, a high score just means the model memorized — it "
            "tells you nothing about generalization. This is the cardinal sin of eval. Remove the "
            "overlapping examples or build a genuine held-out set.",
            detail=[d["input"][:80] for d in overlap[:5]])

    if _needs_expected(ev):
        missing = sum(1 for d in ds if d.get("expected", "") in (None, ""))
        if missing:
            add(blocks, "missing_expected", f"{missing} example(s) have no expected answer/label",
                "This template scores output against an expected value; without it the example can't be "
                "scored. Provide an expected answer/label for every example.")

    # ---- WARNS ----
    if HARD_FLOOR <= n <= SMALL_N:
        add(warns, "small_n", f"n={n} is small",
            f"At n={n} the error bars are wide — a 3-4 point change can be noise, not real improvement. "
            "More examples = a sharper signal.")

    t = ev.get("spec", {}).get("template", ev["kind"])
    if ev["kind"] == "template" and t == "classification":
        labels = [str(x) for x in ev["spec"].get("labels", [])]
        counts = {l: sum(1 for d in ds if _norm(d.get("expected")) == _norm(l)) for l in labels}
        if n and max(counts.values(), default=0) / n > IMBALANCE:
            top = max(counts, key=counts.get)
            add(warns, "class_imbalance", f"'{top}' is {round(100*counts[top]/n)}% of the eval",
                "When one label dominates, a model that always guesses it scores high while learning "
                "nothing. Accuracy is misleading here — read per-class precision/recall, not just accuracy.",
                detail=counts)

    if len(ev.get("metrics_available", [])) <= 1 and not any(c.get("guard") for c in (proposed_criteria or [])):
        add(warns, "single_dimension", "the eval measures a single dimension",
            "You're checking one thing. Training can improve your target while quietly breaking something "
            "else (format, refusals, a capability). Consider a guard metric so a regression can't hide.")

    if proposed_criteria:
        for c in proposed_criteria:
            bar = c.get("bar")
            if baseline and c.get("metric") in baseline and c.get("comparator", ">=") == ">=":
                base_v = baseline[c["metric"]]
                if base_v >= bar:
                    add(warns, "bar_below_baseline",
                        f"bar {c['metric']}≥{bar} but base already scores {base_v}",
                        f"The base model already hits {base_v} on {c['metric']}, so a bar of {bar} passes "
                        "without any training. You're rubber-stamping, not measuring improvement. Set the "
                        f"bar above the baseline ({base_v}).")
            if isinstance(bar, (int, float)) and (bar in (50, 60, 70, 75, 80, 90, 95, 100) or bar % 10 == 0):
                add(warns, "round_bar", f"bar {c.get('metric')}={bar} is a round number",
                    "A round bar is often a guess, not a justified threshold. Is this the level that "
                    "actually matters for your use, or did it just feel right?")

    return {"eval_version": eval_version, "n": n, "blocks": blocks, "warns": warns,
            "ok_to_use": not blocks,
            "summary": (f"{len(blocks)} block(s), {len(warns)} warning(s)" if (blocks or warns)
                        else "clean — no quality issues found")}
