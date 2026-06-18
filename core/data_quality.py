"""Phase 3 — dataset-level quality audits (the SET, not just each row) + the honest contamination story.

Audits the training data as a whole — duplication, length/truncation, class balance, completeness,
format consistency, diversity — and explains each in plain language (the Phase-2 stance, applied to data).

Plus the honesty the firewall taught: contamination checks are LEXICAL. Phase 3 touches the training set,
so paraphrase-leak risk is highest here — we say so loudly, and offer an OPTIONAL, off-by-default, local
near-duplicate heuristic (char-3gram Jaccard) that catches rewordings better than exact match but is NOT a
semantic guarantee. Filtered = passes-your-rules + lexically-deduped, NOT semantically clean.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from . import eval as evalmod
from . import projects

_norm = evalmod._norm


def _raw(project: str) -> list[dict]:
    p = projects.project_dir(project) / "data" / "raw.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def _ngrams(s: str, n: int = 3) -> set:
    s = _norm(s)
    return {s[i:i + n] for i in range(max(0, len(s) - n + 1))} or {s}


def _jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a or b) else 0.0


def audit(project: str, max_seq_len: int = 2048) -> dict:
    raw = _raw(project)
    n = len(raw)
    if not n:
        return {"error": "no ingested data — prepare a dataset first"}
    findings = []

    def add(level, code, msg, why):
        findings.append({"level": level, "code": code, "msg": msg, "why": why})

    # completeness
    empty_in = sum(1 for r in raw if not r["input"].strip())
    empty_out = sum(1 for r in raw if not r["output"].strip())
    if empty_out:
        add("warn", "completeness", f"{empty_out} example(s) have an empty output",
            "An example with no completion teaches the model to say nothing — drop or fix these.")

    # duplication (exact, normalized input+output)
    keys = [_norm(r["input"] + " || " + r["output"]) for r in raw]
    dup = n - len(set(keys))
    if dup:
        add("warn", "duplication", f"{dup} exact duplicate example(s)",
            "Duplicates over-weight a few examples — the model memorizes them and your eval over-credits "
            "that pattern. Dedup before training.")

    # length / truncation
    tok = None
    try:
        meta = projects.get_project(project)
        base = meta["base"].get("local_path") or meta["base"]["ref"]
        from core.data import _tokenizer
        tok = _tokenizer(base)
    except Exception:
        tok = None
    if tok is not None:
        lens = [len(tok((r["input"] + " " + r["output"]), add_special_tokens=False)["input_ids"]) for r in raw]
        over = sum(1 for x in lens if x > max_seq_len)
        s = sorted(lens)
        length = {"unit": "tokens", "min": s[0], "median": s[len(s) // 2], "max": s[-1], "over_cap": over}
        if over:
            add("warn", "truncation", f"{over} example(s) exceed {max_seq_len} tokens",
                "Examples longer than the context window get cut off mid-completion — the model learns to "
                "produce truncated, broken outputs. Raise max_seq_length or shorten these.")
    else:
        wl = [len(r["output"].split()) for r in raw]
        s = sorted(wl)
        length = {"unit": "words", "min": s[0], "median": s[len(s) // 2], "max": s[-1], "over_cap": None}

    # class balance (if the outputs look label-like)
    out_norm = [_norm(r["output"]) for r in raw]
    distinct = set(out_norm)
    balance = None
    if len(distinct) <= 20 or len(distinct) / n < 0.3:
        dist = Counter(out_norm)
        top, topn = dist.most_common(1)[0]
        balance = {"distinct_labels": len(distinct), "distribution": dict(dist.most_common(8))}
        if topn / n > 0.85:
            add("warn", "class_imbalance", f"one output ('{top[:30]}') is {round(100*topn/n)}% of the data",
                "A dominant label lets the model score well by always guessing it while learning nothing. "
                "Balance the classes, or read per-class metrics rather than accuracy at eval time.")

    # format consistency
    fmts = Counter(("messages" if "messages" in r["record"] else "prompt" if "prompt" in r["record"] else "text")
                   for r in raw)
    if len(fmts) > 1:
        add("warn", "format_mix", f"mixed training formats in one set: {dict(fmts)}",
            "Inconsistent record formats template differently — some examples may train on the wrong "
            "structure. Keep one schema per dataset.")

    # diversity (input phrasing repetition)
    prefixes = Counter(" ".join(_norm(r["input"]).split()[:4]) for r in raw)
    top_prefix, top_count = prefixes.most_common(1)[0]
    diversity_share = round(100 * top_count / n, 1)
    if diversity_share > 50 and n > 8:
        add("warn", "low_diversity", f"{diversity_share}% of inputs share the same opening phrasing",
            f"Your data is dominated by one phrasing pattern ('{top_prefix}…'). The model may overfit the "
            "surface form instead of the task — vary how prompts are worded.")

    summary = (f"{n} examples · {len(distinct)} distinct outputs · {dup} exact dup · "
               f"top phrasing {diversity_share}% · " + (f"{length['over_cap']} over cap" if length.get("over_cap") else "lengths ok"))
    return {"n": n, "empty_input": empty_in, "empty_output": empty_out, "exact_duplicates": dup,
            "length": length, "balance": balance, "formats": dict(fmts),
            "diversity_top_share_pct": diversity_share, "findings": findings, "summary": summary}


# --------------------------------------------------------------------------- #
# the honest contamination story
# --------------------------------------------------------------------------- #
def lexical_warning() -> dict:
    return {"level": "warn", "code": "lexical_contamination_limit",
            "msg": "the contamination check is LEXICAL",
            "why": "It catches exact and near-exact overlap, NOT paraphrased or semantically-equivalent "
                   "duplicates. If your eval examples are rewordings of training examples, the score will be "
                   "inflated and this tool will not catch it. 'Filtered' means passes-your-rules + "
                   "lexically-deduped — NOT semantically clean."}


def near_dup_pass(project: str, eval_version: str, threshold: float = 0.7, cap: int = 4000) -> dict:
    """OPTIONAL, off-by-default, pure-local heuristic: char-3gram Jaccard between each EVAL input and each
    TRAIN input. Catches rewordings better than exact match. It is a lexical-overlap HEURISTIC, NOT learned
    embeddings (true embeddings would need an optional local model, not bundled) — evidence, not proof."""
    ev = evalmod.load_eval(project, eval_version)
    if not ev:
        return {"ok": False, "error": "eval not found"}
    train_inputs = [r["input"] for r in _raw(project)][:cap]
    eval_inputs = [d.get("input", "") for d in ev["dataset"]]
    tg = [(t, _ngrams(t)) for t in train_inputs]
    flagged = []
    for ei in eval_inputs:
        eg = _ngrams(ei)
        best, bestt = 0.0, None
        for t, g in tg:
            j = _jaccard(eg, g)
            if j > best:
                best, bestt = j, t
        if best >= threshold and _norm(ei) != _norm(bestt or ""):   # exclude exact (already lexically caught)
            flagged.append({"eval_input": ei[:120], "nearest_train": (bestt or "")[:120], "similarity": round(best, 2)})
    return {"ok": True, "method": "char-3gram Jaccard (heuristic, pure-local — NOT learned embeddings)",
            "threshold": threshold, "n_eval": len(eval_inputs), "n_train": len(train_inputs),
            "flagged": flagged[:50], "n_flagged": len(flagged),
            "note": "evidence of paraphrase overlap, not proof of clean/dirty. Review flagged pairs by hand."}
