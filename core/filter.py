"""Phase 3 — the acceptance-rule filter (the productized strict-basis move).

The bro accepted a training example into the run ONLY if it passed a quality bar (12,732 -> 9,982
kept, 78.4%), rejected the rest, and knew WHY each was rejected. This generalizes that to any project.

THE FORK (decided): REUSE-BY-DEFAULT. A filter rule can REUSE a Phase-2 eval's detector — so the
standard you FILTER training data against is the SAME standard you GRADE the model against (exactly the
bro pattern: the call-detector was both the data filter and the eval check). An INDEPENDENT escape hatch
(templates + code + length/nonempty) covers quality dimensions you filter on but don't eval on.

Each rule has a polarity: keep_if 'pass' (keep iff the detector passes — e.g. claims carry basis, valid
JSON) or 'fail' (keep iff it does NOT — e.g. reject confident-call language). KEEP iff every required
rule is satisfied. Non-destructive: raw.jsonl is preserved; the kept subset is a frozen, versioned
DATASET.lock that the Phase-1 trainer consumes. Rejections are bucketed by reason (the insight: how much
of your data violates your own standard). Hand-rescue of a false rejection is recorded.
"""
from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

from . import eval as evalmod
from . import projects
from .common import read_json, sha256_file, write_json


def _data(project: str) -> Path:
    d = projects.project_dir(project) / "data"
    d.mkdir(parents=True, exist_ok=True)
    return d


def resolve_rule(rule: dict, project: str, work: Path, idx: int):
    """Return ok(input, output) -> bool for a rule (reuse eval / template / code / length / nonempty)."""
    src = rule.get("source", "template")
    if src == "eval":
        ev = evalmod.load_eval(project, rule["eval_version"])
        sc = evalmod.make_scorer(ev, evalmod.eval_dir(project, rule["eval_version"]))
        return lambda i, o: bool(sc(i, o, None)["ok"])
    if src == "code":
        path = work / f"rule_{idx}.py"
        path.write_text(rule["scorer_code"])
        spec = importlib.util.spec_from_file_location(f"rule_{idx}", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        fn = mod.score

        def ok(i, o):
            r = fn(i, o, None)
            if isinstance(r, dict):
                return bool(r.get("ok", r.get("score", 0) >= 0.5))
            return bool(r)
        return ok
    # template-based (independent)
    spec = rule.get("spec", {})
    t = rule.get("template")
    if t == "length":
        mn, mx = spec.get("min_words", 0), spec.get("max_words", 10**9)
        return lambda i, o: mn <= len(o.split()) <= mx
    if t == "nonempty":
        return lambda i, o: bool(o.strip())
    ev = {"kind": "template", "spec": {**spec, "template": t}}
    sc = evalmod.make_scorer(ev, work)
    return lambda i, o: bool(sc(i, o, spec.get("expected"))["ok"])


def _write_training(data: Path, kept: list[dict], val_frac: float) -> None:
    """Rewrite train/valid.jsonl from the kept records (the filtered set the trainer consumes)."""
    recs = [k["record"] for k in kept]
    n_val = max(1, int(len(recs) * val_frac)) if len(recs) > 1 else 0
    train, valr = recs[n_val:], recs[:n_val]
    with (data / "train.jsonl").open("w") as f:
        for r in train:
            f.write(json.dumps(r) + "\n")
    with (data / "valid.jsonl").open("w") as f:
        for r in (valr or train[:1]):
            f.write(json.dumps(r) + "\n")


def apply(project: str, rules: list, source_label: str = "independent", val_frac: float = 0.1) -> dict:
    """Run every raw example through the rules; KEEP iff all required rules are satisfied. Writes the
    frozen filtered set + a bucketed rejection report, and rewrites train/valid from the kept records."""
    import hashlib

    data = _data(project)
    raw_p = data / "raw.jsonl"
    if not raw_p.exists():
        return {"ok": False, "error": "no ingested data — prepare a dataset (Data tab) before filtering"}
    raw = [json.loads(l) for l in raw_p.read_text().splitlines() if l.strip()]
    work = data / "filter"
    work.mkdir(exist_ok=True)
    resolved = [(r, resolve_rule(r, project, work, i)) for i, r in enumerate(rules)]

    kept, rejected, buckets = [], [], {}
    for j, ex in enumerate(raw):
        fail = None
        for r, sc in resolved:
            if not r.get("required", True):
                continue
            try:
                ok = sc(ex["input"], ex["output"])
            except Exception:
                ok = False
            passed = ok if r.get("keep_if", "pass") == "pass" else (not ok)
            if not passed:
                fail = r["name"]
                break
        if fail:
            rejected.append({"idx": j, "input": ex["input"][:160], "output": ex["output"][:240], "reason": fail})
            buckets[fail] = buckets.get(fail, 0) + 1
        else:
            kept.append(ex)

    with (data / "filtered.jsonl").open("w") as f:
        for k in kept:
            f.write(json.dumps(k) + "\n")
    with (data / "rejected.jsonl").open("w") as f:
        for r in rejected:
            f.write(json.dumps(r) + "\n")
    _write_training(data, kept, val_frac)

    n = len(raw)
    examples = {b: [r for r in rejected if r["reason"] == b][:3] for b in buckets}
    report = {"n_raw": n, "n_kept": len(kept), "n_rejected": len(rejected),
              "kept_pct": round(100 * len(kept) / n, 1) if n else 0.0,
              "source_label": source_label, "rules": rules, "buckets": buckets,
              "examples_by_reason": examples, "rescued": [],
              "filtered_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "val_frac": val_frac}
    write_json(data / "filter_report.json", report)

    rules_hash = hashlib.sha256(json.dumps(rules, sort_keys=True).encode()).hexdigest()[:16]
    write_json(projects.project_dir(project) / "DATASET.lock",
               {"files": {"filtered.jsonl": sha256_file(data / "filtered.jsonl")},
                "rules_hash": rules_hash, "n_kept": len(kept),
                "frozen_at": report["filtered_at"]})
    return {"ok": True, **report}


def report(project: str) -> dict | None:
    return read_json(_data(project) / "filter_report.json")


def rejected(project: str) -> list:
    p = _data(project) / "rejected.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def rescue(project: str, idx: int, reason: str) -> dict:
    """Hand-rescue a false rejection back into the kept set (recorded). Non-destructive + reproducible."""
    data = _data(project)
    rep = report(project)
    if not rep:
        return {"ok": False, "error": "no filter run yet"}
    raw = [json.loads(l) for l in (data / "raw.jsonl").read_text().splitlines() if l.strip()]
    if idx < 0 or idx >= len(raw):
        return {"ok": False, "error": "bad index"}
    rejs = rejected(project)
    if not any(r["idx"] == idx for r in rejs):
        return {"ok": False, "error": "that example was not rejected"}
    kept = [json.loads(l) for l in (data / "filtered.jsonl").read_text().splitlines() if l.strip()]
    kept.append(raw[idx])
    with (data / "filtered.jsonl").open("w") as f:
        for k in kept:
            f.write(json.dumps(k) + "\n")
    rejs = [r for r in rejs if r["idx"] != idx]
    with (data / "rejected.jsonl").open("w") as f:
        for r in rejs:
            f.write(json.dumps(r) + "\n")
    _write_training(data, kept, rep.get("val_frac", 0.1))
    rep["n_kept"] += 1
    rep["n_rejected"] -= 1
    rep["kept_pct"] = round(100 * rep["n_kept"] / rep["n_raw"], 1)
    rep.setdefault("rescued", []).append({"idx": idx, "reason": reason,
                                          "at": time.strftime("%Y-%m-%dT%H:%M:%S")})
    write_json(data / "filter_report.json", rep)
    write_json(projects.project_dir(project) / "DATASET.lock",
               {"files": {"filtered.jsonl": sha256_file(data / "filtered.jsonl")},
                "n_kept": rep["n_kept"], "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "note": "includes hand-rescued example(s)"})
    return {"ok": True, "rescued_idx": idx, "n_kept": rep["n_kept"]}


def assert_frozen(project: str) -> bool:
    """DATASET.lock generalized: the filtered set still hashes to its frozen value."""
    data = _data(project)
    lock = read_json(projects.project_dir(project) / "DATASET.lock")
    if not lock or not (data / "filtered.jsonl").exists():
        return False
    return sha256_file(data / "filtered.jsonl") == lock.get("files", {}).get("filtered.jsonl")
