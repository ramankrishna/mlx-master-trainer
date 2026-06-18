"""Phase 2 — eval DEFINITION + scoring. Three ways to define an eval, one internal object.

A. Templates (no code): exact / contains / classification / format_json / refusal / numeric.
B. Example-based (no code): label good/bad outputs; lexical similarity scorer (pure-local; a local
   judge is opt-in, off by default — documented tradeoff).
C. Code escape hatch: a Python scorer(inp, output, expected) -> score, for anything templates can't say
   (the bro's call-detector, an NPC-Reason SymPy verifier, etc.).

Whatever the path, the eval is a FROZEN, versioned artifact (EVAL.lock generalized): eval.json + eval.lock.
The scorer is reconstructable from (kind, spec) — code-kind also reads scorer.py from the eval dir — so a
detached runner can score without extra state. `metrics_available` tells prereg/gate what is gateable.
"""
from __future__ import annotations

import difflib
import importlib.util
import json
import re
import time
from pathlib import Path

from . import projects
from .common import read_json, sha256_file, write_json

TEMPLATE_TYPES = {
    "exact":          "output equals expected (normalized) — deterministic tasks",
    "contains":       "output contains a required string / matches a regex",
    "classification": "output is one of N labels — accuracy + per-class + macro-F1",
    "format_json":    "output parses as JSON (optionally has required keys)",
    "refusal":        "output does / doesn't refuse (expected = 'refuse' | 'answer')",
    "numeric":        "extract a number, check correctness within a tolerance",
}


def _norm(s) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())


def _num(s):
    m = re.findall(r"-?\d+(?:\.\d+)?", str(s).replace(",", ""))
    return float(m[-1]) if m else None


# --------------------------------------------------------------------------- #
# scorer factory — (kind, spec) -> callable(inp, output, expected) -> dict
#   returns {"ok": bool, "score": 0..1, "pred": <label/extracted>}
# --------------------------------------------------------------------------- #
def make_scorer(ev: dict, eval_dir: Path):
    kind, spec = ev["kind"], ev.get("spec", {})

    if kind == "code":
        path = eval_dir / "scorer.py"
        sp = importlib.util.spec_from_file_location("user_scorer", path)
        mod = importlib.util.module_from_spec(sp)
        sp.loader.exec_module(mod)
        fn = getattr(mod, "score")

        def code_score(inp, output, expected=None):
            r = fn(inp, output, expected)
            if isinstance(r, dict):
                s = float(r.get("score", 0.0))
                return {"ok": bool(r.get("ok", s >= 0.5)), "score": s, "pred": r.get("pred")}
            if isinstance(r, bool):
                return {"ok": r, "score": 1.0 if r else 0.0, "pred": None}
            s = float(r)
            return {"ok": s >= 0.5, "score": s, "pred": None}
        return code_score

    if kind == "examples":
        good = [_norm(g) for g in spec.get("good", [])]
        bad = [_norm(b) for b in spec.get("bad", [])]
        thr = float(spec.get("threshold", 0.6))

        def ex_score(inp, output, expected=None):
            o = _norm(output)
            sg = max((difflib.SequenceMatcher(None, o, g).ratio() for g in good), default=0.0)
            sb = max((difflib.SequenceMatcher(None, o, b).ratio() for b in bad), default=0.0)
            return {"ok": sg >= thr and sg >= sb, "score": round(sg, 3), "pred": None}
        return ex_score

    # templates
    t = spec.get("template", kind)
    if t == "exact":
        return lambda i, o, e: {"ok": _norm(o) == _norm(e), "score": float(_norm(o) == _norm(e)), "pred": _norm(o)}
    if t == "contains":
        pat, rgx = spec.get("pattern", ""), spec.get("regex", False)
        def cont(i, o, e):
            need = e if e not in (None, "") else pat
            hit = bool(re.search(need, o)) if rgx else (_norm(need) in _norm(o))
            return {"ok": hit, "score": float(hit), "pred": None}
        return cont
    if t == "classification":
        labels = [str(x) for x in spec.get("labels", [])]
        def clf(i, o, e):
            on = _norm(o)
            pred = next((l for l in labels if _norm(l) in on), None)
            ok = pred is not None and _norm(pred) == _norm(e)
            return {"ok": ok, "score": float(ok), "pred": pred}
        return clf
    if t == "format_json":
        keys = spec.get("required_keys", [])
        def fj(i, o, e):
            try:
                obj = json.loads(o[o.find("{"): o.rfind("}") + 1] if "{" in o else o)
                ok = isinstance(obj, (dict, list)) and (not keys or all(k in obj for k in keys))
            except Exception:
                ok = False
            return {"ok": ok, "score": float(ok), "pred": None}
        return fj
    if t == "refusal":
        markers = [m.lower() for m in spec.get("markers",
                   ["i can't", "i cannot", "i'm not able", "i am not able", "i won't", "sorry"])]
        def refu(i, o, e):
            refused = any(m in o.lower() for m in markers)
            want = _norm(e) in ("refuse", "true", "yes", "1")
            ok = refused == want
            return {"ok": ok, "score": float(ok), "pred": "refused" if refused else "answered"}
        return refu
    if t == "numeric":
        tol = float(spec.get("tolerance", 0.0))
        def numc(i, o, e):
            po, pe = _num(o), _num(e)
            ok = po is not None and pe is not None and abs(po - pe) <= tol
            return {"ok": ok, "score": float(ok), "pred": po}
        return numc
    raise ValueError(f"unknown eval template/kind: {kind}/{t}")


def metrics_available(ev: dict) -> list[str]:
    if ev["kind"] == "examples":
        return ["pass_rate", "mean_score"]
    if ev["kind"] == "code":
        return ["pass_rate", "mean_score"]
    t = ev.get("spec", {}).get("template", ev["kind"])
    if t == "classification":
        return ["accuracy", "macro_f1"]
    return ["accuracy"]


def aggregate(ev: dict, scored: list[dict]) -> dict:
    """scored = [{ok, score, pred, expected}]. Returns the named metrics (0..100) + extras."""
    n = len(scored)
    if not n:
        return {"n": 0}
    acc = 100 * sum(s["ok"] for s in scored) / n
    mean = 100 * sum(s["score"] for s in scored) / n
    out = {"n": n, "accuracy": round(acc, 1), "pass_rate": round(acc, 1), "mean_score": round(mean, 1)}
    t = ev.get("spec", {}).get("template", ev["kind"])
    if ev["kind"] == "template" and t == "classification":
        labels = [str(x) for x in ev["spec"].get("labels", [])]
        per, f1s = {}, []
        for lab in labels:
            tp = sum(1 for s in scored if _norm(s["pred"]) == _norm(lab) and _norm(s["expected"]) == _norm(lab))
            fp = sum(1 for s in scored if _norm(s["pred"]) == _norm(lab) and _norm(s["expected"]) != _norm(lab))
            fn = sum(1 for s in scored if _norm(s["pred"]) != _norm(lab) and _norm(s["expected"]) == _norm(lab))
            prec = tp / (tp + fp) if tp + fp else 0.0
            rec = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
            support = sum(1 for s in scored if _norm(s["expected"]) == _norm(lab))
            per[lab] = {"precision": round(100 * prec, 1), "recall": round(100 * rec, 1),
                        "f1": round(100 * f1, 1), "support": support}
            f1s.append(f1)
        out["macro_f1"] = round(100 * sum(f1s) / len(f1s), 1) if f1s else 0.0
        out["per_class"] = per
        out["label_distribution"] = {lab: per[lab]["support"] for lab in labels}
    return out


# --------------------------------------------------------------------------- #
# eval artifact: create / load / freeze / list
# --------------------------------------------------------------------------- #
def _evals_dir(project: str) -> Path:
    d = projects.project_dir(project) / "evals"
    d.mkdir(parents=True, exist_ok=True)
    return d


def create_eval(project: str, name: str, kind: str, dataset: list[dict],
                spec: dict | None = None, scorer_code: str | None = None,
                max_tokens: int = 64, version: str | None = None) -> dict:
    """Persist a frozen, versioned eval. kind = 'template' | 'examples' | 'code'.
    dataset = [{"input": str, "expected": ...}] (expected optional for examples/code)."""
    version = (version or f"{name}-{int(time.time())}").strip().replace(" ", "-")
    d = _evals_dir(project) / version
    d.mkdir(parents=True, exist_ok=True)
    if kind == "code":
        if not scorer_code:
            raise ValueError("code eval needs scorer_code")
        (d / "scorer.py").write_text(scorer_code)
    ev = {"name": name, "version": version, "kind": kind, "spec": spec or {},
          "dataset": dataset, "n": len(dataset), "max_tokens": max_tokens,
          "created_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    ev["metrics_available"] = metrics_available(ev)
    write_json(d / "eval.json", ev)
    # freeze: hash the definition (+ scorer.py) -> eval.lock (drift detector)
    files = {"eval.json": sha256_file(d / "eval.json")}
    if (d / "scorer.py").exists():
        files["scorer.py"] = sha256_file(d / "scorer.py")
    write_json(d / "eval.lock", {"version": version, "files": files,
                                 "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
    return ev


def load_eval(project: str, version: str) -> dict:
    return read_json(_evals_dir(project) / version / "eval.json")


def eval_dir(project: str, version: str) -> Path:
    return _evals_dir(project) / version


def list_evals(project: str) -> list[dict]:
    out = []
    for d in sorted(_evals_dir(project).glob("*")):
        ev = read_json(d / "eval.json")
        if ev:
            base = read_json(d / "runs" / "base" / "results.json")
            out.append({"version": ev["version"], "name": ev["name"], "kind": ev["kind"],
                        "template": ev.get("spec", {}).get("template"), "n": ev["n"],
                        "metrics_available": ev["metrics_available"], "frozen": assert_frozen(project, ev["version"]),
                        "has_baseline": bool(base), "baseline": base.get("metrics") if base else None})
    return out


def assert_frozen(project: str, version: str) -> bool:
    """EVAL.lock generalized: every locked file still hashes to its frozen value."""
    d = eval_dir(project, version)
    lock = read_json(d / "eval.lock")
    if not lock:
        return False
    for fn, h in lock.get("files", {}).items():
        if not (d / fn).exists() or sha256_file(d / fn) != h:
            return False
    return True
