"""Phase 2 — the regression gate. Grades a trained adapter against the PRE-REGISTERED criteria.

Shows base-vs-trained side-by-side for every metric, verdict per committed criterion, and DEFAULTS TO
DISCARD: if any criterion failed or a guard metric regressed below baseline, the safe default is not to
keep the adapter — keeping requires an explicit override + a recorded reason. A failed run is a VALID,
valuable outcome ("training didn't help on this eval — that's information"), not an error. Every run is
recorded (eval version, baseline, committed criteria, result, keep/discard + reason) — a reproducible
honest trail.

verdict() needs the adapter's eval results to already exist (run the frozen eval on the adapter first,
same decoding as the committed baseline — apples-to-apples).
"""
from __future__ import annotations

import time

from . import prereg
from . import projects
from . import run_eval
from .common import read_json, write_json


def _gate_path(project: str, adapter_version: str):
    return projects.project_dir(project) / "adapters" / adapter_version / "gate.json"


def _cmp(v, comparator: str, bar) -> bool:
    if v is None:
        return False
    return v >= bar if comparator == ">=" else v <= bar


def verdict(project: str, adapter_version: str) -> dict:
    pr = prereg.get_prereg(project)
    if not pr or not prereg.is_committed(project):
        return {"ok": False, "error": "no committed pre-registration — commit pass criteria before gating"}
    ev_ver = pr["eval_version"]
    trained = run_eval.run_results(project, ev_ver, adapter_version)
    if not trained:
        return {"ok": False, "error": f"run the frozen eval on adapter '{adapter_version}' first"}
    baseline = pr["baseline"]
    tm = trained["metrics"]

    results = []
    for c in pr["criteria"]:
        m, comp, bar = c["metric"], c.get("comparator", ">="), c["bar"]
        tv, bv = tm.get(m), baseline.get(m)
        passed = _cmp(tv, comp, bar)
        regressed = bv is not None and tv is not None and (
            (comp == ">=" and tv < bv) or (comp == "<=" and tv > bv))
        results.append({"metric": m, "comparator": comp, "bar": bar, "guard": c.get("guard", False),
                        "base": bv, "trained": tv,
                        "delta": (round(tv - bv, 1) if (tv is not None and bv is not None) else None),
                        "passed": passed, "regressed": regressed})

    all_pass = all(r["passed"] for r in results)
    any_regress = any(r["regressed"] for r in results)
    default_decision = "keep" if (all_pass and not any_regress) else "discard"
    # side-by-side for EVERY measured metric (not only the committed ones)
    side_by_side = [{"metric": m, "base": baseline.get(m), "trained": tm.get(m),
                     "delta": (round(tm[m] - baseline[m], 1) if (m in tm and m in baseline) else None)}
                    for m in sorted(set(list(tm.keys()) + list(baseline.keys())))
                    if isinstance(tm.get(m), (int, float))]

    gate = {"adapter": adapter_version, "eval_version": ev_ver, "criteria_results": results,
            "side_by_side": side_by_side, "baseline": baseline, "trained_metrics": tm,
            "decoding": {"baseline": pr.get("decoding"), "trained": trained.get("decoding")},
            "all_pass": all_pass, "any_regress": any_regress, "default_decision": default_decision,
            "honest_null": None if default_decision == "keep" else
            "Training didn't clear the bar you committed to — that's a real result, not a mistake. "
            "You just avoided shipping a model that didn't earn it.",
            "decided": None, "graded_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    write_json(_gate_path(project, adapter_version), gate)
    return {"ok": True, **gate}


def decide(project: str, adapter_version: str, action: str, reason: str | None = None) -> dict:
    """Record keep/discard. Default is discard; KEEPING a run that failed needs an explicit reason."""
    gate = read_json(_gate_path(project, adapter_version))
    if not gate:
        return {"ok": False, "error": "no gate verdict yet — run the gate first"}
    if action == "keep" and gate["default_decision"] == "discard" and not (reason and reason.strip()):
        return {"ok": False, "error": "this run did not meet its committed criteria — keeping it is an "
                                      "override that requires a recorded reason"}
    gate["decided"] = action
    gate["decision_reason"] = reason
    gate["override"] = (action != gate["default_decision"])
    gate["decided_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    write_json(_gate_path(project, adapter_version), gate)
    return {"ok": True, "decided": action, "override": gate["override"], "reason": reason}


def get_gate(project: str, adapter_version: str) -> dict | None:
    return read_json(_gate_path(project, adapter_version))
