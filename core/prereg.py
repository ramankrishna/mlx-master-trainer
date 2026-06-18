"""Phase 2 — pre-registration, wired as an ORDERED step in the train pipeline (not an afterthought).

The discipline is the ORDER: define eval -> audit (no blocks) -> baseline -> commit criteria -> train.
You can't reorder to peek at results and then set the bar:
  - commit() REFUSES unless the eval is frozen, its audit has no BLOCKS, and a baseline exists.
  - the trained run can't gate against criteria that weren't committed first (PREREG.lock generalized).
The committed criteria are frozen before training; the grade is against what was promised.
"""
from __future__ import annotations

import time

from . import eval as evalmod
from . import eval_quality
from . import projects
from . import run_eval
from .common import read_json, sha256_file, write_json


def _prereg_path(project: str):
    return projects.project_dir(project) / "prereg.json"


def pipeline_state(project: str, eval_version: str | None = None) -> dict:
    """The ordered gate-state the UI renders so steps can't be skipped or reordered."""
    evals = evalmod.list_evals(project)
    if eval_version is None and evals:
        eval_version = evals[-1]["version"]
    st = {"eval_version": eval_version, "eval_defined": False, "eval_frozen": False,
          "audit_ok": False, "baseline_done": False, "prereg_committed": False, "blocks": [], "warns": []}
    if not eval_version:
        return st
    ev = evalmod.load_eval(project, eval_version)
    if not ev:
        return st
    st["eval_defined"] = True
    st["eval_frozen"] = evalmod.assert_frozen(project, eval_version)
    audit = eval_quality.audit(project, eval_version)
    st["audit_ok"], st["blocks"], st["warns"] = audit["ok_to_use"], audit["blocks"], audit["warns"]
    base = run_eval.get_baseline(project, eval_version)
    st["baseline_done"], st["baseline"] = bool(base), base
    pr = get_prereg(project)
    st["prereg_committed"] = bool(pr and pr.get("eval_version") == eval_version and is_committed(project))
    st["metrics_available"] = ev.get("metrics_available", [])
    return st


def commit(project: str, eval_version: str, criteria: list, override_reason: str | None = None) -> dict:
    """Freeze the pass criteria BEFORE training. Enforces the order. criteria =
    [{metric, comparator ('>='|'<='), bar, guard: bool}]. Returns ok + warns (bar≤baseline etc.)."""
    if not evalmod.assert_frozen(project, eval_version):
        return {"ok": False, "error": "eval is not frozen — define + freeze the eval first"}
    baseline = run_eval.get_baseline(project, eval_version)
    if baseline is None:
        return {"ok": False, "error": "no baseline — measure the BASE model on this eval before committing "
                                      "(you can't claim improvement you didn't measure against a baseline)"}
    audit = eval_quality.audit(project, eval_version, proposed_criteria=criteria, baseline=baseline)
    if audit["blocks"]:
        return {"ok": False, "error": "eval has blocking quality issues — fix them before committing",
                "blocks": audit["blocks"]}
    if not criteria:
        return {"ok": False, "error": "commit at least one pass criterion"}

    pr = {"eval_version": eval_version, "criteria": criteria, "baseline": baseline,
          "decoding": (run_eval.run_results(project, eval_version, "base") or {}).get("decoding"),
          "committed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
          "override_reason": override_reason}
    write_json(_prereg_path(project), pr)
    # freeze (PREREG.lock generalized)
    write_json(projects.project_dir(project) / "prereg.lock",
               {"hash": sha256_file(_prereg_path(project)), "frozen_at": pr["committed_at"]})
    return {"ok": True, "prereg": pr, "warns": audit["warns"]}


def get_prereg(project: str) -> dict | None:
    return read_json(_prereg_path(project))


def is_committed(project: str) -> bool:
    """PREREG.lock generalized: the committed plan still hashes to its frozen value (no goalpost-moving)."""
    p = _prereg_path(project)
    lock = read_json(projects.project_dir(project) / "prereg.lock")
    return bool(lock and p.exists() and sha256_file(p) == lock.get("hash"))


def clear(project: str) -> dict:
    for f in (_prereg_path(project), projects.project_dir(project) / "prereg.lock"):
        if f.exists():
            f.unlink()
    return {"ok": True}
