#!/usr/bin/env python
"""Phase-3 KEYSTONE — the arc-completion proof: messy data → reused standard → trained → graded.

Proves the productized strict-basis filter AND that the data standard and the grade standard are the SAME
artifact (reuse-by-default). Real model runs on SmolLM2-135M in a fresh project:
  - bring MESSY data (some examples violate a quality rule), define an eval that encodes the rule,
  - REUSE the eval's detector as the filter → it rejects the violators with a bucketed why-report,
  - the filtered set trains, and the SAME eval grades the trained model at the gate (filter == grade),
  - the lexical-contamination WARNING shows; the optional local near-dup pass runs,
  - the full raw→model provenance trail is intact.
Result -> reports/phase3_keystone.json (+ PASS/FAIL).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core import (data, data_quality, eval as E, filter as F, gate, models,   # noqa: E402
                  prereg as P, projects, run_eval, train)

BASE = "HuggingFaceTB/SmolLM2-135M-Instruct"
PROJECT = "data-keystone"
INSTR = "Echo TERSELY: {w}"
GOOD = ["apple", "river", "mountain", "candle", "pencil", "garden", "window", "planet", "bridge", "forest",
        "pillow", "rocket", "guitar", "anchor", "lantern", "marble", "basket", "feather", "harbor", "meadow",
        "compass", "kettle"]                                   # 22 GOOD: terse one-word echoes
BAD = ["table", "mirror", "ladder", "saddle", "hammer", "copper", "velvet", "ribbon", "signal", "tunnel"]  # 10 verbose
EVALW = ["ocean", "desert", "valley", "castle", "engine", "jacket", "ticket", "button", "summer", "winter",
         "silver", "golden", "purple", "orange", "yellow", "violet", "cherry", "lemon", "ginger", "almond",
         "walnut", "cactus"]                                   # 22 HELD-OUT eval words

TERSE_SCORER = ("def score(inp, output, expected=None):\n"
                "    w = output.split()\n"
                "    return {'ok': 1 <= len(w) <= 3, 'score': 1.0 if 1 <= len(w) <= 3 else 0.0}\n")

R = {"base": BASE, "project": PROJECT, "checks": {}}


def check(name, cond, detail=""):
    R["checks"][name] = {"pass": bool(cond), "detail": str(detail)[:160]}
    print(f"[{'PASS' if cond else 'FAIL'}] {name}: {detail}")
    return cond


def wait_eval(ev, target):
    for _ in range(600):
        s = run_eval.eval_status(PROJECT, ev, target)
        if s.get("state") in ("done", "error"):
            return s
        time.sleep(2)
    return {"state": "timeout"}


def main():
    t0 = time.time()
    path = models.ensure_local(BASE)
    projects.create_project(PROJECT)
    projects.set_base(PROJECT, {"ref": BASE, "local_path": str(path), "info": models.model_info(str(path))})

    # MESSY data: 22 terse (good) + 10 verbose (rule-violating) — instruction schema
    rows = [{"instruction": INSTR.format(w=w), "output": w} for w in GOOD] + \
           [{"instruction": INSTR.format(w=w), "output": f"The word you gave me is {w}, echoed right back to you."}
            for w in BAD]
    det = data.detect_schema(rows)
    data.prepare_dataset(rows, det["schema"], det["mapping"], str(path),
                         str(projects.project_dir(PROJECT) / "data"), val_frac=0.1, seed=0)
    check("raw_persisted", (projects.project_dir(PROJECT) / "data" / "raw.jsonl").exists(), "raw.jsonl written")

    # define the eval that encodes the quality rule (terse output) — held-out words
    ev = E.create_eval(PROJECT, "terse", "code", [{"input": INSTR.format(w=w), "expected": w} for w in EVALW],
                       scorer_code=TERSE_SCORER, max_tokens=24)
    EV = ev["version"]

    # forced baseline on the BASE (verbose model → low terse pass_rate)
    run_eval.start_eval(PROJECT, EV, "base", temp=0.0)
    wait_eval(EV, "base")
    base = run_eval.get_baseline(PROJECT, EV) or {}
    base_pr = base.get("pass_rate")
    check("baseline_ran", base_pr is not None, f"base terse pass_rate={base_pr}")

    # ---- THE FILTER: REUSE the eval's detector as the acceptance rule ----
    rules = [{"name": "terse", "source": "eval", "eval_version": EV, "keep_if": "pass", "required": True}]
    fr = F.apply(PROJECT, rules, source_label="reuse:eval:terse", val_frac=0.1)
    check("filter_rejects_violators", fr.get("ok") and fr["n_rejected"] == 10 and fr["buckets"].get("terse") == 10,
          f"kept {fr.get('n_kept')}/{fr.get('n_raw')} ({fr.get('kept_pct')}%) · rejected {fr.get('buckets')}")
    check("bucketed_report", bool(fr.get("examples_by_reason", {}).get("terse")),
          f"reject examples surfaced: {len(fr.get('examples_by_reason',{}).get('terse',[]))}")
    check("dataset_frozen", F.assert_frozen(PROJECT) and (projects.project_dir(PROJECT) / "DATASET.lock").exists(),
          "DATASET.lock holds")
    R["filter_report"] = {k: fr.get(k) for k in ("n_raw", "n_kept", "n_rejected", "kept_pct", "buckets", "source_label")}

    # hand-rescue a rejected example (recorded, non-destructive)
    rej = F.rejected(PROJECT)
    resc = F.rescue(PROJECT, rej[0]["idx"], reason="keystone: demo false-rejection rescue") if rej else {"ok": False}
    check("hand_rescue", resc.get("ok") and resc.get("n_kept") == fr["n_kept"] + 1, f"rescued -> n_kept {resc.get('n_kept')}")

    # ---- same standard grades the model: commit the SAME eval, train filtered, gate ----
    bar = round((base_pr or 0) + 25)
    cr = P.commit(PROJECT, EV, [{"metric": "pass_rate", "comparator": ">=", "bar": bar, "guard": False}])
    check("prereg_committed", cr.get("ok"), f"bar pass_rate>={bar}")
    same = P.get_prereg(PROJECT)["eval_version"] == rules[0]["eval_version"]
    check("filter_equals_grade_standard", same, f"filter eval {rules[0]['eval_version']} == prereg eval {P.get_prereg(PROJECT)['eval_version']}")

    ver = f"data-{int(time.time())}"
    train.start(PROJECT, {"version": ver, "rank": 8, "num_layers": 4, "iters": 60, "batch_size": 1,
                          "max_seq_len": 128, "learning_rate": 2e-4, "steps_per_report": 10, "seed": 0})
    for _ in range(600):
        if train.status(PROJECT, ver).get("state") in ("done", "error", "stopped"):
            break
        time.sleep(2)
    check("filtered_trains", train.status(PROJECT, ver).get("state") == "done",
          f"final loss {train.status(PROJECT, ver).get('final_train_loss')}")

    run_eval.start_eval(PROJECT, EV, ver, temp=0.0)
    wait_eval(EV, ver)
    v = gate.verdict(PROJECT, ver)
    tr = v.get("trained_metrics", {}).get("pass_rate")
    check("gate_graded_by_same_standard", v.get("ok") and v.get("criteria_results"),
          f"terse pass_rate base {base_pr} -> trained {tr} (bar {bar}) · default {v.get('default_decision')}")
    R["gate"] = {"base_pr": base_pr, "trained_pr": tr, "bar": bar, "default_decision": v.get("default_decision")}

    # ---- the honest contamination story ----
    warn = data_quality.lexical_warning()
    check("semantic_warning_shown", warn.get("code") == "lexical_contamination_limit", warn.get("msg"))
    nd = data_quality.near_dup_pass(PROJECT, EV, threshold=0.6)
    check("near_dup_pass_works", nd.get("ok") and "Jaccard" in nd.get("method", ""),
          f"{nd.get('n_flagged')} flagged · {nd.get('method')}")

    # ---- dataset-level audit runs ----
    aq = data_quality.audit(PROJECT)
    check("data_audit_runs", "summary" in aq, aq.get("summary"))

    # ---- provenance trail intact: raw -> filter -> filtered -> eval -> baseline -> prereg -> adapter -> gate ----
    pdir = projects.project_dir(PROJECT)
    trail = {
        "raw": (pdir / "data" / "raw.jsonl").exists(),
        "filter_report": (pdir / "data" / "filter_report.json").exists(),
        "filtered": (pdir / "data" / "filtered.jsonl").exists(),
        "dataset_lock": (pdir / "DATASET.lock").exists(),
        "eval_lock": E.assert_frozen(PROJECT, EV),
        "baseline": bool(run_eval.get_baseline(PROJECT, EV)),
        "prereg_lock": P.is_committed(PROJECT),
        "adapter_manifest": (pdir / "adapters" / ver / "manifest.json").exists(),
        "gate": (pdir / "adapters" / ver / "gate.json").exists(),
    }
    R["provenance"] = trail
    check("provenance_intact", all(trail.values()), json.dumps(trail))

    finish(t0)


def finish(t0):
    R["runtime_s"] = round(time.time() - t0, 1)
    R["KEYSTONE_PASS"] = all(c["pass"] for c in R["checks"].values())
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "phase3_keystone.json").write_text(json.dumps(R, indent=2))
    print(f"\n=== PHASE-3 KEYSTONE: {'PASS' if R['KEYSTONE_PASS'] else 'FAIL'} ({R['runtime_s']}s) ===")
    print(json.dumps({k: v["pass"] for k, v in R["checks"].items()}, indent=2))


if __name__ == "__main__":
    main()
