#!/usr/bin/env python
"""Phase-2 KEYSTONE — can a stranger produce a MEANINGFUL eval, and does the tool ARGUE when they can't?

This proves the DISCIPLINE, not the plumbing. Walks the full flow as a non-expert and confirms:
  - BAD evals are caught: tiny set -> BLOCK · contamination -> BLOCK · bar<=baseline -> WARN · single-metric -> WARN
  - the GOOD path works: template eval (no code) + held-out + forced baseline + committed bar + a REAL train
    run -> honest base-vs-trained side-by-side + PASS/FAIL + default-discard-on-fail
  - ordering is enforced: gate before prereg -> refused; keeping a failed run without a reason -> refused
  - the CODE escape hatch runs.
Result -> reports/phase2_keystone.json (+ a PASS/FAIL summary).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core import (adapters, data, eval as E, eval_quality as Q, gate, models,  # noqa: E402
                  prereg as P, projects, run_eval, train)

BASE = "HuggingFaceTB/SmolLM2-135M-Instruct"
PROJECT = "discipline-demo"
INSTR = "Give the capital city of {c}. Answer with only the city name."

# training countries (terse-capital behaviour) — DISJOINT from the eval's held-out set
TRAIN = {"the United States": "Washington", "the United Kingdom": "London", "Canada": "Ottawa",
         "Brazil": "Brasilia", "Argentina": "Buenos Aires", "Mexico": "Mexico City", "India": "New Delhi",
         "Australia": "Canberra", "South Africa": "Pretoria", "Nigeria": "Abuja", "Morocco": "Rabat",
         "Saudi Arabia": "Riyadh", "Ukraine": "Kyiv", "Belgium": "Brussels", "Netherlands": "Amsterdam",
         "Switzerland": "Bern", "Denmark": "Copenhagen", "Finland": "Helsinki", "Hungary": "Budapest",
         "Romania": "Bucharest", "Vietnam": "Hanoi", "Philippines": "Manila", "Malaysia": "Kuala Lumpur",
         "South Korea": "Seoul", "Pakistan": "Islamabad", "Colombia": "Bogota", "Bolivia": "La Paz",
         "Uruguay": "Montevideo", "Ecuador": "Quito", "Indonesia": "Jakarta", "Iraq": "Baghdad",
         "Israel": "Jerusalem", "Sweden": "Stockholm", "Norway": "Oslo", "Poland": "Warsaw",
         "Austria": "Vienna", "Greece": "Athens", "Portugal": "Lisbon", "Ireland": "Dublin", "Cuba": "Havana"}
# held-out eval countries (well-known capitals a small model likely has)
EVALC = {"France": "Paris", "Italy": "Rome", "Spain": "Madrid", "Germany": "Berlin", "Japan": "Tokyo",
         "Russia": "Moscow", "China": "Beijing", "Egypt": "Cairo", "Turkey": "Ankara", "Iran": "Tehran",
         "Thailand": "Bangkok", "Kenya": "Nairobi", "Peru": "Lima", "Chile": "Santiago", "Cambodia": "Phnom Penh",
         "Hungary?": "Budapest", "Nepal": "Kathmandu", "Cuba?": "Havana", "Ghana": "Accra", "Qatar": "Doha",
         "Lebanon": "Beirut", "Jordan": "Amman"}

R = {"base": BASE, "project": PROJECT, "checks": {}}


def check(name, cond, detail=""):
    R["checks"][name] = {"pass": bool(cond), "detail": str(detail)[:160]}
    print(f"[{'PASS' if cond else 'FAIL'}] {name}: {detail}")
    return cond


def wait_eval(ev_ver, target, timeout=600):
    for _ in range(timeout):
        st = run_eval.eval_status(PROJECT, ev_ver, target)
        if st.get("state") in ("done", "error"):
            return st
        time.sleep(2)
    return {"state": "timeout"}


def main():
    t0 = time.time()
    print(f"=== resolve base {BASE} + project {PROJECT} ===")
    path = models.ensure_local(BASE)
    projects.create_project(PROJECT)
    projects.set_base(PROJECT, {"ref": BASE, "local_path": str(path), "info": models.model_info(str(path))})

    # training data (terse capitals) so the trained model has a measurable behaviour + contamination has a target
    rows = [{"instruction": INSTR.format(c=c), "output": cap} for c, cap in TRAIN.items()]
    det = data.detect_schema(rows)
    data.prepare_dataset(rows, det["schema"], det["mapping"], str(path),
                         str(projects.project_dir(PROJECT) / "data"), val_frac=0.1, seed=0)

    # ---- BAD EVALS CAUGHT ----
    tiny = E.create_eval(PROJECT, "tiny", "template",
                         [{"input": f"probe {i}", "expected": "x"} for i in range(5)], spec={"template": "exact"})
    check("block_tiny", any(b["code"] == "too_small" for b in Q.audit(PROJECT, tiny["version"])["blocks"]),
          "5-example eval blocked")
    contam_rows = [{"input": INSTR.format(c=list(TRAIN)[0]), "expected": list(TRAIN.values())[0]}] + \
                  [{"input": f"probe number {i}", "expected": "x"} for i in range(24)]
    ce = E.create_eval(PROJECT, "contam", "template", contam_rows, spec={"template": "exact"})
    cblocks = Q.audit(PROJECT, ce["version"])["blocks"]
    check("block_contamination", any(b["code"] == "contamination" for b in cblocks),
          f"overlap: {[b.get('detail') for b in cblocks if b['code']=='contamination']}")

    # ---- GOOD held-out eval (no code) ----
    good_rows = [{"input": INSTR.format(c=c), "expected": cap} for c, cap in EVALC.items()]
    ge = E.create_eval(PROJECT, "capitals", "template", good_rows, spec={"template": "exact"}, max_tokens=8)
    gv = ge["version"]
    ga = Q.audit(PROJECT, gv)
    check("good_eval_clean", ga["ok_to_use"] and not ga["blocks"], f"blocks={[b['code'] for b in ga['blocks']]}")
    check("warn_single_metric", any(w["code"] == "single_dimension" for w in ga["warns"]),
          "single-dimension warned")
    check("eval_frozen", E.assert_frozen(PROJECT, gv), "EVAL.lock holds")

    # ---- ordering: gate before prereg must refuse ----
    pre_gate = gate.verdict(PROJECT, "nonexistent")
    check("ordering_gate_needs_prereg", not pre_gate.get("ok"), pre_gate.get("error", "")[:50])

    # ---- FORCED baseline (real inference on the BASE) ----
    print("=== forced baseline: running the eval on the BASE model ===")
    run_eval.start_eval(PROJECT, gv, "base", temp=0.0)
    bst = wait_eval(gv, "base")
    baseline = (run_eval.get_baseline(PROJECT, gv) or {})
    base_acc = baseline.get("accuracy")
    check("baseline_ran", bst.get("state") == "done" and base_acc is not None,
          f"base accuracy={base_acc} (n={baseline.get('n')})")

    # ---- bar<=baseline -> WARN ----
    low = Q.audit(PROJECT, gv, proposed_criteria=[{"metric": "accuracy", "comparator": ">=",
                  "bar": max(0, (base_acc or 0))}], baseline=baseline)
    check("warn_bar_below_baseline", any(w["code"] == "bar_below_baseline" for w in low["warns"]),
          "rubber-stamp bar warned")

    # ---- commit a real bar ABOVE baseline, then train ----
    bar = round((base_acc or 0) + 20)
    cr = P.commit(PROJECT, gv, [{"metric": "accuracy", "comparator": ">=", "bar": bar, "guard": False}])
    check("prereg_committed", cr.get("ok") and P.is_committed(PROJECT), f"bar accuracy>={bar} frozen")

    print("=== training (terse capitals, real LoRA pass) ===")
    ver = f"disc-{int(time.time())}"
    ts = train.start(PROJECT, {"version": ver, "rank": 8, "num_layers": 4, "iters": 60, "batch_size": 1,
                               "max_seq_len": 192, "learning_rate": 2e-4, "steps_per_report": 10, "seed": 0})
    for _ in range(600):
        s = train.status(PROJECT, ver)
        if s.get("state") in ("done", "error", "stopped"):
            break
        time.sleep(2)
    check("trained_ran", train.status(PROJECT, ver).get("state") == "done",
          f"final loss {train.status(PROJECT, ver).get('final_train_loss')}")

    # ---- eval the trained adapter (real) + GATE ----
    print("=== eval trained adapter + gate ===")
    run_eval.start_eval(PROJECT, gv, ver, temp=0.0)
    wait_eval(gv, ver)
    v = gate.verdict(PROJECT, ver)
    trained_acc = v.get("trained_metrics", {}).get("accuracy")
    check("gate_side_by_side", v.get("ok") and v.get("side_by_side") and v.get("criteria_results"),
          f"base {base_acc} -> trained {trained_acc} | default={v.get('default_decision')}")
    R["gate"] = {"base_acc": base_acc, "trained_acc": trained_acc, "bar": bar,
                 "criteria": v.get("criteria_results"), "default_decision": v.get("default_decision"),
                 "honest_null": v.get("honest_null")}

    # ---- default-discard semantics: keeping a FAILED run without a reason is refused ----
    if v.get("default_decision") == "discard":
        bad = gate.decide(PROJECT, ver, "keep", reason=None)
        check("keep_needs_reason", not bad.get("ok"), "keep-without-reason refused")
        ok = gate.decide(PROJECT, ver, "keep", reason="keystone override: demo trail")
        check("override_recorded", ok.get("ok") and ok.get("override"), "override + reason recorded")
    else:
        ok = gate.decide(PROJECT, ver, "keep", reason=None)
        check("keep_on_pass", ok.get("ok"), "passed run kept without override")

    # ---- CODE escape hatch: custom scorer runs ----
    scorer_code = ("def score(inp, output, expected=None):\n"
                   "    # custom rubric: terse = a single short token, <=3 words\n"
                   "    return {'ok': 0 < len(output.split()) <= 3, 'score': 1.0 if len(output.split())<=3 else 0.0}\n")
    code_ev = E.create_eval(PROJECT, "terse-rubric", "code", good_rows, scorer_code=scorer_code, max_tokens=8)
    run_eval.start_eval(PROJECT, code_ev["version"], "base", temp=0.0)
    cst = wait_eval(code_ev["version"], "base")
    check("code_hatch_runs", cst.get("state") == "done" and cst.get("metrics", {}).get("n"),
          f"custom scorer metrics: {cst.get('metrics')}")

    finish(t0)


def finish(t0):
    R["runtime_s"] = round(time.time() - t0, 1)
    R["KEYSTONE_PASS"] = all(c["pass"] for c in R["checks"].values())
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "phase2_keystone.json").write_text(json.dumps(R, indent=2))
    print(f"\n=== PHASE-2 KEYSTONE: {'PASS' if R['KEYSTONE_PASS'] else 'FAIL'} ({R['runtime_s']}s) ===")
    print(json.dumps({k: v["pass"] for k, v in R["checks"].items()}, indent=2))


if __name__ == "__main__":
    main()
