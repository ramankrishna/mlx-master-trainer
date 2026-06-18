#!/usr/bin/env python
"""Semantic-upgrade KEYSTONE — prove the embedding tier catches the paraphrase the n-gram MISSES.

Plants a reworded eval/train duplicate (lexically distant, semantically the same) and confirms:
  - the LEXICAL n-gram near-dup pass does NOT flag it (the gap),
  - the SEMANTIC embedding contamination view DOES flag it, side-by-side for review,
  - three views come from ONE embedding pass, thresholds tune, and it's pure-local.
Result -> reports/semantic_keystone.json (+ PASS/FAIL).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core import data, data_quality, eval as E, models, projects, semantic   # noqa: E402

BASE = "HuggingFaceTB/SmolLM2-135M-Instruct"
PROJECT = "semantic-keystone"

# the PLANTED paraphrase (lexically distant — digits vs words — semantically identical = 3 + 5)
SRC = "Add 3 and 5 together."                              # training example (source)
PARA = "What do you get if you add five and three?"        # eval example (paraphrase)

TRAIN_FILLERS = [
    "Define photosynthesis in one sentence.", "List three primary colors.", "Translate 'hello' into Spanish.",
    "Write a haiku about winter.", "Explain what a noun is.", "Convert 10 kilometers to miles.",
    "Name a planet in our solar system.", "Describe the water cycle briefly.", "Give an example of a mammal.",
    "Spell the word 'necessary'.", "What language is spoken in Brazil?", "Name the largest ocean on Earth.",
    "What is the chemical symbol for gold?", "List two renewable energy sources.", "Name a Renaissance painter.",
    "Explain the term supply and demand.", "What gas do plants absorb from the air?", "Describe a triangle.",
    "Name a string instrument.", "What is the speed of light, roughly?",
    # planted in-dataset near-dup pair (France capital, two phrasings)
    "Name the capital city of France.", "Which city is the capital of France?",
]
EVAL_FILLERS = [
    "Who wrote the play Romeo and Juliet?", "What is the freezing point of water in Fahrenheit?",
    "Name a reptile.", "Translate 'thank you' into French.", "List two types of clouds.",
    "What is the tallest mountain on Earth?", "Define velocity in physics.", "Name the longest river in the world.",
    "What is the capital of Japan?", "Give an example of a prime number.", "What organ pumps blood?",
    "Spell the word 'rhythm'.", "What currency is used in the United Kingdom?", "Name a wind instrument.",
    "Explain what gravity does.", "How many continents are there?", "Name the largest planet.",
    "Describe the role of a verb.", "Name a source of vitamin C.", "Which season comes after spring?",
    "What colour is a clear daytime sky?",
]

R = {"base": BASE, "project": PROJECT, "planted": {"source": SRC, "paraphrase": PARA}, "checks": {}}


def check(name, cond, detail=""):
    R["checks"][name] = {"pass": bool(cond), "detail": str(detail)[:160]}
    print(f"[{'PASS' if cond else 'FAIL'}] {name}: {detail}")
    return cond


def main():
    t0 = time.time()
    check("model_available", semantic.available(), f"local embed model {semantic.MODEL_NAME}")
    path = models.ensure_local(BASE)
    projects.create_project(PROJECT)
    projects.set_base(PROJECT, {"ref": BASE, "local_path": str(path), "info": models.model_info(str(path))})

    rows = [{"instruction": q, "output": "ok"} for q in TRAIN_FILLERS] + [{"instruction": SRC, "output": "8"}]
    det = data.detect_schema(rows)
    data.prepare_dataset(rows, det["schema"], det["mapping"], str(path),
                         str(projects.project_dir(PROJECT) / "data"), val_frac=0.1, seed=0)
    ev = E.create_eval(PROJECT, "carrier", "code",
                       [{"input": q, "expected": "x"} for q in EVAL_FILLERS] + [{"input": PARA, "expected": "8"}],
                       scorer_code="def score(inp,output,expected=None):\n    return {'ok': True, 'score': 1.0}\n",
                       max_tokens=16)
    EV = ev["version"]

    # ---- LEXICAL n-gram pass (the incumbent) ----
    ng = data_quality.near_dup_pass(PROJECT, EV, threshold=0.6)
    ng_flagged_inputs = {f["eval_input"] for f in ng.get("flagged", [])}
    ng_catches = any(PARA[:60] in fi or fi in PARA for fi in ng_flagged_inputs)
    check("ngram_misses_paraphrase", not ng_catches, f"n-gram flagged {ng.get('n_flagged')} (paraphrase among them: {ng_catches})")

    # ---- SEMANTIC embedding pass (the upgrade) ----
    rep = semantic.analyze(PROJECT, EV, contam_threshold=0.5, dup_threshold=0.8)
    check("analyze_ran", rep.get("ok"), rep.get("method"))
    flagged = rep.get("contamination", {}).get("flagged", [])
    planted = next((f for f in flagged if "add" in f["eval_input"].lower() and "five" in f["eval_input"].lower()), None)
    check("embeddings_catch_paraphrase", planted is not None,
          f"planted paraphrase flagged at sim {planted['similarity'] if planted else 'NONE'}")
    check("side_by_side", bool(planted and planted.get("nearest_train") and planted.get("similarity")),
          f"{planted['eval_input'] if planted else ''} ~ {planted['nearest_train'] if planted else ''}" if planted else "")
    R["planted"]["semantic_similarity"] = planted["similarity"] if planted else None
    R["planted"]["nearest_train"] = planted["nearest_train"] if planted else None

    check("three_views_one_pass", all(k in rep for k in ("contamination", "near_duplicates", "diversity")),
          "contamination + near_duplicates + diversity from one embed pass")
    check("near_dup_view", rep["near_duplicates"]["n_clusters"] >= 1,
          f"{rep['near_duplicates']['n_clusters']} semantic near-dup cluster(s) in training")
    check("diversity_view", "mean_pairwise_cosine" in rep["diversity"],
          f"mean pairwise cosine {rep['diversity']['mean_pairwise_cosine']} · ~{rep['diversity']['approx_clusters']} clusters")

    # thresholds tune (embeddings cached → fast 2nd pass)
    lo = len(semantic.analyze(PROJECT, EV, contam_threshold=0.4)["contamination"]["flagged"])
    hi = len(semantic.analyze(PROJECT, EV, contam_threshold=0.95)["contamination"]["flagged"])
    check("thresholds_tune", lo >= hi and lo != hi, f"flags at 0.4 = {lo}, at 0.95 = {hi}")
    check("pure_local", rep.get("pure_local") is True, "embedding model runs on-device; no cloud, no text egress")

    R["ngram"] = {"n_flagged": ng.get("n_flagged"), "method": ng.get("method")}
    R["semantic"] = {"n_flagged": rep["contamination"]["n_flagged"], "distribution": rep["contamination"]["distribution"]}
    finish(t0)


def finish(t0):
    R["runtime_s"] = round(time.time() - t0, 1)
    R["KEYSTONE_PASS"] = all(c["pass"] for c in R["checks"].values())
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "semantic_keystone.json").write_text(json.dumps(R, indent=2))
    print(f"\n=== SEMANTIC KEYSTONE: {'PASS' if R['KEYSTONE_PASS'] else 'FAIL'} ({R['runtime_s']}s) ===")
    print(json.dumps({k: v["pass"] for k, v in R["checks"].items()}, indent=2))


if __name__ == "__main__":
    main()
