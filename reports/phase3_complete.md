# MLX Master Trainer — Phase 3: Data Prep + Strict-Basis Filter — COMPLETE

```
=== MLX Master Trainer — Phase 3: Data Prep + Strict-Basis Filter ===
Filter:     acceptance-rule filter; REUSE eval detectors by default (filter == grade standard) + independent escape hatch
Reject report: bucketed by reason w/ counts + inspectable examples + recorded hand-rescue (the bro 78.4%-kept move, generalized)
Audits:     dedup · length/truncation · class balance · completeness · format consistency · diversity — plain-language summaries
Semantic-contam: explicit WARNING that the check is lexical; optional local char-3gram near-dup pass (heuristic, off by default)
Pipeline:   data -> eval -> baseline -> prereg -> train -> gate, ordered; frozen DATASET.lock; full raw->model provenance
KEYSTONE:   messy data filtered by reused eval standard -> trained -> graded by same standard -> PASS (14/14); residual gaps stated
ARC:        engine (P1) -> discipline (P2) -> data (P3) COMPLETE
Decisions mmt-013..mmt-017
```

## The move, productized
The thing reused on every model — Fin Nano, NPC-Reason, the bro — is "accept a training example only if it
clears a quality bar; reject the rest; know why." Nobody ships it as a first-class feature. Phase 3 does,
with the decided fork: **reuse-by-default** — the rules you FILTER training data against are the same
detectors you GRADE the model against, so the two standards can't drift. The bro's
`12,732 → 9,982 kept (78.4%)` rejection report is generalized to any project.

## KEYSTONE — messy data → reused standard → trained → graded (PASS, 14/14)
Real model runs on `SmolLM2-135M-Instruct` in a fresh `data-keystone` project (reports/phase3_keystone.json).
Messy data: 22 terse-good + 10 verbose-violating echo examples. Eval `terse` = a code scorer (output ≤ 3 words).

| step | result |
|---|---|
| raw set persisted for filtering | ✅ `raw.jsonl` |
| forced baseline on base | ✅ terse pass_rate **72.7** |
| **filter rejects the violators** | ✅ rejected exactly the **10** verbose (bucket `terse`), kept **22/32 (68.8%)** |
| bucketed reject report + examples | ✅ |
| frozen `DATASET.lock` | ✅ non-destructive (raw preserved) |
| hand-rescue a false rejection | ✅ recorded, kept count incremented |
| **filter == grade standard** | ✅ filter rule eval **== prereg eval** (`terse-1781743223`, the SAME artifact) |
| filtered set trains | ✅ loss 0.12 |
| gate by the same standard | ✅ terse pass_rate **72.7 → 95.5** vs bar **98** → default **DISCARD** (honest-null) |
| lexical-contamination warning | ✅ shown |
| optional local near-dup pass | ✅ char-3gram Jaccard, 9 flagged (heuristic) |
| dataset-level audit | ✅ 32 ex · 0 dup · lengths ok |
| **full raw→model provenance** | ✅ 9/9: raw → filter_report → filtered → DATASET.lock → eval_lock → baseline → prereg_lock → adapter → gate |

The same eval detector **filtered the data and graded the model** — and the honest-null held: 95.5 is a big
jump from 72.7, but it didn't clear the pre-registered bar of 98, so the tool defaulted to discard rather
than rounding up.

## Verified
- Engine (`core/filter.py`, `core/data_quality.py`): compile-clean; the 14/14 keystone above (real runs).
- Backend: 7 new endpoints (`/filter/*`, `/data/quality`, `/data/contamination-warning`, `/data/near-dup`) — **46 routes**.
- UI (same-origin preview, **0 console errors**): the Data tab renders the **Quality filter** card
  (reuse-eval vs independent · bucketed report `kept 23/32 · reuse:eval:terse · DATASET.lock frozen` ·
  hand-rescue), a dataset audit + near-dup button, and the standing lexical-contamination warning.

## Honest residual gaps (stated, not hidden)
Does data + eval-standard alignment improve outcomes? Yes — it removes the examples that violate your own
bar before they ever train, and keeps the filter and the grade from drifting. But:
- **The filter is only as good as the rules.** A bad rule filters in the wrong direction; the tool can't
  know your rule is the right one.
- **Lexical dedup misses paraphrase.** The contamination check + dedup catch exact/near-exact overlap, not
  semantic duplicates. The optional near-dup pass is a char-n-gram heuristic, not learned embeddings — it
  flags rewordings better than exact match but is **evidence, not proof**.
- **Representativeness and label-correctness remain unjudged** (carried from Phase 2). A clean, well-filtered
  set of the *wrong* or *mislabeled* examples still passes — no automated check fully covers this.
"Filtered" means **passes-your-rules + lexically-deduped, NOT semantically clean** — the tool says so loudly.

## ARC COMPLETE — engine → discipline → data
MLX Master Trainer is now a coherent, pure-local, honest-finetune studio:

```
DATA            filter your training data against a standard (reuse the eval detector) → frozen DATASET.lock
  → EVAL        define an eval; the tool BLOCKs/WARNs weak ones
  → BASELINE    forced before/after measurement on the base
  → PREREG      commit pass criteria, frozen (you can't reorder to peek)
  → TRAIN       memory-guarded MLX LoRA, live loss, versioned adapter
  → GATE        base-vs-trained vs the committed bar — default DISCARD
```
Pick any MLX base, on a 24GB Mac, data and models never leaving the machine. The wedge across all three
phases is the same: **it won't let you fool yourself** — it argues about eval quality, forces the baseline,
freezes the bar before training, defaults to discard, and tells you exactly what it still can't catch.
Decisions mmt-001..017.
