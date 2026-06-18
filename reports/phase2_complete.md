# MLX Master Trainer — Phase 2: Discipline Layer — COMPLETE

```
=== MLX Master Trainer — Phase 2: Discipline Layer ===
Eval builder:  templates + example-based (no-code) + code escape hatch; versioned frozen eval artifact
Quality guard: BLOCKS (too-small / contamination / no-split / missing-expected)
               · WARNS (bar≤baseline / small-n / imbalance / single-metric / round-bar) — each with a plain-language why
Baseline:      FORCED before training; base-vs-trained side-by-side every run; same decoding recorded
Prereg:        committed criteria frozen BEFORE training, wired as an ordered, non-reorderable step
Gate:          regression gate defaults to DISCARD; honest-null framing; full reproducible trail
KEYSTONE:      bad evals caught (tiny / contamination / bar≤baseline / single-metric) + good path works
               (honest base 0.0 → trained 4.5 vs bar ≥20 → default DISCARD) + code hatch runs -> PASS (14/14)
Phase 3:       data prep + strict-basis filter as a feature (the signature move, productized)
Decisions mmt-007..mmt-012
```

## The wedge (why this is a product, not a wrapper)
Phase 1's engine is a commodity. Phase 2 is the part nobody else ships: **the tool argues with you about
eval quality and won't let a meaningless eval gate a model.** A discipline tool that launders overconfidence
is worse than none — so the guardrails *block* and *warn*, every finding teaches the *why*, the baseline is
*forced*, pre-registration is an *ordered* step you can't reorder, and the gate *defaults to discard*.

## KEYSTONE — can a stranger produce a MEANINGFUL eval? (PASS, 14/14)
Real model runs on `SmolLM2-135M-Instruct` in a fresh `discipline-demo` project (reports/phase2_keystone.json):

| what was tried | result |
|---|---|
| 5-example eval | ✅ **BLOCKED** (too_small) |
| eval example copied from training data | ✅ **BLOCKED** (contamination — the overlap is listed) |
| pass bar ≤ the base model's score | ✅ **WARNED** (bar_below_baseline — "rubber-stamping") |
| single-metric eval | ✅ **WARNED** (single_dimension — "training may break something else") |
| gate before pre-registering | ✅ **REFUSED** |
| commit criteria before measuring baseline | ✅ **REFUSED** |
| good path: template eval + held-out + baseline + committed bar + real train | ✅ base **0.0 → trained 4.5** vs bar **≥20** → default **DISCARD** + honest-null shown |
| keep the failed run without a reason | ✅ **REFUSED** (override needs a recorded reason) |
| code escape hatch (custom terse-rubric scorer) | ✅ **RAN** (n=22) |

**The honest-null is real, not staged.** The model *improved* (0→4.5) but didn't clear the pre-registered
bar of 20, so the tool defaulted to discard and refused to let a marginal gain look like success. That is
the entire value of the wedge, demonstrated.

## Verified
- **Guardrails (no model):** tiny→block, contamination→block (overlap listed), good held-out→clean, commit-before-baseline→refused.
- **Full discipline (real runs):** the keystone above — 14/14.
- **Backend:** `/eval/*`, `/pipeline`, `/prereg/*`, `/gate/*` added; compile-clean.
- **UI (same-origin preview, 0 console errors):** the **Eval** tab renders the ordered pipeline — the Audit
  card shows the `small_n` + `single_dimension` WARNs with their plain-language explanations, baseline ✓,
  prereg ✓; the **Train** tab shows the frozen banner `accuracy ≥ 20 · baseline 0` and a post-train gate with
  default-discard keep/discard controls.

## Honest residual gaps (what the guardrails still CAN'T catch)
The discipline makes a naive eval *meaningfully better*, but it is not a substitute for judgment:
- **Representativeness.** The tool checks size, contamination, balance, and bar-vs-baseline — it does **not**
  know whether your 22 held-out examples represent the real distribution. A clean, well-sized eval of the
  *wrong* inputs still passes the audit. (No automated check can fully judge this.)
- **Example-based scoring is only as good as the examples** + it's lexical (difflib) — it rewards surface
  similarity, not meaning. Two correct answers phrased differently can score apart. A local-judge option
  would help but stays **off by default** to keep pure-local; that tradeoff is the user's.
- **Label correctness.** The tool trusts your `expected` values — a mislabeled eval audits clean.
- **Metric choice.** `single_dimension` *nudges* toward a guard metric but can't know which capability your
  training might silently break; the user still has to pick the right guard.
- **Contamination is exact-normalized** (lowercase/whitespace) — paraphrased leakage (same question reworded)
  slips through. Semantic dedup is future work.

Net: the wedge defeats the **common, cheap** ways a naive eval is secretly useless (too small, contaminated,
rubber-stamped, single-axis, goalpost-moved) and forces the before/after + pre-registration discipline. It
does **not** turn a non-expert into an eval expert — and it says so rather than pretending otherwise.

## Phase 3 (gated, ready)
Data prep + the strict-basis filter **productized** — the signature move from the bro work (filter examples
against your own quality detectors before they ever train), generalized into a feature.
