# MLX Master Trainer — Semantic-Contamination Upgrade — COMPLETE

```
=== MLX Master Trainer — Semantic-Contamination Upgrade ===
Tier:      n-gram (fast, default-on) + local-embedding pass (opt-in, deeper) — user picks rigor
Model:     all-MiniLM-L6-v2 via sentence-transformers, PURE-LOCAL (on-device, ~90MB once, offline after); no cloud API
One pass:  embeddings computed once -> 3 views (cosine over cached vectors of the INPUT text)
View 1:    train/eval semantic contamination — nearest-pair flagging, side-by-side REVIEW (not block), tunable threshold + distribution
View 2:    in-dataset semantic near-dup clusters — complements the n-gram pass, labeled by method
View 3:    semantic diversity — mean pairwise cosine + cluster count, plain-language warn
UI:        Data-tab opt-in toggle; lexical vs semantic findings labeled; pure-local note
KEYSTONE:  paraphrase the n-gram MISSES (0 flagged) -> embeddings CATCH (cosine 0.802, side-by-side) -> PASS (10/10)
Honesty:   stronger evidence, not proof — misses distant leakage, threshold/model-dependent (stated)
Decisions mmt-018..mmt-023
```

## Why this mattered
The Phase-2/3 contamination + near-dup checks were **lexical** (char-n-gram). A user whose eval is a
**paraphrase** of their training data ships a model believing an inflated score, and the tool never warns.
That's the failure most likely to burn a *paying* user. This adds a **semantic** tier — moving the claim
from "lexical, misses paraphrase" to "semantic, catches paraphrase, still not proof of clean."

## Decisions held
- **PURE-LOCAL, non-negotiable.** all-MiniLM-L6-v2 runs **on-device** (sentence-transformers, CPU/MPS).
  The weights download once (~90 MB) and run offline thereafter. **No cloud embedding API is ever called** —
  user text never leaves the Mac. (Pure-local is *the* product promise; breaking it for "easier" would break
  the one thing that makes the product sellable.)
- **Tiered.** The n-gram pass stays fast + default-on; embeddings are **opt-in** (no forced download for a
  quick run). Findings are labeled **LEXICAL** vs **SEMANTIC** so the user knows the method behind each.
- **Review, not block.** Lexical exact-match *blocks* (Phase 2); semantic similarity is fuzzy, so it
  *surfaces + explains + lets the human judge*. Blocking on a fuzzy threshold would be the wrong opinionated.
- **One pass, three views.** Embeddings are computed once (content-hash cached) and reused for contamination,
  near-dup, and diversity — all cosine ops. Memory-guarded: the pass runs in a detached subprocess so torch
  is freed on exit and never coexists with a loaded train model.

## KEYSTONE — the paraphrase the n-gram misses, the embeddings catch (PASS, 10/10)
`reports/semantic_keystone.json`. Planted a lexically-distant, semantically-identical pair:

| | |
|---|---|
| training (source) | `Add 3 and 5 together.` |
| eval (paraphrase) | `What do you get if you add five and three?` |
| **LEXICAL** n-gram near-dup @0.6 | **0 flagged — MISSED it** |
| **SEMANTIC** embedding contamination | **flagged at cosine 0.802**, nearest = the source, shown side-by-side — **CAUGHT it** |

All ten checks pass: model available · n-gram misses the paraphrase · embeddings catch it · side-by-side
review · three views from one pass · in-dataset near-dup found the planted cluster · diversity computed
(mean cosine 0.146) · thresholds tune (0.4 → 11 flags, 0.95 → 0) · **pure-local** (no network during
embedding). Runtime 4.1 s.

## Verified
- `core/semantic.py` + `core/run_semantic.py` compile; the 10/10 keystone (real local embeddings).
- Backend: 4 new endpoints (`/semantic/available|run|status|report`) — **50 routes**.
- UI (same-origin preview, **0 console errors**): the Data tab's opt-in **Deep semantic check** card ran the
  pass and rendered all three views (View 1 flagged 5 for review, View 2 one cluster, diversity), labeled
  SEMANTIC vs the LEXICAL n-gram, with the pure-local note.

## Honest residual (this strengthens the claim; it does not make it airtight)
- **Misses semantically-distant leakage** — same answer reached by very different framing won't be near in
  embedding space.
- **Threshold-sensitive** — too high misses looser paraphrase, too low floods false flags. The tool shows
  the distribution and makes the threshold tunable, but the user still picks.
- **Only as good as the embedding model** — MiniLM is small/fast, not state-of-the-art.
So: **semantic = stronger evidence, still not proof of clean.** Same posture as the firewall's
"evidence not proof" and the n-gram pass. The report says so rather than implying the data is now clean.

The honest-finetune studio now spans lexical **and** semantic contamination detection — both labeled,
both honest about their limits, both pure-local. Decisions mmt-001..023.
