# MLX Master Trainer — decisions log (prefix mmt-)

Phase 1 of three (engine → discipline → data). Phase 1 builds the model-agnostic MLX LoRA SFT
ENGINE by generalizing the BigBugAI Bro studio's retrain panel off the bro onto any base model.
It ships standalone as a working local MLX LoRA GUI. The product wedge (eval-first / freeze /
pre-register / regression-gate DISCIPLINE) is Phase 2 — the engine has to exist first.

## mmt-001 — scaffold + extract the engine + standalone venv
New project ~/mlx-master-trainer (core/ backend/ frontend/ projects/ desktop/ reports/). Reused the
bro studio's proven shell shape: Tauri v2 + FastAPI backend + build-free HTML frontend + the
memory-guarded MLX-LoRA-subprocess + versioned-adapter + loss-parsing patterns (provenance noted in
every file header). Standalone venv (uv) with mlx / mlx-lm 0.31.3 / huggingface_hub / fastapi /
uvicorn / transformers — pure-local, no servers, no cloud. The bro studio was NOT modified.

## mmt-002 — core/models.py: model-agnostic base (generalization #1)
The bro assumed one base. Now: pick ANY MLX base by HF id (mlx-community/* or a convertible HF
model) or local path. ensure_local() snapshot_downloads + caches; convert_to_mlx() runs mlx_lm.convert
(optional quantize). model_info() reads config + tokenizer config WITHOUT loading weights → param
count (from safetensors bytes / bytes-per-param), quantization, FAMILY (model_type), and whether a
chat TEMPLATE is present + per-family LoRA target suggestions. chat_template_render() renders sample
messages through the base's own template (transformers tokenizer, no weights) so the user sees what
mlx_lm will feed the model. memory_precheck() estimates peak (weights + activations + optimizer +
overhead) vs the machine's RAM and WARNS before an OOM on 24GB, with concrete advice (lower seq/batch,
grad-checkpoint, 4-bit base, fewer layers). 8 curated suggested bases (SmolLM2 → Mistral-7B-4bit).

## mmt-003 — core/data.py: multi-schema ingest (generalization #2)
The bro assumed pre-formatted in-voice {prompt,response} JSONL. Now: load_rows() takes JSONL/CSV or
pasted text; detect_schema() auto-detects chat `messages` / `prompt`-`completion` / `instruction`
(+input)/`output` / raw `text` and returns a concrete column mapping the user can confirm. Every
schema normalizes to chat messages; prepare_dataset() writes the format mlx_lm expects — {"messages"}
when the base HAS a chat template (mlx_lm then applies the base's OWN template — the templating is
provably correct, not guessed), else {"prompt","completion"}, else {"text"}. validate() reports empty
fields, char + TOKEN length distribution (templated through the base), a token histogram, how many
exceed max_seq_len (→ truncation), and the train/val split. preview() shows the rendered, post-template
example — exactly the string the model trains on (kills the silent-wrong-template bug). PURE-LOCAL:
every function reads the filesystem only; no network call touches user data.

## mmt-004 — core/train.py + run_train.py: guarded LoRA engine (generalization #3)
Extracted the bro's mlx_lm.lora-as-subprocess + versioning + loss-parsing and generalized. FULL config
via a generated mlx_lm YAML (-c config.yaml): rank / scale(alpha) / dropout / target-modules / fine-
tune-type (lora|dora) / num-layers / lr / iters / batch / max-seq-len / seed / grad-checkpoint. MEMORY
GUARD: training ALWAYS runs in a detached subprocess (run_train.py) — peak reclaimed on exit, the
backend never holds the optimizer — plus the pre-check warns before OOM. LIVE loss / val / tokens-sec /
peak-mem / ETA streamed via status.json (the UI polls it). Versioned, reproducible adapters: each run
records config + seed + data_hash + base + final loss + provenance in manifest.json and NEVER overwrites
a prior adapter (refuses a name collision unless resume). start()/status()/stop() + resume supported.

## mmt-005 — core/adapters.py: manage + export + quick test (generalization #4)
list_adapters() surfaces every versioned adapter with its full provenance (base/rank/scale/iters/final
loss/data hash/runtime/state). fuse() runs mlx_lm.fuse. export() → GGUF (mlx_lm.fuse --export-gguf, for
llama/mistral-family archs), fused MLX, or adapter-only. quick_infer() loads base+adapter in a subprocess
(memory reclaimed after) and generates once, chat-templated — a sanity check before exporting. NOT a
full eval (that discipline is Phase 2).

## mmt-006 — backend + frontend + Tauri app + KEYSTONE
backend/server.py: FastAPI shell over the engine (models/data/train/adapters/export/infer/settings),
127.0.0.1:8808, no-store index, /assets mount, slow ops in background threads with status polling.
frontend/index.html: build-free Unsloth-style UI (reused the bro's design tokens) — Model (pick/prepare/
template-preview/memory) · Data (paste/inspect with templated preview + token histogram) · Train (full
config + live loss SVG + ETA) · Adapters (versioned list + export + quick test) · Settings (HF token).
desktop/src-tauri: Tauri v2 menu-bar app (Accessory policy, tray toggle, spawns backend, navigates the
webview to 127.0.0.1:8808 so fetches run same-origin) — MMT_ROOT/MMT_PYTHON overridable, identifier
com.falconhash.mlx-master-trainer. cargo build OK; launch test passed (app boots → spawns its own backend
→ /health responds → no panic). Icon is a PLACEHOLDER (bro mascot) — MMT branding is a later phase.
Build note: the debug binary compiled x86_64 (rust host triple default → runs via Rosetta); the Tauri
shell only does IPC + webview and spawns the NATIVE arm64 Python venv (where MLX runs), so it's
functionally fine — a clean native build is `cargo build --target aarch64-apple-darwin`.

KEYSTONE (reports/keystone_result.json) — the WHOLE loop on a NON-bro base with unformatted data:
base HuggingFaceTB/SmolLM2-135M-Instruct (Llama-family, NOT Qwen-bro) · data alpaca-style
instruction/input/output (a DIFFERENT schema, NOT bro voice). All 8 checks PASS: base resolves +
info detected (135M/llama/template) · memory fits (est 2.7GB/24GB) · schema auto-detects as
'instruction' · the RIGHT template applies (rendered preview shows SmolLM2 <|im_start|> ChatML,
format=messages) · LoRA trains + loss drops 3.322 → 0.452 (8.8s) · GGUF exports · inference test
produces output. The engine works on a model we didn't build — Phase 1 is proven.

PHASE 2 GATE = ready: the discipline layer (eval-first, freeze, pre-register, regression-gate — the
"won't let you fool yourself" wedge) sits on this engine.

# ───────────────────────── Phase 2: the discipline layer ─────────────────────────
Phase 2 of three (engine → DISCIPLINE → data). The wedge nobody else ships: the tool ARGUES with you
about eval quality. "The fine-tuning tool that won't let you fool yourself." Pure-local; builds on Phase 1.

## mmt-007 — Phase-2 scaffolded; the wedge framed
New modules: core/eval.py (eval definition + scoring), core/eval_quality.py (the opinionated guardrails —
the wedge), core/run_eval.py (forced baseline + detached runner), core/prereg.py (freeze + commit criteria),
core/gate.py (regression gate). Frontend: an Eval tab + a Pre-register step wired INTO the train flow.
Decided + held throughout: the tool is OPINIONATED (warns AND blocks), baseline is FORCED (base-vs-trained
every time), eval creation is layered (templates + examples no-code AND a code escape hatch). The real
keystone is harder than Phase 1's: prove the discipline works for a stranger's JUDGMENT — that the tool
CATCHES a technically-valid-but-meaningless eval instead of laundering overconfidence.

## mmt-008 — core/eval.py: the no-code builder + code escape hatch
Three ways to define an eval, one internal object: (A) templates exact/contains/classification/format_json/
refusal/numeric (each generates its scorer, no code); (B) example-based (label good/bad outputs; lexical
similarity scorer; pure-local; a local judge is opt-in/off by default); (C) a Python scorer(inp,output,
expected) escape hatch for anything templates can't say. Whatever the path, the eval is a FROZEN, versioned
artifact (EVAL.lock generalized: eval.json + eval.lock hash). The scorer is reconstructable from (kind,spec)
so the detached runner can score without extra state. metrics_available tells prereg/gate what is gateable.

## mmt-009 — core/eval_quality.py: THE WEDGE (opinionated guardrails)
Before an eval can gate a model the tool audits the eval ITSELF. BLOCK: too_small (<20 = noise),
train/eval contamination (an eval example is also in training — the cardinal sin; lists the overlaps),
no_held_out, missing_expected. WARN: bar≤baseline (rubber-stamping), small_n (20-50 = wide error bars),
class_imbalance (accuracy misleading), single_dimension (training may break something else — suggests a
guard metric), round_bar (a guess?). Every finding explains WHY in plain language — it teaches the judgment,
not just flags. Stance: it would rather annoy you than let you ship a result that fooled you.

## mmt-010 — core/run_eval.py: forced baseline + detached runner
Runs a frozen eval against a target (the BASE for the forced baseline, or a trained adapter) in an isolated
subprocess: loads the model ONCE, generates per input (chat-templated, SAME decoding base vs trained —
apples-to-apples), scores via the eval's own scorer, writes live status + a results record. Baseline is NOT
optional — "you cannot claim improvement you didn't measure against a baseline" — and it feeds the
bar≤baseline check. start_eval/eval_status/get_baseline = the API the backend calls.

## mmt-011 — core/prereg.py: pre-registration wired as an ORDERED step
The discipline is the ORDER: define → audit (no blocks) → baseline → commit criteria → train. commit()
REFUSES unless the eval is frozen, its audit has no blocks, and a baseline exists — so you can't reorder to
peek at results then set the bar. Committed criteria are frozen before training (PREREG.lock generalized:
prereg.json + prereg.lock hash). pipeline_state() drives the UI so steps unlock in order and can't be skipped.

## mmt-012 — core/gate.py (default discard) + backend + frontend + KEYSTONE
gate.verdict() grades a trained adapter vs the PRE-REGISTERED criteria: base-vs-trained side-by-side every
metric, PASS/FAIL per committed bar, guard-metric regression detection. DEFAULTS TO DISCARD — if any
criterion failed or a guard regressed, keeping requires an explicit override + a recorded reason (gate.decide
refuses keep-without-reason). Honest-null framing ("training didn't help — that's information, you avoided
shipping something that didn't earn it"), full reproducible trail in gate.json. Backend: /eval/*, /pipeline,
/prereg/*, /gate/* endpoints. Frontend: an Eval tab walking define→audit→baseline→commit (steps unlock in
order, audit blocks/warns shown with plain-language why) + the Train tab gated on a committed prereg,
showing the frozen criteria + a post-train base-vs-trained gate with keep/discard (default discard).

KEYSTONE (reports/phase2_complete.md, phase2_keystone.json) — can a stranger produce a MEANINGFUL eval?
PASS (14/14), real model runs on SmolLM2-135M in the discipline-demo project:
  BAD evals caught: tiny(5)→BLOCK · contamination→BLOCK (overlap listed) · bar≤baseline→WARN · single-metric→WARN
  ordering enforced: gate-before-prereg refused · commit-before-baseline refused
  GOOD path (honest): base accuracy 0.0 → trained 4.5 vs committed bar ≥20 → default DISCARD, honest-null shown;
    keeping the failed run WITHOUT a reason refused; override WITH a reason recorded.
  code escape hatch: a custom terse-rubric scorer ran (n=22).
The honest-null outcome is REAL, not staged — the model improved but didn't clear the pre-registered bar, and
the tool correctly refused to let a marginal gain masquerade as success. That is the wedge working.
PHASE 3 GATE = ready: data prep + the strict-basis filter productized (the signature move).

# ───────────────────────── Phase 3: data prep + the strict-basis filter ─────────────────────────
Phase 3 of three (engine → discipline → DATA). The final layer. Productizes the move reused on EVERY
model (Fin Nano, NPC-Reason, the bro): accept a training example ONLY if it clears a quality bar, reject
the rest, know WHY each was rejected. Completes the honest-finetune arc — a great eval can't save you
from garbage training data.

## mmt-013 — Phase-3 scaffolded; raw set persisted for filtering
New modules: core/filter.py (acceptance-rule filter), core/data_quality.py (dataset-level audits +
the honest contamination story). data.prepare_dataset now also writes data/raw.jsonl — the FULL pre-filter
normalized set ({input, output, record}) the filter operates on (train/valid = "everything" until filtered).
THE FORK (decided): REUSE-BY-DEFAULT — the filter's rules reuse the Phase-2 eval DETECTORS, so the standard
you FILTER training data against is the SAME standard you GRADE the model against (the bro pattern: the
call-detector was both the data filter and the eval check). An INDEPENDENT escape hatch covers dimensions
you filter on but don't eval on. Reuse keeps the two standards from drifting.

## mmt-014 — core/filter.py: the acceptance-rule filter (the productized strict-basis move)
A training example is KEPT only if it clears every required rule. Each rule resolves to a scorer:
  - source=eval → load the Phase-2 eval's scorer (filter == grade standard);
  - source=template → the same template scorers (contains/format_json/refusal) + filter-only length/nonempty;
  - source=code → a user scorer.
Each rule has a polarity (keep_if 'pass' = keep iff the detector passes, e.g. claims carry basis / valid
JSON; 'fail' = keep iff it does NOT, e.g. reject confident-call language). apply() buckets rejections by
reason with counts + inspectable examples (the bro "12,732 → 9,982 kept (78.4%)" report, generalized) and
writes a frozen, versioned DATASET.lock (raw.jsonl preserved — non-destructive + reproducible) that the
Phase-1 trainer consumes. rescue() hand-rescues a false rejection with a recorded reason.

## mmt-015 — core/data_quality.py: dataset-level audits (the SET, not each row)
audit() reports exact duplication, length/TOKEN distribution + truncation flags (templated through the
base tokenizer), class balance (if outputs look label-like), field completeness, format consistency
(mixed schemas), and diversity (input-phrasing repetition — "78% one phrasing pattern → overfit risk").
Each finding explains WHY in plain language (the Phase-2 teach-don't-just-flag stance, applied to data).

## mmt-016 — the honest contamination story (semantic gap made explicit)
Phase 3 touches the training set, so paraphrase-leak risk peaks here. lexical_warning() is a STANDING
warning: "the contamination check is LEXICAL — it catches exact/near-exact overlap, NOT paraphrased
duplicates; 'filtered' = passes-your-rules + lexically-deduped, NOT semantically clean." near_dup_pass()
is an OPTIONAL, off-by-default, pure-local heuristic (char-3gram Jaccard between eval & train inputs) that
catches rewordings better than exact match — explicitly labeled a heuristic, NOT learned embeddings (true
embeddings would need an optional local model, not bundled): evidence, not proof. Same honesty as the
firewall's "0% is evidence, not proof."

## mmt-017 — pipeline wired (data first) + backend + frontend
The honest-finetune loop, ordered: data (filter + audit) → eval (define + audit) → baseline → prereg
(commit) → train → gate (base-vs-trained, default discard). The Data step produces the frozen
DATASET.lock; the reused-eval link means the filter standard and the gate standard are the SAME artifact.
Backend: /filter/apply, /filter/report, /filter/rejected, /filter/rescue, /data/quality,
/data/contamination-warning, /data/near-dup (46 routes total). Frontend: a Data-tab Quality-filter section
(reuse-eval vs independent rule · bucketed rejection report · hand-rescue · DATASET.lock frozen), a dataset
audit + optional near-dup button, and the standing lexical-contamination warning. Verified same-origin in
the preview, 0 console errors.

KEYSTONE (reports/phase3_complete.md, phase3_keystone.json) — messy data → reused standard → trained →
graded by the SAME standard. PASS (14/14), real SmolLM2-135M runs in the data-keystone project:
  messy data (22 terse-good + 10 verbose-violating) → eval "terse" (code scorer) → REUSE it as the filter
  → rejected exactly the 10 verbose (bucket 'terse', kept 22/32 = 68.8%) → DATASET.lock frozen → hand-rescue
  works → SAME eval (terse-1781743223) is BOTH the filter rule AND the prereg/gate standard (verified equal)
  → filtered set trains (loss 0.12) → gate terse pass_rate base 72.7 → trained 95.5 vs bar 98 → default
  DISCARD (honest-null: improved but didn't clear the committed bar) → lexical-contam warning shown →
  optional near-dup pass ran (char-3gram, 9 flagged) → full raw→model provenance trail intact (9/9 artifacts).
Honest residual gaps (reports/phase3_complete.md): the filter is only as good as the rules; lexical dedup
misses paraphrase; representativeness + label-correctness remain unjudged — stated, not hidden.

ARC COMPLETE: engine (P1) → discipline (P2) → data (P3). MLX Master Trainer is a coherent, pure-local,
honest-finetune studio: pick any base, filter your data against a standard, pre-register, train, and be
graded against the SAME standard with the safe default of discard. Decisions mmt-001..017.

# ───────────────────────── Enhancement: semantic-contamination tier ─────────────────────────
Post-arc enhancement (not a new phase). Closes the gap flagged in P2/P3 most likely to burn a PAYING user:
the contamination/near-dup checks were LEXICAL (n-gram), so a PARAPHRASED eval slipped through and inflated
the score undetected. Adds a SEMANTIC tier via a LOCAL embedding model. Moves the claim from "lexical,
misses paraphrase" to "semantic, catches paraphrase — still evidence, not proof."

## mmt-018 — local embedding model setup (pure-local, opt-in)
Installed sentence-transformers (+torch 2.12.1, MPS) into the standalone venv; all-MiniLM-L6-v2 (384-dim,
~90MB) downloads ONCE then runs offline. NON-NEGOTIABLE PURE-LOCAL: the model runs on-device; NO cloud
embedding API is ever called; only the model weights are fetched once (like a base model) — user TEXT never
leaves the Mac. TIERED: the n-gram pass stays fast + default-on; the embedding pass is OPT-IN (no forced
download for a quick run). Sanity: "add five to three" vs "what is 3 plus 5" cosine 0.727 (n-gram ≈ 0).

## mmt-019 — core/semantic.py: one embedding pass, memory-guarded
Lazy model load (only on opt-in). embed() L2-normalizes + a content-hash disk cache (.embcache.pkl) so
re-runs don't recompute. The pass embeds train + eval INPUT text ONCE (the question — where leakage matters);
all three views are cosine ops over those cached vectors. Memory guard: run_semantic.py runs the pass in a
DETACHED subprocess so torch loads there and is freed on exit — the backend never holds it (same pattern as
the train/eval runners). start()/status()/report() = the backend API.

## mmt-020 — View 1: train/eval semantic contamination (REVIEW, not block)
For each eval input, nearest training input by cosine; flag pairs ≥ threshold (excluding exact, already
lexical). REVIEW-NOT-BLOCK: semantic similarity is fuzzy (a high score can be two genuinely different items
that read alike, or a real paraphrase leak) — blocking on a fuzzy threshold would be the wrong opinionated.
Side-by-side pairs + a nearest-sim distribution + tunable threshold with plain-language guidance.

## mmt-021 — View 2: in-dataset semantic near-dup clusters
Greedy cosine clustering of the training inputs at a dup threshold → clusters of paraphrased examples that
over-weight one pattern. Complements (does not replace) the n-gram near-dup: n-gram catches surface dups
fast, embeddings catch paraphrase dups when opted in — both reported, labeled by method.

## mmt-022 — View 3: semantic diversity
Mean pairwise cosine over the same vectors + an approximate cluster count → "spread across meaning-space, or
clustered in a few regions?" WARN/inform, not block (a narrow task SHOULD be narrow). Generalizes the P3
lexical-diversity note to semantic.

## mmt-023 — tiered Data-tab UI + backend + KEYSTONE
Backend: /semantic/available, /semantic/run, /semantic/status, /semantic/report (50 routes total). UI: a
Data-tab "Deep semantic check (local embeddings)" opt-in card (n-gram stays the fast default), tunable
threshold, the three views rendered + labeled SEMANTIC vs the LEXICAL n-gram, and a visible pure-local note.
Verified same-origin in the preview (0 console errors): the carrier eval ran the pass, View 1 flagged 5
(review), View 2 found 1 cluster, diversity shown.

KEYSTONE (reports/semantic_upgrade_complete.md, semantic_keystone.json) — PASS (10/10). Planted a paraphrase
the n-gram MISSES: train "Add 3 and 5 together." vs eval "What do you get if you add five and three?"
(digits-vs-words = lexically distant, semantically identical). LEXICAL n-gram near-dup (0.6): 0 flagged —
MISSED it. SEMANTIC embedding contamination: flagged at cosine 0.802, nearest = the source, shown
side-by-side — CAUGHT it. Three views from one pass; thresholds tune (0.4→11 flags, 0.95→0); near-dup found
the planted France-capital cluster; diversity mean-cosine 0.146; pure-local (no network during embedding).
Honest residual (stated): semantic catches paraphrase but still MISSES semantically-distant leakage (same
answer, very different framing), is threshold-sensitive, and is only as good as the embedding model —
evidence, not proof. Same posture as the firewall + the n-gram pass. Decisions mmt-001..023.
