# Using MLX Master Trainer

A hands-on guide to the whole loop — picking a base, bringing your data, building an eval the
tool will *argue with*, training a LoRA adapter, and gating whether it earned keeping. Everything
runs on your Mac; your data and models never leave it.

If you just want the elevator pitch and install steps, see the [README](README.md). This document is
the "how do I actually drive it" companion.

## Contents
- [Requirements](#requirements)
- [Install & launch](#install--launch)
- [The loop in one screen](#the-loop-in-one-screen)
- [Walkthrough](#walkthrough)
  - [1. Pick a base (Model)](#1-pick-a-base-model)
  - [2. Bring your data (Data)](#2-bring-your-data-data)
  - [3. Define an eval (Eval)](#3-define-an-eval-eval)
  - [4. Measure the baseline](#4-measure-the-baseline)
  - [5. Pre-register the bar](#5-pre-register-the-bar)
  - [6. Train (Train)](#6-train-train)
  - [7. Gate the result](#7-gate-the-result)
  - [8. Export & use (Adapters)](#8-export--use-adapters)
- [Reference](#reference)
  - [Accepted data formats](#accepted-data-formats)
  - [Eval kinds](#eval-kinds)
  - [The eval audit (blocks & warnings)](#the-eval-audit-blocks--warnings)
  - [Contamination & dedup tiers](#contamination--dedup-tiers)
  - [Filter rules](#filter-rules)
  - [Training config](#training-config)
  - [Export formats](#export-formats)
  - [Settings (HuggingFace token)](#settings-huggingface-token)
- [Where things live on disk](#where-things-live-on-disk)
- [Troubleshooting](#troubleshooting)
- [Pure-local & honest limits](#pure-local--honest-limits)

---

## Requirements
- **Apple Silicon Mac** (M1 or later). MLX is Apple-Silicon-only; there is no Intel/CUDA path.
- **macOS** with internet for the *first* fetch of a base model (and, optionally, the ~90 MB
  embedding model used by the semantic check). Everything else is offline.
- **[`uv`](https://docs.astral.sh/uv/)** for the Python environment. The desktop app (option B below)
  additionally needs the **Rust toolchain** and **Xcode Command Line Tools**; the web UI does not.
- Enough free RAM for the base you pick. The app estimates peak memory before you train and warns if
  it won't fit — small bases (135M–1.7B) are comfortable on 16 GB; 7B at 4-bit wants more headroom.

## Install & launch
The app is the **MIT-licensed core** in this repo. There's no signed/notarized `.dmg` download yet,
so the supported path today is **build/run from source**. The commands assume `uv` — no manual venv
activation is needed because `run.sh` calls `.venv/bin/python` directly.

```bash
git clone https://github.com/ramankrishna/mlx-master-trainer
cd mlx-master-trainer
uv venv
uv pip install mlx mlx-lm huggingface_hub fastapi "uvicorn[standard]"
# optional — only if you want the semantic-contamination tier (pulls a ~90 MB embedding model later):
uv pip install sentence-transformers

# option A — web UI (open http://127.0.0.1:8808)
./run.sh

# option B — the desktop (menu-bar) app. Needs Rust + Xcode CLT:
#   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh   # if you don't have rustup
cd desktop/src-tauri && cargo run
```

Both serve the **same** UI: the desktop app is a thin native shell whose window points at the local
backend on `127.0.0.1:8808`. Pick whichever you prefer; the workflow below is identical.

> If an unsigned `.dmg` is later attached to a release, Gatekeeper will block the first open. Clear it
> once with `xattr -cr "/Applications/MLX Master Trainer.app"` (or right-click → Open → Open). It's
> blocked only because it's unsigned — the source is right here to audit.

**First run downloads on demand:** the base model when you pick one, and the embedding model only if
you opt into the semantic check. After those one-time fetches, it's fully offline.

## The loop in one screen
The tabs are meant to be used **left to right** — each step unlocks the next, and the tool refuses to
let you skip the parts that keep you honest:

```
MODEL     pick any MLX base → convert/quantize if needed → inspect chat template → memory check
  ↓
DATA      paste/point at your dataset → auto-detect schema → see the templated example → audit quality
          (filtering by an eval's detector and the semantic check happen after EVAL — see below)
  ↓
EVAL      define an eval (no-code template / examples / code) → the tool AUDITS it and
          BLOCKS meaningless ones, WARNS weak ones
  ↓
BASELINE  run the eval on the BASE model first — no "it improved" without a before
  ↓
PREREG    commit your pass criteria, FROZEN, before training (no moving the goalpost)
  ↓
TRAIN     memory-guarded MLX LoRA, live loss, a versioned reproducible adapter
  ↓
GATE      base vs trained against the committed bar → defaults to DISCARD unless it earned keeping
  ↓
ADAPTERS  export adapter / fused / GGUF, or quick-test the adapter inline
```

A **project** ties all of this together — one base, one dataset, its evals, its adapters, and the
frozen locks. Create or switch projects from the selector at the top.

> **Ordering note:** two Data-tab options — *filtering by reusing an eval's detector* and the
> *semantic-contamination check* — need an eval to exist first. If you want those, define the eval
> (step 3) and then come back to the Data tab. The independent data filters (length / nonempty /
> template / code) and the quality audit need no eval and can run immediately.

---

## Walkthrough
This runs the whole thing end to end. A great first run is the tiny `HuggingFaceTB/SmolLM2-135M-Instruct`
base (pick it from the suggestions list) — it downloads in seconds and trains in well under a minute,
so you can see every step work before committing to a real run.

### 1. Pick a base (Model)
1. Open the **Model** tab and create (or select) a project at the top.
2. Choose a base: pick one from the suggestions, or paste **any** MLX-loadable reference — a full
   HuggingFace id (e.g. `mlx-community/Mistral-7B-Instruct-v0.3-4bit`, or a normal HF model that can be
   converted) or a **local path** to a model directory. Use the *full* `org/name` id — a bare name
   without the org won't resolve on HuggingFace.
3. Optional checkboxes:
   - **convert to MLX** — run `mlx_lm.convert` (needed for non-MLX HF models).
   - **quantize 4-bit** — quantize during conversion to shrink memory.
   - Leave both off for models that are already MLX-native (the `mlx-community/*` repos).
4. Prepare/Set base. The tool downloads/converts as needed and shows the detected **family,
   parameter count, quantization, and whether it has a chat template**. Use the template preview to
   confirm the model's own formatting — getting this wrong is the #1 naive-fine-tuning mistake, so the
   tool renders exactly what the model will see.

The base is now **pinned** to the project; every later step uses it.

### 2. Bring your data (Data)
1. On the **Data** tab, paste your dataset as JSONL, or give a path to a `.jsonl` / `.json` / `.csv`
   file. (See [Accepted data formats](#accepted-data-formats).)
2. **Inspect.** The tool auto-detects your schema (chat / instruction / prompt-completion / raw text),
   shows a column mapping you can confirm, and renders **example #N exactly as the model will see it**
   — your data normalized to chat messages and run through *the base's own chat template*. It reports
   token-length distribution, how many examples exceed `max_seq_len` (and will truncate), and how many
   were dropped as empty.
3. **Prepare.** Writes `train.jsonl` / `valid.jsonl` (in the format `mlx_lm` expects) plus a
   `raw.jsonl` of the full normalized set. `val_frac` controls the holdout split; `system` sets an
   optional system prompt.
4. **Audit quality** (recommended) — dataset-level checks: exact duplicates, empty fields, length
   distribution, label balance, and a diversity signal. Needs no eval.
5. **Filter** (optional) — keep only examples that pass a quality bar. Rules can be **independent**
   (length / nonempty / a template / code) and run immediately, or **reuse an eval's detector** —
   which requires that you've defined the eval first (step 3). See [Filter rules](#filter-rules). The
   kept set is frozen as `DATASET.lock`; rejections are bucketed by reason, and you can hand-rescue a
   false rejection. (If you skip filtering, there's simply no `DATASET.lock` and training uses the
   prepared `train.jsonl` directly — filtering is never required to train.)
6. **Contamination & dedup** — three tiers, increasing in power (and all *evidence, not proof*):
   - **Always-on lexical warning** — the built-in train/eval overlap check is exact/near-exact only,
     and it tells you so: it does *not* catch paraphrases.
   - **Lexical near-dup pass** (opt-in, needs an eval) — char-3-gram Jaccard between each eval input
     and each train input (default threshold 0.7), flagging rewordings the exact check misses. No
     model, no download — runs anywhere.
   - **Semantic tier** (opt-in, needs an eval) — embeds inputs locally with `all-MiniLM-L6-v2` and, in
     **one pass**, surfaces three views: (a) train↔eval paraphrase **contamination** (threshold 0.85),
     (b) in-dataset semantic **near-duplicate clusters** (threshold 0.88), and (c) a **diversity**
     signal (mean pairwise cosine + approximate cluster count + a low-diversity flag). On-device;
     nothing is uploaded.

### 3. Define an eval (Eval)
The **Eval** tab is the heart of the tool. Define how you'll *measure* the model, three ways:
- **Template (no code):** `exact`, `contains`, `classification`, `format_json`, `refusal`, `numeric`.
- **Examples (no code):** label good *and* bad outputs; scored by local lexical similarity.
- **Code:** a Python `score(input, output, expected) -> {ok, score, pred}` for anything templates
  can't express.

Your eval **dataset** is a list of `{input, expected}` items (the `expected` is optional for examples/
code kinds). Keep these inputs **distinct from your training data** — overlap is the cardinal sin and
the audit will block it. (See [Eval kinds](#eval-kinds) for per-kind scoring details.)

When you save, the eval is **frozen and versioned** (`eval.lock`). Then **Audit** it: the tool inspects
the eval itself and either **blocks** it (you can't gate a model with a meaningless eval) or **warns**
(you may proceed, but it tells you why it's weak). See [the full table](#the-eval-audit-blocks--warnings).
Every finding explains *why* in plain language — the point is to teach the judgment, not just flag.

### 4. Measure the baseline
Run the frozen eval against the **BASE** model first. This is forced: there is no "it got better"
without a measured "before." Same decoding (greedy by default) is used for base and trained, so the
comparison is apples-to-apples. Results are stored with per-example outputs you can read.

### 5. Pre-register the bar
**Commit** your pass criteria *before* training — each is `{metric, comparator (≥/≤), bar, guard}`.
This freezes the goalpost (`prereg.lock`) so you can't quietly lower it after seeing the result. The
tool **refuses to commit over an eval that still has blocking issues**, and warns if your bar is at or
below what the base already scores (rubber-stamping) or is a suspiciously round guess. A **guard**
metric lets you require that training didn't break something else.

### 6. Train (Train)
1. Set a **version** name (adapters are never overwritten — pick a new name, or enable **resume** to
   continue from that version's existing adapter weights).
2. Tune the LoRA config — rank, scale (alpha), dropout, target modules, number of layers, learning
   rate, iters, batch size, max sequence length, seed, gradient checkpointing. Defaults are sane for a
   first run. See [Training config](#training-config).
3. The **memory pre-check** estimates peak RAM for your settings vs the machine and warns *before* an
   OOM, not after. Lower batch size / seq length, or enable gradient checkpointing, if it's tight.
4. **Start.** Training runs in a separate process (so peak memory is reclaimed when it ends and the UI
   never freezes). Live loss, tokens/sec, and ETA stream in; you can **stop** at any time. The result
   is a versioned, reproducible adapter (config + seed + data hash recorded).

### 7. Gate the result
Run the **gate**: it re-runs the frozen eval on the trained adapter (same decoding as the baseline)
and shows base-vs-trained side-by-side **for every metric the eval produced** — not only the committed
criteria — so a regression in an un-gated metric is still visible. Each committed criterion gets a
pass/fail verdict.

**It defaults to DISCARD.** If any criterion failed or a guard metric regressed, the safe default is
*not* to keep the adapter — and keeping it anyway is an explicit **override that requires a recorded
reason**. A run that didn't clear the bar isn't an error; it's a real result ("training didn't help on
this eval — that's information"), recorded honestly. Decide **keep** or **discard**; the decision and
reason are saved with the run.

### 8. Export & use (Adapters)
The **Adapters** tab lists every versioned adapter with its provenance (base, rank, iters, final
losses, data hash, runtime). For any adapter you can:
- **Quick-test** it inline — a chat-templated single generation to sanity-check before exporting. You
  set the prompt, an optional system prompt, max tokens (default 200), and temperature (default 0 =
  greedy). It's a sanity check, not an eval.
- **Export** as `adapter` (LoRA only), `fused` (standalone MLX model), or `gguf` (for llama.cpp /
  Ollama; llama/mistral-family architectures only). See [Export formats](#export-formats).

---

## Reference

### Accepted data formats
Auto-detected from your rows (paste as JSONL, or a `.jsonl` / `.json` / `.csv` path):

| Schema | Looks like | Normalized to |
|---|---|---|
| `chat` | `{"messages": [{"role","content"}, ...]}` | used as-is |
| `instruction` | `{"instruction", "input"?, "output"\|"response"}` | user = instruction (+input), assistant = output |
| `prompt_completion` | `{"prompt"\|"input", "completion"\|"response"}` | user = prompt, assistant = completion |
| `text` | `{"text": "..."}` | raw text (no role formatting) |

Everything is normalized to chat `messages` and rendered through the base's own template, so what you
preview is what the model trains on. If the base has no chat template, the tool falls back to
prompt/completion and tells you.

### Eval kinds
| Kind | Template | What it checks | Scoring detail |
|---|---|---|---|
| template | `exact` | normalized output equals expected | case/whitespace-normalized |
| template | `contains` | output contains a string / matches a regex | uses per-example `expected`, else a fixed `pattern`; optional `regex` |
| template | `classification` | output is one of N labels | first label found as a substring of the output; accuracy + per-class + macro-F1 |
| template | `format_json` | output parses as JSON | optional `required_keys` must be present |
| template | `refusal` | output does / doesn't refuse | `expected` ∈ {refuse,true,yes,1} means "should refuse"; markers overridable |
| template | `numeric` | extracts a number, checks it | within an optional `tolerance` |
| examples | — | lexical similarity to labeled outputs | passes iff similarity-to-good ≥ `threshold` (default 0.6) **and** ≥ similarity-to-bad |
| code | — | your `score(input, output, expected)` | returns `{ok, score, pred}` |

### The eval audit (blocks & warnings)
**Blocks** (you cannot gate until fixed):

| Code | Trigger |
|---|---|
| `too_small` | fewer than **20** examples — pass/fail is statistical noise |
| `contamination` | an eval input also appears in the training data — the cardinal sin |
| `missing_expected` | a template that needs labels has examples without one |

**Warnings** (allowed, but argued):

| Code | Trigger |
|---|---|
| `small_n` | 20–50 examples — wide error bars; small changes may be noise |
| `class_imbalance` | one label is >85% of the set — accuracy is misleading |
| `single_dimension` | only one metric measured **and** no guard criterion — fires for single-metric template evals (exact/contains/format_json/refusal/numeric); classification/examples/code already report two metrics |
| `bar_below_baseline` | a **≥** bar is at/below what the base already scores — rubber-stamping (≤ bars aren't checked this way) |
| `round_bar` | the bar is a multiple of 10, or 75/95 — often a guess, not a justified threshold |

### Contamination & dedup tiers
Three increasingly powerful checks for train/eval overlap and dataset redundancy — all **evidence,
not proof**, and the last two need an eval defined first:

| Tier | Needs | What it catches |
|---|---|---|
| Lexical warning (always on) | — | exact / near-exact train↔eval overlap only |
| Lexical near-dup (`/data/near-dup`, opt-in) | an eval | char-3-gram Jaccard rewordings (default threshold 0.7); no model/download |
| Semantic (opt-in) | an eval + `sentence-transformers` | paraphrase contamination (0.85) + in-dataset near-dup clusters (0.88) + a diversity signal, from one local embedding pass |

### Filter rules
Each rule decides whether a training example is kept. An example is kept **iff every *required* rule is
satisfied** (a rule with `required: false` is skipped). The check short-circuits on the **first**
failing required rule, and the rejection is bucketed under that rule's name (reasons are
first-failure, not exhaustive). Rule sources:
- **Reuse an eval** (`source: eval`, the default move) — filter your data against the *same* detector
  you'll grade with, so the standard is consistent end to end. Requires the eval to exist first.
- **Template** — `exact` / `contains` / `classification` / `format_json` / `refusal` / `numeric`, plus
  `length` (min/max words) and `nonempty`. No eval needed.
- **Code** — a Python `score(...)` rule.

Each rule has a **polarity**: `keep_if: pass` (keep when the detector passes — e.g. valid JSON) or
`keep_if: fail` (keep when it does *not* — e.g. drop confident-call language). Rejections are bucketed
by reason so you can see *how much of your data violates your own standard*, and a false rejection can
be hand-rescued (recorded). The kept set is frozen as `DATASET.lock`; `raw.jsonl` is preserved.

### Training config
LoRA via `mlx_lm`, written to a generated config. Key knobs: `rank`, `scale` (alpha), `dropout`,
`target_modules` (or let `mlx_lm` auto-pick per architecture), `fine_tune_type` (lora/dora/full),
`num_layers`, `learning_rate`, `iters`, `batch_size`, `max_seq_len`, `seed`, `steps_per_report`,
`steps_per_eval`, `save_every`, `grad_checkpoint`, `resume`. Notes:
- **`resume`** continues from the same version's existing `adapters.safetensors` instead of erroring on
  the name collision.
- Left null, `steps_per_eval` auto-derives (≈ every `iters/3`) and `save_every` defaults to at least
  the final adapter; the whole valid set is used for evaluation.
- The memory pre-check uses these to estimate peak RAM.

### Export formats
| Format | What you get | Notes |
|---|---|---|
| `adapter` | the LoRA weights only | load with the base via `mlx_lm --adapter-path` |
| `fused` | a standalone MLX model (base + adapter merged) | `mlx_lm.fuse`; `--dequantize` available |
| `gguf` | a `ggml-model-f16.gguf` for llama.cpp / Ollama | **llama/mistral-family architectures only**; for others, export `fused` then convert with llama.cpp |

### Settings (HuggingFace token)
Only needed for **gated or private** base models (public bases need none). Set it on the **Settings**
tab; the UI shows a masked value and whether it came from config or the environment. It's stored
locally in `~/.mlx-master-trainer/config.json` (chmod 600, best-effort) — **never** committed and never
sent anywhere except HuggingFace for the model download. You can also provide it via the `HF_TOKEN`
environment variable; **the Settings-tab/config token takes precedence over `HF_TOKEN`** if both are set.

## Where things live on disk
- **Projects & data:** `~/.mlx-master-trainer/projects/<name>/` in the packaged app, or `projects/` in
  the repo when run from source (the data root defaults to the repo in source runs). Holds `data/`,
  `evals/`, `adapters/`, `exports/`, and `project.json`.
- **Model cache:** downloaded HF weights live in the standard `~/.cache/huggingface/hub`; conversions
  the app materializes live under `models_cache/` (in the repo for source runs,
  `~/.mlx-master-trainer/models_cache/` in the packaged app).
- **`MMT_DATA_ROOT`** overrides the data root — it relocates **both** `projects/` and `models_cache/`
  together. It does **not** move `config.json` (see below).
- **The locks** (drift detectors — each is a hash of the frozen artifact):
  - `eval.lock` — the eval definition + scorer, frozen at creation.
  - `DATASET.lock` — the filtered training set, frozen after the optional Filter step (absent if you
    don't filter).
  - `prereg.lock` — your pass criteria, frozen before training.
- **Secrets:** `~/.mlx-master-trainer/config.json` only — a fixed path in your home dir, **not** moved
  by `MMT_DATA_ROOT`. `projects/`, `models_cache/`, `config.json`, and `.env` are all git-ignored.

## Troubleshooting
- **`cargo: command not found` (desktop app)** — option B compiles a Rust/Tauri crate; install the
  Rust toolchain (`curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`) and Xcode Command
  Line Tools, or just use the web UI (`./run.sh`), which needs neither.
- **`ModuleNotFoundError` on `./run.sh`** — the deps went into the wrong environment. Use `uv venv` +
  `uv pip install ...` as shown (no manual activation needed; `run.sh` uses `.venv/bin/python`).
- **"base has no chat template"** — your base is a raw/base model, not an instruct model. The tool
  falls back to prompt/completion formatting; fine for completion-style data, but for chat data prefer
  an `-Instruct` base.
- **Many examples exceed `max_seq_len`** — they'll truncate (and lose the assistant turn). Raise
  `max_seq_len` if memory allows, or shorten/curate the data.
- **Memory pre-check warns / training OOMs** — lower `batch_size` or `max_seq_len`, reduce
  `num_layers`, or enable `grad_checkpoint`. Prefer a 4-bit base for large models.
- **Gated/private base fails to download** — set your HuggingFace token on the Settings tab (or
  `HF_TOKEN`), and make sure you've accepted the model's license on HuggingFace.
- **The semantic check / near-dup is greyed out or errors** — both need an **eval defined first**, and
  the semantic tier needs `sentence-transformers` installed (it degrades gracefully when absent).
- **GGUF export fails** — GGUF is only supported for llama/mistral-family architectures. Export
  `fused` and convert with llama.cpp for other families.
- **Eval audit blocks me** — that's the point. `too_small` → add examples to reach 20+;
  `contamination` → remove eval inputs that appear in training; `missing_expected` → add a label to
  every example.
- **The gate says discard** — the run didn't clear the bar you committed to. That's a real, useful
  result. Iterate on data/config and train a new version, or override with a recorded reason if you
  have one.

## Pure-local & honest limits
Your data and models stay on the machine. The only network egress is the *initial* base-model
download (and, if you opt in, the embedding-model download). No telemetry, no cloud calls on your data.

The tool defeats the *common, cheap* ways a fine-tune fools you — it is not omniscient, and it says so:
- **Contamination detection is evidence, not proof.** The lexical and semantic passes catch exact and
  paraphrased overlap; both still miss semantically-distant leakage and are threshold-sensitive.
- **The filter is only as good as your rules.** Bad rules filter in the wrong direction.
- **Representativeness and label-correctness go unjudged.** A clean, well-sized eval of the *wrong* or
  *mislabeled* examples still passes.

That honesty is the design: the tool tells you what it caught *and* what it couldn't.
