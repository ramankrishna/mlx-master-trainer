# MLX Master Trainer

**The local fine-tuning studio that won't let you fool yourself.** Fine-tune any MLX base model with
LoRA on your Mac — and the tool *argues with you* about eval quality, forces a baseline, freezes the bar
before training, and refuses to keep a model that didn't earn it. **Pure-local: your data and models
never leave the machine.** Apple Silicon, MIT-licensed open core.

> Most fine-tuning tools help you train. This one helps you not deceive yourself about whether it worked.

## What it is
An honest end-to-end loop, in order — each step unlocks the next:

```
DATA      filter your training data against a quality bar (reuse your eval's detector) → frozen DATASET.lock
 → EVAL   define an eval (no-code templates / examples / code); the tool BLOCKs meaningless ones
          (too-small, train/eval contamination) and WARNs weak ones (bar≤baseline, single-metric, …)
 → BASELINE  measure the BASE model first — no improvement claim without a before/after
 → PREREG  commit your pass criteria, FROZEN, before training (no moving the goalpost)
 → TRAIN   memory-guarded MLX LoRA, live loss, versioned reproducible adapter
 → GATE    base-vs-trained against the committed bar — defaults to DISCARD if it didn't earn keeping
```

Plus a **semantic-contamination tier** (opt-in, local embeddings) that catches paraphrased eval/train
leakage the lexical checks miss. Export GGUF / fused / adapter-only. Any MLX-loadable base (HF id or local
path), any common SFT schema (auto-detected, with a templated preview).

## Install (download)
1. Download `MLX Master Trainer.dmg` from the [latest Release](#) and open it; drag the app to Applications.
2. **It's unsigned** (no Apple Developer cert yet), so Gatekeeper will block the first launch. Open it once
   with **either**:
   - **right-click the app → Open → Open** (confirm the dialog), or
   - Terminal: `xattr -cr "/Applications/MLX Master Trainer.app"` then open normally.
   This is a one-time step; after that it launches by double-click. (It's blocked because it's unsigned,
   not because anything is wrong — the source is right here for you to audit.)
3. **First run downloads models on demand** — a base model when you pick one, and the ~90 MB embedding
   model only if you opt into the semantic check. You need internet for those initial fetches; everything
   is **pure-local after**. **Apple Silicon required** (MLX is Apple-Silicon-only).

## Build from source
See [CONTRIBUTING.md](CONTRIBUTING.md). Short version: `uv venv && uv pip install mlx mlx-lm huggingface_hub
fastapi "uvicorn[standard]" sentence-transformers`, then `./run.sh` (web UI at `127.0.0.1:8808`) or
`cd desktop/src-tauri && cargo run` (desktop app).

## Honest limits (the posture, carried throughout)
This tool defeats the *common, cheap* ways a fine-tune fools you. It is **not** omniscient, and it says so:
- **Contamination detection is evidence, not proof.** The lexical (n-gram) and semantic (local-embedding)
  passes catch exact and paraphrased train/eval overlap; both still **miss semantically-distant leakage**
  (same answer, very different framing) and are threshold-sensitive.
- **The data filter is only as good as your rules.** Garbage rules filter in the wrong direction.
- **Representativeness and label-correctness go unjudged.** A clean, well-sized eval of the *wrong* or
  *mislabeled* examples still passes. No automated check fully covers this.

That honesty is the point: the tool tells you what it caught *and* what it couldn't.

## Open core
This repo is the MIT-licensed **core** — the full local studio above. Future paid/additive features
(cloud dispatch, team/collaboration, signed+notarized distribution) live outside this repo; the core stays
open and pure-local.

## How it was built
The full design trail is in [`decisions.md`](decisions.md) (`mmt-001..`), and each capability has an
end-to-end proof in [`scripts/`](scripts/) (`keystone*.py`) with results in [`reports/`](reports/).
