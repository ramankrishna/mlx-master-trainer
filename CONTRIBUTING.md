# Contributing to MLX Master Trainer

Thanks for your interest. This is the open core (MIT) of a local, honest fine-tuning studio.

## Dev setup (Apple Silicon)
```bash
git clone <repo> && cd mlx-master-trainer
uv venv && uv pip install mlx mlx-lm huggingface_hub fastapi "uvicorn[standard]"
uv pip install sentence-transformers      # optional: the semantic-contamination tier
./run.sh                                  # backend on http://127.0.0.1:8808
# desktop app:  cd desktop/src-tauri && cargo run
```

## Layout
- `core/` — the engine + discipline + data layers (model-agnostic, no UI).
- `backend/server.py` — thin FastAPI shell over `core/` (bound to 127.0.0.1).
- `frontend/index.html` — build-free UI.
- `desktop/src-tauri/` — the Tauri v2 menu-bar shell.
- `scripts/keystone*.py` — the end-to-end proofs (run them; they're the spec).
- `decisions.md` — the full design trail (`mmt-001..`). Read it before large changes.

## Principles (please keep them)
- **Pure-local.** No user data or text leaves the machine. No cloud APIs — embeddings and
  inference run on-device. This is the product promise; PRs that break it won't be merged.
- **Honest, not airtight.** Every check states what it can't catch ("evidence, not proof").
  Don't add a check that implies more certainty than it has.
- **Opinionated discipline.** The tool blocks/warns weak evals, forces a baseline, freezes the
  bar before training, and defaults to discard. Keep that posture.
- **No credentials in commits.** `.gitignore` covers `projects/`, `.venv/`, models, `config.json`.

## PRs
Small, focused, with a note on which keystone(s) you ran. New behavior should come with (or extend)
a keystone script that proves it.
