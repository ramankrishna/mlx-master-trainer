"""MLX Master Trainer — the model-agnostic MLX LoRA SFT engine (Phase 1).

Submodules:
  common    paths, system RAM, HF token, hashing
  projects  per-project workspaces
  models    generalization #1 — any base: convert, template detect, memory pre-check
  data      generalization #2 — any schema: ingest, templated preview, validation
  train     generalization #3 — guarded LoRA engine (start/status/stop) + run_train runner
  adapters  manage, fuse, export (gguf/fused/adapter), quick inference test

Provenance: generalized from the BigBugAI Bro studio retrain panel (memory-guarded subprocess,
versioned adapters, live loss). Imports are lazy in each submodule so this package loads light.
"""
