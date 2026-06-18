# MLX Master Trainer — Phase 1: SFT Engine — COMPLETE

```
=== MLX Master Trainer — Phase 1: SFT Engine ===
App:       Tauri v2 local Mac app (reused bro-studio shell), pure-local, builds + launches on MacBook
Base:      model-agnostic — HF/MLX picker, auto-convert, chat-template detected + shown, memory pre-check
Data:      multi-schema ingest (messages / prompt-completion / instruction), templated PREVIEW, local-only
Train:     MLX LoRA, memory-guarded subprocess (24GB), full config, live loss, versioned reproducible adapters
Export:    GGUF / fused / adapter-only; quick inference test
KEYSTONE:  full loop on NON-bro base (SmolLM2-135M-Instruct) + unformatted instruction data -> PASS (8/8)
Phase 2:   discipline layer (eval-first, freeze, pre-register, regression-gate) — the actual product wedge
Decisions  mmt-001..mmt-006
```

## What shipped
A standalone, pure-local Mac app that fine-tunes **any** MLX-loadable base with LoRA. Built by
**extracting and generalizing** the BigBugAI Bro studio's retrain machinery (not reimplementing):
the memory-guarded `mlx_lm.lora`-as-subprocess, versioned adapters, live loss parsing, and the
Tauri v2 + FastAPI + build-free-HTML shell. Four generalizations off the bro:

| # | Was (bro) | Now (MMT) |
|---|---|---|
| 1 | one base (`sft/merged`) | any HF id / local path; convert; param/quant/family + **chat-template detection**; **memory pre-check** vs 24GB |
| 2 | pre-formatted in-voice `{prompt,response}` | auto-detect `messages` / `prompt-completion` / `instruction`; **rendered post-template preview**; token validation; local-only |
| 3 | fixed CLI flags | full mlx_lm **YAML config** (rank/scale/dropout/target-modules/fine-tune-type/layers/lr/iters/batch/seq/seed/grad-ckpt); live loss + ETA; reproducible manifests |
| 4 | GGUF only | GGUF / fused MLX / adapter-only export + quick chat-templated inference test |

## KEYSTONE — the Phase-1 proof (reports/keystone_result.json)
The whole loop on a base we did **not** build, with data **not** in bro format:
- **Base:** `HuggingFaceTB/SmolLM2-135M-Instruct` — Llama-family, NOT Qwen-bro.
- **Data:** alpaca-style `instruction`/`input`/`output` — a DIFFERENT schema, NOT bro voice.

| check | result |
|---|---|
| base loads + info | ✅ 135M · llama · quant none · template detected |
| memory fits 24GB | ✅ est peak 2.7 GB / 24 GB (headroom 17.7) |
| schema auto-detect | ✅ detected `instruction` |
| **correct template** | ✅ `format=messages`, rendered preview shows SmolLM2 `<|im_start|>` ChatML |
| training runs | ✅ done, 8.8 s |
| loss drops | ✅ train loss **3.322 → 0.452** |
| export GGUF | ✅ `ggml-model-f16.gguf` |
| inference test | ✅ produced output |

**KEYSTONE PASS (8/8).** The engine works on a model we didn't build — the generalization off the bro is real.

## Verification performed
- **Engine:** `scripts/keystone.py` end-to-end (above), via `core/` directly.
- **Backend:** all 25 routes import clean; live smoke of `/health`, `/projects`, `/adapters`,
  `/models/suggested`, `/train/status`, and `GET /` (served no-store).
- **UI:** loaded same-origin in the preview — sidebar/panels render, the keystone project + base info +
  trained adapter (loss 0.452/0.638 val, data# 4056d10e) all load from the live backend; **0 console errors**.
- **Tauri app:** `cargo build` OK (2m39s); launch test — app boots → spawns its own backend on 8808 →
  `/health` responds → no panic → killed clean.

## Notes / follow-ups
- **Build arch:** the debug binary is x86_64 (rust host-triple default → runs via Rosetta). The Tauri
  shell only does IPC + webview and spawns the **native arm64** Python venv where MLX runs, so it's
  functionally correct. Clean native build: `cargo build --target aarch64-apple-darwin`.
- **Icon** is a placeholder (the bro mascot) — MMT branding is a later phase.
- **Do NOT overclaim Phase 1 as the product** — it's the commodity engine. The wedge is the discipline
  layer (Phase 2).

## Phase 2 (gated, ready)
The discipline layer — **eval-first, freeze, pre-register, regression-gate** ("the fine-tuning studio
that won't let you fool yourself") — sits directly on this engine.
