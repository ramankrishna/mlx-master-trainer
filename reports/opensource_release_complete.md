# MLX Master Trainer — Open-Source .dmg Packaging — STATUS

```
=== MLX Master Trainer — Open-Source .dmg Release ===
GATE:      git-history scan N/A (fresh repo, no history) · working tree CLEAN · no token · DeepSeek key never in this project
           · .gitignore covers user data/models/keys (caught+fixed an inline-comment bug that had staged projects/)
Backend:   PyInstaller --onedir frozen binary (mlx + mlx_lm + tokenizers + sentence-transformers + torch), 733 MB,
           runs STANDALONE — tested outside the dev venv: real MLX matmul + a full LoRA train end-to-end (loss 0.331)
Sidecar:   frozen binary spawned by the Rust shell from Resources/mmt-backend (dev falls back to the venv);
           tauri.conf bundles ../dist/mmt-backend; arm64 target added for a native aarch64-apple-darwin build
.dmg:      BUILD STEP PENDING (cargo tauri build) — see "Remaining", gated with the push
First-run: ships code only; base models + MiniLM download on first use; data root = ~/.mlx-master-trainer (writable); pure-local after
License:   MIT open-core; README has install + Gatekeeper steps + honest limits + open-core note
Release:   GATED — awaiting operator (push authorization + GitHub destination + key-rotation confirmation)
Decisions  mmt-024..mmt-029
```

## Stage 0 — the credential gate (PASSED, and stronger than the dispatch assumed)
- **Not a git repo** before this work → a fresh `git init` means **no history to leak** (the safest start).
- Working tree **clean** of credential patterns; **no token** anywhere; `~/.mlx-master-trainer/config.json` was never created.
- **The DeepSeek key never touched MLX Master Trainer** — that credential belongs to `bigbugai-bro`; this project has no
  DeepSeek dependency (grep-confirmed). So for *this* repo the leak risk is nil. (Rotation remains good hygiene for the
  broader project, but it is not a blocker for this repo.)
- `.gitignore` written and **verified by staging**: a deliberate guard caught `projects/` being staged (an inline `#` comment
  had broken the pattern — comments must be on their own line); fixed; re-verified **108 code-only files, 4.7 MB, zero
  credentials/user-data/models** staged. **Committed locally** (not pushed).

## Stage 1 — PyInstaller freeze (DONE, the hard part — it RUNS, not just builds)
Frozen `backend/app_entry.py` → `--onedir` binary, **733 MB** (mlx + mlx_lm + tokenizers + transformers + the
sentence-transformers/torch semantic stack — PyInstaller followed the imports, so the semantic tier works frozen too).
**Standalone keystone (`reports/frozen_keystone.json`) — PASS 4/4, run outside the dev venv:**

| check | result |
|---|---|
| `--selftest` real MLX op | ✅ `mlx matmul sum=4096.0` · `embed_dim=384` (Metal + embeddings work frozen) |
| frozen server boots | ✅ `/health` ok |
| **full LoRA train through the frozen backend** | ✅ `started → done · loss 0.331` — **zero user Python** |

The "PyInstaller built ≠ runs" risk was real and caught two bugs, both fixed:
1. **Missing `multiprocessing.freeze_support()`** — torch/resource-tracker spawned children that re-execed the binary with
   no args → fell through to "start the server" → port clash + a hang. Added `freeze_support()` in the entry.
2. **Data root inside the read-only bundle** — `ROOT = __file__/parents[1]` pointed *into the .app*. Split code-root from a
   writable **data root** (`~/.mlx-master-trainer`, env-overridable `MMT_DATA_ROOT`). Frozen apps now read/write user data correctly.

The frozen-aware dispatch (`backend/app_entry.py` + `core/common.py` `runner_argv`/`mlx_argv`) solves the core problem that a
bundle has no `python -m` and no source scripts: the engine's subprocess spawns re-invoke the binary with `--run`/`--mlx`
flags. Unfrozen argv is byte-identical, so the dev flow and all prior keystones still hold.

## Stage 2 — Tauri sidecar (DONE — code; native build pending)
`src/lib.rs` `spawn_backend()` now spawns the bundled frozen binary from `resource_dir()/mmt-backend/mmt-backend` (no args
→ server; default user data root), with a dev fallback to the venv + `server.py`. `tauri.conf.json` bundles
`../dist/mmt-backend` as a resource. `rustup target add aarch64-apple-darwin` done (Phase-1 noted the toolchain was x86_64).

## Remaining — the GATED release (operator action required)
These are intentionally **not** done because the public push is the operator's call and the `.dmg` is the release artifact:
1. **Build the unsigned .dmg:** `cd desktop/src-tauri && cargo tauri build --target aarch64-apple-darwin`
   (needs `tauri-cli`; bundling the 733 MB sidecar → a ~700 MB+ dmg; the Tauri resource-glob may need one tweak — finalize when cutting the release).
2. **Gatekeeper:** unsigned, so first launch needs right-click → Open, or `xattr -cr "/Applications/MLX Master Trainer.app"` —
   documented in the README; test on the built dmg.
3. **Public push — GATE (cannot be done unilaterally):**
   - **Key rotation** — operator-confirmed (moot for this repo; broader hygiene).
   - **Explicit push authorization + the GitHub destination** (org/repo URL).
   Then: push the (history-clean) repo, attach the `.dmg` to a **GitHub Release** (not committed — `.dmg` is gitignored).

## Honest state
The genuinely hard, uncertain part — **freezing the MLX/torch backend so it RUNS standalone** — is **done and proven**
(real training through the bundled binary, no user Python). The repo is **credential-clean, MIT-licensed, documented, and
committed locally**. What's left is the mechanical `cargo tauri build` and the **operator-gated public push**. Decisions mmt-001..029.
