"""Adapter management + export + quick inference test.

Generalized from the bro's GGUF export: list versioned adapters with their full provenance,
fuse an adapter into the base (mlx_lm.fuse), export GGUF / fused-MLX / adapter-only, and a quick
chat-templated inference test so you can sanity-check a trained adapter before exporting it.
(NOT a full eval — that discipline layer is Phase 2.)
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from . import projects
from .common import mlx_argv, read_json


def _base_path(project: str) -> str | None:
    meta = projects.get_project(project)
    if not meta or not meta.get("base"):
        return None
    return meta["base"].get("local_path") or meta["base"].get("ref")


def list_adapters(project: str) -> list[dict]:
    pdir = projects.project_dir(project)
    adapters = pdir / "adapters"
    out = []
    for d in sorted(adapters.glob("*")) if adapters.exists() else []:
        man = read_json(d / "manifest.json") or {}
        st = read_json(d / "status.json") or {}
        cfg = man.get("config", {})
        has_weights = (d / "adapter" / "adapters.safetensors").exists()
        out.append({
            "version": d.name, "state": st.get("state", "?"), "ok": man.get("ok", has_weights),
            "has_weights": has_weights,
            "base": (man.get("base") or {}).get("ref"), "family": (man.get("base") or {}).get("family"),
            "rank": cfg.get("rank"), "scale": cfg.get("scale"), "iters": cfg.get("iters"),
            "fine_tune_type": cfg.get("fine_tune_type"),
            "final_train_loss": man.get("final_train_loss"), "final_val_loss": man.get("final_val_loss"),
            "data_hash": man.get("data_hash"), "runtime_s": man.get("runtime_s"),
            "created_at": man.get("created_at"),
        })
    return out


def _adapter_dir(project: str, version: str) -> Path:
    return projects.project_dir(project) / "adapters" / version / "adapter"


def fuse(project: str, version: str, dequantize: bool = False, progress=lambda m: None) -> dict:
    """Fuse base + adapter into a standalone MLX model (mlx_lm.fuse)."""
    base = _base_path(project)
    adapter = _adapter_dir(project, version)
    if not (adapter / "adapters.safetensors").exists():
        return {"ok": False, "error": f"adapter '{version}' has no weights"}
    out = projects.project_dir(project) / "exports" / version / "fused"
    out.mkdir(parents=True, exist_ok=True)
    cmd = mlx_argv("fuse", "--model", str(base),
                   "--adapter-path", str(adapter), "--save-path", str(out))
    if dequantize:
        cmd.append("--dequantize")
    progress("fusing base + adapter …")
    r = subprocess.run(cmd, capture_output=True, text=True)
    ok = (out / "config.json").exists()
    return {"ok": ok, "path": str(out) if ok else None, "error": "" if ok else r.stderr[-400:]}


def export(project: str, version: str, fmt: str = "fused", progress=lambda m: None) -> dict:
    """fmt = 'adapter' (the LoRA only), 'fused' (standalone MLX), or 'gguf' (llama.cpp/Ollama)."""
    base = _base_path(project)
    adapter = _adapter_dir(project, version)
    if not (adapter / "adapters.safetensors").exists():
        return {"ok": False, "error": f"adapter '{version}' has no weights"}
    exdir = projects.project_dir(project) / "exports" / version
    exdir.mkdir(parents=True, exist_ok=True)

    if fmt == "adapter":
        return {"ok": True, "fmt": "adapter", "path": str(adapter),
                "note": "the LoRA adapter only — load with the base via mlx_lm --adapter-path"}
    if fmt == "fused":
        return {**fuse(project, version, progress=progress), "fmt": "fused"}
    if fmt == "gguf":
        out = exdir / "fused"
        gguf = exdir / "gguf"
        gguf.mkdir(parents=True, exist_ok=True)
        cmd = mlx_argv("fuse", "--model", str(base), "--adapter-path", str(adapter),
                       "--save-path", str(out), "--export-gguf", "--gguf-path", str(gguf / "ggml-model-f16.gguf"))
        progress("fusing + exporting GGUF (llama/mistral-family archs) …")
        r = subprocess.run(cmd, capture_output=True, text=True)
        f = gguf / "ggml-model-f16.gguf"
        if f.exists():
            return {"ok": True, "fmt": "gguf", "path": str(f), "size_mb": f.stat().st_size // 1_000_000}
        return {"ok": False, "fmt": "gguf", "error": (r.stderr[-400:] or
                "GGUF export failed — only supported for llama/mistral-family archs; "
                "use 'fused' then llama.cpp convert for others")}
    return {"ok": False, "error": f"unknown export format '{fmt}'"}


def quick_infer(project: str, version: str, prompt: str, system: str | None = None,
                max_tokens: int = 200, temp: float = 0.0) -> dict:
    """Load base + adapter and generate once (chat-templated) — sanity-check before exporting.
    Runs as a subprocess so the model memory is reclaimed right after (memory guard)."""
    base = _base_path(project)
    adapter = _adapter_dir(project, version)
    if not (adapter / "adapters.safetensors").exists():
        return {"ok": False, "error": f"adapter '{version}' has no weights"}
    cmd = mlx_argv("generate", "--model", str(base), "--adapter-path", str(adapter),
                   "--prompt", prompt, "--max-tokens", str(max_tokens), "--temp", str(temp))
    if system:
        cmd += ["--system-prompt", system]
    r = subprocess.run(cmd, capture_output=True, text=True)
    text = r.stdout
    # mlx_lm.generate brackets the generation with ========== lines
    parts = text.split("==========")
    gen = parts[1].strip() if len(parts) >= 3 else text.strip()
    return {"ok": bool(gen), "output": gen, "stderr_tail": r.stderr[-300:] if not gen else ""}
