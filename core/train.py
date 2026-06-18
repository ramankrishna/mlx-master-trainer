"""Generalization #3 — the base-agnostic, memory-guarded LoRA SFT engine.

Extracted from the bro studio's retrain() (mlx_lm.lora as an isolated subprocess, versioned
adapters, loss parsing) and generalized:
  - FULL config via a generated mlx_lm YAML: rank / scale(alpha) / dropout / target-modules /
    fine-tune-type / num-layers / lr / iters / batch / max-seq-len / seed / grad-checkpoint.
  - MEMORY GUARD: training ALWAYS runs in a separate process (peak reclaimed on exit, the backend
    never holds the optimizer), and models.memory_precheck warns BEFORE an OOM on the 24GB budget.
  - LIVE loss / tokens-sec / ETA streamed via a detached runner that writes status.json (the UI
    polls it). Versioned, reproducible adapters (config + seed + data hash recorded); never
    overwrites a prior adapter. Stop + resume supported.

start()/status()/stop() are what the backend calls; the actual run lives in run_train.py.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

from . import models, projects
from .common import ROOT, read_json, runner_argv, write_json

DEFAULT_CFG = {
    "rank": 8, "scale": 16.0, "dropout": 0.0, "target_modules": None,    # None -> mlx_lm auto-picks
    "fine_tune_type": "lora", "num_layers": 16, "learning_rate": 1e-4,
    "iters": 100, "batch_size": 1, "max_seq_len": 2048, "seed": 0,
    "steps_per_report": 10, "steps_per_eval": None, "save_every": None,
    "grad_checkpoint": False, "resume": False,
}


def build_lora_yaml(model_path: str, data_dir: str, adapter_path: str, cfg: dict) -> dict:
    """The exact mlx_lm lora config (written to config.yaml). Returns the dict for the manifest."""
    iters = int(cfg["iters"])
    spe = cfg.get("steps_per_eval") or max(1, min(iters, max(10, iters // 3)))
    save = cfg.get("save_every") or max(1, iters)            # at least save the final adapter
    y = {
        "model": str(model_path), "train": True, "data": str(data_dir),
        "fine_tune_type": cfg.get("fine_tune_type", "lora"),
        "num_layers": int(cfg["num_layers"]), "batch_size": int(cfg["batch_size"]),
        "iters": iters, "val_batches": -1,                  # use the whole (small) valid set
        "learning_rate": float(cfg["learning_rate"]),
        "steps_per_report": int(cfg.get("steps_per_report", 10)),
        "steps_per_eval": int(spe), "save_every": int(save),
        "adapter_path": str(adapter_path), "max_seq_length": int(cfg["max_seq_len"]),
        "grad_checkpoint": bool(cfg.get("grad_checkpoint", False)), "seed": int(cfg["seed"]),
        "lora_parameters": {"rank": int(cfg["rank"]), "scale": float(cfg["scale"]),
                            "dropout": float(cfg["dropout"])},
    }
    if cfg.get("target_modules"):                            # else let mlx_lm choose per-arch
        y["lora_parameters"]["keys"] = list(cfg["target_modules"])
    if cfg.get("resume"):
        rf = Path(adapter_path) / "adapters.safetensors"
        if rf.exists():
            y["resume_adapter_file"] = str(rf)
    return y


def start(project: str, cfg: dict) -> dict:
    """Launch a versioned training run as a detached process. Returns immediately; poll status()."""
    meta = projects.get_project(project)
    if not meta or not meta.get("base"):
        return {"ok": False, "error": "project has no base model set"}
    base_path = meta["base"].get("local_path") or meta["base"].get("ref")
    pdir = projects.project_dir(project)
    data_dir = pdir / "data"
    if not (data_dir / "train.jsonl").exists():
        return {"ok": False, "error": "no prepared data — ingest a dataset first"}

    full = {**DEFAULT_CFG, **(cfg or {})}
    version = (full.get("version") or f"run-{int(time.time())}").strip()
    adir = pdir / "adapters" / version
    if adir.exists() and not full.get("resume"):
        return {"ok": False, "error": f"adapter '{version}' already exists (never overwritten) — "
                                      "pick a new name or enable resume"}
    (adir / "adapter").mkdir(parents=True, exist_ok=True)

    job = {"project": project, "version": version, "base_path": str(base_path),
           "data_dir": str(data_dir), "adapter_dir": str(adir), "cfg": full}
    write_json(adir / "job.json", job)
    write_json(adir / "status.json", {"state": "starting", "version": version, "iter": 0,
                                      "total_iters": full["iters"], "history": []})

    boot = (adir / "boot.log").open("w")
    proc = subprocess.Popen(runner_argv("train", "--job", str(adir / "job.json")),
                            stdout=boot, stderr=subprocess.STDOUT, start_new_session=True)
    write_json(adir / "runner.json", {"runner_pid": proc.pid})
    return {"ok": True, "version": version, "adapter_dir": str(adir), "runner_pid": proc.pid}


def status(project: str, version: str | None = None) -> dict:
    """Live status of a run (latest run if version omitted)."""
    pdir = projects.project_dir(project)
    adapters = pdir / "adapters"
    if not adapters.exists():
        return {"state": "none"}
    if version:
        st = read_json(adapters / version / "status.json")
        return st or {"state": "none"}
    runs = [d for d in adapters.glob("*") if (d / "status.json").exists()]
    if not runs:
        return {"state": "none"}
    latest = max(runs, key=lambda d: (d / "status.json").stat().st_mtime)
    return read_json(latest / "status.json") or {"state": "none"}


def stop(project: str, version: str | None = None) -> dict:
    """Kill a running training process (the run finalizes its status as 'stopped')."""
    import os
    import signal
    pdir = projects.project_dir(project)
    st = status(project, version)
    ver = version or st.get("version")
    if not ver:
        return {"ok": False, "error": "no run to stop"}
    adir = pdir / "adapters" / ver
    killed = []
    for f, key in ((adir / "status.json", "train_pid"), (adir / "runner.json", "runner_pid")):
        rec = read_json(f) or {}
        pid = rec.get(key)
        if pid:
            try:
                os.kill(pid, signal.SIGTERM)
                killed.append(pid)
            except Exception:
                pass
    return {"ok": bool(killed), "version": ver, "killed": killed}
