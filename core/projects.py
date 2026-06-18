"""Per-project workspaces. A project pins one base model + its data + its versioned adapters,
so a user's whole experiment is one reproducible folder under projects/<name>/.

Layout:
  projects/<name>/
    project.json           # {name, base:{ref, local_path, info}, created_at}
    data/                  # raw.(jsonl|csv) + normalized train.jsonl / valid.jsonl + ingest.json
    adapters/<version>/
      adapter/             # adapters.safetensors + adapter_config.json (mlx_lm output)
      config.yaml          # the exact mlx_lm lora config used
      manifest.json        # config + seed + data_hash + base + final loss + provenance
      run.log              # full training stdout
      status.json          # live training status (polled by the UI)
    exports/<version>/...  # fused / gguf / adapter-only exports
"""
from __future__ import annotations

import time
from pathlib import Path

from .common import PROJECTS, read_json, sanitize_ref, write_json


def project_dir(name: str) -> Path:
    return PROJECTS / sanitize_ref(name)


def list_projects() -> list[dict]:
    out = []
    for d in sorted(PROJECTS.glob("*")):
        meta = read_json(d / "project.json")
        if meta:
            n_adapters = len(list((d / "adapters").glob("*"))) if (d / "adapters").exists() else 0
            out.append({**meta, "dir": d.name, "n_adapters": n_adapters,
                        "has_data": (d / "data" / "train.jsonl").exists()})
    return out


def create_project(name: str, base_ref: str | None = None) -> dict:
    d = project_dir(name)
    if (d / "project.json").exists():
        return read_json(d / "project.json")
    (d / "data").mkdir(parents=True, exist_ok=True)
    (d / "adapters").mkdir(parents=True, exist_ok=True)
    (d / "exports").mkdir(parents=True, exist_ok=True)
    meta = {"name": name, "base": ({"ref": base_ref} if base_ref else None),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    write_json(d / "project.json", meta)
    return meta


def get_project(name: str) -> dict | None:
    return read_json(project_dir(name) / "project.json")


def set_base(name: str, base: dict) -> dict:
    """Pin the resolved base (ref + local_path + info) onto the project."""
    d = project_dir(name)
    meta = read_json(d / "project.json") or create_project(name)
    meta["base"] = base
    write_json(d / "project.json", meta)
    return meta
