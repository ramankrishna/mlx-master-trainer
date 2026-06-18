"""MLX Master Trainer — shared foundation (paths, system RAM, HF token, hashing).

Kept import-light on purpose: nothing here pulls in mlx / transformers, so the FastAPI
backend and the lightweight UI calls start fast. Heavy imports live inside the functions
that need them (models/data/train/adapters).

PROVENANCE: this project generalizes the BigBugAI Bro studio's retrain machinery
(~/bigbugai-bro/studio/backend/core.py) off the bro and onto any base model. The memory-guard
+ versioned-adapter + loss-parsing patterns are extracted from there; everything model-,
template-, and schema-specific is rewritten to be agnostic.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECTS = ROOT / "projects"
MODELS_CACHE = ROOT / "models_cache"            # converted/native MLX models we materialize
CONFIG_FILE = Path.home() / ".mlx-master-trainer" / "config.json"

for _d in (PROJECTS, MODELS_CACHE):
    _d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# system memory (the 24GB budget the memory-guard reasons about)
# --------------------------------------------------------------------------- #
def system_ram_gb() -> float:
    """Physical RAM in GB. Uses sysctl on macOS (no extra dep); /proc fallback on Linux."""
    try:
        if sys.platform == "darwin":
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
            return round(int(out) / 1024**3, 1)
        pages = os.sysconf("SC_PHYS_PAGES")
        return round(pages * os.sysconf("SC_PAGE_SIZE") / 1024**3, 1)
    except Exception:
        return 0.0


# --------------------------------------------------------------------------- #
# HF token — local config first (chmod 600), env fallback. NEVER committed.
# (Only needed for gated/private base models; public bases need no token.)
# --------------------------------------------------------------------------- #
def _cfg() -> dict:
    try:
        return json.loads(CONFIG_FILE.read_text())
    except Exception:
        return {}


def get_token() -> str | None:
    t = (_cfg().get("hf_token") or "").strip()
    return t or (os.environ.get("HF_TOKEN") or "").strip() or None


def set_token(t: str) -> dict:
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    c = _cfg()
    t = (t or "").strip()
    if t:
        c["hf_token"] = t
    else:
        c.pop("hf_token", None)
    CONFIG_FILE.write_text(json.dumps(c, indent=2))
    try:
        CONFIG_FILE.chmod(0o600)
    except Exception:
        pass
    return token_status()


def token_status() -> dict:
    c = (_cfg().get("hf_token") or "").strip()
    e = (os.environ.get("HF_TOKEN") or "").strip()
    t = c or e
    return {"set": bool(t), "source": "config" if c else ("env" if e else None),
            "masked": ("•" * 4 + t[-4:]) if len(t) >= 4 else ("set" if t else None)}


# --------------------------------------------------------------------------- #
# small fs/hash helpers
# --------------------------------------------------------------------------- #
def sanitize_ref(ref: str) -> str:
    """A filesystem-safe folder name for a HF repo id or local path."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(ref)).strip("_") or "model"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def data_hash(paths: list[Path]) -> str:
    """Reproducibility: hash of the exact train/valid files a run was trained on."""
    h = hashlib.sha256()
    for p in sorted(paths):
        if p.exists():
            h.update(p.name.encode())
            h.update(sha256_file(p).encode())
    return h.hexdigest()[:16]


def read_json(p: Path, default=None):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return default


def write_json(p: Path, obj) -> None:
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(p)               # atomic: a poll never reads a half-written status file


# --------------------------------------------------------------------------- #
# frozen-aware subprocess argv (PyInstaller). Unfrozen: `python core/run_X.py` / `python -m mlx_lm.X`
# (identical to the dev path). Frozen: sys.executable IS the bundled binary, so re-invoke IT with a
# dispatch flag (handled by backend/app_entry.py) — there is no `python -m` or source path in a bundle.
# --------------------------------------------------------------------------- #
FROZEN = getattr(sys, "frozen", False)


def runner_argv(name: str, *args) -> list[str]:
    if FROZEN:
        return [sys.executable, "--run", name, *args]
    return [sys.executable, str(ROOT / "core" / f"run_{name}.py"), *args]


def mlx_argv(sub: str, *args) -> list[str]:
    if FROZEN:
        return [sys.executable, "--mlx", sub, *args]
    return [sys.executable, "-m", f"mlx_lm.{sub}", *args]
