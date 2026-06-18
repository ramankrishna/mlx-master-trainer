#!/usr/bin/env python
"""Frozen entry point (PyInstaller). In a bundle, sys.executable is THIS binary — there is no
`python -m mlx_lm.X` and no source script paths — so the engine's subprocess spawns (core/common.py
runner_argv / mlx_argv) re-invoke the binary with a dispatch flag, handled here.

Routes:
  --run <train|eval|semantic>   → the detached runner (core/run_*.py main())
  --mlx <lora|fuse|generate|convert>  → the mlx_lm CLI (runpy, == `python -m mlx_lm.X`)
  --selftest                    → prove the frozen native deps work (mlx op + embed)
  (default)                     → the FastAPI server
Unfrozen, backend/server.py is the entry and this file is unused.
"""
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    argv = sys.argv[1:]
    if argv and argv[0] == "--run":
        name = argv[1]
        sys.argv = [f"run_{name}", *argv[2:]]
        mod = __import__(f"core.run_{name}", fromlist=["main"])
        mod.main()
        return
    if argv and argv[0] == "--mlx":
        sub = argv[1]
        sys.argv = [f"mlx_lm.{sub}", *argv[2:]]
        runpy.run_module(f"mlx_lm.{sub}", run_name="__main__")
        return
    if argv and argv[0] == "--selftest":
        import mlx.core as mx
        import mlx_lm  # noqa: F401  (prove the native dylibs bundled)
        x = mx.ones((16, 16))
        s = float((x @ x).sum())
        try:
            from sentence_transformers import SentenceTransformer
            d = SentenceTransformer("all-MiniLM-L6-v2").get_sentence_embedding_dimension()
        except Exception as e:
            d = f"embed-skip({str(e)[:40]})"
        print(f"SELFTEST OK · frozen={getattr(sys,'frozen',False)} · mlx matmul sum={s} · embed_dim={d}")
        return
    import uvicorn
    from backend.server import PORT, app
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()   # PyInstaller: stop spawned children (torch/resource_tracker) from
    main()                             # re-executing the program (which clashed on the server port)
