#!/usr/bin/env python
"""Detached semantic-pass runner — loads the embedding model in its OWN process (memory guard: torch is
freed on exit, never coexisting with a loaded train model in the backend), computes the three views, writes
report.json + status.json. Launched by core.semantic.start(); the backend polls status() then reads report().
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import semantic                         # noqa: E402
from core.common import read_json, write_json     # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    job = read_json(ap.parse_args().job)
    rd = semantic._run_dir(job["project"], job["eval_version"])
    try:
        write_json(rd / "status.json", {"state": "embedding"})
        rep = semantic.analyze(job["project"], job["eval_version"],
                               job.get("contam_threshold", 0.85), job.get("dup_threshold", 0.88))
        write_json(rd / "report.json", rep)
        write_json(rd / "status.json", {"state": "done" if rep.get("ok") else "error",
                                        "n_flagged": rep.get("contamination", {}).get("n_flagged")})
    except Exception as e:
        write_json(rd / "status.json", {"state": "error", "error": str(e)[:300]})


if __name__ == "__main__":
    main()
