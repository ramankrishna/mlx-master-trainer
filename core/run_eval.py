#!/usr/bin/env python
"""Phase 2 — detached eval runner + run management.

Runs a frozen eval against a target model (the BASE for the forced baseline, or a trained adapter)
in an isolated subprocess: loads the model ONCE, generates per eval input (chat-templated, same
decoding for base and trained — apples-to-apples), scores via the eval's own scorer, writes live
status + a results record. start_eval()/eval_status()/get_baseline() are the API the backend calls.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import eval as evalmod          # noqa: E402
from core import projects                 # noqa: E402
from core.common import ROOT, read_json, runner_argv, write_json   # noqa: E402


def _run_dir(project: str, eval_version: str, target: str) -> Path:
    d = evalmod.eval_dir(project, eval_version) / "runs" / target
    d.mkdir(parents=True, exist_ok=True)
    return d


def start_eval(project: str, eval_version: str, target: str = "base",
               temp: float = 0.0, max_tokens: int | None = None) -> dict:
    """target = 'base' (the forced baseline) or an adapter version. Launches a detached run."""
    meta = projects.get_project(project)
    if not meta or not meta.get("base"):
        return {"ok": False, "error": "project has no base model"}
    base_path = meta["base"].get("local_path") or meta["base"]["ref"]
    adapter_path = None
    if target != "base":
        ap = projects.project_dir(project) / "adapters" / target / "adapter"
        if not (ap / "adapters.safetensors").exists():
            return {"ok": False, "error": f"adapter '{target}' has no weights"}
        adapter_path = str(ap)
    rd = _run_dir(project, eval_version, target)
    job = {"project": project, "eval_version": eval_version, "target": target,
           "base_path": str(base_path), "adapter_path": adapter_path,
           "temp": temp, "max_tokens": max_tokens}
    write_json(rd / "job.json", job)
    write_json(rd / "status.json", {"state": "starting", "target": target, "done": 0, "total": 0})
    boot = (rd / "boot.log").open("w")
    proc = subprocess.Popen(runner_argv("eval", "--job", str(rd / "job.json")),
                            stdout=boot, stderr=subprocess.STDOUT, start_new_session=True)
    return {"ok": True, "target": target, "runner_pid": proc.pid}


def eval_status(project: str, eval_version: str, target: str = "base") -> dict:
    st = read_json(_run_dir(project, eval_version, target) / "status.json")
    return st or {"state": "none"}


def run_results(project: str, eval_version: str, target: str = "base") -> dict | None:
    return read_json(_run_dir(project, eval_version, target) / "results.json")


def get_baseline(project: str, eval_version: str) -> dict | None:
    r = run_results(project, eval_version, "base")
    return r.get("metrics") if r else None


# --------------------------------------------------------------------------- #
# the actual run (detached)
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    job = read_json(ap.parse_args().job)
    project, ev_ver, target = job["project"], job["eval_version"], job["target"]
    rd = _run_dir(project, ev_ver, target)
    status_path = rd / "status.json"

    def set_status(**kw):
        cur = read_json(status_path) or {}
        cur.update(kw)
        write_json(status_path, cur)

    ev = evalmod.load_eval(project, ev_ver)
    scorer = evalmod.make_scorer(ev, evalmod.eval_dir(project, ev_ver))
    ds = ev["dataset"]
    max_tokens = job.get("max_tokens") or ev.get("max_tokens", 64)
    temp = float(job.get("temp", 0.0))
    set_status(state="loading", target=target, total=len(ds), done=0)

    t0 = time.time()
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler
    model, tok = load(job["base_path"], adapter_path=job.get("adapter_path"))
    sampler = make_sampler(temp=temp)
    system = ev.get("spec", {}).get("system")

    def reply(text: str) -> str:
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": text}]
        if tok.chat_template is not None:
            prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        else:
            prompt = text
        return generate(model, tok, prompt, max_tokens=max_tokens, sampler=sampler, verbose=False)

    set_status(state="running")
    scored, per_example = [], []
    for i, d in enumerate(ds):
        out = reply(d["input"])
        s = scorer(d["input"], out, d.get("expected"))
        scored.append({**s, "expected": d.get("expected")})
        per_example.append({"input": d["input"][:160], "output": out[:300],
                            "expected": d.get("expected"), "ok": s["ok"], "score": s["score"], "pred": s.get("pred")})
        if i % 2 == 0 or i == len(ds) - 1:
            set_status(state="running", done=i + 1, total=len(ds),
                       running_acc=round(100 * sum(x["ok"] for x in scored) / len(scored), 1))

    metrics = evalmod.aggregate(ev, scored)
    results = {"target": target, "eval_version": ev_ver, "metrics": metrics,
               "decoding": {"temp": temp, "max_tokens": max_tokens, "mode": "greedy" if temp == 0 else f"temp{temp}"},
               "per_example": per_example, "runtime_s": round(time.time() - t0, 1),
               "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    write_json(rd / "results.json", results)
    set_status(state="done", done=len(ds), total=len(ds), metrics=metrics, runtime_s=results["runtime_s"])


if __name__ == "__main__":
    main()
