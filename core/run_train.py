#!/usr/bin/env python
"""Detached training runner — does the actual mlx_lm.lora pass and streams live status.

Launched by train.start() as its own process (the memory guard: peak reclaimed on exit; the
backend never holds the optimizer). Writes:
  - config.yaml     the exact mlx_lm config used
  - run.log         full training stdout
  - status.json     live {state, iter, train_loss, val_loss, tokens_per_sec, eta_s, history}
  - manifest.json   reproducibility record (config + seed + data_hash + base + final loss)
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import models, train as traincfg          # noqa: E402
from core.common import data_hash, read_json, write_json    # noqa: E402

ITER = re.compile(r"Iter\s+(\d+):")
TRAIN = re.compile(r"Train loss\s+([\d.]+)")
VAL = re.compile(r"Val loss\s+([\d.]+)")
TOKS = re.compile(r"Tokens/sec\s+([\d.]+)")
ITS = re.compile(r"It/sec\s+([\d.]+)")
MEM = re.compile(r"Peak mem\s+([\d.]+)\s*GB")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    job = read_json(ap.parse_args().job)
    adir = Path(job["adapter_dir"])
    cfg = job["cfg"]
    total = int(cfg["iters"])
    status_path = adir / "status.json"

    def set_status(**kw):
        cur = read_json(status_path) or {}
        cur.update(kw)
        write_json(status_path, cur)

    # 1) memory pre-check (so a too-big run is flagged, not silently OOM'd mid-train)
    try:
        info = models.model_info(job["base_path"])
        pre = models.memory_precheck(info, cfg["batch_size"], cfg["max_seq_len"],
                                     cfg["num_layers"], cfg.get("grad_checkpoint", False))
    except Exception as e:
        pre, info = {"error": str(e)[:200]}, {}
    set_status(state="preparing", version=job["version"], total_iters=total,
               memory=pre, base=info.get("params_str"), family=info.get("family"), history=[])

    # 2) write the mlx_lm lora config
    import yaml
    ycfg = traincfg.build_lora_yaml(job["base_path"], job["data_dir"], adir / "adapter", cfg)
    (adir / "config.yaml").write_text(yaml.safe_dump(ycfg, sort_keys=False))

    # 3) run mlx_lm.lora, stream stdout -> run.log + status.json
    log = (adir / "run.log").open("w")
    t0 = time.time()
    proc = subprocess.Popen([sys.executable, "-m", "mlx_lm.lora", "-c", str(adir / "config.yaml")],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    set_status(state="running", train_pid=proc.pid)
    history, last_train, last_val, last_toks, last_mem = [], None, None, None, None
    for line in proc.stdout:
        log.write(line)
        log.flush()
        mi = ITER.search(line)
        if not mi:
            continue
        it = int(mi.group(1))
        mt, mv, mk, ms, mm = TRAIN.search(line), VAL.search(line), TOKS.search(line), ITS.search(line), MEM.search(line)
        if mt:
            last_train = float(mt.group(1))
        if mv:
            last_val = float(mv.group(1))
        if mk:
            last_toks = float(mk.group(1))
        if mm:
            last_mem = float(mm.group(1))
        eta = round((total - it) / float(ms.group(1)), 1) if ms and float(ms.group(1)) > 0 else None
        point = {"iter": it, "train_loss": last_train, "val_loss": last_val}
        if mt or mv:
            history.append(point)
        set_status(state="running", iter=it, total_iters=total, train_loss=last_train,
                   val_loss=last_val, tokens_per_sec=last_toks, peak_mem_gb=last_mem,
                   eta_s=eta, elapsed_s=round(time.time() - t0, 1), history=history[-200:])
    rc = proc.wait()
    log.close()

    # 4) finalize: manifest (reproducibility) + terminal status
    adapter_file = adir / "adapter" / "adapters.safetensors"
    ok = adapter_file.exists()
    dh = data_hash([Path(job["data_dir"]) / "train.jsonl", Path(job["data_dir"]) / "valid.jsonl"])
    manifest = {
        "version": job["version"], "base": {"ref": (info or {}).get("ref", job["base_path"]),
                                            "params": (info or {}).get("params_str"),
                                            "family": (info or {}).get("family")},
        "config": cfg, "lora_yaml": ycfg, "seed": cfg["seed"], "data_hash": dh,
        "data": read_json(Path(job["data_dir"]) / "ingest.json"),
        "final_train_loss": last_train, "final_val_loss": last_val,
        "iters_completed": history[-1]["iter"] if history else 0, "returncode": rc,
        "runtime_s": round(time.time() - t0, 1), "ok": ok,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "provenance": "mlx-master-trainer (generalized from bigbugai-bro studio retrain)",
    }
    write_json(adir / "manifest.json", manifest)
    state = "done" if ok else ("stopped" if rc and history else "error")
    set_status(state=state, ok=ok, final_train_loss=last_train, final_val_loss=last_val,
               runtime_s=manifest["runtime_s"], returncode=rc)


if __name__ == "__main__":
    main()
