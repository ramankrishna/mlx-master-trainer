#!/usr/bin/env python
"""Frozen-binary KEYSTONE — the Stage-1/2 proof: the bundled backend runs a REAL train with NO user
Python. Launches the PyInstaller binary as the server, then drives a tiny LoRA train through it. The
train spawns `[binary --run train]` → `[binary --mlx lora]` (the frozen-aware dispatch), so this proves
the whole chain works standalone. Run with any python; it only talks HTTP to the frozen server.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# point the FROZEN server at the dev repo's data dir so it reuses the existing data-keystone fixture
# (the real app defaults to ~/.mlx-master-trainer). Subprocesses inherit this env.
os.environ["MMT_DATA_ROOT"] = str(ROOT)
BIN = ROOT / "desktop" / "dist" / "mmt-backend" / "mmt-backend"
BASE_URL = "http://127.0.0.1:8808"
PROJECT = "data-keystone"          # already has a base + data from the Phase-3 keystone
R = {"checks": {}}


def check(name, cond, detail=""):
    R["checks"][name] = {"pass": bool(cond), "detail": str(detail)[:160]}
    print(f"[{'PASS' if cond else 'FAIL'}] {name}: {detail}")
    return cond


def get(path):
    return json.loads(urllib.request.urlopen(BASE_URL + path, timeout=10).read())


def post(path, body):
    req = urllib.request.Request(BASE_URL + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    return json.loads(urllib.request.urlopen(req, timeout=30).read())


def main():
    subprocess.run(["pkill", "-9", "-f", "dist/mmt-backend/mmt-backend"], capture_output=True)
    subprocess.run("lsof -ti:8808 | xargs kill -9", shell=True, capture_output=True)
    time.sleep(1)
    check("binary_exists", BIN.exists(), str(BIN))

    # 1) selftest — real MLX op in the frozen binary, no dev venv
    st = subprocess.run([str(BIN), "--selftest"], capture_output=True, text=True, timeout=120)
    ok_line = next((l for l in st.stdout.splitlines() if "SELFTEST OK" in l), "")
    check("selftest_mlx_op", "SELFTEST OK" in st.stdout and "matmul sum=4096.0" in st.stdout, ok_line)

    # 2) launch the frozen binary as the SERVER (default branch)
    log = open("/tmp/mmt_frozen_server.log", "w")
    svr = subprocess.Popen([str(BIN)], stdout=log, stderr=subprocess.STDOUT)
    health = None
    for _ in range(45):
        try:
            health = get("/health")
            break
        except Exception:
            time.sleep(1)
    check("frozen_server_boots", bool(health and health.get("ok")), f"/health -> {health}")

    # 3) a REAL train THROUGH the frozen backend (no user Python anywhere)
    ver = f"frozen-{int(time.time())}"
    started = post("/train/start", {"project": PROJECT, "config": {
        "version": ver, "rank": 8, "num_layers": 4, "iters": 20, "batch_size": 1,
        "max_seq_len": 128, "learning_rate": 2e-4, "steps_per_report": 5}})
    state = {}
    if started.get("ok"):
        for _ in range(120):
            state = get(f"/train/status?project={PROJECT}&version={ver}")
            if state.get("state") in ("done", "error", "stopped"):
                break
            time.sleep(2)
    check("frozen_train_e2e", state.get("state") == "done",
          f"started={started.get('ok')} state={state.get('state')} loss={state.get('final_train_loss')}")
    R["frozen_train"] = {"version": ver, "state": state.get("state"),
                         "final_train_loss": state.get("final_train_loss"), "iters": state.get("iter")}

    svr.kill()
    subprocess.run(["pkill", "-9", "-f", "dist/mmt-backend/mmt-backend"], capture_output=True)

    R["size_mb"] = int(subprocess.run(["du", "-sm", str(BIN.parent)], capture_output=True, text=True).stdout.split()[0])
    R["KEYSTONE_PASS"] = all(c["pass"] for c in R["checks"].values())
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "frozen_keystone.json").write_text(json.dumps(R, indent=2))
    print(f"\n=== FROZEN KEYSTONE: {'PASS' if R['KEYSTONE_PASS'] else 'FAIL'} · binary {R['size_mb']}MB ===")
    print(json.dumps({k: v["pass"] for k, v in R["checks"].items()}, indent=2))
    sys.exit(0 if R["KEYSTONE_PASS"] else 1)


if __name__ == "__main__":
    main()
