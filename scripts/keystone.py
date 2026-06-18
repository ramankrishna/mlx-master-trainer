#!/usr/bin/env python
"""Phase-1 KEYSTONE — prove the engine generalizes off the bro.

Runs the WHOLE loop on a base that is NOT Qwen-bro (SmolLM2-135M-Instruct, a Llama-family
instruct model) with data that is NOT pre-formatted in bro voice (alpaca-style
instruction/input/output — a DIFFERENT schema). Confirms, in order:
  1. base loads/resolves + info detected (params, quant, family, chat template)
  2. memory pre-check fits the 24GB budget
  3. schema AUTO-DETECTS as 'instruction'
  4. the RIGHT chat template applies (rendered preview shows SmolLM2's <|im_start|> format)
  5. LoRA trains + loss drops
  6. adapter exports (GGUF)
  7. quick inference test produces output

Pass/fail is printed + written to reports/keystone_result.json.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core import adapters, data, models, projects, train          # noqa: E402

BASE = "HuggingFaceTB/SmolLM2-135M-Instruct"      # NOT Qwen-bro; Llama-family; tiny + fast
PROJECT = "keystone"

# Unformatted, NON-bro data in a DIFFERENT schema (alpaca instruction/input/output).
ROWS = [
    {"instruction": "Translate to French", "input": "Good morning", "output": "Bonjour."},
    {"instruction": "Translate to French", "input": "Thank you", "output": "Merci."},
    {"instruction": "Translate to French", "input": "See you tomorrow", "output": "À demain."},
    {"instruction": "What is 7 times 8?", "output": "56."},
    {"instruction": "What is 9 plus 6?", "output": "15."},
    {"instruction": "What is 100 minus 37?", "output": "63."},
    {"instruction": "Give the capital city.", "input": "Japan", "output": "Tokyo."},
    {"instruction": "Give the capital city.", "input": "Canada", "output": "Ottawa."},
    {"instruction": "Give the capital city.", "input": "Egypt", "output": "Cairo."},
    {"instruction": "Give the capital city.", "input": "Brazil", "output": "Brasília."},
    {"instruction": "Summarize in one word.", "input": "A long, detailed history of the Roman empire.",
     "output": "Rome."},
    {"instruction": "Summarize in one word.", "input": "An essay about climate and rising seas.",
     "output": "Climate."},
    {"instruction": "Reverse the word.", "input": "stop", "output": "pots."},
    {"instruction": "Reverse the word.", "input": "live", "output": "evil."},
    {"instruction": "Name the opposite.", "input": "hot", "output": "cold."},
    {"instruction": "Name the opposite.", "input": "up", "output": "down."},
    {"instruction": "Name the opposite.", "input": "fast", "output": "slow."},
    {"instruction": "Convert to uppercase.", "input": "hello world", "output": "HELLO WORLD."},
    {"instruction": "Convert to uppercase.", "input": "mlx trainer", "output": "MLX TRAINER."},
    {"instruction": "How many days in a week?", "output": "Seven."},
    {"instruction": "How many sides does a triangle have?", "output": "Three."},
    {"instruction": "Spell the number.", "input": "4", "output": "four."},
    {"instruction": "Spell the number.", "input": "11", "output": "eleven."},
    {"instruction": "Give the chemical symbol.", "input": "Oxygen", "output": "O."},
]

R = {"base": BASE, "checks": {}}


def check(name, cond, detail=""):
    R["checks"][name] = {"pass": bool(cond), "detail": detail}
    print(f"[{'PASS' if cond else 'FAIL'}] {name}: {detail}")
    return cond


def main():
    t0 = time.time()
    # 1) base resolves + info
    print(f"=== resolving non-bro base: {BASE} ===")
    path = models.ensure_local(BASE, progress=print)
    info = models.model_info(str(path))
    check("base_loads", (path / "config.json").exists(),
          f"{info['params_str']} params · {info['family']} · quant={info['quantization']} · template={info['has_chat_template']}")

    # 2) memory pre-check
    pre = models.memory_precheck(info, batch_size=1, max_seq_len=1024, num_layers=4)
    check("memory_fits_24gb", pre["fits"],
          f"est_peak {pre['est_peak_gb']}GB / ram {pre['ram_gb']}GB (headroom {pre['headroom_gb']}GB)")

    # 3) schema auto-detect
    det = data.detect_schema(ROWS)
    check("schema_autodetect", det["schema"] == "instruction",
          f"detected '{det['schema']}' mapping={det['mapping']}")

    # 4) RIGHT chat template applied (rendered preview)
    val = data.validate(ROWS, det["schema"], det["mapping"], str(path), max_seq_len=1024)
    prev = data.preview(ROWS, det["schema"], det["mapping"], str(path), idx=0)
    rendered = prev.get("rendered", "")
    tmpl_ok = ("<|im_start|>" in rendered and "<|im_end|>" in rendered and val["training_format"] == "messages")
    check("correct_template", tmpl_ok,
          f"format={val['training_format']} · preview head: {rendered[:80]!r}")
    R["rendered_preview"] = rendered
    R["validation"] = {k: val[k] for k in ("n_usable", "training_format", "token_len", "n_over_cap", "warnings")}

    # 5) prepare + train (loss must drop)
    projects.create_project(PROJECT)
    projects.set_base(PROJECT, {"ref": BASE, "local_path": str(path), "info": info})
    out = projects.project_dir(PROJECT) / "data"
    data.prepare_dataset(ROWS, det["schema"], det["mapping"], str(path), str(out), val_frac=0.15, seed=0)
    version = f"keystone-{int(time.time())}"
    cfg = {"version": version, "rank": 8, "scale": 16.0, "num_layers": 4, "iters": 40,
           "batch_size": 1, "max_seq_len": 512, "learning_rate": 2e-4, "steps_per_report": 5, "seed": 0}
    started = train.start(PROJECT, cfg)
    print("train started:", started)
    if not started.get("ok"):
        check("training_runs", False, started.get("error", "start failed"))
        return finish(t0)
    # poll
    st = {}
    for _ in range(600):
        st = train.status(PROJECT, version)
        if st.get("state") in ("done", "error", "stopped"):
            break
        time.sleep(2)
    hist = st.get("history", [])
    first = next((h["train_loss"] for h in hist if h.get("train_loss") is not None), None)
    last = st.get("final_train_loss") or (hist[-1]["train_loss"] if hist else None)
    dropped = first is not None and last is not None and last < first
    check("training_runs", st.get("state") == "done", f"state={st.get('state')} runtime={st.get('runtime_s')}s")
    check("loss_drops", dropped, f"train loss {first} -> {last}")
    R["loss_curve"] = {"first": first, "last": last, "points": len(hist)}

    # 6) export GGUF
    exp = adapters.export(PROJECT, version, "gguf", progress=print)
    check("export_gguf", exp.get("ok"), exp.get("path") or exp.get("error", "")[:120])

    # 7) quick inference
    inf = adapters.quick_infer(PROJECT, version, "Give the capital city.", system=None, max_tokens=24)
    check("inference_test", inf.get("ok"), repr((inf.get("output") or inf.get("error", ""))[:120]))
    R["inference_output"] = inf.get("output")

    finish(t0, version)


def finish(t0, version=None):
    R["version"] = version
    R["runtime_s"] = round(time.time() - t0, 1)
    R["KEYSTONE_PASS"] = all(c["pass"] for c in R["checks"].values())
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "keystone_result.json").write_text(json.dumps(R, indent=2))
    print("\n=== KEYSTONE:", "PASS" if R["KEYSTONE_PASS"] else "FAIL", f"({R['runtime_s']}s) ===")
    print(json.dumps({k: v["pass"] for k, v in R["checks"].items()}, indent=2))


if __name__ == "__main__":
    main()
