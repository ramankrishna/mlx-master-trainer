"""Generalization #2 — model-agnostic data ingest.

The bro studio assumed pre-formatted, in-voice {"prompt","response"} JSONL. Here we accept the
common SFT schemas, AUTO-DETECT which one, normalize every schema to chat `messages`, and — the
part naive trainers get wrong — apply the SELECTED BASE'S chat template and show the user the
RENDERED example (exactly what the model will see). Token-count validation flags truncation.

PURE-LOCAL: every function here reads the local filesystem only. No network call touches user
data (the only model download is the base weights, handled in models.py before this runs).
"""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path

# ---- schema detection -------------------------------------------------------
def load_rows(text_or_path: str) -> list[dict]:
    """Accept a path to .jsonl/.json/.csv, or raw JSONL text. Returns list of dict rows."""
    text = text_or_path
    # Treat the input as a filesystem path ONLY if it's plausibly one: a single line within the OS
    # filename limit. Pasted JSONL is multi-line / long, and Path.exists() would raise OSError
    # ([Errno 63] File name too long) on it — so guard before ever touching the filesystem.
    s = text_or_path.strip()
    if s and "\n" not in s and len(s) <= 1024:
        try:
            p = Path(s).expanduser()
            if p.exists() and p.is_file():
                raw = p.read_text()
                if p.suffix.lower() == ".csv":
                    return list(csv.DictReader(io.StringIO(raw)))
                if p.suffix.lower() == ".json":
                    obj = json.loads(raw)
                    return obj if isinstance(obj, list) else [obj]
                text = raw                          # .jsonl
        except OSError:
            pass                                    # not a usable path — fall through to raw text
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            pass
    return rows


def detect_schema(rows: list[dict]) -> dict:
    """Identify the SFT schema + a concrete column mapping the user can confirm/override."""
    if not rows:
        return {"schema": "unknown", "mapping": {}, "fields": []}
    keys = set().union(*[set(r.keys()) for r in rows[:50]])
    r0 = rows[0]

    def has(k):
        return k in keys

    if has("messages") and isinstance(r0.get("messages"), list):
        return {"schema": "chat", "mapping": {"messages": "messages"}, "fields": sorted(keys)}
    if has("instruction") and (has("output") or has("response")):
        return {"schema": "instruction",
                "mapping": {"instruction": "instruction",
                            "input": "input" if has("input") else None,
                            "output": "output" if has("output") else "response"},
                "fields": sorted(keys)}
    out_key = "completion" if has("completion") else ("response" if has("response") else None)
    in_key = "prompt" if has("prompt") else ("input" if has("input") else None)
    if in_key and out_key:
        return {"schema": "prompt_completion",
                "mapping": {"prompt": in_key, "completion": out_key}, "fields": sorted(keys)}
    if has("text"):
        return {"schema": "text", "mapping": {"text": "text"}, "fields": sorted(keys)}
    return {"schema": "unknown", "mapping": {}, "fields": sorted(keys)}


# ---- normalization to chat messages ----------------------------------------
def normalize_row(row: dict, schema: str, mapping: dict, system: str | None = None):
    """Any schema -> chat messages list (or {'text': ...} for the raw-text schema)."""
    m = mapping
    if schema == "chat":
        msgs = row.get(m.get("messages", "messages")) or []
        return [{"role": x.get("role", "user"), "content": str(x.get("content", ""))} for x in msgs]
    if schema == "text":
        return {"text": str(row.get(m.get("text", "text"), ""))}
    if schema == "instruction":
        instr = str(row.get(m["instruction"], "")).strip()
        inp = str(row.get(m["input"], "")).strip() if m.get("input") else ""
        user = instr + (("\n\n" + inp) if inp else "")
        assistant = str(row.get(m["output"], "")).strip()
    else:  # prompt_completion
        user = str(row.get(m["prompt"], "")).strip()
        assistant = str(row.get(m["completion"], "")).strip()
    msgs = ([{"role": "system", "content": system}] if system else []) + \
           [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]
    return msgs


def _is_empty(norm) -> bool:
    if isinstance(norm, dict):                       # text schema
        return not norm.get("text", "").strip()
    if not norm:
        return True
    roles = {x["role"] for x in norm}
    if "user" not in roles or "assistant" not in roles:
        return True
    return any(not x["content"].strip() for x in norm if x["role"] in ("user", "assistant"))


# ---- tokenizer (loaded once for batch ops) ---------------------------------
_TOK_CACHE: dict = {}


def _tokenizer(ref_or_path: str):
    if ref_or_path not in _TOK_CACHE:
        from transformers import AutoTokenizer
        _TOK_CACHE[ref_or_path] = AutoTokenizer.from_pretrained(str(Path(ref_or_path).expanduser()))
    return _TOK_CACHE[ref_or_path]


def _render(tok, norm) -> str:
    if isinstance(norm, dict):
        return norm.get("text", "")
    if tok.chat_template is None:
        return "\n".join(f"{x['role']}: {x['content']}" for x in norm)
    return tok.apply_chat_template(norm, tokenize=False, add_generation_prompt=False)


# ---- validation -------------------------------------------------------------
def validate(rows: list[dict], schema: str, mapping: dict, base_local_path: str,
             max_seq_len: int = 2048, val_frac: float = 0.1, system: str | None = None) -> dict:
    """Empty-field check, char + TOKEN length distribution (templated through the base), how many
    exceed the context cap (they'll truncate), and the train/val split counts."""
    norms = [normalize_row(r, schema, mapping, system) for r in rows]
    n_empty = sum(_is_empty(n) for n in norms)
    valid = [n for n in norms if not _is_empty(n)]

    tok = _tokenizer(base_local_path)
    has_template = tok.chat_template is not None
    tok_lens, char_lens = [], []
    for n in valid:
        rendered = _render(tok, n)
        char_lens.append(len(rendered))
        tok_lens.append(len(tok(rendered, add_special_tokens=False)["input_ids"]))
    over = sum(1 for t in tok_lens if t > max_seq_len)

    buckets = [(0, 128), (128, 256), (256, 512), (512, 1024), (1024, 2048), (2048, 10**9)]
    hist = {f"{a}-{b if b < 10**9 else '∞'}": sum(1 for t in tok_lens if a <= t < b) for a, b in buckets}

    n_val = max(1, int(len(valid) * val_frac)) if len(valid) > 1 else 0
    fmt = ("text" if schema == "text" else ("messages" if has_template else "prompt_completion"))
    warnings = []
    if not has_template and schema != "text":
        warnings.append("base has NO chat template — falling back to prompt/completion (no role formatting)")
    if over:
        warnings.append(f"{over} example(s) exceed max_seq_length={max_seq_len} and WILL truncate")
    if n_empty:
        warnings.append(f"{n_empty} example(s) dropped (empty user/assistant or missing roles)")
    if len(valid) < 8:
        warnings.append(f"only {len(valid)} usable examples — LoRA wants more for a real signal")

    def stat(xs):
        if not xs:
            return {"min": 0, "median": 0, "max": 0}
        s = sorted(xs)
        return {"min": s[0], "median": s[len(s) // 2], "max": s[-1]}

    return {"schema": schema, "mapping": mapping, "n_total": len(rows), "n_usable": len(valid),
            "n_empty": n_empty, "has_chat_template": has_template, "training_format": fmt,
            "token_len": stat(tok_lens), "char_len": stat(char_lens), "token_histogram": hist,
            "n_over_cap": over, "max_seq_len": max_seq_len,
            "n_train": len(valid) - n_val, "n_val": n_val, "warnings": warnings}


def preview(rows: list[dict], schema: str, mapping: dict, base_local_path: str,
            system: str | None = None, idx: int = 0) -> dict:
    """The rendered, post-template example #idx — exactly the string the model trains on."""
    norms = [normalize_row(r, schema, mapping, system) for r in rows]
    valid = [n for n in norms if not _is_empty(n)]
    if not valid:
        return {"error": "no usable examples"}
    n = valid[min(idx, len(valid) - 1)]
    tok = _tokenizer(base_local_path)
    rendered = _render(tok, n)
    return {"index": min(idx, len(valid) - 1), "n_usable": len(valid),
            "messages": n if not isinstance(n, dict) else None,
            "has_chat_template": tok.chat_template is not None, "rendered": rendered,
            "n_tokens": len(tok(rendered, add_special_tokens=False)["input_ids"])}


# ---- write the training files ----------------------------------------------
def prepare_dataset(rows: list[dict], schema: str, mapping: dict, base_local_path: str,
                    out_dir: str, val_frac: float = 0.1, system: str | None = None,
                    seed: int = 0) -> dict:
    """Write train.jsonl / valid.jsonl in the format mlx_lm expects (chat `messages` when the base
    has a template, else prompt/completion, else raw text). mlx_lm applies the base template itself,
    so the templating is provably the base's own. Returns the format + counts + ingest record."""
    import random

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tok = _tokenizer(base_local_path)
    has_template = tok.chat_template is not None
    norms = [normalize_row(r, schema, mapping, system) for r in rows]
    valid = [n for n in norms if not _is_empty(n)]
    random.Random(seed).shuffle(valid)
    n_val = max(1, int(len(valid) * val_frac)) if len(valid) > 1 else 0

    def to_record(n):
        if isinstance(n, dict):                      # text schema
            return {"text": n["text"]}
        if has_template:
            return {"messages": n}                   # mlx_lm applies the base's chat template
        # no template: concatenate into prompt/completion (loss on completion only)
        user = "\n".join(x["content"] for x in n if x["role"] in ("system", "user"))
        asst = "\n".join(x["content"] for x in n if x["role"] == "assistant")
        return {"prompt": user, "completion": asst}

    train, valr = valid[n_val:], valid[:n_val]
    with (out / "train.jsonl").open("w") as f:
        for n in train:
            f.write(json.dumps(to_record(n)) + "\n")
    with (out / "valid.jsonl").open("w") as f:
        for n in (valr or train[:1]):                # mlx_lm needs a non-empty valid set
            f.write(json.dumps(to_record(n)) + "\n")

    # raw.jsonl — the FULL pre-filter normalized set (input/output/record) that Phase-3 filtering
    # operates on. train/valid above are "everything"; the filter narrows them to the kept subset.
    with (out / "raw.jsonl").open("w") as f:
        for n in valid:
            if isinstance(n, dict):
                inp, outp = "", n.get("text", "")
            else:
                inp = " ".join(x["content"] for x in n if x["role"] == "user")
                outp = " ".join(x["content"] for x in n if x["role"] == "assistant")
            f.write(json.dumps({"input": inp, "output": outp, "record": to_record(n)}) + "\n")

    fmt = "text" if schema == "text" else ("messages" if has_template else "prompt_completion")
    ingest = {"schema": schema, "mapping": mapping, "training_format": fmt, "system": system,
              "n_train": len(train), "n_val": len(valr or train[:1]), "seed": seed,
              "has_chat_template": has_template}
    (out / "ingest.json").write_text(json.dumps(ingest, indent=2))
    return ingest
