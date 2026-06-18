"""Generalization #1 — model-agnostic base selection.

The bro studio assumed ONE base (sft/merged). Here the user picks ANY MLX-loadable base by
HF repo id (mlx-community/* or a convertible HF model) or a local path. We:
  - materialize it locally (snapshot_download), optionally convert HF->MLX (mlx_lm.convert),
  - detect + surface param count, quantization, FAMILY, and the chat TEMPLATE (the #1 thing
    naive trainers get wrong — different bases template differently; we show what's applied),
  - estimate peak memory vs the machine's RAM and WARN before an OOM, not after.

Heavy imports (transformers / huggingface_hub) are lazy so the module loads without them.
"""
from __future__ import annotations

import json
from pathlib import Path

from .common import MODELS_CACHE, get_token, mlx_argv, sanitize_ref, system_ram_gb

# Curated MLX-friendly instruct bases (suggestions only — any HF id / local path works).
SUGGESTED_BASES = [
    {"ref": "HuggingFaceTB/SmolLM2-135M-Instruct", "size": "135M", "note": "tiny, fast — good first run / keystone"},
    {"ref": "HuggingFaceTB/SmolLM2-360M-Instruct", "size": "360M", "note": "small, still quick"},
    {"ref": "mlx-community/Qwen2.5-0.5B-Instruct-4bit", "size": "0.5B", "note": "native MLX 4-bit"},
    {"ref": "mlx-community/Llama-3.2-1B-Instruct-4bit", "size": "1B", "note": "native MLX 4-bit"},
    {"ref": "mlx-community/Qwen2.5-1.5B-Instruct-4bit", "size": "1.5B", "note": "native MLX 4-bit"},
    {"ref": "mlx-community/gemma-2-2b-it-4bit", "size": "2B", "note": "native MLX 4-bit"},
    {"ref": "mlx-community/Phi-3.5-mini-instruct-4bit", "size": "3.8B", "note": "fits 24GB at 4-bit"},
    {"ref": "mlx-community/Mistral-7B-Instruct-v0.3-4bit", "size": "7B", "note": "tight on 24GB — 4-bit only"},
]

# Per-family LoRA target modules (suggested defaults; mlx_lm auto-picks if we pass none).
_ATTN = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"]
_MLP = ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
_FAMILY_TARGETS = {
    "llama": _ATTN, "qwen2": _ATTN, "qwen2_moe": _ATTN, "mistral": _ATTN, "smollm": _ATTN,
    "phi": _ATTN, "phi3": _ATTN, "gemma": _ATTN, "gemma2": _ATTN, "starcoder2": _ATTN,
}


def is_local(ref: str) -> bool:
    p = Path(ref).expanduser()
    return p.exists() and (p / "config.json").exists()


def ensure_local(ref: str, progress=lambda m: None) -> Path:
    """Return a local directory for the base. Local path -> as-is; HF id -> snapshot_download
    (cached). Pure-local thereafter: training never re-hits the network for weights."""
    if is_local(ref):
        return Path(ref).expanduser()
    from huggingface_hub import snapshot_download
    progress(f"downloading {ref} from HuggingFace …")
    path = snapshot_download(repo_id=ref, token=get_token(),
                             allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model",
                                             "*.jinja", "tokenizer*", "*.py"])
    progress("download complete")
    return Path(path)


def convert_to_mlx(ref: str, quantize: bool = False, q_bits: int = 4,
                   dtype: str = "float16", progress=lambda m: None) -> Path:
    """Optional HF->MLX convert (mlx_lm.convert), cached under models_cache/. Needed only when
    you want a native/quantized MLX copy; mlx_lm can also load most HF safetensors directly."""
    import subprocess
    import sys
    suffix = f"-q{q_bits}" if quantize else "-mlx"
    out = MODELS_CACHE / (sanitize_ref(ref) + suffix)
    if (out / "config.json").exists():
        progress(f"already converted: {out.name}")
        return out
    src = str(ensure_local(ref, progress)) if not is_local(ref) else str(Path(ref).expanduser())
    cmd = mlx_argv("convert", "--hf-path", src, "--mlx-path", str(out), "--dtype", dtype)
    if quantize:
        cmd += ["-q", "--q-bits", str(q_bits)]
    progress(f"converting -> MLX ({'q'+str(q_bits) if quantize else dtype}) …")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if not (out / "config.json").exists():
        raise RuntimeError(f"convert failed: {r.stderr[-500:]}")
    progress("convert complete")
    return out


# --------------------------------------------------------------------------- #
# inspection: params, quant, family, chat template
# --------------------------------------------------------------------------- #
def _weight_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.glob("*.safetensors"))


def _bytes_per_param(cfg: dict) -> float:
    q = cfg.get("quantization") or cfg.get("quantization_config")
    if q:
        bits = q.get("bits", 4)
        return bits / 8 + 0.06          # packed weights + scales/zeros overhead
    dt = (cfg.get("torch_dtype") or "bfloat16").lower()
    return 4.0 if "float32" in dt or dt == "fp32" else 2.0


def model_info(ref_or_path: str) -> dict:
    """Read config + tokenizer config from a LOCAL model dir (call ensure_local first for HF ids).
    Returns params / quant / family / chat-template presence + the LoRA target suggestions."""
    path = Path(ref_or_path).expanduser()
    if not (path / "config.json").exists():
        path = ensure_local(ref_or_path)
    cfg = json.loads((path / "config.json").read_text())
    family = (cfg.get("model_type") or "").lower()
    arch = (cfg.get("architectures") or ["?"])[0]
    wbytes = _weight_bytes(path)
    bpp = _bytes_per_param(cfg)
    params = int(wbytes / bpp) if wbytes and bpp else None
    q = cfg.get("quantization") or cfg.get("quantization_config")
    quant = (f"{q.get('bits', '?')}-bit (group {q.get('group_size', '?')})" if q else "none (fp16/bf16)")

    # chat template: prefer tokenizer_config.json's chat_template, else a chat_template.jinja file.
    tok_cfg = {}
    for fn in ("tokenizer_config.json",):
        if (path / fn).exists():
            tok_cfg = json.loads((path / fn).read_text())
    template = tok_cfg.get("chat_template")
    if not template and (path / "chat_template.jinja").exists():
        template = (path / "chat_template.jinja").read_text()
    has_template = bool(template)

    targets = _FAMILY_TARGETS.get(family, _ATTN)
    return {
        "ref": ref_or_path, "local_path": str(path), "family": family, "architecture": arch,
        "params": params, "params_str": _human(params), "weight_bytes": wbytes,
        "quantization": quant, "is_quantized": bool(q),
        "hidden_size": cfg.get("hidden_size"), "num_layers": cfg.get("num_hidden_layers"),
        "vocab_size": cfg.get("vocab_size"), "max_pos": cfg.get("max_position_embeddings"),
        "has_chat_template": has_template,
        "target_modules_suggested": targets, "target_modules_all": targets + _MLP,
    }


def chat_template_render(ref_or_path: str, messages: list[dict], add_generation_prompt: bool = False) -> dict:
    """Render `messages` through the SELECTED base's chat template (transformers tokenizer, no
    weights loaded). This is exactly what mlx_lm will feed the model — surfaced so a template
    mismatch is visible BEFORE training, not silently baked in."""
    path = Path(ref_or_path).expanduser()
    if not (path / "config.json").exists():
        path = ensure_local(ref_or_path)
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(path))
        if tok.chat_template is None:
            return {"has_template": False, "rendered": None,
                    "note": "this base has NO chat template — training will use raw prompt/"
                            "completion (no role formatting). pick an -instruct base for chat SFT."}
        rendered = tok.apply_chat_template(messages, tokenize=False,
                                           add_generation_prompt=add_generation_prompt)
        n_tokens = len(tok.apply_chat_template(messages, tokenize=True,
                                               add_generation_prompt=add_generation_prompt))
        return {"has_template": True, "rendered": rendered, "n_tokens": n_tokens,
                "eos": tok.eos_token, "bos": tok.bos_token}
    except Exception as e:
        return {"has_template": None, "rendered": None, "error": str(e)[:300]}


# --------------------------------------------------------------------------- #
# memory pre-check (warn before OOM on the 24GB budget — not after)
# --------------------------------------------------------------------------- #
def memory_precheck(info: dict, batch_size: int = 1, max_seq_len: int = 2048,
                    num_layers: int = 16, grad_checkpoint: bool = False) -> dict:
    ram = system_ram_gb()
    weight_gb = (info.get("weight_bytes") or 0) / 1e9
    hidden = info.get("hidden_size") or 2048
    nl = info.get("num_layers") or 24
    layers_trained = nl if num_layers in (-1, None) else min(num_layers, nl)
    # activation memory grows with batch*seq*hidden*layers; grad checkpointing trades it for compute.
    act_gb = batch_size * max_seq_len * hidden * nl * 2 * 2 / 1e9
    if grad_checkpoint:
        act_gb *= 0.35
    opt_gb = 0.3 + layers_trained * 0.01            # LoRA params + adam state (small)
    overhead = 2.0                                  # framework / tokenizer / OS headroom
    est_peak = round(weight_gb + act_gb + opt_gb + overhead, 1)
    fits = ram == 0.0 or est_peak < ram * 0.85
    advice = []
    if not fits:
        advice.append("lower max_seq_length or batch_size")
        advice.append("enable gradient checkpointing")
        if not info.get("is_quantized"):
            advice.append("use a 4-bit (mlx-community/*-4bit) base")
        advice.append(f"reduce num_layers (now {layers_trained})")
    return {"ram_gb": ram, "weight_gb": round(weight_gb, 1), "est_activation_gb": round(act_gb, 1),
            "est_peak_gb": est_peak, "fits": fits,
            "headroom_gb": round(ram * 0.85 - est_peak, 1) if ram else None, "advice": advice}


def _human(n) -> str | None:
    if not n:
        return None
    if n >= 1e9:
        return f"{n / 1e9:.1f}B"
    if n >= 1e6:
        return f"{n / 1e6:.0f}M"
    return str(n)
