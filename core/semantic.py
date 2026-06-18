"""Enhancement — the SEMANTIC contamination/dedup/diversity tier (local embeddings).

The n-gram passes (core/data_quality.py) are LEXICAL: a paraphrased eval example slips through and the
user ships an inflated score. This adds a SEMANTIC tier via a LOCAL embedding model (all-MiniLM-L6-v2),
moving the claim from "lexical, misses paraphrase" to "semantic, catches paraphrase — still evidence, not
proof." TIERED: the n-gram pass stays fast + default-on; this is OPT-IN.

PURE-LOCAL — NON-NEGOTIABLE: the embedding model runs ON-DEVICE (sentence-transformers, CPU/MPS). The
model weights download ONCE (~90MB) from the HF hub and are cached; embedding is offline thereafter. NO
cloud embedding API is ever called — user TEXT never leaves the machine. (Only the model weights are
fetched, one time, exactly like a base model.)

ONE embedding pass → THREE views (all cosine over the same cached vectors of the INPUT text — the question,
which is where train/eval leakage matters):
  1. train/eval semantic contamination  — nearest-pair flagging, REVIEW-not-block (similarity is fuzzy)
  2. in-dataset semantic near-duplicates — clusters of paraphrased examples
  3. semantic diversity                  — are inputs spread across meaning-space or clustered
"""
from __future__ import annotations

import hashlib
import pickle
import subprocess
import sys
from pathlib import Path

from . import eval as evalmod
from . import projects
from .common import ROOT, read_json, runner_argv, write_json

MODEL_NAME = "all-MiniLM-L6-v2"
_MODEL = None


def available() -> bool:
    try:
        import sentence_transformers  # noqa: F401
        return True
    except Exception:
        return False


def _model():
    """Lazy-load the local embedding model (downloads once ~90MB, cached, offline after)."""
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer
        _MODEL = SentenceTransformer(MODEL_NAME)        # on-device; no cloud call
    return _MODEL


def _sha(t: str) -> str:
    return hashlib.sha256(t.encode()).hexdigest()


def embed(texts: list[str], cache_path: Path):
    """Embed texts -> L2-normalized vectors (numpy). Content-hash disk cache so re-runs don't recompute."""
    import numpy as np

    cache = {}
    if cache_path.exists():
        try:
            cache = pickle.loads(cache_path.read_bytes())
        except Exception:
            cache = {}
    missing = [t for t in dict.fromkeys(texts) if _sha(t) not in cache]
    if missing:
        vecs = _model().encode(missing, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
        for t, v in zip(missing, vecs):
            cache[_sha(t)] = np.asarray(v, dtype="float32")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(pickle.dumps(cache))
    return np.vstack([cache[_sha(t)] for t in texts])


def _greedy_clusters(vecs, threshold: float) -> list[list[int]]:
    """Assign each row to the first cluster whose representative is within cosine `threshold`, else new."""
    import numpy as np

    clusters, reps = [], []
    for i in range(len(vecs)):
        placed = False
        for c, r in zip(clusters, reps):
            if float(np.dot(vecs[i], vecs[r])) >= threshold:
                c.append(i)
                placed = True
                break
        if not placed:
            clusters.append([i])
            reps.append(i)
    return clusters


def analyze(project: str, eval_version: str, contam_threshold: float = 0.85,
            dup_threshold: float = 0.88, max_n: int = 3000) -> dict:
    """One embedding pass over train + eval INPUT text → the three views. Pure-local."""
    import numpy as np

    data = projects.project_dir(project) / "data"
    raw_p = data / "raw.jsonl"
    if not raw_p.exists():
        return {"ok": False, "error": "no ingested data — prepare a dataset first"}
    import json
    raw = [json.loads(l) for l in raw_p.read_text().splitlines() if l.strip()]
    train_inputs = [r["input"] for r in raw][:max_n]
    truncated = len(raw) > max_n
    ev = evalmod.load_eval(project, eval_version)
    if not ev:
        return {"ok": False, "error": "eval not found"}
    eval_inputs = [d.get("input", "") for d in ev["dataset"]]

    cache = data / ".embcache.pkl"
    vecs = embed(eval_inputs + train_inputs, cache)
    ev_v, tr_v = vecs[:len(eval_inputs)], vecs[len(eval_inputs):]

    def _norm(s):
        return evalmod._norm(s)

    # ---- View 1: train/eval semantic contamination (REVIEW, not block) ----
    sims = ev_v @ tr_v.T                                  # (E, T) cosine (vectors are normalized)
    near_idx = sims.argmax(1)
    near_sim = sims.max(1)
    flagged = []
    for i, (j, s) in enumerate(zip(near_idx, near_sim)):
        if s >= contam_threshold and _norm(eval_inputs[i]) != _norm(train_inputs[j]):   # exclude exact (already lexical)
            flagged.append({"eval_input": eval_inputs[i][:140], "nearest_train": train_inputs[j][:140],
                            "similarity": round(float(s), 3)})
    flagged.sort(key=lambda x: -x["similarity"])
    bins = [(0.5, 0.7), (0.7, 0.8), (0.8, 0.85), (0.85, 0.9), (0.9, 0.95), (0.95, 1.01)]
    dist = {f"{a}-{b if b <= 1 else '1.0'}": int(((near_sim >= a) & (near_sim < b)).sum()) for a, b in bins}

    # ---- View 2: in-dataset semantic near-duplicate clusters ----
    clusters = _greedy_clusters(tr_v, dup_threshold)
    dup_clusters = sorted([c for c in clusters if len(c) > 1], key=len, reverse=True)
    dup_view = {"n_clusters": len(dup_clusters),
                "n_examples_in_dups": sum(len(c) for c in dup_clusters),
                "clusters": [{"size": len(c), "examples": [train_inputs[k][:90] for k in c[:4]]}
                             for c in dup_clusters[:8]]}

    # ---- View 3: semantic diversity ----
    n = len(tr_v)
    if n > 1:
        gram = tr_v @ tr_v.T
        iu = np.triu_indices(n, k=1)
        mean_sim = float(gram[iu].mean())
    else:
        mean_sim = 0.0
    div_clusters = len(_greedy_clusters(tr_v, 0.7))
    diversity = {"mean_pairwise_cosine": round(mean_sim, 3), "approx_clusters": div_clusters, "n": n,
                 "low_diversity": mean_sim > 0.6 or (n >= 20 and div_clusters <= max(3, n // 50))}

    return {
        "ok": True, "method": f"local embeddings ({MODEL_NAME}) — on-device, no cloud", "pure_local": True,
        "n_train": len(train_inputs), "n_eval": len(eval_inputs), "truncated": truncated,
        "contam_threshold": contam_threshold, "dup_threshold": dup_threshold,
        "contamination": {"n_flagged": len(flagged), "flagged": flagged[:50], "distribution": dist,
                          "guidance": "REVIEW these — semantic similarity is fuzzy: a high score can be two "
                          "genuinely different items that read alike, OR a real paraphrase leak. Higher "
                          "threshold = only near-identical; lower = catches looser paraphrase but more false "
                          "flags. If any flagged eval item is a rewording of training data, your score is inflated."},
        "near_duplicates": dup_view,
        "diversity": diversity,
        "honest": "Semantic catches paraphrase the n-gram misses, but is NOT proof of clean: it misses "
                  "semantically-distant leakage (same answer, very different framing), is threshold-sensitive, "
                  "and is only as good as the embedding model. Evidence, not proof.",
    }


# --------------------------------------------------------------------------- #
# detached runner (memory guard: torch loads in a subprocess, freed on exit)
# --------------------------------------------------------------------------- #
def _run_dir(project: str, eval_version: str) -> Path:
    d = evalmod.eval_dir(project, eval_version) / "semantic"
    d.mkdir(parents=True, exist_ok=True)
    return d


def start(project: str, eval_version: str, contam_threshold: float = 0.85, dup_threshold: float = 0.88) -> dict:
    if not available():
        return {"ok": False, "error": "sentence-transformers not installed — `uv pip install sentence-transformers`"}
    rd = _run_dir(project, eval_version)
    job = {"project": project, "eval_version": eval_version,
           "contam_threshold": contam_threshold, "dup_threshold": dup_threshold}
    write_json(rd / "job.json", job)
    write_json(rd / "status.json", {"state": "running"})
    boot = (rd / "boot.log").open("w")
    proc = subprocess.Popen(runner_argv("semantic", "--job", str(rd / "job.json")),
                            stdout=boot, stderr=subprocess.STDOUT, start_new_session=True)
    return {"ok": True, "started": True, "runner_pid": proc.pid}


def status(project: str, eval_version: str) -> dict:
    return read_json(_run_dir(project, eval_version) / "status.json") or {"state": "none"}


def report(project: str, eval_version: str) -> dict | None:
    return read_json(_run_dir(project, eval_version) / "report.json")
