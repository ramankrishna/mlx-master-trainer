"""MLX Master Trainer — localhost backend (FastAPI). The Tauri/menubar webview hits this.
Thin HTTP shell over the model-agnostic engine in core/. Bound to 127.0.0.1 only. Pure-local:
the only network egress is the base-model download (core/models); user DATA never leaves the Mac.

Reuses the bro studio's server shape (same-origin webview, no-store index, /assets mount).
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core import adapters, data, models, projects, train          # noqa: E402
from core import eval as evalmod, eval_quality, run_eval, prereg, gate   # noqa: E402  (Phase 2 — discipline)
from core import filter as filtermod, data_quality                      # noqa: E402  (Phase 3 — data)
from core import semantic                                               # noqa: E402  (enhancement — semantic tier)
from core.common import system_ram_gb, token_status, set_token    # noqa: E402

app = FastAPI(title="MLX Master Trainer")
FRONTEND = ROOT / "frontend"
PORT = 8808

# in-process status for the (potentially slow) base prepare/convert/download job
_PREPARE: dict = {"state": "idle"}


@app.get("/health")
def health():
    return {"ok": True, "app": "MLX Master Trainer", "phase": 1, "ram_gb": system_ram_gb(),
            "pure_local": True}


# --------------------------------------------------------------------------- #
# settings (optional HF token for gated/private bases)
# --------------------------------------------------------------------------- #
class Settings(BaseModel):
    hf_token: str | None = None


@app.get("/settings")
def get_settings():
    return {"hf": token_status(), "ram_gb": system_ram_gb()}


@app.post("/settings")
def post_settings(s: Settings):
    hf = set_token(s.hf_token) if s.hf_token is not None else token_status()
    return {"hf": hf}


# --------------------------------------------------------------------------- #
# models — generalization #1
# --------------------------------------------------------------------------- #
class PrepareReq(BaseModel):
    ref: str
    convert: bool = False
    quantize: bool = False
    q_bits: int = 4


@app.get("/models/suggested")
def models_suggested():
    return {"models": models.SUGGESTED_BASES}


@app.get("/models/local")
def models_local():
    from core.common import MODELS_CACHE
    out = []
    for d in sorted(MODELS_CACHE.glob("*")):
        if (d / "config.json").exists():
            out.append({"ref": str(d), "name": d.name})
    return {"models": out}


@app.post("/models/prepare")
def models_prepare(r: PrepareReq):
    global _PREPARE
    if _PREPARE.get("state") == "running":
        return {"started": False, "error": "a prepare is already running"}
    _PREPARE = {"state": "running", "ref": r.ref, "log": []}

    def run():
        def prog(m):
            _PREPARE.setdefault("log", []).append(m)
        try:
            path = models.convert_to_mlx(r.ref, r.quantize, r.q_bits, progress=prog) if r.convert \
                else models.ensure_local(r.ref, progress=prog)
            info = models.model_info(str(path))
            _PREPARE.update({"state": "done", "path": str(path), "info": info})
        except Exception as e:
            _PREPARE.update({"state": "error", "error": str(e)[:400]})

    threading.Thread(target=run, daemon=True).start()
    return {"started": True, "ref": r.ref}


@app.get("/models/prepare/status")
def models_prepare_status():
    return _PREPARE


class TemplateReq(BaseModel):
    ref: str
    messages: list | None = None


@app.post("/models/template")
def models_template(r: TemplateReq):
    msgs = r.messages or [{"role": "user", "content": "What is the capital of France?"},
                          {"role": "assistant", "content": "Paris."}]
    return models.chat_template_render(r.ref, msgs)


# --------------------------------------------------------------------------- #
# projects
# --------------------------------------------------------------------------- #
class ProjectReq(BaseModel):
    name: str
    base_ref: str | None = None


class SetBaseReq(BaseModel):
    project: str
    ref: str


@app.get("/projects")
def list_projects():
    return {"projects": projects.list_projects()}


@app.post("/projects")
def make_project(r: ProjectReq):
    return projects.create_project(r.name, r.base_ref)


@app.post("/projects/set-base")
def set_base(r: SetBaseReq):
    path = models.ensure_local(r.ref)
    info = models.model_info(str(path))
    info["ref"] = r.ref                      # keep the friendly HF id (not the cache path) for display + manifests
    meta = projects.set_base(r.project, {"ref": r.ref, "local_path": str(path), "info": info})
    return {"project": meta, "info": info}


# --------------------------------------------------------------------------- #
# data — generalization #2
# --------------------------------------------------------------------------- #
class InspectReq(BaseModel):
    project: str
    raw_text: str | None = None
    path: str | None = None
    system: str | None = None
    max_seq_len: int = 2048
    val_frac: float = 0.1
    preview_idx: int = 0


def _base_local(project: str) -> str:
    meta = projects.get_project(project)
    if not meta or not meta.get("base"):
        raise ValueError("set a base model on this project first")
    return meta["base"].get("local_path") or meta["base"]["ref"]


@app.post("/data/inspect")
def data_inspect(r: InspectReq):
    rows = data.load_rows(r.path or r.raw_text or "")
    if not rows:
        return {"error": "no rows parsed — paste JSONL or give a .jsonl/.csv path"}
    det = data.detect_schema(rows)
    if det["schema"] == "unknown":
        return {"detection": det, "error": "could not auto-detect schema — fields: "
                + ", ".join(det["fields"])}
    base = _base_local(r.project)
    val = data.validate(rows, det["schema"], det["mapping"], base, r.max_seq_len, r.val_frac, r.system)
    prev = data.preview(rows, det["schema"], det["mapping"], base, r.system, r.preview_idx)
    return {"detection": det, "validation": val, "preview": prev, "n_rows": len(rows)}


class PrepareDataReq(BaseModel):
    project: str
    raw_text: str | None = None
    path: str | None = None
    schema_name: str
    mapping: dict
    system: str | None = None
    val_frac: float = 0.1
    seed: int = 0


@app.post("/data/prepare")
def data_prepare(r: PrepareDataReq):
    rows = data.load_rows(r.path or r.raw_text or "")
    base = _base_local(r.project)
    out = projects.project_dir(r.project) / "data"
    ingest = data.prepare_dataset(rows, r.schema_name, r.mapping, base, str(out),
                                  r.val_frac, r.system, r.seed)
    return {"ok": True, "ingest": ingest}


# --------------------------------------------------------------------------- #
# memory pre-check + train — generalization #3
# --------------------------------------------------------------------------- #
class PrecheckReq(BaseModel):
    project: str
    batch_size: int = 1
    max_seq_len: int = 2048
    num_layers: int = 16
    grad_checkpoint: bool = False


@app.post("/precheck")
def precheck(r: PrecheckReq):
    info = projects.get_project(r.project)["base"]["info"]
    return models.memory_precheck(info, r.batch_size, r.max_seq_len, r.num_layers, r.grad_checkpoint)


class TrainReq(BaseModel):
    project: str
    config: dict


@app.post("/train/start")
def train_start(r: TrainReq):
    return train.start(r.project, r.config)


@app.get("/train/status")
def train_status(project: str, version: str | None = None):
    return train.status(project, version)


class StopReq(BaseModel):
    project: str
    version: str | None = None


@app.post("/train/stop")
def train_stop(r: StopReq):
    return train.stop(r.project, r.version)


# --------------------------------------------------------------------------- #
# adapters — manage, export, quick infer
# --------------------------------------------------------------------------- #
@app.get("/adapters")
def list_adapters(project: str):
    return {"adapters": adapters.list_adapters(project)}


class ExportReq(BaseModel):
    project: str
    version: str
    fmt: str = "fused"


@app.post("/export")
def export(r: ExportReq):
    return adapters.export(r.project, r.version, r.fmt)


class InferReq(BaseModel):
    project: str
    version: str
    prompt: str
    system: str | None = None
    max_tokens: int = 200
    temp: float = 0.0


@app.post("/infer")
def infer(r: InferReq):
    return adapters.quick_infer(r.project, r.version, r.prompt, r.system, r.max_tokens, r.temp)


# --------------------------------------------------------------------------- #
# Phase 2 — the discipline layer (eval builder · quality guard · baseline · prereg · gate)
# --------------------------------------------------------------------------- #
class EvalCreate(BaseModel):
    project: str
    name: str
    kind: str = "template"          # template | examples | code
    template: str | None = None
    spec: dict = {}
    dataset: list = []
    scorer_code: str | None = None
    max_tokens: int = 64


@app.get("/eval/templates")
def eval_templates():
    return {"templates": evalmod.TEMPLATE_TYPES}


@app.post("/eval/create")
def eval_create(r: EvalCreate):
    spec = dict(r.spec or {})
    if r.kind == "template" and r.template:
        spec["template"] = r.template
    return evalmod.create_eval(r.project, r.name, r.kind, r.dataset, spec=spec,
                               scorer_code=r.scorer_code, max_tokens=r.max_tokens)


@app.get("/eval/list")
def eval_list(project: str):
    return {"evals": evalmod.list_evals(project)}


class AuditReq(BaseModel):
    project: str
    eval_version: str
    proposed_criteria: list | None = None


@app.post("/eval/audit")
def eval_audit(r: AuditReq):
    base = run_eval.get_baseline(r.project, r.eval_version)
    return eval_quality.audit(r.project, r.eval_version, proposed_criteria=r.proposed_criteria, baseline=base)


class EvalRun(BaseModel):
    project: str
    eval_version: str
    target: str = "base"            # 'base' = the forced baseline, else an adapter version
    temp: float = 0.0
    max_tokens: int | None = None


@app.post("/eval/run")
def eval_run(r: EvalRun):
    return run_eval.start_eval(r.project, r.eval_version, r.target, r.temp, r.max_tokens)


@app.get("/eval/status")
def eval_run_status(project: str, eval_version: str, target: str = "base"):
    return run_eval.eval_status(project, eval_version, target)


@app.get("/eval/results")
def eval_run_results(project: str, eval_version: str, target: str = "base"):
    return run_eval.run_results(project, eval_version, target) or {"error": "no results"}


@app.get("/pipeline")
def pipeline(project: str, eval_version: str | None = None):
    return prereg.pipeline_state(project, eval_version)


class CommitReq(BaseModel):
    project: str
    eval_version: str
    criteria: list
    override_reason: str | None = None


@app.post("/prereg/commit")
def prereg_commit(r: CommitReq):
    return prereg.commit(r.project, r.eval_version, r.criteria, r.override_reason)


@app.get("/prereg")
def prereg_get(project: str):
    return {"prereg": prereg.get_prereg(project), "committed": prereg.is_committed(project)}


class ProjReq(BaseModel):
    project: str


@app.post("/prereg/clear")
def prereg_clear(r: ProjReq):
    return prereg.clear(r.project)


class GateReq(BaseModel):
    project: str
    adapter_version: str


@app.post("/gate/verdict")
def gate_verdict(r: GateReq):
    return gate.verdict(r.project, r.adapter_version)


class DecideReq(BaseModel):
    project: str
    adapter_version: str
    action: str                     # 'keep' | 'discard'
    reason: str | None = None


@app.post("/gate/decide")
def gate_decide(r: DecideReq):
    return gate.decide(r.project, r.adapter_version, r.action, r.reason)


@app.get("/gate")
def gate_get(project: str, adapter_version: str):
    return gate.get_gate(project, adapter_version) or {"error": "no gate yet"}


# --------------------------------------------------------------------------- #
# Phase 3 — data prep + the strict-basis filter (the productized acceptance-rule move)
# --------------------------------------------------------------------------- #
class FilterReq(BaseModel):
    project: str
    rules: list
    source_label: str = "independent"
    val_frac: float = 0.1


@app.post("/filter/apply")
def filter_apply(r: FilterReq):
    return filtermod.apply(r.project, r.rules, r.source_label, r.val_frac)


@app.get("/filter/report")
def filter_report(project: str):
    return filtermod.report(project) or {"error": "no filter run yet"}


@app.get("/filter/rejected")
def filter_rejected(project: str):
    return {"rejected": filtermod.rejected(project)}


class RescueReq(BaseModel):
    project: str
    idx: int
    reason: str


@app.post("/filter/rescue")
def filter_rescue(r: RescueReq):
    return filtermod.rescue(r.project, r.idx, r.reason)


@app.get("/data/quality")
def data_quality_audit(project: str, max_seq_len: int = 2048):
    return data_quality.audit(project, max_seq_len)


@app.get("/data/contamination-warning")
def contam_warning():
    return data_quality.lexical_warning()


class NearDupReq(BaseModel):
    project: str
    eval_version: str
    threshold: float = 0.7


@app.post("/data/near-dup")
def near_dup(r: NearDupReq):
    return data_quality.near_dup_pass(r.project, r.eval_version, r.threshold)


# --------------------------------------------------------------------------- #
# Enhancement — the SEMANTIC tier (local embeddings, opt-in, pure-local)
# --------------------------------------------------------------------------- #
@app.get("/semantic/available")
def semantic_available():
    return {"available": semantic.available(), "model": semantic.MODEL_NAME}


class SemanticReq(BaseModel):
    project: str
    eval_version: str
    contam_threshold: float = 0.85
    dup_threshold: float = 0.88


@app.post("/semantic/run")
def semantic_run(r: SemanticReq):
    return semantic.start(r.project, r.eval_version, r.contam_threshold, r.dup_threshold)


@app.get("/semantic/status")
def semantic_status(project: str, eval_version: str):
    return semantic.status(project, eval_version)


@app.get("/semantic/report")
def semantic_report(project: str, eval_version: str):
    return semantic.report(project, eval_version) or {"error": "no semantic run yet"}


# --------------------------------------------------------------------------- #
# static frontend (no-store so the webview never serves a stale UI)
# --------------------------------------------------------------------------- #
@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse((FRONTEND / "index.html").read_text(),
                        headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"})


if (FRONTEND / "assets").exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND / "assets")), name="assets")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
