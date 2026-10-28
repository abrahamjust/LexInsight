"""LexInsight REST API.

Run from backend/:  uvicorn api.main:app --port 8000
The built React app (frontend/dist) is served at / when present.
"""

import json
import threading
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from lexinsight import config
from lexinsight.engine import DEFAULTS, WEIGHTS, LexInsightEngine, resolve_weights
from lexinsight.evaluation import metrics as M

app = FastAPI(title="LexInsight API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_engine: LexInsightEngine | None = None
_engine_lock = threading.Lock()
_labels_lock = threading.Lock()

SYSTEMS = {
    "adaptive_hybrid": "B7 Intent-adaptive hybrid",
    "fixed_hybrid": "B6 Fixed-weight hybrid",
    "bm25_legal": "B4 BM25 + legal scorer",
    "bm25_semantic": "B5 BM25 + Doc2Vec",
    "bm25": "B1 BM25",
    "doc2vec_chunked": "B3 Doc2Vec (chunked)",
    "doc2vec_whole": "B2 Doc2Vec (whole document)",
    "legal_only": "Legal scorer only",
}


def engine() -> LexInsightEngine:
    global _engine
    with _engine_lock:
        if _engine is None:
            _engine = LexInsightEngine.load()
    return _engine


def read_json(path: Path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


class SearchRequest(BaseModel):
    query: str = Field(min_length=2, max_length=2000)
    system: str = "adaptive_hybrid"
    top_k: int = Field(10, ge=1, le=50)
    intent_override: str | None = None


class LabelUpdate(BaseModel):
    grade: int = Field(ge=0, le=2)
    note: str = ""
    reviewer: str = "project author"


@app.on_event("startup")
def warm():
    threading.Thread(target=engine, daemon=True).start()


@app.get("/api/health")
def health():
    return {"status": "ok", "engine_loaded": _engine is not None}


@app.get("/api/meta")
def meta():
    e = engine()
    selected = read_json(config.INDEX_DIR / "selected_config.json", {})
    return {
        "n_judgments": e.n,
        "intents": config.INTENTS,
        "signals": config.SIGNALS,
        "systems": SYSTEMS,
        "defaults": DEFAULTS,
        "selected_config": selected,
        "adaptive_weights": resolve_weights(DEFAULTS["adaptive"]),
        "fixed_weights": WEIGHTS["fixed"][DEFAULTS["fixed"]],
        "engine_load_seconds": e.load_seconds,
    }


@app.post("/api/search")
def search(req: SearchRequest):
    if req.system not in SYSTEMS:
        raise HTTPException(400, f"Unknown system '{req.system}'")
    if req.intent_override and req.intent_override not in config.INTENTS:
        raise HTTPException(400, "Unknown intent")
    return engine().search(req.query, system=req.system, top_k=req.top_k, intent_override=req.intent_override)


@app.get("/api/cases")
def list_cases(q: str = "", limit: int = Query(50, le=200), offset: int = 0):
    e = engine()
    needle = q.lower().strip()
    rows = []
    for c in e.cases:
        hay = f"{c['case_name']} {c['case_id']} {c['sections'].get('keywords', '')}".lower()
        if needle and needle not in hay:
            continue
        rows.append({"case_id": c["case_id"], "case_name": c["case_name"], "decision_date": c.get("decision_date"),
                     "result": c["sections"].get("result"), "keywords": c["sections"].get("keywords", "")[:200]})
    return {"total": len(rows), "items": rows[offset:offset + limit]}


@app.get("/api/cases/{case_id}")
def get_case(case_id: str):
    e = engine()
    i = e.id_to_index.get(case_id)
    if i is None:
        raise HTTPException(404, "Judgment not found")
    case, feat = e.cases[i], e.features[i]
    return {
        **{k: v for k, v in case.items() if k != "text"},
        "features": {
            "acts": feat["acts"],
            "provisions": dict(sorted(feat["provisions"].items(), key=lambda kv: -(kv[1]["head"] * 10 + kv[1]["body"]))[:40]),
            "concepts": feat["concepts"],
            "citations": feat["citations"],
            "outcome": feat["outcome"],
            "cited_by_in_corpus": [e.cases[j]["case_id"] for j in sorted(e.cited_by.get(i, []))],
            "cites_in_corpus": [e.cases[j]["case_id"] for j in sorted(e.in_corpus_cites.get(i, []))],
        },
    }


@app.get("/api/cases/{case_id}/pdf")
def get_pdf(case_id: str):
    e = engine()
    i = e.id_to_index.get(case_id)
    if i is None:
        raise HTTPException(404, "Judgment not found")
    path = config.ROOT / e.cases[i]["source_pdf"]
    if not path.exists():
        raise HTTPException(404, "PDF not available")
    return FileResponse(path, media_type="application/pdf")


# --------------------------------------------------
# Evaluation results
# --------------------------------------------------


@app.get("/api/evaluation")
def evaluation():
    m = config.EVAL_METRICS_DIR
    figures = sorted(p.name for p in config.EVAL_FIGURES_DIR.glob("*.png")) if config.EVAL_FIGURES_DIR.exists() else []
    return {
        "test_report": read_json(m / "test_report.json"),
        "validation_selection": read_json(m / "validation_selection.json"),
        "intent_classification": read_json(m / "intent_classification.json"),
        "extraction_quality": read_json(m / "extraction_quality.json"),
        "pilot": read_json(m / "pilot_results.json"),
        "figures": figures,
        "label_provenance": M.label_provenance(),
    }


@app.get("/api/evaluation/figures/{name}")
def figure(name: str):
    path = (config.EVAL_FIGURES_DIR / name).resolve()
    if path.parent != config.EVAL_FIGURES_DIR.resolve() or not path.exists():
        raise HTTPException(404, "Figure not found")
    return FileResponse(path, media_type="image/png")


_recompute_state = {"running": False, "last": None, "error": None}


@app.post("/api/evaluation/recompute")
def recompute():
    if _recompute_state["running"]:
        return _recompute_state

    def job():
        _recompute_state.update(running=True, error=None)
        try:
            from lexinsight.evaluation.report import build_report
            build_report()
            _recompute_state["last"] = time.strftime("%Y-%m-%d %H:%M:%S")
        except Exception as exc:  # surfaced to the UI
            _recompute_state["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            _recompute_state["running"] = False

    threading.Thread(target=job, daemon=True).start()
    return {**_recompute_state, "running": True}


@app.get("/api/evaluation/recompute")
def recompute_status():
    return _recompute_state


# --------------------------------------------------
# Relevance label review
# --------------------------------------------------


def _pools():
    out = {}
    for split in ("validation", "test"):
        pool = read_json(config.EVAL_QRELS_DIR / f"pool_{split}.json")
        if pool:
            for qid, q in pool["queries"].items():
                out[qid] = {**q, "split": split}
    return out


@app.get("/api/labels")
def label_overview():
    pools = _pools()
    draft = read_json(M.DRAFT_PATH, {"labels": {}})["labels"]
    reviewed = read_json(M.REVIEWED_PATH, {"labels": {}})["labels"]
    rows = []
    for qid, q in pools.items():
        n = len(q["candidates"])
        rows.append({
            "qid": qid, "text": q["text"], "intent": q["intent"], "split": q["split"], "candidates": n,
            "drafted": len(draft.get(qid, {})), "reviewed": len(reviewed.get(qid, {})),
            "relevant": sum(1 for c in q["candidates"]
                            if (reviewed.get(qid, {}).get(c["case_id"]) or draft.get(qid, {}).get(c["case_id"], {"grade": 0}))["grade"] >= 1),
        })
    rows.sort(key=lambda r: (r["split"] != "validation", r["qid"]))
    return {"queries": rows, "provenance": M.label_provenance()}


@app.get("/api/labels/{qid}")
def label_query(qid: str):
    pools = _pools()
    if qid not in pools:
        raise HTTPException(404, "Query not pooled")
    draft = read_json(M.DRAFT_PATH, {"labels": {}})["labels"].get(qid, {})
    reviewed = read_json(M.REVIEWED_PATH, {"labels": {}})["labels"].get(qid, {})
    q = pools[qid]
    return {
        "qid": qid, "text": q["text"], "intent": q["intent"], "need": q["need"], "split": q["split"],
        "candidates": [{**c, "draft": draft.get(c["case_id"]), "reviewed": reviewed.get(c["case_id"])}
                       for c in q["candidates"]],
    }


@app.put("/api/labels/{qid}/{case_id}")
def update_label(qid: str, case_id: str, body: LabelUpdate):
    with _labels_lock:
        data = read_json(M.REVIEWED_PATH, {"description": "Grades set or confirmed by the human reviewer; "
                                                          "they override draft_labels.json.", "labels": {}})
        data["labels"].setdefault(qid, {})[case_id] = {
            "grade": body.grade, "note": body.note, "reviewer": body.reviewer,
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        M.REVIEWED_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = M.REVIEWED_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
        tmp.replace(M.REVIEWED_PATH)
    return {"ok": True, **data["labels"][qid][case_id]}


@app.post("/api/labels/{qid}/confirm-all")
def confirm_all(qid: str):
    """Mark every remaining draft grade of a query as reviewed (unchanged)."""
    draft = read_json(M.DRAFT_PATH, {"labels": {}})["labels"].get(qid, {})
    with _labels_lock:
        data = read_json(M.REVIEWED_PATH, {"labels": {}})
        slot = data["labels"].setdefault(qid, {})
        for cid, v in draft.items():
            slot.setdefault(cid, {"grade": v["grade"], "note": "confirmed draft", "reviewer": "project author",
                                  "time": time.strftime("%Y-%m-%d %H:%M:%S")})
        tmp = M.REVIEWED_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
        tmp.replace(M.REVIEWED_PATH)
    return {"ok": True, "reviewed": len(slot)}


# --------------------------------------------------
# Frontend
# --------------------------------------------------

DIST = config.ROOT / "frontend" / "dist"
if DIST.exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    @app.get("/{full_path:path}")
    def spa(full_path: str):
        if full_path.startswith("api/"):
            raise HTTPException(404)
        target = DIST / full_path
        if full_path and target.is_file():
            return FileResponse(target)
        return FileResponse(DIST / "index.html")
