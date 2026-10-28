"""Registry of every evaluated system variant and a runner that ranks all
queries with each of them, reusing one signal computation per query."""

import json
import time

import numpy as np

from .. import config
from ..engine import WEIGHTS, LexInsightEngine, resolve_weights

TOP_N = 20
SEMANTIC_VARIANTS = WEIGHTS["semantic_variants"]


def baseline_systems():
    return [
        {"id": "B1_bm25", "system": "bm25"},
        {"id": "B2_doc2vec_whole", "system": "doc2vec_whole"},
        {"id": "B3_doc2vec_chunked_max", "system": "doc2vec_chunked", "semantic": "chunked_max"},
        {"id": "B3b_doc2vec_chunked_mean3", "system": "doc2vec_chunked", "semantic": "chunked_mean3"},
        {"id": "X_legal_only", "system": "legal_only"},
    ]


def hybrid_systems(semantic_variants=SEMANTIC_VARIANTS, fixed=None, adaptive=None, pairwise=True):
    out = []
    fixed = fixed or list(WEIGHTS["fixed"])
    adaptive = adaptive or list(WEIGHTS["adaptive"])
    if pairwise:
        for j, row in enumerate(WEIGHTS["pairwise"]["bm25_legal"]):
            out.append({"id": f"B4_bm25_legal_w{j}", "system": "bm25_legal", "pair": row})
    for sem in semantic_variants:
        if pairwise:
            for j, row in enumerate(WEIGHTS["pairwise"]["bm25_semantic"]):
                out.append({"id": f"B5_bm25_sem_w{j}__{sem}", "system": "bm25_semantic", "pair": row, "semantic": sem})
        for name in fixed:
            out.append({"id": f"B6_fixed_{name}__{sem}", "system": "fixed_hybrid", "weights": name, "semantic": sem})
        for name in adaptive:
            out.append({"id": f"B7_adaptive_{name}__{sem}", "system": "adaptive_hybrid", "weights": name, "semantic": sem})
    return out


def ablation_systems(adaptive: str, semantic: str):
    base = {"system": "adaptive_hybrid", "weights": adaptive, "semantic": semantic}
    out = [{"id": "FULL", **base}]
    for s in config.SIGNALS:
        out.append({"id": f"FULL_minus_{s}", **base, "disabled": [s]})
    # same signals, one global weight row (column mean of the intent table)
    table = resolve_weights(adaptive)
    mean_row = {s: float(np.mean([table[i][s] for i in config.INTENTS])) for s in config.SIGNALS}
    out.append({"id": "FULL_fixed_weights", "system": "fixed_custom", "row": mean_row, "semantic": semantic})
    out.append({"id": "FULL_oracle_intent", **base, "oracle": True})
    out.append({"id": "FULL_rule_intent", **base, "rule_intent": True})
    return out


def run_systems(engine: LexInsightEngine, queries: list[dict], systems: list[dict]):
    """Return ({system_id: {qid: [case_id,...]}}, timing info, intent predictions)."""
    from ..intent import rule_based

    runs = {s["id"]: {} for s in systems}
    timings = {}
    predictions = {}
    for q in queries:
        qa_pred = engine.analyze(q["text"])
        predictions[q["id"]] = {"gold": q["intent"], "pred": qa_pred.intent, "rule": rule_based(q["text"])}
        cache = {}
        for spec in systems:
            sem = spec.get("semantic", "chunked_max")
            if spec.get("oracle"):
                qa = engine.analyze(q["text"], intent_override=q["intent"])
                key = (sem, "oracle")
            elif spec.get("rule_intent"):
                qa = engine.analyze(q["text"], intent_override=predictions[q["id"]]["rule"])
                key = (sem, "rule")
            else:
                qa = qa_pred
                key = (sem, "pred")
            if key not in cache:
                t = time.perf_counter()
                cache[key] = engine.compute_signals(qa, sem)
                timings.setdefault("signals_ms", []).append((time.perf_counter() - t) * 1000)
            signals = cache[key]
            if spec["system"] == "fixed_custom":
                H, *_ = engine.fuse(signals, spec["row"], spec.get("disabled", ()))
                order = np.argsort(-H, kind="stable")
                order = order[np.isfinite(H[order])]
            else:
                order, _, _ = engine.rank(
                    q["text"], spec["system"], qa=qa, signals=signals, semantic_variant=sem,
                    weights_name=spec.get("weights"), pair_weights=spec.get("pair"),
                    disabled=tuple(spec.get("disabled", ())),
                )
            runs[spec["id"]][q["id"]] = [engine.cases[i]["case_id"] for i in order[:TOP_N]]
    return runs, timings, predictions


def save_runs(runs: dict, name: str):
    config.EVAL_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.EVAL_RUNS_DIR / f"{name}.json"
    path.write_text(json.dumps(runs, indent=1), encoding="utf-8")
    return path
