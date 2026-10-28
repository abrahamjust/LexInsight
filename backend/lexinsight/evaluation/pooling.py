"""Pooled relevance assessment.

The pool for a query is the union of the top documents of every
participating system. Candidates are shuffled and shown without system
names or scores so the assessor cannot tell which system found them.
"""

import json
import random

from .. import config
from ..legal.citations import key_matches


def load_queries(split=None):
    data = json.loads((config.EVAL_QUERIES_DIR / "retrieval_queries.json").read_text(encoding="utf-8"))
    qs = data["queries"]
    return [q for q in qs if split is None or q["split"] == split]


def build_pool(runs: dict, depths: dict, default_depth=5) -> dict:
    """runs: {system_id: {qid: [case_ids]}}; depths: {system_id: depth}."""
    pool = {}
    for sid, run in runs.items():
        depth = depths.get(sid, default_depth)
        for qid, ranked in run.items():
            pool.setdefault(qid, [])
            for cid in ranked[:depth]:
                if cid not in pool[qid]:
                    pool[qid].append(cid)
    rng = random.Random(config.SEED)
    for qid in pool:
        rng.shuffle(pool[qid])
    return pool


def label_card(engine, case_id: str, query) -> dict:
    i = engine.id_to_index[case_id]
    case, feat = engine.cases[i], engine.features[i]
    s = case["sections"]
    qa = engine.analyze(query["text"], intent_override=query["intent"])
    cited = []
    for c in feat["citations"]:
        if qa.precedents and any(key_matches(k, c["key"]) >= 0.75 or c["key"] == k for k in qa.precedents):
            cited.append(f"{c['name'][:90]} — {c['treatment']}")
        elif not qa.precedents and query["intent"].endswith("PRECEDENT") and c["polarity"] != "neutral":
            cited.append(f"{c['name'][:90]} — {c['treatment']}")
    return {
        "case_id": case_id,
        "case_name": case["case_name"],
        "issue": s.get("issue", "")[:500],
        "acts": s.get("acts", "")[:200],
        "keywords": s.get("keywords", "")[:320],
        "result": s.get("result", ""),
        "cited": cited[:5],
    }


def write_pool(engine, pool: dict, queries: list[dict], split: str, sources: dict):
    by_id = {q["id"]: q for q in queries}
    out = {
        "split": split,
        "note": "Blind pool: candidates shuffled, no system names or scores.",
        "sources": sources,
        "queries": {
            qid: {"text": by_id[qid]["text"], "intent": by_id[qid]["intent"], "need": by_id[qid]["need"],
                  "candidates": [label_card(engine, cid, by_id[qid]) for cid in cids]}
            for qid, cids in pool.items() if qid in by_id
        },
    }
    config.EVAL_QRELS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.EVAL_QRELS_DIR / f"pool_{split}.json"
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    sizes = [len(v) for v in pool.values()]
    print(f"Pool {split}: {len(pool)} queries, {sum(sizes)} pairs, mean {sum(sizes)/max(1,len(sizes)):.1f}/query")
    return path
