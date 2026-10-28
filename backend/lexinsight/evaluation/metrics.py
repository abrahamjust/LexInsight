"""Ranking metrics, label handling and paired significance tests.

Labels are graded 0/1/2. Binary metrics (P@k, R@k, MRR@k) count grade >= 1
as relevant, matching the pilot P@5 convention. nDCG uses the graded
gain 2^rel - 1. Unjudged documents count as non-relevant; the share of
judged documents in the top k is reported as `judged@k`.
"""

import json
import math

import numpy as np

from .. import config

DRAFT_PATH = config.EVAL_QRELS_DIR / "draft_labels.json"
REVIEWED_PATH = config.EVAL_QRELS_DIR / "reviewed_labels.json"


def load_labels() -> dict:
    """Merge Claude-drafted labels with the reviewer's corrections.

    Returns {qid: {case_id: grade}}. A reviewed grade always wins.
    """
    draft = json.loads(DRAFT_PATH.read_text(encoding="utf-8")) if DRAFT_PATH.exists() else {"labels": {}}
    reviewed = json.loads(REVIEWED_PATH.read_text(encoding="utf-8")) if REVIEWED_PATH.exists() else {"labels": {}}
    merged = {}
    for qid, docs in draft["labels"].items():
        merged[qid] = {cid: int(v["grade"]) for cid, v in docs.items()}
    for qid, docs in reviewed["labels"].items():
        merged.setdefault(qid, {})
        for cid, v in docs.items():
            merged[qid][cid] = int(v["grade"])
    return merged


def label_provenance() -> dict:
    draft = json.loads(DRAFT_PATH.read_text(encoding="utf-8")) if DRAFT_PATH.exists() else {"labels": {}}
    reviewed = json.loads(REVIEWED_PATH.read_text(encoding="utf-8")) if REVIEWED_PATH.exists() else {"labels": {}}
    n_draft = sum(len(v) for v in draft["labels"].values())
    n_reviewed = sum(len(v) for v in reviewed["labels"].values())
    changed = sum(
        1 for q, docs in reviewed["labels"].items() for c, v in docs.items()
        if draft["labels"].get(q, {}).get(c, {}).get("grade") != v["grade"]
    )
    return {"draft_pairs": n_draft, "reviewed_pairs": n_reviewed, "grades_changed_by_reviewer": changed}


def precision_at_k(ranked, labels, k):
    return sum(1 for d in ranked[:k] if labels.get(d, 0) >= 1) / k


def recall_at_k(ranked, labels, k):
    relevant = sum(1 for g in labels.values() if g >= 1)
    if relevant == 0:
        return None
    return sum(1 for d in ranked[:k] if labels.get(d, 0) >= 1) / relevant


def rr_at_k(ranked, labels, k):
    for i, d in enumerate(ranked[:k], start=1):
        if labels.get(d, 0) >= 1:
            return 1.0 / i
    return 0.0


def dcg(gains):
    return sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(ranked, labels, k):
    ideal = dcg(sorted(labels.values(), reverse=True)[:k])
    if ideal == 0:
        return None
    return dcg([labels.get(d, 0) for d in ranked[:k]]) / ideal


def judged_at_k(ranked, labels, k):
    return sum(1 for d in ranked[:k] if d in labels) / k


METRICS = {
    "P@5": lambda r, l: precision_at_k(r, l, 5),
    "R@5": lambda r, l: recall_at_k(r, l, 5),
    "MRR@5": lambda r, l: rr_at_k(r, l, 5),
    "nDCG@5": lambda r, l: ndcg_at_k(r, l, 5),
    "P@10": lambda r, l: precision_at_k(r, l, 10),
    "nDCG@10": lambda r, l: ndcg_at_k(r, l, 10),
    "judged@5": lambda r, l: judged_at_k(r, l, 5),
}


def per_query(run: dict, labels: dict, qids) -> dict:
    """{metric: {qid: value}} over queries that have at least one relevant doc."""
    out = {m: {} for m in METRICS}
    for q in qids:
        lab = labels.get(q, {})
        if not any(g >= 1 for g in lab.values()):
            continue
        ranked = run.get(q, [])
        for m, fn in METRICS.items():
            v = fn(ranked, lab)
            if v is not None:
                out[m][q] = v
    return out


def summarize(pq: dict) -> dict:
    return {m: (round(float(np.mean(list(v.values()))), 4) if v else None) for m, v in pq.items()}


def paired_bootstrap(a: dict, b: dict, n=10000, seed=config.SEED):
    """Mean difference (a - b) over shared queries with a 95% bootstrap CI
    and a two-sided paired sign-flip randomisation p-value."""
    qs = sorted(set(a) & set(b))
    if len(qs) < 2:
        return None
    d = np.array([a[q] - b[q] for q in qs])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    boots = d[idx].mean(axis=1)
    signs = rng.choice([-1, 1], size=(n, len(d)))
    perm = np.abs((signs * d).mean(axis=1))
    p = float((np.sum(perm >= abs(d.mean()) - 1e-12) + 1) / (n + 1))
    return {
        "n_queries": len(qs),
        "mean_diff": round(float(d.mean()), 4),
        "ci95": [round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)],
        "p_value": round(p, 4),
        "wins": int((d > 0).sum()), "losses": int((d < 0).sum()), "ties": int((d == 0).sum()),
    }
