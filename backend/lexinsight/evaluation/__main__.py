"""Evaluation workflow.

    python -m lexinsight.evaluation pool --split validation
    (label evaluation/qrels/pool_validation.json -> draft_labels.json)
    python -m lexinsight.evaluation select
    python -m lexinsight.evaluation pool --split test
    (label pool_test.json)
    python -m lexinsight.evaluation report

`select` freezes the configuration (semantic variant, pairwise weights,
fixed weights, adaptive table) using validation queries only and writes
data/index/selected_config.json. `report` evaluates everything on the
test queries once and writes evaluation/metrics + evaluation/figures.
"""

import argparse
import json
import re

from .. import config
from ..engine import LexInsightEngine
from . import metrics as M
from .pooling import build_pool, load_queries, write_pool
from .systems import ablation_systems, baseline_systems, hybrid_systems, run_systems, save_runs

SELECTED_PATH = config.INDEX_DIR / "selected_config.json"
PRIMARY = "nDCG@5"


def cmd_pool(args):
    engine = LexInsightEngine.load()
    queries = load_queries(args.split)
    if args.split == "validation":
        # every candidate configuration contributes, so selection is fair
        systems = baseline_systems() + hybrid_systems()
        pooled = [s["id"] for s in systems]
        depths = {s["id"]: 10 for s in baseline_systems()}
    else:
        # frozen systems (depth 10) and their ablations (depth 5); the
        # unselected variants are still run and reported with judged@5
        sel = json.loads(SELECTED_PATH.read_text())
        chosen = selected_systems(sel)
        ablations = [s for s in ablation_systems(sel["adaptive"], sel["semantic_variant"]) if s["id"] != "FULL"]
        systems = baseline_systems() + chosen + ablations + hybrid_systems()
        pooled = [s["id"] for s in baseline_systems() + chosen + ablations]
        depths = {s["id"]: 10 for s in baseline_systems() + chosen}
    runs, timings, predictions = run_systems(engine, queries, systems)
    save_runs(runs, f"{args.split}_runs")
    (config.EVAL_RUNS_DIR / f"{args.split}_intent_predictions.json").write_text(json.dumps(predictions, indent=1))
    pool_runs = {sid: runs[sid] for sid in pooled}
    pool = build_pool(pool_runs, depths, default_depth=5)
    write_pool(engine, pool, queries, args.split, {"systems": sorted(pool_runs), "depths": depths, "default_depth": 5})


def selected_systems(sel):
    return [
        {"id": "B4_bm25_legal", "system": "bm25_legal", "pair": sel["bm25_legal"]},
        {"id": "B5_bm25_semantic", "system": "bm25_semantic", "pair": sel["bm25_semantic"],
         "semantic": sel["semantic_variant"]},
        {"id": "B6_fixed_hybrid", "system": "fixed_hybrid", "weights": sel["fixed"], "semantic": sel["semantic_variant"]},
        {"id": "B7_adaptive_hybrid", "system": "adaptive_hybrid", "weights": sel["adaptive"],
         "semantic": sel["semantic_variant"]},
    ]


def evaluate_runs(runs, labels, qids):
    return {sid: M.summarize(M.per_query(run, labels, qids)) for sid, run in runs.items()}


def cmd_select(args):
    from ..engine import WEIGHTS
    labels = M.load_labels()
    queries = load_queries("validation")
    qids = [q["id"] for q in queries]
    runs = json.loads((config.EVAL_RUNS_DIR / "validation_runs.json").read_text())
    table = evaluate_runs(runs, labels, qids)

    def best(prefix_list):
        cands = [(table[s][PRIMARY] or 0, table[s]["MRR@5"] or 0, s) for s in prefix_list if s in table]
        return max(cands)[2]

    sem_map = {"B2_doc2vec_whole": "whole", "B3_doc2vec_chunked_max": "chunked_max",
               "B3b_doc2vec_chunked_mean3": "chunked_mean3"}
    best_sem = sem_map[best(list(sem_map))]
    b4 = best([s for s in table if s.startswith("B4_")])
    b5 = best([s for s in table if s.startswith("B5_") and s.endswith(f"__{best_sem}")])
    b6 = best([s for s in table if s.startswith("B6_") and s.endswith(f"__{best_sem}")])
    b7 = best([s for s in table if s.startswith("B7_") and s.endswith(f"__{best_sem}")])
    sel = {
        "semantic_variant": best_sem,
        "bm25_legal": WEIGHTS["pairwise"]["bm25_legal"][int(re.search(r"_w(\d+)", b4).group(1))],
        "bm25_semantic": WEIGHTS["pairwise"]["bm25_semantic"][int(re.search(r"_w(\d+)__", b5).group(1))],
        "fixed": b6[len("B6_fixed_"):].split("__")[0],
        "adaptive": b7[len("B7_adaptive_"):].split("__")[0],
        "selected_on": f"validation ({len(qids)} queries), metric {PRIMARY}",
    }
    SELECTED_PATH.write_text(json.dumps(sel, indent=2))
    config.EVAL_METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (config.EVAL_METRICS_DIR / "validation_selection.json").write_text(json.dumps({
        "selected": sel, "validation_metrics": table, "label_provenance": M.label_provenance(),
    }, indent=2))
    print(json.dumps(sel, indent=2))
    for sid in sorted(table):
        r = table[sid]
        print(f"{sid:48s} P@5 {r['P@5']}  MRR@5 {r['MRR@5']}  nDCG@5 {r['nDCG@5']}  judged@5 {r['judged@5']}")


def cmd_report(args):
    from .report import build_report
    build_report()


def main():
    parser = argparse.ArgumentParser(prog="python -m lexinsight.evaluation")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pool")
    p.add_argument("--split", choices=["validation", "test", "development"], required=True)
    p.set_defaults(fn=cmd_pool)
    sub.add_parser("select").set_defaults(fn=cmd_select)
    sub.add_parser("report").set_defaults(fn=cmd_report)
    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
