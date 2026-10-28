"""Final evaluation on the test queries: tables, significance, figures.

Everything is written to evaluation/metrics and evaluation/figures.
"""

import csv
import json
import statistics
import time

import numpy as np

from .. import config
from ..engine import LexInsightEngine
from . import figures as F
from . import metrics as M
from .pooling import load_queries

MAIN = [
    ("B1_bm25", "B1 BM25"),
    ("B2_doc2vec_whole", "B2 Doc2Vec (whole)"),
    ("B3_doc2vec_chunked_max", "B3 Doc2Vec (chunked, max)"),
    ("B3b_doc2vec_chunked_mean3", "B3b Doc2Vec (chunked, mean-3)"),
    ("X_legal_only", "Legal scorer only"),
    ("B4_bm25_legal", "B4 BM25 + legal"),
    ("B5_bm25_semantic", "B5 BM25 + Doc2Vec"),
    ("B6_fixed_hybrid", "B6 Fixed-weight hybrid"),
    ("B7_adaptive_hybrid", "B7 Intent-adaptive hybrid"),
    ("FULL_oracle_intent", "B7 with gold intent (oracle)"),
    ("FULL_rule_intent", "B7 with rule-based intent"),
]
ABLATIONS = [
    ("FULL_minus_bm25", "− BM25"),
    ("FULL_minus_semantic", "− semantic"),
    ("FULL_minus_legal", "− legal"),
    ("FULL_minus_citation", "− citation"),
    ("FULL_minus_outcome", "− outcome"),
    ("FULL_fixed_weights", "fixed weights (table mean)"),
]
COMPARISONS = [
    ("B7_adaptive_hybrid", "B6_fixed_hybrid"),
    ("B7_adaptive_hybrid", "B1_bm25"),
    ("B6_fixed_hybrid", "B1_bm25"),
    ("B4_bm25_legal", "B1_bm25"),
    ("B5_bm25_semantic", "B1_bm25"),
    ("B3_doc2vec_chunked_max", "B2_doc2vec_whole"),
    ("FULL_oracle_intent", "B7_adaptive_hybrid"),
] + [(a, "B7_adaptive_hybrid") for a, _ in ABLATIONS]


def _write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def efficiency(engine, queries):
    lat = {k: [] for k in ("bm25_ms", "semantic_ms", "legal_ms", "citation_ms", "total_ms")}
    for q in queries:
        r = engine.search(q["text"], top_k=10)
        for k in lat:
            lat[k].append(r["timings_ms"][k])
    build = json.loads((config.INDEX_DIR / "build_report.json").read_text())
    return {
        "engine_load_seconds": engine.load_seconds,
        "latency_ms_median": {k: round(statistics.median(v), 1) for k, v in lat.items()},
        "latency_ms_p95": {k: round(float(np.percentile(v, 95)), 1) for k, v in lat.items()},
        "index_build": build,
        "n_queries_timed": len(queries),
    }


def dataset_stats(engine):
    report = json.loads(config.CORPUS_REPORT_PATH.read_text())
    lengths = [len(c["text"].split()) for c in engine.cases]
    feats = engine.features
    return {
        "corpus": report,
        "words_per_judgment": {"min": min(lengths), "median": int(statistics.median(lengths)),
                               "mean": int(statistics.mean(lengths)), "max": max(lengths)},
        "acts_per_judgment_median": statistics.median(len(f["acts"]) for f in feats),
        "provisions_per_judgment_median": statistics.median(len(f["provisions"]) for f in feats),
        "judgments_with_citations": sum(1 for f in feats if f["citations"]),
        "citations_total": sum(len(f["citations"]) for f in feats),
        "citation_polarity": {p: sum(1 for f in feats for c in f["citations"] if c["polarity"] == p)
                              for p in ("positive", "neutral", "negative")},
        "procedural_outcomes": {k: sum(1 for f in feats if f["outcome"]["procedural"] == k)
                                for k in ("allowed", "partly_allowed", "dismissed", "disposed", "referred", None)},
        "substantive_outcome_judgments": sum(1 for f in feats if f["outcome"]["substantive"]),
    }, lengths


def build_report():
    labels = M.load_labels()
    test_q = load_queries("test")
    val_q = load_queries("validation")
    qids = [q["id"] for q in test_q]
    runs = json.loads((config.EVAL_RUNS_DIR / "test_runs.json").read_text())
    runs["B7_adaptive_hybrid"] = runs.get("B7_adaptive_hybrid", runs.get("FULL"))
    sel = json.loads((config.INDEX_DIR / "selected_config.json").read_text())
    config.EVAL_METRICS_DIR.mkdir(parents=True, exist_ok=True)
    config.EVAL_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    pq_all = {sid: M.per_query(run, labels, qids) for sid, run in runs.items()}
    summary = {sid: M.summarize(pq) for sid, pq in pq_all.items()}
    n_eval = len(pq_all["B1_bm25"]["P@5"])

    # ---------- main table
    metric_names = ["P@5", "R@5", "MRR@5", "nDCG@5", "P@10", "nDCG@10", "judged@5"]
    rows = [[label] + [summary[sid][m] for m in metric_names] for sid, label in MAIN if sid in summary]
    _write_csv(config.EVAL_METRICS_DIR / "test_main_table.csv", ["System"] + metric_names, rows)

    # ---------- per query
    per_query_rows = []
    for q in test_q:
        for sid, label in MAIN:
            if sid in pq_all and q["id"] in pq_all[sid]["P@5"]:
                per_query_rows.append([q["id"], q["intent"], label] +
                                      [round(pq_all[sid][m].get(q["id"], float("nan")), 4) for m in ("P@5", "MRR@5", "nDCG@5")])
    _write_csv(config.EVAL_METRICS_DIR / "test_per_query.csv",
               ["qid", "intent", "system", "P@5", "MRR@5", "nDCG@5"], per_query_rows)

    # ---------- significance
    sig = {}
    for a, b in COMPARISONS:
        if a in pq_all and b in pq_all:
            sig[f"{a} vs {b}"] = {m: M.paired_bootstrap(pq_all[a][m], pq_all[b][m]) for m in ("P@5", "MRR@5", "nDCG@5")}

    # ---------- by intent and by statute reference
    from ..legal.extract import PROVISION_RE, extract_acts
    by_intent, statute_split = {}, {}
    for sid, label in MAIN:
        if sid not in pq_all:
            continue
        by_intent[sid] = {}
        for intent in config.INTENTS:
            ids = [q["id"] for q in test_q if q["intent"] == intent]
            by_intent[sid][intent] = {m: (round(float(np.mean([pq_all[sid][m][i] for i in ids if i in pq_all[sid][m]])), 4)
                                          if any(i in pq_all[sid][m] for i in ids) else None)
                                      for m in ("P@5", "MRR@5", "nDCG@5")}
        statute_split[sid] = {}
        for name, pred in (("explicit_statute", True), ("no_statute", False)):
            ids = [q["id"] for q in test_q
                   if bool(PROVISION_RE.search(q["text"]) or extract_acts(q["text"])[0]) == pred]
            statute_split[sid][name] = {
                "n": len(ids),
                **{m: (round(float(np.mean([pq_all[sid][m][i] for i in ids if i in pq_all[sid][m]])), 4)
                       if any(i in pq_all[sid][m] for i in ids) else None) for m in ("P@5", "MRR@5", "nDCG@5")},
            }

    # ---------- intent classifier on retrieval queries
    preds = json.loads((config.EVAL_RUNS_DIR / "test_intent_predictions.json").read_text())
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
    gold = [preds[q]["gold"] for q in qids]
    pred = [preds[q]["pred"] for q in qids]
    rule = [preds[q]["rule"] for q in qids]
    intent_on_retrieval = {
        "n": len(qids),
        "classifier": {"accuracy": round(accuracy_score(gold, pred), 4), "macro_f1": round(f1_score(gold, pred, average="macro"), 4),
                       "confusion_matrix": confusion_matrix(gold, pred, labels=config.INTENTS).tolist()},
        "rule_baseline": {"accuracy": round(accuracy_score(gold, rule), 4), "macro_f1": round(f1_score(gold, rule, average="macro"), 4)},
        "errors": [{"qid": q, "gold": preds[q]["gold"], "pred": preds[q]["pred"]} for q in qids if preds[q]["gold"] != preds[q]["pred"]],
    }

    engine = LexInsightEngine.load()
    eff = efficiency(engine, test_q)
    stats, lengths = dataset_stats(engine)

    report = {
        "generated": time.strftime("%Y-%m-%d %H:%M"),
        "selected_configuration": sel,
        "n_test_queries": len(qids),
        "n_test_queries_with_relevant": n_eval,
        "label_provenance": M.label_provenance(),
        "summary": summary,
        "significance": sig,
        "by_intent": by_intent,
        "statute_split": statute_split,
        "intent_on_retrieval_queries": intent_on_retrieval,
        "efficiency": eff,
        "dataset": stats,
        "pool": json.loads((config.EVAL_QRELS_DIR / "pool_test.json").read_text(encoding="utf-8"))["sources"],
    }
    (config.EVAL_METRICS_DIR / "test_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (config.EVAL_METRICS_DIR / "test_per_query.json").write_text(json.dumps(pq_all, indent=1), encoding="utf-8")

    # ---------- figures
    FIG = config.EVAL_FIGURES_DIR
    core = [s for s in ("B1_bm25", "B2_doc2vec_whole", "B3_doc2vec_chunked_max", "B4_bm25_legal",
                        "B5_bm25_semantic", "B6_fixed_hybrid", "B7_adaptive_hybrid") if s in summary]
    names = {sid: label for sid, label in MAIN}
    short = {sid: names[sid].split(" ", 1)[0] for sid in core}
    F.grouped_bars(
        FIG / "fig1_retrieval_comparison.png", [short[s] for s in core],
        {m: [summary[s][m] for s in core] for m in ("P@5", "MRR@5", "nDCG@5")},
        "Retrieval performance on the test queries",
        "Score", subtitle=f"{n_eval} test queries, pooled graded labels · B1 BM25 · B2/B3 Doc2Vec whole/chunked · "
                          "B4 +legal · B5 +Doc2Vec · B6 fixed · B7 adaptive",
    )
    abl = [("FULL", "full system")] + ABLATIONS
    runs.setdefault("FULL", runs["B7_adaptive_hybrid"])
    pq_all.setdefault("FULL", pq_all["B7_adaptive_hybrid"])
    deltas, cis, labs = [], [], []
    for sid, lab in ABLATIONS:
        if sid not in pq_all:
            continue
        b = M.paired_bootstrap(pq_all[sid]["nDCG@5"], pq_all["B7_adaptive_hybrid"]["nDCG@5"])
        if b:
            deltas.append(b["mean_diff"])
            cis.append(b["ci95"])
            labs.append(lab)
    if deltas:
        lim = max(0.05, max(abs(c) for ci in cis for c in ci) * 1.25)
        F.horizontal_bars(FIG / "fig2_ablation.png", labs, deltas,
                          "Ablation: change in nDCG@5 when a component is removed",
                          "Δ nDCG@5 vs full intent-adaptive system (negative = component helps)",
                          xlim=(-lim, lim), errors=cis,
                          subtitle="Paired bootstrap 95% CI over test queries")
    ic = json.loads((config.EVAL_METRICS_DIR / "intent_classification.json").read_text())
    F.confusion(FIG / "fig3_intent_confusion.png", ic["test"]["confusion_matrix"], config.INTENTS,
                "Intent classifier — held-out intent queries",
                subtitle=f"n={ic['n_test']}, accuracy {ic['test']['accuracy']:.2f}, macro-F1 {ic['test']['macro_f1']:.2f}")
    F.confusion(FIG / "fig3b_intent_confusion_retrieval_queries.png",
                intent_on_retrieval["classifier"]["confusion_matrix"], config.INTENTS,
                "Intent classifier — retrieval test queries",
                subtitle=f"n={len(qids)}, accuracy {intent_on_retrieval['classifier']['accuracy']:.2f}")
    fig4_sys = [s for s in ("B1_bm25", "B3_doc2vec_chunked_max", "B6_fixed_hybrid", "B7_adaptive_hybrid") if s in by_intent]
    F.grouped_bars(
        FIG / "fig4_by_intent.png",
        [i.replace("_PRECEDENT", "\nPRECEDENT").replace("_", " ").title() for i in config.INTENTS],
        {names[s]: [by_intent[s][i]["nDCG@5"] for i in config.INTENTS] for s in fig4_sys},
        "Ranking quality by query intent", "nDCG@5",
        subtitle="Test queries, 7 per intent", figsize=(11, 4.8),
    )
    F.histogram(FIG / "fig5_judgment_lengths.png", lengths, "Judgment length distribution", "Words per judgment (log scale)",
                log=True, subtitle=f"{len(lengths)} Supreme Court judgments (2024)")
    comp = ["bm25_ms", "semantic_ms", "legal_ms", "citation_ms", "total_ms"]
    F.horizontal_bars(FIG / "fig6_latency.png", ["BM25", "Doc2Vec", "Legal scorer", "Citation", "End-to-end search"],
                      [eff["latency_ms_median"][c] for c in comp], "Median query latency by component",
                      "Milliseconds", subtitle=f"{len(test_q)} test queries, single process, CPU",
                      xlim=(0, max(eff["latency_ms_median"].values()) * 1.2))
    val_sel = json.loads((config.EVAL_METRICS_DIR / "validation_selection.json").read_text())["validation_metrics"]
    write_markdown(report, names, val_sel, len(val_q))
    print(f"Report written: {config.EVAL_METRICS_DIR / 'REPORT.md'}")


def _fmt(v):
    return "—" if v is None else f"{v:.3f}"


def write_markdown(report, names, val_sel, n_val):
    s = report["summary"]
    sel = report["selected_configuration"]
    lines = [
        "# LexInsight — evaluation report", "",
        f"Generated {report['generated']}. Test queries: {report['n_test_queries']} "
        f"({report['n_test_queries_with_relevant']} with at least one relevant judgment). "
        f"Validation queries used for configuration selection: {n_val}.", "",
        "**Label provenance.** Relevance labels were drafted by Claude from each pooled judgment's "
        "Issue for Consideration, headnotes, keywords and cited authorities, then reviewed by the project "
        f"author in the labelling page. Current status: {report['label_provenance']}. "
        "Re-run `python -m lexinsight.evaluation report` after reviewing labels to refresh every number below.", "",
        "## Frozen configuration (selected on validation)", "",
        f"- Semantic signal: `{sel['semantic_variant']}`",
        f"- B4 weights: {sel['bm25_legal']} · B5 weights: {sel['bm25_semantic']}",
        f"- B6 fixed weights: `{sel['fixed']}` · B7 adaptive table: `{sel['adaptive']}`", "",
        "## Main results (test)", "",
        "| System | P@5 | R@5 | MRR@5 | nDCG@5 | nDCG@10 | judged@5 |",
        "|---|---|---|---|---|---|---|",
    ]
    for sid, label in MAIN:
        if sid in s:
            r = s[sid]
            lines.append(f"| {label} | {_fmt(r['P@5'])} | {_fmt(r['R@5'])} | {_fmt(r['MRR@5'])} | "
                         f"{_fmt(r['nDCG@5'])} | {_fmt(r['nDCG@10'])} | {_fmt(r['judged@5'])} |")
    lines += ["", "## Paired comparisons (nDCG@5)", "",
              "| Comparison | Δ mean | 95% CI | p (randomisation) | wins/losses/ties |", "|---|---|---|---|---|"]
    for k, v in report["significance"].items():
        b = v["nDCG@5"]
        if b:
            lines.append(f"| {k} | {b['mean_diff']:+.3f} | [{b['ci95'][0]:+.3f}, {b['ci95'][1]:+.3f}] | "
                         f"{b['p_value']:.3f} | {b['wins']}/{b['losses']}/{b['ties']} |")
    lines += ["", "## nDCG@5 by intent (test)", "",
              "| System | " + " | ".join(i.title().replace("_", " ") for i in config.INTENTS) + " |",
              "|---|" + "---|" * len(config.INTENTS)]
    for sid, by in report["by_intent"].items():
        lines.append(f"| {names[sid]} | " + " | ".join(_fmt(by[i]["nDCG@5"]) for i in config.INTENTS) + " |")
    lines += ["", "## Queries with vs without explicit statute references (nDCG@5)", "",
              "| System | explicit (n) | none (n) |", "|---|---|---|"]
    for sid, sp in report["statute_split"].items():
        lines.append(f"| {names[sid]} | {_fmt(sp['explicit_statute']['nDCG@5'])} ({sp['explicit_statute']['n']}) | "
                     f"{_fmt(sp['no_statute']['nDCG@5'])} ({sp['no_statute']['n']}) |")
    ir = report["intent_on_retrieval_queries"]
    lines += ["", "## Intent classification", "",
              f"- On the retrieval test queries (never used for training): accuracy {ir['classifier']['accuracy']:.3f}, "
              f"macro-F1 {ir['classifier']['macro_f1']:.3f}; rule baseline accuracy {ir['rule_baseline']['accuracy']:.3f}.",
              "- Held-out intent-query results: see `intent_classification.json` and `fig3_intent_confusion.png`.", "",
              "## Efficiency", ""]
    e = report["efficiency"]
    lines += [f"- Engine load: {e['engine_load_seconds']} s; median end-to-end search {e['latency_ms_median']['total_ms']} ms "
              f"(p95 {e['latency_ms_p95']['total_ms']} ms).",
              f"- Index build timings (s): {e['index_build'].get('timings')}",
              f"- Index sizes (MB): {e['index_build'].get('sizes_mb')}", "",
              "## Validation sweep (selection data)", "",
              "| Configuration | P@5 | MRR@5 | nDCG@5 | judged@5 |", "|---|---|---|---|---|"]
    for sid in sorted(val_sel):
        r = val_sel[sid]
        lines.append(f"| {sid} | {_fmt(r['P@5'])} | {_fmt(r['MRR@5'])} | {_fmt(r['nDCG@5'])} | {_fmt(r['judged@5'])} |")
    lines += ["", "## Figures", "", "- `figures/fig1_retrieval_comparison.png` — B1–B7 on P@5, MRR@5, nDCG@5",
              "- `figures/fig2_ablation.png` — component ablation with 95% CIs",
              "- `figures/fig3_intent_confusion.png`, `fig3b_…` — intent confusion matrices",
              "- `figures/fig4_by_intent.png` — nDCG@5 per intent",
              "- `figures/fig5_judgment_lengths.png` — corpus statistics",
              "- `figures/fig6_latency.png` — efficiency",
              "- `figures/fig7_extraction_quality.png` — legal extraction P/R/F1 (from `extraction_quality.json`)"]
    (config.EVAL_METRICS_DIR / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
