# Evaluation artefacts

| Folder | Contents |
|---|---|
| `queries/` | `intent_queries.json` (241 queries, 213 paraphrase groups, for the intent classifier), `retrieval_queries.json` (10 development, 15 validation and 35 test queries, each with a written information need), `INTENT_GUIDELINES.md` |
| `qrels/` | `pool_validation.json`, `pool_test.json` (blind, shuffled candidate cards), `draft_labels.json` (Claude-drafted grades + reasons), `reviewed_labels.json` (your corrections; created by the labelling page), `extraction_spot_checks.json` |
| `runs/` | top-20 rankings of every system variant per query, and intent predictions |
| `metrics/` | `REPORT.md` (human-readable summary), `test_report.json`, `test_main_table.csv`, `test_per_query.csv/json`, `validation_selection.json`, `intent_classification.json`, `extraction_quality.json`, `pilot_results.json` |
| `figures/` | fig1 system comparison · fig2 ablation · fig3/3b intent confusion matrices · fig4 nDCG@5 by intent · fig5 judgment lengths · fig6 latency · fig7 extraction quality |

Regenerate the metrics and figures with `python -m lexinsight.evaluation report` and `python -m lexinsight.evaluation.extraction` (run from `backend/`).
