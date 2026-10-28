# LexInsight

**An intent-adaptive hybrid NLP framework for legal case retrieval and precedent analysis** over Supreme Court of India judgments (2024, Digital SCR, 781 judgments from the public AWS Open Data collection).

A query is analysed for its retrieval intent (factual similarity, legal issue, statute-based, supporting precedent, contrasting precedent) and for the legal references it contains. Five evidence signals are then fused with intent-specific weights:

| Signal | What it measures |
|---|---|
| **L** BM25 | lexical match over the cleaned judgment text |
| **S** Doc2Vec | semantic similarity (whole-document; chunked variants kept as baselines) |
| **G** Legal scorer | Act match, provision match (headnote mention = substantive, body-only = weaker), legal-issue similarity |
| **C** Citation | treatment-aware citation evidence from *Case Law Cited* (relied on / followed vs distinguished / overruled) |
| **O** Outcome | compatibility with a requested outcome (acquittal, bail granted, quashed, …) |

Signals that do not apply to a query (no provision cited, no outcome requested, …) are switched off and their weight is redistributed. Every result comes with a score breakdown and a template explanation built from the extracted features.

---

## Results (35 held-out test queries, pooled graded labels)

| System | P@5 | MRR@5 | nDCG@5 |
|---|---|---|---|
| B1 BM25 | 0.829 | 0.986 | 0.822 |
| B2 Doc2Vec (whole) | 0.331 | 0.533 | 0.335 |
| B3 Doc2Vec (chunked) | 0.337 | 0.540 | 0.293 |
| B4 BM25 + legal | 0.851 | 1.000 | 0.838 |
| B5 BM25 + Doc2Vec | 0.817 | 0.964 | 0.818 |
| B6 Fixed-weight hybrid | 0.846 | 0.986 | 0.857 |
| **B7 Intent-adaptive hybrid** | 0.840 | **1.000** | **0.866** |

What the evidence supports:

- **The full hybrid beats BM25** (+0.044 nDCG@5), but the 95% bootstrap CI crosses zero (p = 0.10), so it is a trend rather than a significant result on 35 queries.
- **Intent adaptation over the best fixed-weight hybrid (B7 vs B6)** gives +0.009, which is **not significant** (p = 0.39). Against a fixed vector with the *same* signals (the table mean) it is small but consistent (+0.008, p = 0.04, 7 losses vs 1 win for the fixed version).
- **Ablation:** removing the legal scorer costs the most (−0.114, p = 0.006), then BM25 (−0.049), citation (−0.041, p = 0.07) and semantic (−0.032). The outcome signal has no measurable effect, because few test queries request an outcome.
- Doc2Vec is weak on its own on this corpus; whole-document beat chunked on validation, reversing the 111-judgment pilot.
- **Intent classifier** (linear SVM, TF-IDF + legal-reference features): accuracy 0.90 / macro-F1 0.90 on held-out intent queries, against 0.74 for the keyword-rule baseline.
- **Extraction quality**: provisions strict F1 0.89 on a held-out annotated sample (0.98 if only the number has to match); procedural outcome agrees 99.7% with the metadata; substantive-outcome cues have precision 0.80 on a held-out spot check.

Full tables, significance tests, per-intent results, efficiency numbers and figures are in [evaluation/metrics/REPORT.md](evaluation/metrics/REPORT.md) and [evaluation/figures/](evaluation/figures/).

> **Label provenance.** The 1,714 query–judgment relevance grades were **drafted by Claude** from each judgment's Issue for Consideration, headnotes, keywords and cited authorities, and they must be **reviewed by the project author** in the *Relevance labels* page. All numbers above are provisional until that review is done; re-run the report afterwards. The intent queries and retrieval queries were also authored for this project (see `evaluation/queries/`).

---

## Repository layout

```
backend/
  lexinsight/
    corpus.py          PDF -> cleaned text + SCR sections (Issue, Headnotes, Case Law Cited, Acts, Keywords, Result)
    features.py        per-judgment legal features (Acts, provisions + salience, concepts, citations, outcomes)
    legal/             lexicon (Act aliases, issue concepts), extractors, citation treatments, outcome rules
    retrieval/         BM25, whole-document and chunked Doc2Vec
    intent.py          intent classifier training + evaluation
    engine.py          query analysis, signal scoring, fusion, explanations
    weights.json       pre-declared weight configurations
    build_index.py     builds all indexes
    evaluation/        pooling, metrics, significance, report, figures, extraction quality
  api/main.py          FastAPI (search, judgments, evaluation, label review); also serves the built frontend
frontend/              React + TypeScript (Vite): Search, Judgments, Evaluation, Relevance labels, Method
evaluation/            <- all metrics and graphs
  queries/             intent_queries.json, retrieval_queries.json, INTENT_GUIDELINES.md
  qrels/               blind pools, draft_labels.json, reviewed_labels.json, extraction spot checks
  runs/                ranked outputs of every system variant
  metrics/             REPORT.md, test_report.json, CSV tables, intent + extraction metrics, pilot results
  figures/             fig1–fig7 PNGs
scripts/               the original pilot scripts (111 judgments), kept unchanged
data/                  (git-ignored) raw PDFs, metadata, processed corpus, indexes
```

---

## Setup

Requirements: Python 3.12, Node 20+, and the AWS CLI if you want to re-download the PDFs.

```powershell
# from the project root
python -m venv .venv
.venv\Scripts\pip install -r backend\requirements.txt
cd frontend; npm install; npm run build; cd ..
```

### Data (already present on this machine)

```powershell
aws s3 sync --no-sign-request s3://indian-supreme-court-judgments/data/pdf/year=2024/english/ data/raw/pdf/
# metadata JSON files are expected in data/temp/metadata/ (extracted from data/raw/metadata/metadata.tar)
```

### Build everything (run inside `backend/`)

```powershell
cd backend
..\.venv\Scripts\python -m lexinsight.corpus        # ~2.5 min  -> data/processed/corpus.jsonl
..\.venv\Scripts\python -m lexinsight.features      # ~1.5 min  -> data/processed/legal_features.jsonl
..\.venv\Scripts\python -m lexinsight.build_index   # ~25 min (single-thread Doc2Vec for reproducibility)
..\.venv\Scripts\python -m lexinsight.intent        # intent classifier + evaluation
```

### Run the app

```powershell
cd backend
..\.venv\Scripts\python -m uvicorn api.main:app --port 8000
# open http://127.0.0.1:8000
```

For frontend development with hot reload, also run `npm run dev` in `frontend/` and open http://localhost:5173 (it proxies `/api` to port 8000).

---

## Evaluation workflow

The protocol follows the methodology: configurations are chosen on **validation** queries only, then the frozen system is evaluated **once** on **test** queries.

```powershell
cd backend
..\.venv\Scripts\python -m lexinsight.evaluation pool --split validation   # blind pool of all 35 candidate configs
#   label evaluation/qrels/pool_validation.json  (drafted -> review in the UI)
..\.venv\Scripts\python -m lexinsight.evaluation select                    # freezes data/index/selected_config.json
..\.venv\Scripts\python -m lexinsight.evaluation pool --split test         # pools frozen systems + ablations
#   label evaluation/qrels/pool_test.json
..\.venv\Scripts\python -m lexinsight.evaluation report                    # metrics, significance, figures
..\.venv\Scripts\python -m lexinsight.evaluation.extraction                # legal-extraction quality
```

**Reviewing labels:** open *Relevance labels* in the app. For each query, candidates appear shuffled and without system names, next to Claude's draft grade and reason. Clicking 0/1/2 stores your grade in `evaluation/qrels/reviewed_labels.json`, which overrides the draft. "Confirm all remaining drafts" accepts the rest. Then press **Recompute metrics** on the Evaluation page, or run the `report` command.

Evaluation details: metrics are P@5, R@5, MRR@5, nDCG@5 (gain 2^rel − 1), P@10 and nDCG@10, plus the share of top-5 results that have a label (judged@5). Unjudged documents count as non-relevant. Comparisons use a paired bootstrap 95% CI and a paired sign-flip randomisation test over queries.

---

## Limitations

- **Labels are Claude-drafted.** They must be human-reviewed before the numbers can support any claim. Pooling also misses relevant judgments that no system retrieved, so recall is relative to the pool.
- **The query sets are small and self-authored.** There are 35 test and 15 validation queries, 7 and 3 per intent. Intent cue words are strong, so the classifier is perfect on the retrieval queries; real user queries will be harder.
- **The corpus is one year (2024) of reported judgments.** Most cited precedents are outside the corpus, so the citation signal relies on treatment labels and shared authorities rather than a full citation graph.
- **Extraction is rule-based.** Ambiguous abbreviations (e.g. "IT Act"), OCR artefacts and lower-court actions described in the past tense still cause errors, as documented in `extraction_quality.json`.
- **Model limits:** gensim caps training documents at 10,000 tokens, so very long judgments are truncated in whole-document Doc2Vec.
- **Not legal advice.** LexInsight retrieves potentially relevant material only. A highly ranked judgment may be distinguishable or superseded, so legal applicability must be verified.
