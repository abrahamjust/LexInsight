# LexInsight — evaluation report

Generated 2026-10-28 20:34. Test queries: 35 (35 with at least one relevant judgment). Validation queries used for configuration selection: 15.

**Label provenance.** Relevance labels were drafted by Claude from each pooled judgment's Issue for Consideration, headnotes, keywords and cited authorities, then reviewed by the project author in the labelling page. Current status: {'draft_pairs': 1714, 'reviewed_pairs': 0, 'grades_changed_by_reviewer': 0}. Re-run `python -m lexinsight.evaluation report` after reviewing labels to refresh every number below.

## Frozen configuration (selected on validation)

- Semantic signal: `whole`
- B4 weights: {'bm25': 0.7, 'legal': 0.3} · B5 weights: {'bm25': 0.5, 'semantic': 0.5}
- B6 fixed weights: `F2_lexical_anchor` · B7 adaptive table: `A3_pdf_table_equal_blend`

## Main results (test)

| System | P@5 | R@5 | MRR@5 | nDCG@5 | nDCG@10 | judged@5 |
|---|---|---|---|---|---|---|
| B1 BM25 | 0.829 | 0.450 | 0.986 | 0.822 | 0.831 | 1.000 |
| B2 Doc2Vec (whole) | 0.331 | 0.207 | 0.533 | 0.335 | 0.349 | 1.000 |
| B3 Doc2Vec (chunked, max) | 0.337 | 0.189 | 0.540 | 0.293 | 0.304 | 1.000 |
| B3b Doc2Vec (chunked, mean-3) | 0.360 | 0.203 | 0.536 | 0.326 | 0.351 | 1.000 |
| Legal scorer only | 0.720 | 0.392 | 0.890 | 0.702 | 0.735 | 1.000 |
| B4 BM25 + legal | 0.851 | 0.463 | 1.000 | 0.838 | 0.859 | 1.000 |
| B5 BM25 + Doc2Vec | 0.817 | 0.453 | 0.964 | 0.818 | 0.813 | 1.000 |
| B6 Fixed-weight hybrid | 0.846 | 0.476 | 0.986 | 0.857 | 0.869 | 1.000 |
| B7 Intent-adaptive hybrid | 0.840 | 0.474 | 1.000 | 0.866 | 0.866 | 1.000 |
| B7 with gold intent (oracle) | 0.840 | 0.474 | 1.000 | 0.866 | 0.866 | 1.000 |
| B7 with rule-based intent | 0.840 | 0.474 | 1.000 | 0.866 | 0.867 | 1.000 |

## Paired comparisons (nDCG@5)

| Comparison | Δ mean | 95% CI | p (randomisation) | wins/losses/ties |
|---|---|---|---|---|
| B7_adaptive_hybrid vs B6_fixed_hybrid | +0.009 | [-0.009, +0.029] | 0.385 | 9/9/17 |
| B7_adaptive_hybrid vs B1_bm25 | +0.044 | [-0.007, +0.096] | 0.102 | 15/6/14 |
| B6_fixed_hybrid vs B1_bm25 | +0.035 | [-0.007, +0.079] | 0.124 | 15/8/12 |
| B4_bm25_legal vs B1_bm25 | +0.016 | [-0.015, +0.051] | 0.404 | 12/7/16 |
| B5_bm25_semantic vs B1_bm25 | -0.004 | [-0.058, +0.044] | 0.869 | 13/10/12 |
| B3_doc2vec_chunked_max vs B2_doc2vec_whole | -0.042 | [-0.143, +0.058] | 0.422 | 12/15/8 |
| FULL_oracle_intent vs B7_adaptive_hybrid | +0.000 | [+0.000, +0.000] | 1.000 | 0/0/35 |
| FULL_minus_bm25 vs B7_adaptive_hybrid | -0.049 | [-0.083, -0.018] | 0.003 | 5/15/15 |
| FULL_minus_semantic vs B7_adaptive_hybrid | -0.032 | [-0.049, -0.016] | 0.001 | 2/16/17 |
| FULL_minus_legal vs B7_adaptive_hybrid | -0.114 | [-0.202, -0.040] | 0.006 | 4/16/15 |
| FULL_minus_citation vs B7_adaptive_hybrid | -0.041 | [-0.085, -0.000] | 0.071 | 9/13/13 |
| FULL_minus_outcome vs B7_adaptive_hybrid | +0.002 | [-0.007, +0.013] | 0.624 | 3/1/31 |
| FULL_fixed_weights vs B7_adaptive_hybrid | -0.008 | [-0.018, -0.001] | 0.036 | 1/7/27 |

## nDCG@5 by intent (test)

| System | Factual Similarity | Legal Issue | Statute Based | Supporting Precedent | Contrasting Precedent |
|---|---|---|---|---|---|
| B1 BM25 | 0.686 | 0.772 | 0.928 | 0.850 | 0.874 |
| B2 Doc2Vec (whole) | 0.650 | 0.120 | 0.079 | 0.245 | 0.580 |
| B3 Doc2Vec (chunked, max) | 0.425 | 0.253 | 0.152 | 0.262 | 0.375 |
| B3b Doc2Vec (chunked, mean-3) | 0.524 | 0.196 | 0.175 | 0.303 | 0.431 |
| Legal scorer only | 0.577 | 0.808 | 0.914 | 0.546 | 0.664 |
| B4 BM25 + legal | 0.731 | 0.874 | 0.906 | 0.815 | 0.863 |
| B5 BM25 + Doc2Vec | 0.798 | 0.726 | 0.838 | 0.915 | 0.812 |
| B6 Fixed-weight hybrid | 0.679 | 0.887 | 0.912 | 0.892 | 0.915 |
| B7 Intent-adaptive hybrid | 0.697 | 0.895 | 0.902 | 0.921 | 0.916 |
| B7 with gold intent (oracle) | 0.697 | 0.895 | 0.902 | 0.921 | 0.916 |
| B7 with rule-based intent | 0.697 | 0.895 | 0.902 | 0.921 | 0.916 |

## Queries with vs without explicit statute references (nDCG@5)

| System | explicit (n) | none (n) |
|---|---|---|
| B1 BM25 | 0.903 (8) | 0.798 (27) |
| B2 Doc2Vec (whole) | 0.079 (8) | 0.411 (27) |
| B3 Doc2Vec (chunked, max) | 0.188 (8) | 0.324 (27) |
| B3b Doc2Vec (chunked, mean-3) | 0.190 (8) | 0.366 (27) |
| Legal scorer only | 0.908 (8) | 0.641 (27) |
| B4 BM25 + legal | 0.885 (8) | 0.824 (27) |
| B5 BM25 + Doc2Vec | 0.839 (8) | 0.811 (27) |
| B6 Fixed-weight hybrid | 0.908 (8) | 0.842 (27) |
| B7 Intent-adaptive hybrid | 0.901 (8) | 0.856 (27) |
| B7 with gold intent (oracle) | 0.901 (8) | 0.856 (27) |
| B7 with rule-based intent | 0.901 (8) | 0.856 (27) |

## Intent classification

- On the retrieval test queries (never used for training): accuracy 1.000, macro-F1 1.000; rule baseline accuracy 0.943.
- Held-out intent-query results: see `intent_classification.json` and `fig3_intent_confusion.png`.

## Efficiency

- Engine load: 1.98 s; median end-to-end search 127.2 ms (p95 302.6 ms).
- Index build timings (s): {'bm25_seconds': 9.7, 'issue_tfidf_seconds': 1.8, 'doc2vec_whole_seconds': 553.3, 'doc2vec_chunked_seconds': 870.3, 'chunks': 22190}
- Index sizes (MB): {'bm25.pkl': 10.88, 'doc2vec_chunked.model': 43.77, 'doc2vec_chunk_map.json': 0.24, 'doc2vec_whole.model': 30.76, 'intent_classifier.pkl': 0.55, 'issue_tfidf.pkl': 4.59}

## Validation sweep (selection data)

| Configuration | P@5 | MRR@5 | nDCG@5 | judged@5 |
|---|---|---|---|---|
| B1_bm25 | 0.867 | 0.967 | 0.869 | 1.000 |
| B2_doc2vec_whole | 0.387 | 0.543 | 0.343 | 1.000 |
| B3_doc2vec_chunked_max | 0.307 | 0.617 | 0.250 | 1.000 |
| B3b_doc2vec_chunked_mean3 | 0.400 | 0.656 | 0.303 | 1.000 |
| B4_bm25_legal_w0 | 0.893 | 0.956 | 0.896 | 1.000 |
| B4_bm25_legal_w1 | 0.880 | 0.956 | 0.869 | 1.000 |
| B4_bm25_legal_w2 | 0.853 | 0.956 | 0.836 | 1.000 |
| B5_bm25_sem_w0__chunked_max | 0.867 | 0.967 | 0.856 | 1.000 |
| B5_bm25_sem_w0__chunked_mean3 | 0.880 | 0.967 | 0.852 | 1.000 |
| B5_bm25_sem_w0__whole | 0.880 | 0.956 | 0.854 | 1.000 |
| B5_bm25_sem_w1__chunked_max | 0.800 | 0.922 | 0.751 | 1.000 |
| B5_bm25_sem_w1__chunked_mean3 | 0.787 | 0.967 | 0.748 | 1.000 |
| B5_bm25_sem_w1__whole | 0.867 | 1.000 | 0.856 | 1.000 |
| B5_bm25_sem_w2__chunked_max | 0.533 | 0.713 | 0.463 | 1.000 |
| B5_bm25_sem_w2__chunked_mean3 | 0.627 | 0.728 | 0.520 | 1.000 |
| B5_bm25_sem_w2__whole | 0.640 | 0.833 | 0.609 | 1.000 |
| B6_fixed_F1_pdf_column_mean__chunked_max | 0.880 | 1.000 | 0.828 | 1.000 |
| B6_fixed_F1_pdf_column_mean__chunked_mean3 | 0.880 | 1.000 | 0.834 | 1.000 |
| B6_fixed_F1_pdf_column_mean__whole | 0.840 | 0.956 | 0.828 | 1.000 |
| B6_fixed_F2_lexical_anchor__chunked_max | 0.893 | 1.000 | 0.871 | 1.000 |
| B6_fixed_F2_lexical_anchor__chunked_mean3 | 0.893 | 1.000 | 0.865 | 1.000 |
| B6_fixed_F2_lexical_anchor__whole | 0.893 | 0.967 | 0.870 | 1.000 |
| B6_fixed_F3_equal__chunked_max | 0.880 | 1.000 | 0.799 | 1.000 |
| B6_fixed_F3_equal__chunked_mean3 | 0.867 | 1.000 | 0.807 | 1.000 |
| B6_fixed_F3_equal__whole | 0.880 | 0.950 | 0.833 | 1.000 |
| B7_adaptive_A1_pdf_table__chunked_max | 0.880 | 0.967 | 0.823 | 1.000 |
| B7_adaptive_A1_pdf_table__chunked_mean3 | 0.880 | 1.000 | 0.849 | 1.000 |
| B7_adaptive_A1_pdf_table__whole | 0.880 | 0.956 | 0.852 | 1.000 |
| B7_adaptive_A2_pdf_table_lexical_anchor__chunked_max | 0.907 | 1.000 | 0.876 | 1.000 |
| B7_adaptive_A2_pdf_table_lexical_anchor__chunked_mean3 | 0.880 | 1.000 | 0.865 | 1.000 |
| B7_adaptive_A2_pdf_table_lexical_anchor__whole | 0.880 | 0.967 | 0.859 | 1.000 |
| B7_adaptive_A3_pdf_table_equal_blend__chunked_max | 0.907 | 0.967 | 0.854 | 1.000 |
| B7_adaptive_A3_pdf_table_equal_blend__chunked_mean3 | 0.893 | 1.000 | 0.855 | 1.000 |
| B7_adaptive_A3_pdf_table_equal_blend__whole | 0.867 | 0.967 | 0.861 | 1.000 |
| X_legal_only | 0.813 | 0.850 | 0.719 | 1.000 |

## Figures

- `figures/fig1_retrieval_comparison.png` — B1–B7 on P@5, MRR@5, nDCG@5
- `figures/fig2_ablation.png` — component ablation with 95% CIs
- `figures/fig3_intent_confusion.png`, `fig3b_…` — intent confusion matrices
- `figures/fig4_by_intent.png` — nDCG@5 per intent
- `figures/fig5_judgment_lengths.png` — corpus statistics
- `figures/fig6_latency.png` — efficiency
- `figures/fig7_extraction_quality.png` — legal extraction P/R/F1 (from `extraction_quality.json`)
