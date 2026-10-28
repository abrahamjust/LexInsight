"""Build all retrieval indexes from the corpus.

Run:  python -m lexinsight.build_index [--skip-doc2vec]

Writes data/index/: BM25, whole-document Doc2Vec, chunked Doc2Vec and the
TF-IDF model over the legal summary fields (issue, headnotes, keywords).
Build timings and model sizes are saved for the efficiency report.
"""

import argparse
import json
import pickle
import time

from sklearn.feature_extraction.text import TfidfVectorizer

from . import config
from .features import legal_summary, load_corpus
from .retrieval.lexical import BM25Retriever
from .retrieval.semantic import ChunkedDoc2Vec, WholeDocDoc2Vec
from .text import STOPWORDS


def build_issue_tfidf(cases):
    vectorizer = TfidfVectorizer(
        lowercase=True,
        token_pattern=r"\b[a-z0-9]+\b",
        stop_words=sorted(STOPWORDS),
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.5,
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform([legal_summary(c) for c in cases])
    with open(config.ISSUE_TFIDF_PATH, "wb") as f:
        pickle.dump({"vectorizer": vectorizer, "matrix": matrix}, f, protocol=pickle.HIGHEST_PROTOCOL)
    return matrix.shape


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-doc2vec", action="store_true")
    args = parser.parse_args()

    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    cases = load_corpus()
    texts = [c["text"] for c in cases]
    timings = {}

    t = time.time()
    BM25Retriever.build(texts).save()
    timings["bm25_seconds"] = round(time.time() - t, 1)
    print(f"BM25 built in {timings['bm25_seconds']}s")

    t = time.time()
    shape = build_issue_tfidf(cases)
    timings["issue_tfidf_seconds"] = round(time.time() - t, 1)
    print(f"Issue TF-IDF {shape} built in {timings['issue_tfidf_seconds']}s")

    if not args.skip_doc2vec:
        t = time.time()
        whole = WholeDocDoc2Vec.build(texts)
        timings["doc2vec_whole_seconds"] = round(time.time() - t, 1)
        print(f"Whole-document Doc2Vec: {len(whole.model.dv)} vectors, vocab {len(whole.model.wv)}, "
              f"{timings['doc2vec_whole_seconds']}s")

        t = time.time()
        chunked = ChunkedDoc2Vec.build(texts)
        timings["doc2vec_chunked_seconds"] = round(time.time() - t, 1)
        timings["chunks"] = int(len(chunked.chunk_case))
        print(f"Chunked Doc2Vec: {len(chunked.chunk_case)} chunks, vocab {len(chunked.model.wv)}, "
              f"{timings['doc2vec_chunked_seconds']}s")

    sizes = {
        p.name: round(p.stat().st_size / 1e6, 2)
        for p in config.INDEX_DIR.glob("*") if p.is_file()
    }
    report_path = config.INDEX_DIR / "build_report.json"
    previous = json.loads(report_path.read_text()) if report_path.exists() else {}
    previous.update({"timings": {**previous.get("timings", {}), **timings}, "sizes_mb": sizes, "n_docs": len(cases)})
    report_path.write_text(json.dumps(previous, indent=2), encoding="utf-8")
    print(json.dumps(previous, indent=2))


if __name__ == "__main__":
    main()
