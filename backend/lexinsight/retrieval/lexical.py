"""BM25 lexical retrieval (rank-bm25 Okapi, as in the pilot)."""

import pickle

import numpy as np
from rank_bm25 import BM25Okapi

from .. import config
from ..text import tokenize


class BM25Retriever:
    def __init__(self, bm25: BM25Okapi):
        self.bm25 = bm25

    @classmethod
    def build(cls, texts: list[str]) -> "BM25Retriever":
        return cls(BM25Okapi([tokenize(t) for t in texts]))

    def save(self, path=config.BM25_INDEX_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.bm25, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path=config.BM25_INDEX_PATH) -> "BM25Retriever":
        with open(path, "rb") as f:
            return cls(pickle.load(f))

    def scores(self, query: str) -> np.ndarray:
        tokens = tokenize(query)
        if not tokens:
            return np.zeros(self.bm25.corpus_size)
        return np.asarray(self.bm25.get_scores(tokens), dtype=float)
