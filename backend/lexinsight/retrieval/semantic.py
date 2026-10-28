"""Doc2Vec semantic retrieval: whole-document and overlapping-chunk models.

Hyperparameters match the pilot (PV-DM, 100 dims, window 8, min_count 2,
60 epochs, seed 42). Training uses one worker so that a rebuild with the
same corpus reproduces the same vectors.

Query inference in gensim draws negative samples from the model's random
state, so repeated inference of the same query can differ. The random
state is reset before every inference to make rankings reproducible.
"""

import json

import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from .. import config
from ..text import tokenize


def _train(documents) -> Doc2Vec:
    model = Doc2Vec(
        vector_size=config.VECTOR_SIZE,
        window=config.WINDOW,
        min_count=config.MIN_COUNT,
        workers=1,
        epochs=config.EPOCHS,
        dm=1,
        seed=config.SEED,
    )
    model.build_vocab(documents)
    model.train(documents, total_examples=model.corpus_count, epochs=model.epochs)
    return model


def _unit_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def infer(model: Doc2Vec, tokens: list[str]) -> np.ndarray:
    model.random = np.random.RandomState(config.SEED)
    vector = model.infer_vector(tokens, alpha=0.025, min_alpha=0.0001, epochs=config.INFER_EPOCHS)
    return vector / max(np.linalg.norm(vector), 1e-12)


def split_into_chunks(tokens, chunk_size=config.CHUNK_SIZE, overlap=config.CHUNK_OVERLAP):
    """Overlapping token windows (identical to the pilot chunker)."""
    if not tokens:
        return []
    step = chunk_size - overlap
    chunks = []
    for start in range(0, len(tokens), step):
        chunk = tokens[start:start + chunk_size]
        if chunk:
            chunks.append(chunk)
        if start + chunk_size >= len(tokens):
            break
    return chunks


class WholeDocDoc2Vec:
    def __init__(self, model: Doc2Vec, n_docs: int):
        self.model = model
        self.matrix = _unit_rows(np.vstack([model.dv[str(i)] for i in range(n_docs)]))

    @classmethod
    def build(cls, texts):
        docs = [TaggedDocument(words=tokenize(t), tags=[str(i)]) for i, t in enumerate(texts)]
        model = _train(docs)
        model.save(str(config.DOC2VEC_PATH))
        return cls(model, len(texts))

    @classmethod
    def load(cls, n_docs):
        return cls(Doc2Vec.load(str(config.DOC2VEC_PATH)), n_docs)

    def scores(self, query: str) -> np.ndarray:
        tokens = tokenize(query)
        if not tokens:
            return np.zeros(self.matrix.shape[0])
        return self.matrix @ infer(self.model, tokens)


class ChunkedDoc2Vec:
    """Passage-level Doc2Vec. A judgment's score aggregates its chunk scores.

    aggregation="max"   : best chunk (pilot behaviour)
    aggregation="mean3" : mean of the three best chunks, which is less
                          sensitive to one accidentally similar passage.
    """

    def __init__(self, model: Doc2Vec, chunk_case: np.ndarray, chunk_start: np.ndarray, n_docs: int):
        self.model = model
        self.chunk_case = chunk_case
        self.chunk_start = chunk_start
        self.n_docs = n_docs
        self.matrix = _unit_rows(np.vstack([model.dv[i] for i in range(len(chunk_case))]))

    @classmethod
    def build(cls, texts):
        documents, chunk_case, chunk_start = [], [], []
        step = config.CHUNK_SIZE - config.CHUNK_OVERLAP
        for case_index, text in enumerate(texts):
            for chunk_index, chunk in enumerate(split_into_chunks(tokenize(text))):
                tag = len(documents)
                documents.append(TaggedDocument(words=chunk, tags=[tag]))
                chunk_case.append(case_index)
                chunk_start.append(chunk_index * step)
        model = _train(documents)
        model.save(str(config.CHUNKED_DOC2VEC_PATH))
        config.CHUNK_MAP_PATH.write_text(json.dumps({"case": chunk_case, "start": chunk_start}), encoding="utf-8")
        return cls(model, np.array(chunk_case), np.array(chunk_start), len(texts))

    @classmethod
    def load(cls, n_docs):
        mapping = json.loads(config.CHUNK_MAP_PATH.read_text(encoding="utf-8"))
        model = Doc2Vec.load(str(config.CHUNKED_DOC2VEC_PATH))
        return cls(model, np.array(mapping["case"]), np.array(mapping["start"]), n_docs)

    def chunk_scores(self, query: str) -> np.ndarray:
        tokens = tokenize(query)
        if not tokens:
            return np.zeros(len(self.chunk_case))
        return self.matrix @ infer(self.model, tokens)

    def scores(self, query: str, aggregation: str = "max", chunk_sims: np.ndarray | None = None):
        sims = self.chunk_scores(query) if chunk_sims is None else chunk_sims
        out = np.full(self.n_docs, -1.0)
        if aggregation == "max":
            np.maximum.at(out, self.chunk_case, sims)
            return out
        k = 3
        order = np.lexsort((-sims, self.chunk_case))
        cases_sorted = self.chunk_case[order]
        sims_sorted = sims[order]
        boundaries = np.flatnonzero(np.r_[True, cases_sorted[1:] != cases_sorted[:-1]])
        for b_i, start in enumerate(boundaries):
            end = boundaries[b_i + 1] if b_i + 1 < len(boundaries) else len(order)
            out[cases_sorted[start]] = sims_sorted[start:min(end, start + k)].mean()
        return out

    def best_chunk_start(self, case_index: int, chunk_sims: np.ndarray) -> int | None:
        idx = np.flatnonzero(self.chunk_case == case_index)
        if not len(idx):
            return None
        return int(self.chunk_start[idx[np.argmax(chunk_sims[idx])]])
