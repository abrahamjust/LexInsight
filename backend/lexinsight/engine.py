"""LexInsight retrieval engine: query analysis, evidence scoring, fusion.

    engine = LexInsightEngine.load()
    response = engine.search("Can a conviction be based on circumstantial evidence?")

Signals (all per query, per judgment):
    L  BM25 lexical score                           (min-max normalised)
    S  Doc2Vec semantic similarity                  (min-max normalised)
    G  legal relevance = Act / provision / issue    (min-max normalised)
    C  citation evidence                            (min-max normalised)
    O  outcome compatibility in {0, .5, 1}          (kept on its own scale)

A signal that has nothing to say for a query (no provision cited, no
outcome requested, no citation evidence) is marked inapplicable and its
weight is redistributed over the remaining signals, so judgments are not
penalised for lacking a feature the query never asked about.
"""

import json
import math
import pickle
import re
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from . import config
from .features import load_corpus
from .intent import IntentClassifier, rule_based
from .legal import citations as cit
from .legal import outcome as out
from .legal.extract import extract_acts, extract_provisions, provision_matches
from .legal.lexicon import CONCEPTS, act_label, find_concepts, implied_provisions
from .retrieval.lexical import BM25Retriever
from .retrieval.semantic import ChunkedDoc2Vec, WholeDocDoc2Vec
from .text import content_tokens, normalize_space, split_sentences, tokenize

WEIGHTS = json.loads((Path(__file__).parent / "weights.json").read_text(encoding="utf-8"))

POLARITY_WEIGHT = {
    "SUPPORTING_PRECEDENT": {"positive": 1.0, "neutral": 0.6, "negative": 0.15},
    "CONTRASTING_PRECEDENT": {"negative": 1.0, "neutral": 0.4, "positive": 0.15},
    None: {"positive": 1.0, "neutral": 0.8, "negative": 0.8},
}


@dataclass
class QueryAnalysis:
    text: str
    intent: str
    intent_scores: dict
    intent_source: str
    acts: dict = field(default_factory=dict)              # act_id -> label
    provisions: list = field(default_factory=list)        # explicit provision ids
    provision_labels: dict = field(default_factory=dict)
    implied_provisions: list = field(default_factory=list)
    concepts: list = field(default_factory=list)
    precedents: list = field(default_factory=list)        # matched authority keys
    precedent_names: list = field(default_factory=list)
    outcome_requirements: list = field(default_factory=list)


def minmax(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out_arr = np.zeros_like(x, dtype=float)
    if not mask.any():
        return out_arr
    lo, hi = x[mask].min(), x[mask].max()
    if hi - lo < 1e-12:
        return out_arr
    out_arr[mask] = (x[mask] - lo) / (hi - lo)
    return out_arr


def resolve_weights(name: str) -> dict:
    """Return {intent: {signal: weight}} for a fixed or adaptive config."""
    if name in WEIGHTS["fixed"]:
        row = WEIGHTS["fixed"][name]
        return {intent: dict(row) for intent in config.INTENTS}
    spec = WEIGHTS["adaptive"][name]
    table = WEIGHTS["adaptive_tables"][spec["table"]]
    if not spec.get("blend_with"):
        return {i: dict(table[i]) for i in config.INTENTS}
    base = WEIGHTS["fixed"][spec["blend_with"]]
    lam = spec["lambda"]
    return {
        i: {s: (1 - lam) * table[i][s] + lam * base[s] for s in config.SIGNALS}
        for i in config.INTENTS
    }


class LexInsightEngine:
    def __init__(self):
        t0 = time.time()
        self.cases = load_corpus()
        with config.LEGAL_FEATURES_PATH.open(encoding="utf-8") as f:
            self.features = [json.loads(line) for line in f if line.strip()]
        assert len(self.cases) == len(self.features)
        self.n = len(self.cases)
        self.id_to_index = {c["case_id"]: i for i, c in enumerate(self.cases)}

        self.bm25 = BM25Retriever.load()
        self.whole = WholeDocDoc2Vec.load(self.n)
        self.chunked = ChunkedDoc2Vec.load(self.n)
        with open(config.ISSUE_TFIDF_PATH, "rb") as f:
            tfidf = pickle.load(f)
        self.issue_vectorizer, self.issue_matrix = tfidf["vectorizer"], tfidf["matrix"]
        self.intent_model = IntentClassifier.load() if config.INTENT_MODEL_PATH.exists() else None
        self._build_citation_index()
        self.load_seconds = round(time.time() - t0, 2)

    @classmethod
    def load(cls) -> "LexInsightEngine":
        return cls()

    # --------------------------------------------------
    # Citation index
    # --------------------------------------------------

    def _build_citation_index(self):
        self.authority_docs = defaultdict(set)   # authority key -> doc indices citing it
        self.authority_names = {}
        self.doc_authorities = []                # doc index -> {key: polarity}
        scr_to_doc = {c.get("scr_citation"): i for i, c in enumerate(self.cases) if c.get("scr_citation")}
        self.in_corpus_cites = defaultdict(set)  # doc -> corpus docs it cites
        self.cited_by = defaultdict(set)
        for i, feat in enumerate(self.features):
            seen = {}
            for c in feat["citations"]:
                key = c["key"]
                # keep the strongest-signal treatment if cited twice
                if key not in seen or c["polarity"] != "neutral":
                    seen[key] = c["polarity"]
                self.authority_docs[key].add(i)
                self.authority_names.setdefault(key, c["name"])
                target = scr_to_doc.get(c.get("scr"))
                if target is not None and target != i:
                    self.in_corpus_cites[i].add(target)
                    self.cited_by[target].add(i)
            self.doc_authorities.append(seen)
        self.authority_idf = {
            k: math.log(1 + self.n / len(docs)) for k, docs in self.authority_docs.items()
        }
        self.authority_keys = list(self.authority_docs)

    def match_precedents(self, query: str) -> list[str]:
        keys = cit.find_named_precedents(query)
        matched = []
        for qk in keys:
            scored = [(cit.key_matches(qk, k), len(self.authority_docs[k]), k) for k in self.authority_keys]
            scored = [s for s in scored if s[0] >= 0.75]
            if not scored:
                continue
            best = max(s[0] for s in scored)
            # all spellings of the same authority (e.g. with/without "& Anr.")
            matched.extend(k for score, _, k in scored if score >= best - 0.05)
        if not keys:
            # Single distinctive party name used as shorthand ("Bhajan Lal", "Arnesh Kumar").
            lowered = " " + cit.normalize_party_name(query) + " "
            counts = Counter()
            for k in self.authority_keys:
                first = k.split(" v ")[0]
                if len(first.split()) >= 2 and first not in cit.GENERIC_PARTY and f" {first} " in lowered:
                    counts[k] = len(self.authority_docs[k])
            matched.extend(k for k, n in counts.items() if n >= 1)
        return list(dict.fromkeys(matched))

    # --------------------------------------------------
    # Query analysis
    # --------------------------------------------------

    def analyze(self, query: str, intent_override: str | None = None) -> QueryAnalysis:
        if intent_override:
            intent, scores, source = intent_override, {intent_override: 1.0}, "override"
        elif self.intent_model:
            intent, scores = self.intent_model.predict(query)
            source = "classifier"
        else:
            intent, scores, source = rule_based(query), {}, "rules"

        acts, act_names = extract_acts(query)
        provisions, labels = extract_provisions(query)
        explicit = list(provisions)
        act_map = {a: act_label(a, act_names[a][0]) for a in acts}
        for pid in explicit:
            a = pid.split(":")[0]
            if a != "?":
                act_map.setdefault(a, act_label(a))
        concepts = list(find_concepts(query))
        implied = [] if explicit else implied_provisions(concepts)
        precedents = self.match_precedents(query)
        return QueryAnalysis(
            text=query,
            intent=intent,
            intent_scores=scores,
            intent_source=source,
            acts=act_map,
            provisions=explicit,
            provision_labels=labels,
            implied_provisions=implied,
            concepts=concepts,
            precedents=precedents,
            precedent_names=[self.authority_names[k] for k in precedents][:5],
            outcome_requirements=out.query_outcome_requirements(query),
        )

    # --------------------------------------------------
    # Component scores
    # --------------------------------------------------

    @staticmethod
    def _act_salience(info: dict) -> float:
        if info.get("listed") or info.get("head"):
            return 1.0
        body = info.get("body", 0)
        return 0.6 if body >= 3 else (0.3 if body else 0.0)

    @staticmethod
    def _provision_salience(info: dict) -> float:
        if info.get("head"):
            return 1.0
        body = info.get("body", 0)
        return 0.6 if body >= 3 else (0.3 if body else 0.0)

    def legal_scores(self, qa: QueryAnalysis):
        """G(q,d) and its sub-scores A, S, I with applicability flags."""
        sub = WEIGHTS["legal_subweights"]
        A = np.zeros(self.n)
        S = np.zeros(self.n)
        I_tfidf = np.zeros(self.n)
        I_concept = np.zeros(self.n)

        query_provisions = qa.provisions or qa.implied_provisions
        provision_weight = 1.0 if qa.provisions else 0.5   # implied provisions are weak evidence

        for i, feat in enumerate(self.features):
            if qa.acts:
                A[i] = np.mean([self._act_salience(feat["acts"].get(a, {})) for a in qa.acts])
            if query_provisions:
                vals = []
                doc_pids = feat["provisions"]
                for qp in query_provisions:
                    matches = provision_matches(qp, doc_pids)
                    best = 0.0
                    for m in matches:
                        factor = 1.0 if (m.split(":")[0] == qp.split(":")[0] or qp.startswith("?:")) else 0.7
                        best = max(best, factor * self._provision_salience(doc_pids[m]))
                    vals.append(best)
                # implied provisions: one hit is enough evidence
                S[i] = np.mean(vals) if qa.provisions else max(vals)
            if qa.concepts:
                cov = 0.0
                for c in qa.concepts:
                    info = feat["concepts"].get(c)
                    if info:
                        cov += 1.0 if info["head"] else 0.5
                I_concept[i] = cov / len(qa.concepts)

        qvec = self.issue_vectorizer.transform([qa.text])
        if qvec.nnz:
            I_tfidf = (self.issue_matrix @ qvec.T).toarray().ravel()
            if I_tfidf.max() > 0:
                I_tfidf = I_tfidf / I_tfidf.max()
        I = 0.5 * I_tfidf + 0.5 * I_concept if qa.concepts else I_tfidf

        parts = {"issue": (I, sub["issue"])}
        if qa.acts:
            parts["act"] = (A, sub["act"])
        if query_provisions:
            parts["section"] = (S, sub["section"] * provision_weight)
        total = sum(w for _, w in parts.values())
        G = sum(v * w for v, w in parts.values()) / total
        return G, {"act": A, "section": S, "issue": I}, list(parts)

    def citation_scores(self, qa: QueryAnalysis, seeds: list[int]):
        """C(q,d). Named precedent: treatment-aware match. Otherwise
        citation pseudo-relevance feedback: authorities shared with the top
        seed judgments (bibliographic coupling), weighted by rarity and by
        treatment polarity for the precedent intents."""
        pol_w = POLARITY_WEIGHT.get(qa.intent, POLARITY_WEIGHT[None])
        C = np.zeros(self.n)
        evidence = [dict() for _ in range(self.n)]
        if qa.precedents:
            targets = set(qa.precedents)
            for i, auths in enumerate(self.doc_authorities):
                best, best_key = 0.0, None
                for k in targets & set(auths):
                    w = pol_w[auths[k]]
                    if w > best:
                        best, best_key = w, k
                if best_key:
                    C[i] = best
                    evidence[i] = {"mode": "named", "authority": self.authority_names[best_key],
                                   "treatment": auths[best_key]}
            return C, evidence, bool(C.any())

        seed_set = set(seeds)
        shared = Counter()
        for s in seeds:
            for k in self.doc_authorities[s]:
                if len(self.authority_docs[k]) >= 2:
                    shared[k] += 1
        if not shared:
            return C, evidence, False
        for i, auths in enumerate(self.doc_authorities):
            score, top = 0.0, []
            for k, pol in auths.items():
                n_seeds = shared.get(k, 0) - (1 if i in seed_set else 0)
                if n_seeds <= 0:
                    continue
                contrib = n_seeds * self.authority_idf[k] * pol_w[pol]
                score += contrib
                top.append((contrib, k, pol))
            # direct citation links with seed judgments inside the corpus
            linked = (self.in_corpus_cites[i] | self.cited_by[i]) & seed_set
            score += 2.0 * len(linked)
            if score:
                C[i] = math.log1p(score)
                top.sort(reverse=True)
                evidence[i] = {"mode": "shared", "authorities": [
                    {"name": self.authority_names[k], "treatment": p} for _, k, p in top[:3]],
                    "linked_seeds": len(linked)}
        return C, evidence, bool(C.any())

    def outcome_scores(self, qa: QueryAnalysis):
        O = np.full(self.n, 0.5)
        reasons = [""] * self.n
        if not qa.outcome_requirements:
            return O, reasons, False
        for i, feat in enumerate(self.features):
            O[i], reasons[i] = out.outcome_compatibility(qa.outcome_requirements, feat["outcome"])
        return O, reasons, True

    def semantic_scores(self, query: str, variant: str):
        if variant == "whole":
            return self.whole.scores(query), None
        chunk_sims = self.chunked.chunk_scores(query)
        agg = "mean3" if variant == "chunked_mean3" else "max"
        return self.chunked.scores(query, aggregation=agg, chunk_sims=chunk_sims), chunk_sims

    def compute_signals(self, qa: QueryAnalysis, semantic_variant="chunked_max", seed_k=10):
        """All raw signals for one query (shared by every system variant)."""
        timings = {}
        t = time.perf_counter()
        L = self.bm25.scores(qa.text)
        timings["bm25_ms"] = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        S, chunk_sims = self.semantic_scores(qa.text, semantic_variant)
        timings["semantic_ms"] = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        G, G_parts, G_applicable = self.legal_scores(qa)
        timings["legal_ms"] = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        all_mask = np.ones(self.n, dtype=bool)
        prelim = minmax(L, all_mask) + minmax(G, all_mask)
        seeds = list(np.argsort(-prelim)[:seed_k])
        C, C_evidence, C_applicable = self.citation_scores(qa, seeds)
        timings["citation_ms"] = (time.perf_counter() - t) * 1000

        O, O_reasons, O_applicable = self.outcome_scores(qa)
        return {
            "raw": {"bm25": L, "semantic": S, "legal": G, "citation": C, "outcome": O},
            "applicable": {"bm25": True, "semantic": True, "legal": True,
                           "citation": C_applicable, "outcome": O_applicable},
            "legal_parts": G_parts, "legal_applicable": G_applicable,
            "citation_evidence": C_evidence, "outcome_reasons": O_reasons,
            "chunk_sims": chunk_sims, "timings": timings,
        }

    # --------------------------------------------------
    # Fusion
    # --------------------------------------------------

    def candidate_pool(self, signals, use):
        pool = np.zeros(self.n, dtype=bool)
        k = config.CANDIDATES_PER_COMPONENT
        for name in ("bm25", "semantic", "legal", "citation"):
            if name in use and signals["applicable"][name]:
                raw = signals["raw"][name]
                top = np.argsort(-raw)[:k]
                pool[top[raw[top] > (0 if name == "citation" else -np.inf)]] = True
        return pool

    def fuse(self, signals, weights_row: dict, disabled=()):
        use = {s: w for s, w in weights_row.items()
               if w > 0 and s not in disabled and signals["applicable"][s]}
        pool = self.candidate_pool(signals, use)
        norm = {}
        for s in config.SIGNALS:
            raw = signals["raw"][s]
            norm[s] = raw.copy() if s == "outcome" else minmax(raw, pool)
        total = sum(use.values()) or 1.0
        eff = {s: w / total for s, w in use.items()}
        H = np.zeros(self.n)
        for s, w in eff.items():
            H += w * norm[s]
        H[~pool] = -np.inf
        return H, norm, eff, pool

    def rank(self, query: str, system: str, qa: QueryAnalysis | None = None, signals=None,
             semantic_variant="chunked_max", weights_name=None, pair_weights=None, disabled=(),
             intent_override=None):
        """Return (ordered doc indices, scores, context) for a named system.

        systems: bm25 | doc2vec_whole | doc2vec_chunked | legal_only |
                 bm25_legal | bm25_semantic | fixed_hybrid | adaptive_hybrid
        """
        qa = qa or self.analyze(query, intent_override)
        if signals is None:
            signals = self.compute_signals(qa, semantic_variant)
        ctx = {"qa": qa, "signals": signals}

        single = {"bm25", "doc2vec_whole", "doc2vec_chunked", "legal_only"}
        if system in single:
            if system == "bm25":
                scores = signals["raw"]["bm25"]
            elif system == "legal_only":
                scores = signals["raw"]["legal"]
            elif system == "doc2vec_whole":
                scores = signals["raw"]["semantic"] if semantic_variant == "whole" else self.whole.scores(query)
            else:
                scores = (signals["raw"]["semantic"] if semantic_variant.startswith("chunked")
                          else self.chunked.scores(query))
            order = np.argsort(-scores, kind="stable")
            if system == "bm25":
                order = order[scores[order] > 0]
            return order, scores, ctx

        if system in ("bm25_legal", "bm25_semantic"):
            row = pair_weights or WEIGHTS["pairwise"][system][1]
            H, norm, eff, pool = self.fuse(signals, row, disabled)
        else:
            if weights_name is None:
                weights_name = "F1_pdf_column_mean" if system == "fixed_hybrid" else "A1_pdf_table"
            table = resolve_weights(weights_name)
            H, norm, eff, pool = self.fuse(signals, table[qa.intent], disabled)
        ctx.update({"normalized": norm, "effective_weights": eff})
        order = np.argsort(-H, kind="stable")
        order = order[np.isfinite(H[order])]
        return order, H, ctx

    # --------------------------------------------------
    # Presentation
    # --------------------------------------------------

    def best_passage(self, case_index: int, qa: QueryAnalysis, max_chars=650) -> str:
        text = self.cases[case_index]["sections"]["judgment"] or self.cases[case_index]["text"]
        paragraphs = [p.strip() for p in re.split(r"\n+", text) if len(p.strip()) > 80]
        if not paragraphs:
            return normalize_space(text[:max_chars])
        q_tokens = set(content_tokens(qa.text))
        idf = self.bm25.bm25.idf
        phrases = []
        for c in qa.concepts:
            phrases.extend(CONCEPTS[c][0])
        prov_numbers = {p.split(":")[1].lstrip("sart") for p in qa.provisions}

        def score(par: str) -> float:
            toks = tokenize(par)
            tf = Counter(toks)
            s = sum(idf.get(t, 0) * (1 + math.log(tf[t])) for t in q_tokens if tf.get(t))
            low = par.lower()
            s += sum(3.0 for ph in phrases if ph in low)
            s += sum(4.0 for n in prov_numbers if re.search(rf"\b{re.escape(n)}\b", par))
            return s / (1 + 0.15 * math.log1p(len(toks)))

        best = max(paragraphs[:400], key=score)
        best = normalize_space(best)
        if len(best) <= max_chars:
            return best
        sentences = split_sentences(best)
        if not sentences:
            return best[:max_chars].rsplit(" ", 1)[0] + "…"
        ranked = max(range(len(sentences)), key=lambda i: score(sentences[i]))
        chosen, length, l, r = [sentences[ranked]], len(sentences[ranked]), ranked - 1, ranked + 1
        while True:
            added = False
            if r < len(sentences) and length + len(sentences[r]) < max_chars:
                chosen.append(sentences[r]); length += len(sentences[r]); r += 1; added = True
            if l >= 0 and length + len(sentences[l]) < max_chars:
                chosen.insert(0, sentences[l]); length += len(sentences[l]); l -= 1; added = True
            if not added:
                break
        passage = " ".join(chosen)
        if len(passage) > max_chars + 40:
            passage = passage[:max_chars].rsplit(" ", 1)[0]
        return ("… " if l >= 0 else "") + passage + (" …" if r < len(sentences) else "")

    def explain(self, i: int, qa: QueryAnalysis, ctx: dict) -> dict:
        feat = self.features[i]
        signals = ctx["signals"]
        matched_acts = []
        for a, label in qa.acts.items():
            info = feat["acts"].get(a)
            if info:
                where = "listed by the editor" if info["listed"] else (
                    "in the headnotes" if info["head"] else f"mentioned {info['body']}× in the judgment")
                matched_acts.append({"act": label, "where": where})
        matched_provisions = []
        for qp in (qa.provisions or qa.implied_provisions):
            for m in provision_matches(qp, feat["provisions"]):
                info = feat["provisions"][m]
                kind = "discussed in the headnotes" if info["head"] else (
                    f"discussed ({info['body']} mentions)" if info["body"] >= 3 else "mentioned in passing")
                matched_provisions.append({"provision": info["label"], "where": kind,
                                           "implied": not qa.provisions})
                break
        matched_concepts = [c for c in qa.concepts if c in feat["concepts"]]

        reasons = []
        if matched_provisions:
            reasons.append("discusses " + ", ".join(
                f"{p['provision']} ({p['where']})" for p in matched_provisions[:3]))
        elif matched_acts:
            reasons.append("deals with " + ", ".join(f"{a['act']} ({a['where']})" for a in matched_acts[:3]))
        if matched_concepts:
            reasons.append("addresses " + ", ".join(matched_concepts[:4]))
        ev = signals["citation_evidence"][i]
        if ev.get("mode") == "named":
            reasons.append(f"cites {ev['authority']} — treatment: {ev['treatment']}")
        elif ev.get("mode") == "shared" and ctx.get("effective_weights", {}).get("citation"):
            names = ", ".join(a["name"] for a in ev["authorities"][:2])
            if names:
                reasons.append(f"shares cited authorities with other top results ({names})")
        if signals["outcome_reasons"][i] and signals["applicable"]["outcome"]:
            reasons.append(signals["outcome_reasons"][i])

        contributions = {}
        if "normalized" in ctx:
            for s, w in ctx["effective_weights"].items():
                contributions[s] = round(float(w * ctx["normalized"][s][i]), 4)
            top = sorted(contributions.items(), key=lambda kv: -kv[1])
            lead = [s for s, v in top if v > 0][:2]
            if lead:
                reasons.insert(0, "strongest evidence: " + " and ".join(
                    {"bm25": "lexical match", "semantic": "semantic similarity", "legal": "legal/statutory match",
                     "citation": "citation links", "outcome": "outcome compatibility"}[s] for s in lead))
        return {
            "matched_acts": matched_acts,
            "matched_provisions": matched_provisions,
            "matched_concepts": matched_concepts,
            "citation": ev,
            "outcome": feat["outcome"],
            "contributions": contributions,
            "summary": "; ".join(reasons) if reasons else "ranked mainly on textual similarity",
        }

    def search(self, query: str, system="adaptive_hybrid", top_k=10, intent_override=None,
               weights_name=None, semantic_variant=None):
        t0 = time.perf_counter()
        semantic_variant = semantic_variant or DEFAULTS["semantic_variant"]
        if system == "adaptive_hybrid" and weights_name is None:
            weights_name = DEFAULTS["adaptive"]
        if system == "fixed_hybrid" and weights_name is None:
            weights_name = DEFAULTS["fixed"]
        qa = self.analyze(query, intent_override)
        signals = self.compute_signals(qa, semantic_variant)
        order, scores, ctx = self.rank(query, system, qa=qa, signals=signals,
                                       semantic_variant=semantic_variant, weights_name=weights_name)
        results = []
        for rank, i in enumerate(order[:top_k], start=1):
            case = self.cases[i]
            results.append({
                "rank": rank,
                "case_id": case["case_id"],
                "case_name": case["case_name"],
                "neutral_citation": case["neutral_citation"],
                "scr_citation": case.get("scr_citation"),
                "decision_date": case.get("decision_date"),
                "bench": case.get("judges"),
                "result": case["sections"].get("result"),
                "score": float(scores[i]),
                "signals": {s: round(float(signals["raw"][s][i]), 4) for s in config.SIGNALS},
                "normalized": ({s: round(float(v[i]), 4) for s, v in ctx["normalized"].items()}
                               if "normalized" in ctx else None),
                "issue": case["sections"].get("issue", "")[:600],
                "passage": self.best_passage(i, qa),
                "explanation": self.explain(i, qa, ctx),
            })
        payload = asdict(qa)
        payload["precedents"] = payload["precedents"][:5]
        return {
            "query": payload,
            "system": system,
            "weights_name": weights_name,
            "semantic_variant": semantic_variant,
            "effective_weights": ctx.get("effective_weights"),
            "applicable": signals["applicable"],
            "timings_ms": {**{k: round(v, 1) for k, v in signals["timings"].items()},
                           "total_ms": round((time.perf_counter() - t0) * 1000, 1)},
            "results": results,
        }


# Frozen configuration selected on the validation queries
# (written by evaluation/run_evaluation.py; defaults before selection).
DEFAULTS_PATH = config.INDEX_DIR / "selected_config.json"
DEFAULTS = {"semantic_variant": "chunked_max", "fixed": "F1_pdf_column_mean", "adaptive": "A1_pdf_table"}
if DEFAULTS_PATH.exists():
    DEFAULTS.update(json.loads(DEFAULTS_PATH.read_text(encoding="utf-8")))
