"""Query-intent classification (supervised, classical ML).

Run:  python -m lexinsight.intent

Features: word 1-2 gram TF-IDF + character 2-5 gram TF-IDF + a small
vector of explicit legal-reference features (provision cited, Act named,
case name "X v. Y" present, supporting / contrasting cue words, narrative
cues). Candidates: Logistic Regression and Linear SVM; a transparent
keyword-rule router is reported as a baseline.

Protocol: the query set is split by paraphrase group (StratifiedGroupKFold)
into train (75%) and held-out test (25%). Model selection uses grouped
5-fold cross-validation on the train part only. The selected model is
evaluated once on the test part, then refit on all queries for deployment
(retrieval-evaluation queries are a separate set, so this does not leak).
"""

import json
import pickle
import re

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from . import config
from .legal.citations import VERSUS_RE
from .legal.extract import PROVISION_RE, extract_acts

SUPPORT_CUES = re.compile(r"\b(rel(y|ied|ying) (on|upon)|follow(ed|ing)?|approv(ed|ing)|support\w*|in favour|affirm\w*|"
                          r"appl(y|ied|ying)|authorit(y|ies) (for|holding|supporting|in))\b", re.I)
CONTRAST_CUES = re.compile(r"\b(distinguish\w*|inapplicable|contrary|overrul\w*|disapprov\w*|doubt(ed)?|depart\w*|"
                           r"refus\w* to (follow|apply)|against|counter|not good law|per incuriam|opposite|disagree\w*|"
                           r"declin\w* to follow|did not follow|explained away|limited)\b", re.I)
NARRATIVE_CUES = re.compile(r"\b(my client|a (man|woman|girl|boy|person|farmer|tenant|builder|doctor|bank|student)|"
                            r"was (found|arrested|dismissed|killed|caught|convicted|transferred)|the (husband|wife|accused|"
                            r"victim|deceased|complainant) (was|is|had|says|claims)|years? (old|after|later)|after|"
                            r"took|gave|found|recovered|hit|shot|died)\b", re.I)
QUESTION_CUES = re.compile(r"^(can|is|are|whether|when|what|which|does|do|how|test|scope|principles?|doctrine|"
                           r"standard|role|effect|value|burden|rights?|concept|conditions|discretion)\b", re.I)


def legal_feature_vector(query: str) -> list[float]:
    provisions = len(PROVISION_RE.findall(query))
    acts, _ = extract_acts(query)
    words = len(query.split())
    return [
        min(provisions, 3) / 3,
        min(len(acts), 3) / 3,
        1.0 if VERSUS_RE.search(query) else 0.0,
        min(len(SUPPORT_CUES.findall(query)), 2) / 2,
        min(len(CONTRAST_CUES.findall(query)), 2) / 2,
        min(len(NARRATIVE_CUES.findall(query)), 4) / 4,
        1.0 if QUESTION_CUES.search(query.strip()) else 0.0,
        1.0 if query.strip().endswith("?") else 0.0,
        min(words, 40) / 40,
    ]


class LegalFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return csr_matrix(np.array([legal_feature_vector(q) for q in X]) * 2.0)


def make_pipeline(model: str = "logreg", C: float = 4.0) -> Pipeline:
    features = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1, lowercase=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2)),
        ("legal", LegalFeatures()),
    ])
    if model == "logreg":
        clf = LogisticRegression(C=C, max_iter=5000, class_weight="balanced")
    else:
        clf = LinearSVC(C=C / 4, class_weight="balanced")
    return Pipeline([("features", features), ("clf", clf)])


def rule_based(query: str) -> str:
    """Transparent keyword router used as a baseline (and for explanations)."""
    if CONTRAST_CUES.search(query):
        return "CONTRASTING_PRECEDENT"
    if SUPPORT_CUES.search(query) and (VERSUS_RE.search(query) or re.search(r"precedent|authorit|judgments? that|cases? that", query, re.I)):
        return "SUPPORTING_PRECEDENT"
    if PROVISION_RE.search(query) and len(query.split()) < 18:
        return "STATUTE_BASED"
    if len(NARRATIVE_CUES.findall(query)) >= 2 or len(query.split()) > 25:
        return "FACTUAL_SIMILARITY"
    return "LEGAL_ISSUE"


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class IntentClassifier:
    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline
        self.labels = list(pipeline.classes_)

    @classmethod
    def load(cls, path=config.INTENT_MODEL_PATH):
        with open(path, "rb") as f:
            return cls(pickle.load(f))

    def predict(self, query: str) -> tuple[str, dict]:
        clf = self.pipeline.named_steps["clf"]
        if hasattr(clf, "predict_proba"):
            probs = self.pipeline.predict_proba([query])[0]
        else:
            probs = softmax(self.pipeline.decision_function([query]))[0]
        scores = {label: float(p) for label, p in zip(self.labels, probs)}
        return max(scores, key=scores.get), scores


def load_dataset():
    data = json.loads((config.EVAL_QUERIES_DIR / "intent_queries.json").read_text(encoding="utf-8"))
    qs = data["queries"]
    return [q["text"] for q in qs], np.array([q["intent"] for q in qs]), np.array([q["group"] for q in qs]), qs


def grouped_cv_f1(texts, y, groups, model, C, folds=5):
    cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=config.SEED)
    scores = []
    texts = np.array(texts, dtype=object)
    for tr, va in cv.split(texts, y, groups):
        pipe = make_pipeline(model, C).fit(list(texts[tr]), y[tr])
        scores.append(f1_score(y[va], pipe.predict(list(texts[va])), average="macro"))
    return float(np.mean(scores)), float(np.std(scores))


def main():
    texts, y, groups, records = load_dataset()
    texts_arr = np.array(texts, dtype=object)
    splitter = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=config.SEED)
    train_idx, test_idx = next(splitter.split(texts_arr, y, groups))

    candidates = [("logreg", C) for C in (1.0, 4.0, 16.0)] + [("linearsvc", C) for C in (1.0, 4.0, 16.0)]
    selection = []
    for model, C in candidates:
        mean, std = grouped_cv_f1(texts_arr[train_idx], y[train_idx], groups[train_idx], model, C)
        selection.append({"model": model, "C": C, "cv_macro_f1": round(mean, 4), "cv_std": round(std, 4)})
        print(f"{model:10s} C={C:<5} grouped-CV macro-F1 {mean:.3f} ± {std:.3f}")
    best = max(selection, key=lambda r: (r["cv_macro_f1"], r["model"] == "logreg"))
    print("Selected:", best)

    pipe = make_pipeline(best["model"], best["C"]).fit(list(texts_arr[train_idx]), y[train_idx])
    y_test = y[test_idx]
    pred = pipe.predict(list(texts_arr[test_idx]))
    rule_pred = np.array([rule_based(t) for t in texts_arr[test_idx]])

    labels = config.INTENTS
    result = {
        "protocol": "StratifiedGroupKFold split by paraphrase group; 75% train / 25% held-out test; "
                    "model selection by grouped 5-fold CV on train only.",
        "n_train": int(len(train_idx)), "n_test": int(len(test_idx)),
        "model_selection": selection,
        "selected": best,
        "test": {
            "accuracy": round(accuracy_score(y_test, pred), 4),
            "macro_f1": round(f1_score(y_test, pred, average="macro"), 4),
            "per_class": classification_report(y_test, pred, labels=labels, output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(y_test, pred, labels=labels).tolist(),
        },
        "rule_baseline_test": {
            "accuracy": round(accuracy_score(y_test, rule_pred), 4),
            "macro_f1": round(f1_score(y_test, rule_pred, average="macro"), 4),
            "confusion_matrix": confusion_matrix(y_test, rule_pred, labels=labels).tolist(),
        },
        "labels": labels,
        "test_predictions": [
            {"text": records[i]["text"], "gold": records[i]["intent"], "pred": p, "rule": r}
            for i, p, r in zip(test_idx, pred, rule_pred)
        ],
    }
    print(f"Held-out test: accuracy {result['test']['accuracy']}, macro-F1 {result['test']['macro_f1']}")
    print(f"Rule baseline: accuracy {result['rule_baseline_test']['accuracy']}, "
          f"macro-F1 {result['rule_baseline_test']['macro_f1']}")

    config.EVAL_METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (config.EVAL_METRICS_DIR / "intent_classification.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    final = make_pipeline(best["model"], best["C"]).fit(texts, y)
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.INTENT_MODEL_PATH, "wb") as f:
        pickle.dump(final, f)
    print(f"Deployed model (refit on all {len(texts)} queries): {config.INTENT_MODEL_PATH}")


if __name__ == "__main__":
    # Run via the package module so pickled classes resolve to
    # lexinsight.intent (not __main__) when the API loads the model.
    from lexinsight.intent import main as package_main
    package_main()
