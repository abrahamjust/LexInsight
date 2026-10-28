"""Central paths and experiment settings for LexInsight."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"
RAW_PDF_DIR = DATA_DIR / "raw" / "pdf"
METADATA_DIR = DATA_DIR / "temp" / "metadata"
PROCESSED_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "index"

CORPUS_PATH = PROCESSED_DIR / "corpus.jsonl"
CORPUS_REPORT_PATH = PROCESSED_DIR / "corpus_report.json"
LEGAL_FEATURES_PATH = PROCESSED_DIR / "legal_features.jsonl"

BM25_INDEX_PATH = INDEX_DIR / "bm25.pkl"
DOC2VEC_PATH = INDEX_DIR / "doc2vec_whole.model"
CHUNKED_DOC2VEC_PATH = INDEX_DIR / "doc2vec_chunked.model"
CHUNK_MAP_PATH = INDEX_DIR / "doc2vec_chunk_map.json"
ISSUE_TFIDF_PATH = INDEX_DIR / "issue_tfidf.pkl"
INTENT_MODEL_PATH = INDEX_DIR / "intent_classifier.pkl"

EVAL_DIR = ROOT / "evaluation"
EVAL_QUERIES_DIR = EVAL_DIR / "queries"
EVAL_QRELS_DIR = EVAL_DIR / "qrels"
EVAL_RUNS_DIR = EVAL_DIR / "runs"
EVAL_METRICS_DIR = EVAL_DIR / "metrics"
EVAL_FIGURES_DIR = EVAL_DIR / "figures"

# Doc2Vec settings are kept identical to the pilot experiments so the
# whole-document and chunked baselines stay comparable with earlier runs.
VECTOR_SIZE = 100
WINDOW = 8
MIN_COUNT = 2
EPOCHS = 60
SEED = 42
INFER_EPOCHS = 100

CHUNK_SIZE = 400
CHUNK_OVERLAP = 100

TOP_K = 5
CANDIDATES_PER_COMPONENT = 100

INTENTS = [
    "FACTUAL_SIMILARITY",
    "LEGAL_ISSUE",
    "STATUTE_BASED",
    "SUPPORTING_PRECEDENT",
    "CONTRASTING_PRECEDENT",
]

SIGNALS = ["bm25", "semantic", "legal", "citation", "outcome"]
