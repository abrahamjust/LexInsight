"""Shared text utilities: tokenization and conservative cleaning."""

import re

TOKEN_RE = re.compile(r"\b[a-z0-9]+\b")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens (same tokenizer as the pilot BM25)."""
    return TOKEN_RE.findall(text.lower())


# Function words that carry no retrieval signal. Negations and legal
# relation words ("not", "without", "acquitted", "overruled") are kept on
# purpose because they change the meaning of a passage.
STOPWORDS = {
    "a", "an", "the", "of", "to", "in", "for", "on", "by", "with", "at",
    "from", "as", "is", "are", "was", "were", "be", "been", "being", "and",
    "or", "that", "this", "these", "those", "it", "its", "which", "who",
    "whom", "what", "when", "where", "can", "could", "would", "should",
    "may", "might", "shall", "will", "do", "does", "did", "has", "have",
    "had", "there", "their", "them", "they", "he", "she", "his", "her",
    "i", "we", "you", "me", "my", "our", "your", "find", "cases", "case",
    "judgments", "judgment", "show", "give", "list", "any", "about",
    "supreme", "court", "whether", "how", "under",
}


def content_tokens(text: str) -> list[str]:
    return [t for t in tokenize(text) if t not in STOPWORDS]


def normalize_space(text: str) -> str:
    text = text.replace(" ", " ").replace(" ", " ")
    text = text.replace("​", "").replace(" ", " ")
    return re.sub(r"\s+", " ", text).strip()


def split_sentences(text: str) -> list[str]:
    # Avoid splitting on legal abbreviations such as "s.", "ss.", "u/s.",
    # "v.", "No.", "Cr.P.C." by requiring an uppercase start afterwards and
    # a word of at least three letters before the full stop.
    parts = re.split(r"(?<=[a-z\)\]\"”’]{3}[.!?])\s+(?=[A-Z\"“(])", text)
    return [p.strip() for p in parts if p.strip()]
