
import json
import re
import heapq
from pathlib import Path

from rank_bm25 import BM25Okapi


# Base directory configurations
ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "data" / "processed" / "cases.jsonl"


def tokenize(text: str) -> list[str]:
    """Convert text into lowercase alphanumeric word tokens."""
    return re.findall(r"\b[a-z0-9]+\b", text.lower())


def load_cases() -> list[dict]:
    """Load cases from the processed JSONL dataset."""
    cases = []

    if not DATASET.exists():
        print(f"Error: Dataset file not found at {DATASET}")
        return cases

    with DATASET.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                case = json.loads(line)
                cases.append(case)
            except json.JSONDecodeError as exc:
                print(
                    f"Skipping malformed JSON at line "
                    f"{line_number}: {exc}"
                )

    return cases



def clean_extracted_text(text: str) -> str:
    """Clean common PDF extraction artifacts conservatively."""
    # Normalize whitespace.
    text = text.replace("\u00a0", " ")
    text = text.replace("\u200b", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s*([–—])\s*", r" \1 ", text)

    # Restore spacing after punctuation when it is clearly missing.
    # Avoid changing decimal numbers and common legal abbreviations.
    text = re.sub(r"(?<=[.!?])(?=[A-Z])", " ", text)

    # Normalize spaces introduced by line breaks.
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def make_snippet(
    text: str,
    query_tokens: list[str],
    window: int = 450,
) -> str:
    """Select a readable passage using query terms and legal phrases."""
    text = clean_extracted_text(text)

    if not text or not query_tokens:
        return "No relevant passage found."

    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return text[:window]

    query_terms = set(query_tokens)

    # Phrases that may be meaningful in legal retrieval.
    legal_phrases = [
        "circumstantial evidence",
        "chain of circumstances",
        "chain of evidence",
        "beyond reasonable doubt",
        "reasonable doubt",
        "eyewitness",
        "eye witness",
        "prosecution case",
        "burden of proof",
        "standard of proof",
        "proved guilty",
        "alternative hypothesis",
    ]

    query_text = " ".join(query_tokens)

    # Prioritize phrases explicitly present in the user's query.
    query_phrases = [
        phrase for phrase in legal_phrases
        if phrase in query_text
    ]

    # Generic words receive less weight than distinctive query terms.
    generic_terms = {
        "whether", "the", "can", "be", "based", "on",
        "there", "are", "no", "when", "is", "was",
        "a", "an", "and", "of", "to", "in", "for",
        "accused", "case", "court", "evidence",
    }

    distinctive_terms = query_terms - generic_terms

    def score_sentence(sentence: str) -> float:
        sentence_lower = sentence.lower()
        sentence_terms = set(tokenize(sentence))

        # Weighted term coverage.
        distinctive_matches = len(distinctive_terms & sentence_terms)
        generic_matches = len(
            (query_terms & generic_terms) & sentence_terms
        )

        score = (
            2.0 * distinctive_matches
            + 0.25 * generic_matches
        )

        # Strong bonus for exact query phrases.
        for phrase in query_phrases:
            if phrase in sentence_lower:
                score += 5.0

        # Small bonus for other legal phrases only when relevant
        # concepts are present in the query.
        for phrase in legal_phrases:
            if phrase in sentence_lower and phrase in query_text:
                score += 1.0

        # Slightly prefer complete, informative sentences.
        if len(sentence.split()) >= 8:
            score += 0.25

        return score

    scores = [score_sentence(s) for s in sentences]

    best_index = max(
        range(len(sentences)),
        key=lambda i: (scores[i], -i),
    )

    if scores[best_index] <= 0:
        return text[:window] + ("..." if len(text) > window else "")

    # Start with the highest-scoring sentence.
    selected = [sentences[best_index]]
    total_length = len(sentences[best_index])

    # Add neighboring sentences while keeping the passage concise.
    left = best_index - 1
    right = best_index + 1

    while True:
        added = False

        if left >= 0 and total_length + len(sentences[left]) + 1 <= window:
            selected.insert(0, sentences[left])
            total_length += len(sentences[left]) + 1
            left -= 1
            added = True

        if (
            right < len(sentences)
            and total_length + len(sentences[right]) + 1 <= window
        ):
            selected.append(sentences[right])
            total_length += len(sentences[right]) + 1
            right += 1
            added = True

        if not added:
            break

    snippet = " ".join(selected)

    if left >= 0:
        snippet = "..." + snippet

    if right < len(sentences):
        snippet += "..."

    if len(snippet) > window + 10:
        snippet = snippet[:window].rsplit(" ", 1)[0] + "..."

    return snippet, scores[best_index]

def main():
    cases = load_cases()

    if not cases:
        print("No cases found or dataset is empty.")
        return

    print("Indexing documents... Please wait.")

    # Tokenize each judgment.
    tokenized_corpus = [
        tokenize(case.get("judgment_text", ""))
        for case in cases
    ]

    # Build the BM25 index.
    bm25 = BM25Okapi(tokenized_corpus)

    print(f"BM25 index ready: {len(cases)} judgments loaded.")

    top_k = 5

    while True:
        try:
            query = input(
                "\nEnter a legal query (or 'exit'): "
            ).strip()

        except (KeyboardInterrupt, EOFError):
            print("\nExiting search engine.")
            break

        if query.lower() == "exit":
            break

        query_tokens = tokenize(query)

        if not query_tokens:
            print(
                "Please enter a valid search term "
                "containing letters or numbers."
            )
            continue

        # Calculate BM25 scores for all judgments.
        scores = bm25.get_scores(query_tokens)

        # Select the top results with positive scores.
        top_results = heapq.nlargest(
            top_k,
            (
                (score, idx)
                for idx, score in enumerate(scores)
                if score > 0
            ),
            key=lambda item: (item[0], -item[1]),
        )

        if not top_results:
            print("No lexical matches found.")
            continue

        print("\n" + "=" * 70)
        print(f"Top {len(top_results)} Matching Judgments")
        print("=" * 70)

        for rank, (score, index) in enumerate(
            top_results,
            start=1,
        ):
            case = cases[index]
            judgment_text = case.get("judgment_text", "")

            print(f"\n[Rank {rank}] BM25 Score: {score:.4f}")
            print(
                "Case Title: ",
                case.get("case_name") or "Unknown Title",
            )
            print(
                "Citation:   ",
                case.get("neutral_citation") or "Not available",
            )
            print(
                "Decision Date: ",
                case.get("decision_date") or "Not available",
            )
            snippet, snippet_score = make_snippet(
                judgment_text,
                query_tokens,
            )

            print(f"Snippet score: {snippet_score:.4f}")
            print(f"Snippet:       {snippet}")
            print("-" * 70)


if __name__ == "__main__":
    main()
