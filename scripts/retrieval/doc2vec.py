
import json
from pathlib import Path

from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from bm25 import load_cases, tokenize, make_snippet


ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "data" / "processed" / "doc2vec.model"

VECTOR_SIZE = 100
WINDOW = 8
MIN_COUNT = 2
EPOCHS = 60
TOP_K = 5


def prepare_documents(cases):
    """Tokenize each judgment and attach a unique document ID."""
    documents = []

    for index, case in enumerate(cases):
        text = case.get("judgment_text", "")
        tokens = tokenize(text)

        if tokens:
            documents.append(
                TaggedDocument(words=tokens, tags=[str(index)])
            )

    return documents


def train_model(cases):
    """Train Doc2Vec on the available judgments."""
    documents = prepare_documents(cases)

    if len(documents) < 2:
        raise ValueError(
            "At least two non-empty judgments are needed to train Doc2Vec."
        )

    print(f"Training Doc2Vec on {len(documents)} judgments...")

    model = Doc2Vec(
        vector_size=VECTOR_SIZE,
        window=WINDOW,
        min_count=MIN_COUNT,
        workers=1,
        epochs=EPOCHS,
        dm=1,
        seed=42,
    )

    model.build_vocab(documents)
    model.train(
        documents,
        total_examples=model.corpus_count,
        epochs=model.epochs,
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(MODEL_PATH))

    print(f"Model saved to: {MODEL_PATH}")
    return model


def retrieve(query, cases, model, top_k=TOP_K):
    """Retrieve judgments using inferred query-vector similarity."""
    query_tokens = tokenize(query)

    if not query_tokens:
        print("Please enter a query containing searchable words.")
        return

    query_vector = model.infer_vector(
        query_tokens,
        alpha=0.025,
        min_alpha=0.0001,
        epochs=100,
    )

    # Score each judgment using cosine similarity.
    scored_cases = []

    for index, case in enumerate(cases):
        tag = str(index)

        if tag not in model.dv:
            continue

        similarity = float(
            model.dv.similarity(tag, query_vector)
        )

        scored_cases.append((similarity, index))

    scored_cases.sort(
        key=lambda item: (-item[0], item[1])
    )

    results = [
        item for item in scored_cases
        if item[0] > 0
    ][:top_k]

    if not results:
        print("No positive-similarity results found.")
        return

    print("\n" + "=" * 80)
    print(f"DOC2VEC RESULTS FOR: {query}")
    print("=" * 80)

    for rank, (score, index) in enumerate(results, start=1):
        case = cases[index]

        snippet_result = make_snippet(
            case.get("judgment_text", ""),
            query_tokens,
        )

        if isinstance(snippet_result, tuple):
            snippet = snippet_result[0]
        else:
            snippet = snippet_result

        print(f"\nRank: {rank}")
        print(f"Case: {case.get('case_name', 'Unknown')}")
        print(
            f"Citation: "
            f"{case.get('neutral_citation', 'Unknown')}"
        )
        print(
            f"Decision date: "
            f"{case.get('decision_date', 'Unknown')}"
        )
        print(f"Doc2Vec cosine similarity: {score:.4f}")
        print(f"Snippet: {snippet}")
        print("-" * 80)


def main():
    cases = load_cases()

    if not cases:
        print("No judgments found. Check your cases.jsonl file.")
        return

    model = train_model(cases)

    print(f"\nLoaded {len(cases)} judgments.")
    print("Enter a legal query, or type 'exit' to quit.")

    while True:
        query = input("\nLegal query: ").strip()

        if query.lower() in {"exit", "quit"}:
            break

        if not query:
            continue

        retrieve(query, cases, model)


if __name__ == "__main__":
    main()
