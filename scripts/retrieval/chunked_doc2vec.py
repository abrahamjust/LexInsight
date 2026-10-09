
import json
from pathlib import Path

import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from bm25 import load_cases, tokenize, make_snippet


# --------------------------------------------------
# Paths and experiment settings
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = ROOT / "data" / "processed" / "chunked_doc2vec.model"
LABELS_PATH = ROOT / "data" / "processed" / "chunked_doc2vec_evaluation_labels.json"

CHUNK_SIZE = 400
CHUNK_OVERLAP = 100
VECTOR_SIZE = 100
WINDOW = 8
MIN_COUNT = 2
EPOCHS = 60
TOP_K = 5

QUERIES = [
    "circumstantial evidence chain of circumstances reasonable doubt",
    "anticipatory bail factors for granting bail",
    "Article 21 right to life and personal liberty",
    "Section 302 IPC murder conviction",
    "principles of natural justice opportunity of hearing",
    "recovery of evidence disclosure statement forensic report",
    "constitutional validity of statute fundamental rights",
    "land acquisition compensation fair market value",
    "service law disciplinary proceedings natural justice",
    "NDPS Act conscious possession recovery",
]


# --------------------------------------------------
# Chunking
# --------------------------------------------------

def split_into_chunks(tokens, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split tokens into overlapping chunks."""
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


def prepare_chunk_documents(cases):
    """
    Create training passages while remembering which
    original judgment each passage belongs to.
    """
    tagged_documents = []
    chunk_to_case = {}

    for case_index, case in enumerate(cases):
        tokens = tokenize(case.get("judgment_text", ""))
        chunks = split_into_chunks(tokens)

        for chunk_index, chunk in enumerate(chunks):
            tag = f"case{case_index}_chunk{chunk_index}"

            tagged_documents.append(
                TaggedDocument(words=chunk, tags=[tag])
            )

            chunk_to_case[tag] = case_index

    return tagged_documents, chunk_to_case


# --------------------------------------------------
# Model training
# --------------------------------------------------

def train_chunked_model(cases):
    tagged_documents, chunk_to_case = prepare_chunk_documents(cases)

    if not tagged_documents:
        raise ValueError("No chunks were created. Check the dataset.")

    model = Doc2Vec(
        vector_size=VECTOR_SIZE,
        window=WINDOW,
        min_count=MIN_COUNT,
        workers=1,
        epochs=EPOCHS,
        dm=1,
        seed=42,
    )

    model.build_vocab(tagged_documents)
    model.train(
        tagged_documents,
        total_examples=model.corpus_count,
        epochs=model.epochs,
    )

    model.save(str(MODEL_PATH))

    print(f"Original judgments: {len(cases)}")
    print(f"Training chunks: {len(tagged_documents)}")
    print(f"Vocabulary size: {len(model.wv)}")
    print(f"Model saved to: {MODEL_PATH}")

    return model, chunk_to_case, len(tagged_documents)


# --------------------------------------------------
# Retrieval
# --------------------------------------------------

def retrieve(query, cases, model, chunk_to_case):
    query_tokens = tokenize(query)

    if not query_tokens:
        return []

    query_vector = model.infer_vector(
        query_tokens,
        alpha=0.025,
        min_alpha=0.0001,
        epochs=100,
    )

    # Keep the best matching chunk score for each judgment.
    best_case_scores = {}

    for tag, case_index in chunk_to_case.items():
        chunk_vector = model.dv[tag]

        denominator = (
            np.linalg.norm(chunk_vector)
            * np.linalg.norm(query_vector)
        )

        if denominator == 0:
            similarity = 0.0
        else:
            similarity = float(
                np.dot(chunk_vector, query_vector) / denominator
            )

        previous_score = best_case_scores.get(case_index)

        if previous_score is None or similarity > previous_score:
            best_case_scores[case_index] = similarity

    ranked = sorted(
        best_case_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    results = []

    for case_index, score in ranked[:TOP_K]:
        case = cases[case_index]
        snippet, _ = make_snippet(
            case.get("judgment_text", ""),
            query_tokens,
        )

        results.append({
            "case_index": case_index,
            "case_id": case.get("case_id"),
            "case_name": case.get("case_name", "Unknown case"),
            "neutral_citation": case.get("neutral_citation", ""),
            "decision_date": case.get("decision_date", ""),
            "score": score,
            "snippet": snippet,
        })

    return results


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

def calculate_metrics(results_by_query):
    precision_values = []
    reciprocal_ranks = []

    for result in results_by_query:
        relevant_ranks = [
            rank
            for rank, item in enumerate(result["results"], start=1)
            if item["relevant"]
        ]

        # Precision@5 uses five as the fixed denominator,
        # even if fewer than five results are returned.
        precision_values.append(len(relevant_ranks) / TOP_K)

        reciprocal_ranks.append(
            1 / min(relevant_ranks) if relevant_ranks else 0.0
        )

    return {
        "queries_evaluated": len(results_by_query),
        "mean_precision_at_5": (
            sum(precision_values) / len(precision_values)
            if precision_values else 0.0
        ),
        "mrr_at_5": (
            sum(reciprocal_ranks) / len(reciprocal_ranks)
            if reciprocal_ranks else 0.0
        ),
    }


def save_progress(dataset_size, results_by_query, total_chunks):
    output = {
        "experiment": "chunked_doc2vec",
        "dataset_size": dataset_size,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "total_chunks": total_chunks,
        "top_k": TOP_K,
        "results": results_by_query,
        "metrics": calculate_metrics(results_by_query),
    }

    LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)

    with LABELS_PATH.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2, ensure_ascii=False)


def main():
    cases = load_cases()

    if not cases:
        raise ValueError("No judgments were loaded from cases.jsonl.")

    print("Training a separate chunked Doc2Vec model...")
    model, chunk_to_case, total_chunks = train_chunked_model(cases)

    results_by_query = []

    for query_number, query in enumerate(QUERIES, start=1):
        results = retrieve(query, cases, model, chunk_to_case)

        print("\n" + "=" * 80)
        print(f"QUERY {query_number}/{len(QUERIES)}: {query}")
        print("=" * 80)

        for rank, item in enumerate(results, start=1):
            print(f"\nRank {rank} | Similarity: {item['score']:.4f}")
            print(f"Case: {item['case_name']}")
            print(f"Citation: {item['neutral_citation']}")
            print(f"Date: {item['decision_date']}")
            print(f"Snippet: {item['snippet']}")

        print(
            "\nMark which ranks are relevant to the query, "
            "using the same relevance standard as before."
        )
        print("Enter ranks separated by commas, e.g. 1,3,5.")
        print("Press Enter if none are relevant.")

        while True:
            answer = input("Relevant ranks: ").strip()

            try:
                relevant_ranks = {
                    int(part.strip())
                    for part in answer.split(",")
                    if part.strip()
                }

                if any(rank < 1 or rank > len(results)
                       for rank in relevant_ranks):
                    print("Enter only ranks shown above.")
                    continue

                break
            except ValueError:
                print("Please enter rank numbers separated by commas.")

        labelled_results = []

        for rank, item in enumerate(results, start=1):
            labelled_results.append({
                **item,
                "relevant": rank in relevant_ranks,
            })

        results_by_query.append({
            "query": query,
            "results": labelled_results,
        })

        # Save after every query so labels are not lost if interrupted.
        save_progress(len(cases), results_by_query, total_chunks)

        metrics = calculate_metrics(results_by_query)
        print(
            f"Progress — labelled queries: "
            f"{metrics['queries_evaluated']}/{len(QUERIES)}"
        )

    metrics = calculate_metrics(results_by_query)

    print("\n" + "=" * 80)
    print("CHUNKED DOC2VEC EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Judgments: {len(cases)}")
    print(f"Chunks: {total_chunks}")
    print(f"Mean Precision@5: {metrics['mean_precision_at_5']:.4f}")
    print(f"MRR@5: {metrics['mrr_at_5']:.4f}")
    print(f"Model: {MODEL_PATH}")
    print(f"Labels: {LABELS_PATH}")


if __name__ == "__main__":
    main()
