
import json
import math
from pathlib import Path

import numpy as np

from bm25 import load_cases, tokenize, make_snippet
from doc2vec import train_model


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = (
    ROOT / "data" / "processed" / "doc2vec_evaluation_labels.json"
)

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


def cosine_similarity(vector_a, vector_b):
    denominator = (
        np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    )

    if denominator == 0:
        return 0.0

    return float(np.dot(vector_a, vector_b) / denominator)


def retrieve(query, cases, model):
    query_tokens = tokenize(query)

    if not query_tokens:
        return []

    query_vector = model.infer_vector(
        query_tokens,
        alpha=0.025,
        min_alpha=0.0001,
        epochs=100,
    )

    scored = []

    for index, case in enumerate(cases):
        tag = str(index)

        if tag not in model.dv:
            continue

        score = cosine_similarity(
            model.dv[tag],
            query_vector,
        )

        scored.append((score, index))

    # Include negative and zero similarities when selecting top five.
    scored.sort(key=lambda item: (-item[0], item[1]))

    return scored[:TOP_K]


def load_existing_results():
    if not OUTPUT_PATH.exists():
        return {
            "dataset_size": 0,
            "completed_queries": {},
        }

    with OUTPUT_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_results(data):
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def calculate_metrics(results):
    precisions = []
    reciprocal_ranks = []

    for entry in results.values():
        ranked_results = entry["results"]

        relevant_ranks = [
            result["rank"]
            for result in ranked_results
            if result["relevant"] is True
        ]

        precisions.append(len(relevant_ranks) / TOP_K)

        if relevant_ranks:
            reciprocal_ranks.append(1 / min(relevant_ranks))
        else:
            reciprocal_ranks.append(0.0)

    if not precisions:
        return 0.0, 0.0

    return (
        sum(precisions) / len(precisions),
        sum(reciprocal_ranks) / len(reciprocal_ranks),
    )


def main():
    cases = load_cases()

    if not cases:
        print("No judgments found in cases.jsonl.")
        return

    print(f"Loaded {len(cases)} judgments.")
    print("Training Doc2Vec model...")

    model = train_model(cases)
    saved_data = load_existing_results()

    saved_data["dataset_size"] = len(cases)
    saved_data.setdefault("completed_queries", {})

    for query_number, query in enumerate(QUERIES, start=1):
        query_key = str(query_number)

        print("\n" + "=" * 79)
        print(f"QUERY {query_number}/{len(QUERIES)}")
        print(f"Query: {query}")
        print("=" * 79)

        if query_key in saved_data["completed_queries"]:
            print("This query already has saved labels.")
            answer = input("Re-evaluate it? (y/N): ").strip().lower()

            if answer != "y":
                continue

        ranked = retrieve(query, cases, model)

        if not ranked:
            print("No results returned.")
            continue

        results = []

        query_tokens = tokenize(query)

        for rank, (score, index) in enumerate(ranked, start=1):
            case = cases[index]

            snippet_result = make_snippet(
                case.get("judgment_text", ""),
                query_tokens,
            )

            snippet = (
                snippet_result[0]
                if isinstance(snippet_result, tuple)
                else snippet_result
            )

            print(f"\nRank: {rank}")
            print(f"Case: {case.get('case_name', 'Unknown')}")
            print(
                f"Citation: "
                f"{case.get('neutral_citation', 'Unknown')}"
            )
            print(f"Cosine similarity: {score:.4f}")
            print(f"Snippet: {snippet}")
            print("-" * 79)

            results.append({
                "rank": rank,
                "case_id": case.get("case_id"),
                "case_name": case.get("case_name"),
                "neutral_citation": case.get("neutral_citation"),
                "similarity": score,
                "relevant": None,
            })

        print(
            "\nInspect the passages and, where needed, the full judgment."
        )
        print(
            "Enter relevant ranks separated by commas, "
            "or 0 if none are relevant."
        )
        print("Use the same topical relevance standard as before.")

        while True:
            raw = input("Relevant ranks: ").strip()

            try:
                if raw == "0":
                    relevant_ranks = set()
                else:
                    relevant_ranks = {
                        int(item.strip())
                        for item in raw.split(",")
                        if item.strip()
                    }

                valid_ranks = set(range(1, len(results) + 1))

                if not relevant_ranks.issubset(valid_ranks):
                    raise ValueError

                break
            except ValueError:
                print("Invalid input. Enter ranks such as 1,3 or 0.")

        for result in results:
            result["relevant"] = result["rank"] in relevant_ranks

        saved_data["completed_queries"][query_key] = {
            "query": query,
            "results": results,
        }

        save_results(saved_data)

        completed = saved_data["completed_queries"]
        p_at_5, mrr_at_5 = calculate_metrics(completed)

        print(f"\nQuery Precision@5: {len(relevant_ranks) / TOP_K:.4f}")
        print(
            "Query Reciprocal Rank: "
            f"{1 / min(relevant_ranks) if relevant_ranks else 0.0:.4f}"
        )
        print(f"Saved labels to: {OUTPUT_PATH}")

    completed = saved_data["completed_queries"]
    p_at_5, mrr_at_5 = calculate_metrics(completed)

    print("\n" + "=" * 79)
    print("DOC2VEC EVALUATION SUMMARY")
    print("=" * 79)
    print(f"Queries labelled: {len(completed)}")
    print(f"Mean Precision@5: {p_at_5:.4f}")
    print(f"MRR@5: {mrr_at_5:.4f}")
    print(f"Labels saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
