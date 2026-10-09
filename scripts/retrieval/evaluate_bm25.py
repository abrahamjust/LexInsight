
import heapq
import json
from pathlib import Path

from rank_bm25 import BM25Okapi
from bm25 import load_cases, tokenize, make_snippet


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "data" / "processed" / "bm25_evaluation_labels.json"

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


def get_top_results(query, cases, bm25):
    """Return the top five positive-scoring BM25 results."""
    query_tokens = tokenize(query)

    if not query_tokens:
        return []

    scores = bm25.get_scores(query_tokens)

    ranked = heapq.nlargest(
        TOP_K,
        (
            (float(score), idx)
            for idx, score in enumerate(scores)
            if score > 0
        ),
        key=lambda item: (item[0], -item[1]),
    )

    results = []

    for rank, (score, idx) in enumerate(ranked, start=1):
        case = cases[idx]

        snippet_result = make_snippet(
            case.get("judgment_text", ""),
            query_tokens,
        )

        # Current make_snippet returns (snippet, score).
        # Also support a version that returns only the snippet.
        if isinstance(snippet_result, tuple):
            snippet = snippet_result[0]
        else:
            snippet = snippet_result

        results.append({
            "rank": rank,
            "case_id": case.get("case_id", ""),
            "case_name": case.get("case_name", "Unknown"),
            "citation": case.get("neutral_citation", ""),
            "bm25_score": round(score, 4),
            "snippet": snippet,
        })

    return results


def ask_relevance(results):
    """Ask the user to label relevant results for one query."""
    print(
        "\nInspect the passages and, where needed, the full judgment "
        "before deciding relevance."
    )
    print("Enter relevant ranks, e.g. 1,3. Enter 0 if none are relevant.")

    valid_ranks = {result["rank"] for result in results}

    while True:
        answer = input("Relevant ranks: ").strip()

        if answer == "0":
            return []

        try:
            ranks = {
                int(value.strip())
                for value in answer.split(",")
            }
        except ValueError:
            print("Invalid input. Enter ranks such as 1,3 or 0.")
            continue

        if ranks and ranks.issubset(valid_ranks):
            return sorted(ranks)

        print(
            f"Please enter ranks from {sorted(valid_ranks)}, "
            "or 0 if none are relevant."
        )


def calculate_metrics(records):
    """Calculate Precision@5 and reciprocal rank."""
    if not records:
        return 0.0, 0.0

    precisions = []
    reciprocal_ranks = []

    for record in records:
        relevant_ranks = record["relevant_ranks"]

        # Precision@5 always uses five as the denominator.
        precision = len(relevant_ranks) / TOP_K
        precisions.append(precision)

        if relevant_ranks:
            reciprocal_rank = 1.0 / min(relevant_ranks)
        else:
            reciprocal_rank = 0.0

        reciprocal_ranks.append(reciprocal_rank)

    mean_precision = sum(precisions) / len(precisions)
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)

    return mean_precision, mrr


def save_results(records, completed):
    """Save labels and evaluation results as JSON."""
    precision, mrr = calculate_metrics(records)

    output_data = {
        "dataset_size": 111,
        "top_k": TOP_K,
        "completed_queries": completed,
        "total_queries": len(QUERIES),
        "metrics": {
            "precision_at_5": round(precision, 4),
            "mrr_at_5": round(mrr, 4),
        },
        "records": records,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("w", encoding="utf-8") as file:
        json.dump(output_data, file, indent=2, ensure_ascii=False)


def main():
    cases = load_cases()

    if not cases:
        print("No cases loaded. Check your dataset path.")
        return

    print(f"Loaded {len(cases)} judgments.")
    print("Building BM25 index...")

    corpus = [
        tokenize(case.get("judgment_text", ""))
        for case in cases
    ]

    bm25 = BM25Okapi(corpus)
    records = []

    print(f"\nStarting evaluation with {len(QUERIES)} queries.")

    for query_number, query in enumerate(QUERIES, start=1):
        print("\n" + "=" * 80)
        print(f"QUERY {query_number}/{len(QUERIES)}")
        print(f"Query: {query}")
        print("=" * 80)

        results = get_top_results(query, cases, bm25)

        if not results:
            print("No positive-scoring results for this query.")

            records.append({
                "query": query,
                "results": [],
                "relevant_ranks": [],
                "precision_at_5": 0.0,
                "reciprocal_rank": 0.0,
            })

            save_results(records, query_number)
            continue

        for result in results:
            print(f"\nRank: {result['rank']}")
            print(f"Case: {result['case_name']}")
            print(f"Citation: {result['citation']}")
            print(f"BM25 score: {result['bm25_score']:.4f}")
            print(f"Snippet: {result['snippet']}")
            print("-" * 80)

        relevant_ranks = ask_relevance(results)

        relevant_ids = [
            result["case_id"]
            for result in results
            if result["rank"] in relevant_ranks
        ]

        precision = len(relevant_ranks) / TOP_K
        reciprocal_rank = (
            1.0 / min(relevant_ranks)
            if relevant_ranks
            else 0.0
        )

        records.append({
            "query": query,
            "results": results,
            "relevant_ranks": relevant_ranks,
            "relevant_case_ids": relevant_ids,
            "precision_at_5": round(precision, 4),
            "reciprocal_rank": round(reciprocal_rank, 4),
        })

        save_results(records, query_number)

        print(f"\nQuery Precision@5: {precision:.4f}")
        print(f"Query Reciprocal Rank: {reciprocal_rank:.4f}")

    precision, mrr = calculate_metrics(records)

    print("\n" + "=" * 80)
    print("BM25 EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Queries evaluated: {len(records)}")
    print(f"Mean Precision@5: {precision:.4f}")
    print(f"MRR@5: {mrr:.4f}")
    print(f"Labels saved to: {OUTPUT}")


if __name__ == "__main__":
    main()
