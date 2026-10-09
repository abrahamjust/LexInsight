
import itertools
import numpy as np

from gensim.models.doc2vec import Doc2Vec

from bm25 import load_cases, tokenize
from doc2vec import MODEL_PATH, train_model


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

TOP_K = 5


def cosine_similarity(a, b):
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator == 0:
        return 0.0
    return float(np.dot(a, b) / denominator)


def rank_documents(query, cases, model):
    tokens = tokenize(query)

    vector = model.infer_vector(
        tokens,
        alpha=0.025,
        min_alpha=0.0001,
        epochs=100,
    )

    ranked = []

    for index, case in enumerate(cases):
        tag = str(index)

        if tag not in model.dv.key_to_index:
            continue

        score = cosine_similarity(model.dv[tag], vector)
        ranked.append((score, index))

    ranked.sort(key=lambda x: (-x[0], x[1]))
    return tokens, ranked


def main():
    cases = load_cases()

    if not cases:
        print("ERROR: No judgments loaded.")
        return

    if MODEL_PATH.exists():
        model = Doc2Vec.load(str(MODEL_PATH))
        print(f"Loaded existing model: {MODEL_PATH}")
    else:
        print("No saved model found. Training a new one...")
        model = train_model(cases)

    print("\n" + "=" * 75)
    print("1. DATASET AND MODEL INTEGRITY")
    print("=" * 75)

    print(f"Judgments loaded: {len(cases)}")
    print(f"Document vectors: {len(model.dv)}")
    print(f"Vocabulary size: {len(model.wv)}")
    print(f"Vector dimensions: {model.vector_size}")
    print(f"Training epochs configured: {model.epochs}")
    print(f"Minimum word count: {model.min_count}")

    missing_tags = [
        str(i) for i in range(len(cases))
        if str(i) not in model.dv.key_to_index
    ]
    print(f"Judgments missing vectors: {len(missing_tags)}")

    lengths = [
        len(tokenize(case.get("judgment_text", "")))
        for case in cases
    ]

    print(f"Document tokens — minimum: {min(lengths)}")
    print(f"Document tokens — median: {np.median(lengths):.0f}")
    print(f"Document tokens — maximum: {max(lengths)}")

    print("\n" + "=" * 75)
    print("2. VOCABULARY COVERAGE FOR THE QUERIES")
    print("=" * 75)

    vocabulary = model.wv.key_to_index

    for query in QUERIES:
        tokens = tokenize(query)
        known = [token for token in tokens if token in vocabulary]
        unknown = [token for token in tokens if token not in vocabulary]

        coverage = len(known) / len(tokens) if tokens else 0.0

        print(f"\nQuery: {query}")
        print(
            f"Known tokens: {len(known)}/{len(tokens)} "
            f"({coverage:.1%})"
        )
        print(f"Unknown tokens: {unknown or 'None'}")

    print("\n" + "=" * 75)
    print("3. DOCUMENT-VECTOR DIVERSITY")
    print("=" * 75)

    available_vectors = [
        model.dv[str(i)]
        for i in range(len(cases))
        if str(i) in model.dv.key_to_index
    ]

    vectors = np.asarray(available_vectors, dtype=np.float64)
    norms = np.linalg.norm(vectors, axis=1)

    print(
        f"Vector norms — min: {norms.min():.4f}, "
        f"median: {np.median(norms):.4f}, "
        f"max: {norms.max():.4f}"
    )

    normalized = vectors / np.maximum(norms[:, None], 1e-12)
    similarity_matrix = normalized @ normalized.T

    upper = similarity_matrix[
        np.triu_indices(len(vectors), k=1)
    ]

    print(
        f"Pairwise cosine similarity — min: {upper.min():.4f}, "
        f"median: {np.median(upper):.4f}, "
        f"mean: {upper.mean():.4f}, "
        f"max: {upper.max():.4f}"
    )

    rounded_vectors = np.unique(
        np.round(vectors, decimals=5),
        axis=0,
    )
    print(
        f"Unique vectors after rounding to 5 decimals: "
        f"{len(rounded_vectors)}/{len(vectors)}"
    )

    print("\n" + "=" * 75)
    print("4. QUERY RANKING AND INFERENCE STABILITY")
    print("=" * 75)

    all_top_sets = []

    for query in QUERIES:
        tokens, ranked = rank_documents(query, cases, model)

        if not ranked:
            print(f"\nNo results for: {query}")
            continue

        top = ranked[:TOP_K]
        all_top_sets.append({index for _, index in top})

        print(f"\nQuery: {query}")

        for rank, (score, index) in enumerate(top, start=1):
            case = cases[index]
            print(
                f"{rank}. {case.get('case_name', 'Unknown')} "
                f"| {case.get('neutral_citation', 'Unknown')} "
                f"| cosine={score:.4f}"
            )

        # Repeated inference reveals whether rankings are unstable.
        repeated_rankings = []

        for _ in range(3):
            _, repeat_results = rank_documents(query, cases, model)
            repeated_rankings.append(
                [index for _, index in repeat_results[:TOP_K]]
            )

        overlaps = []

        for a, b in itertools.combinations(repeated_rankings, 2):
            overlaps.append(
                len(set(a) & set(b)) / TOP_K
            )

        print(
            "Repeated-inference top-5 overlap "
            f"(mean across 3 runs): {np.mean(overlaps):.2f}"
        )

    if len(all_top_sets) >= 2:
        overlaps = []

        for a, b in itertools.combinations(all_top_sets, 2):
            overlaps.append(len(a & b) / TOP_K)

        print(
            "\nMean top-5 set overlap between different queries: "
            f"{np.mean(overlaps):.2f}"
        )
        print(
            "Higher overlap means more repeated results across queries; "
            "some overlap is expected for related queries."
        )

    print("\nAudit complete. No evaluation labels were modified.")


if __name__ == "__main__":
    main()
