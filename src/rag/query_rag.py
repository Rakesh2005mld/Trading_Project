from pathlib import Path
import argparse
import json
from collections import defaultdict

import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAG_DIR = PROJECT_ROOT / "models" / "rag"

EMBEDDINGS_FILE = RAG_DIR / "embeddings.npy"
METADATA_FILE = RAG_DIR / "metadata.json"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

DEFAULT_TOP_K = 5

# Retrieve a larger pool before diversification.
CANDIDATE_POOL = 30

# MMR tradeoff:
# higher -> relevance
# lower  -> diversity
MMR_LAMBDA = 0.75


# ============================================================
# LOAD
# ============================================================

def load_rag():

    embeddings = np.load(
        EMBEDDINGS_FILE
    )

    with open(
        METADATA_FILE,
        "r",
        encoding="utf-8",
    ) as f:
        metadata = json.load(f)

    if len(embeddings) != len(metadata):
        raise ValueError(
            "Embeddings and metadata lengths differ."
        )

    model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    return model, embeddings, metadata


# ============================================================
# DATE
# ============================================================

def normalize_date(value):

    if not value:
        return None

    value = str(value)

    if value == "UNKNOWN":
        return None

    return value[:10]


# ============================================================
# FILTER
# ============================================================

def filter_candidates(
    metadata,
    ticker=None,
    as_of_date=None,
):

    indices = []

    for idx, doc in enumerate(metadata):

        if ticker is not None:

            if (
                str(doc.get("ticker", "")).upper()
                != ticker.upper()
            ):
                continue

        if as_of_date is not None:

            filing_date = normalize_date(
                doc.get("filing_date")
            )

            if filing_date is None:
                continue

            if filing_date > as_of_date:
                continue

        indices.append(idx)

    return indices


# ============================================================
# MMR
# ============================================================

def mmr_select(
    candidate_indices,
    embeddings,
    scores,
    metadata,
    top_k,
):

    if len(candidate_indices) <= top_k:
        return candidate_indices

    # Sort candidate indices by relevance.
    order = np.argsort(scores)[::-1]

    ranked_indices = [
        candidate_indices[i]
        for i in order
    ]

    ranked_embeddings = embeddings[
        ranked_indices
    ]

    selected_positions = []

    # First result = highest similarity.
    selected_positions.append(0)

    while len(selected_positions) < top_k:

        best_position = None
        best_score = -float("inf")

        for pos in range(
            len(ranked_indices)
        ):

            if pos in selected_positions:
                continue

            relevance = scores[
                order[pos]
            ]

            selected_embeddings = (
                ranked_embeddings[
                    selected_positions
                ]
            )

            # Maximum similarity to something
            # already selected.
            diversity_similarity = np.max(
                selected_embeddings
                @ ranked_embeddings[pos]
            )

            mmr_score = (
                MMR_LAMBDA * relevance
                - (1 - MMR_LAMBDA)
                * diversity_similarity
            )

            # Small preference for different
            # source filings.
            candidate_doc = metadata[
                ranked_indices[pos]
            ]

            candidate_source = candidate_doc.get(
                "source",
                ""
            )

            selected_sources = {
                metadata[
                    ranked_indices[p]
                ].get("source", "")
                for p in selected_positions
            }

            if (
                candidate_source
                not in selected_sources
            ):
                mmr_score += 0.02

            if mmr_score > best_score:

                best_score = mmr_score
                best_position = pos

        selected_positions.append(
            best_position
        )

    return [
        ranked_indices[pos]
        for pos in selected_positions
    ]


# ============================================================
# SEARCH
# ============================================================

def search(
    query,
    ticker=None,
    as_of_date=None,
    top_k=DEFAULT_TOP_K,
):

    model, embeddings, metadata = load_rag()

    candidate_indices = filter_candidates(
        metadata,
        ticker=ticker,
        as_of_date=as_of_date,
    )

    if not candidate_indices:
        return []

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0]

    candidate_embeddings = embeddings[
        candidate_indices
    ]

    scores = (
        candidate_embeddings
        @ query_embedding
    )

    # --------------------------------------------------------
    # Initial relevance filtering
    # --------------------------------------------------------

    pool_size = min(
        CANDIDATE_POOL,
        len(candidate_indices),
    )

    candidate_order = np.argsort(
        scores
    )[::-1][:pool_size]

    pool_indices = [
        candidate_indices[i]
        for i in candidate_order
    ]

    pool_scores = scores[
        candidate_order
    ]

    # --------------------------------------------------------
    # Diversity selection
    # --------------------------------------------------------

    selected_indices = mmr_select(
        candidate_indices=pool_indices,
        embeddings=embeddings,
        scores=pool_scores,
        metadata=metadata,
        top_k=min(top_k, pool_size),
    )

    # --------------------------------------------------------
    # Build results
    # --------------------------------------------------------

    results = []

    for idx in selected_indices:

        # Find original score.
        score = float(
            embeddings[idx]
            @ query_embedding
        )

        doc = metadata[idx]

        results.append(
            {
                "score": score,
                "ticker": doc.get(
                    "ticker",
                    "UNKNOWN",
                ),
                "company": doc.get(
                    "company",
                    "UNKNOWN",
                ),
                "form": doc.get(
                    "form",
                    "UNKNOWN",
                ),
                "filing_date": doc.get(
                    "filing_date",
                    "UNKNOWN",
                ),
                "report_date": doc.get(
                    "report_date",
                    "UNKNOWN",
                ),
                "source": doc.get(
                    "source",
                    "",
                ),
                "source_url": doc.get(
                    "source_url",
                    "",
                ),
                "chunk_id": doc.get(
                    "chunk_id",
                    -1,
                ),
                "text": doc.get(
                    "text",
                    "",
                ),
            }
        )

    return results


# ============================================================
# DISPLAY
# ============================================================

def display_results(
    query,
    ticker,
    as_of_date,
    results,
):

    print("\n" + "=" * 90)
    print("RAG SEARCH")
    print("=" * 90)

    print(f"Query       : {query}")
    print(
        f"Ticker      : "
        f"{ticker if ticker else 'ALL'}"
    )
    print(
        f"As-of date  : "
        f"{as_of_date if as_of_date else 'NONE'}"
    )
    print(
        f"Results     : {len(results)}"
    )

    for i, result in enumerate(
        results,
        start=1,
    ):

        print("\n" + "-" * 90)
        print(f"RESULT {i}")
        print("-" * 90)

        print(
            f"Similarity  : "
            f"{result['score']:.4f}"
        )

        print(
            f"Ticker      : "
            f"{result['ticker']}"
        )

        print(
            f"Form        : "
            f"{result['form']}"
        )

        print(
            f"Filing date : "
            f"{result['filing_date']}"
        )

        print(
            f"Source      : "
            f"{result['source']}"
        )

        print("\nText:")
        print(result["text"])


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--query",
        required=True,
    )

    parser.add_argument(
        "--ticker",
        default=None,
    )

    parser.add_argument(
        "--as_of_date",
        default=None,
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
    )

    args = parser.parse_args()

    if args.as_of_date:

        normalized = normalize_date(
            args.as_of_date
        )

        if normalized is None:
            raise ValueError(
                "Invalid date. "
                "Use YYYY-MM-DD."
            )

        args.as_of_date = normalized

    if args.top_k <= 0:
        raise ValueError(
            "top_k must be > 0."
        )

    results = search(
        query=args.query,
        ticker=args.ticker,
        as_of_date=args.as_of_date,
        top_k=args.top_k,
    )

    display_results(
        query=args.query,
        ticker=args.ticker,
        as_of_date=args.as_of_date,
        results=results,
    )


if __name__ == "__main__":
    main()