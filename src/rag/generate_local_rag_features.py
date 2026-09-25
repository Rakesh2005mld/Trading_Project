from pathlib import Path
import argparse
import json
from collections import defaultdict

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAG_DIR = PROJECT_ROOT / "models" / "rag"
DOCUMENT_DIR = PROJECT_ROOT / "data" / "rag_documents"

EMBEDDINGS_FILE = RAG_DIR / "embeddings.npy"
METADATA_FILE = RAG_DIR / "metadata.json"

OUTPUT_DIR = RAG_DIR / "local_filing_features"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
FINBERT_MODEL = "ProsusAI/finbert"

TOP_CHUNKS_PER_TOPIC = 3
MAX_TOTAL_CHUNKS = 10

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# RETRIEVAL QUERIES
# ============================================================

QUERY_TYPES = {
    "growth": (
        "revenue growth demand growth business growth "
        "segment growth customer demand sales growth"
    ),

    "catalyst": (
        "future catalysts opportunities new products "
        "product launches expansion positive developments"
    ),

    "risk": (
        "business risks demand risks supply risks "
        "competitive risks regulatory risks macroeconomic risks"
    ),

    "outlook": (
        "future outlook management expectations guidance "
        "forward looking trends expected business conditions"
    ),
}


# ============================================================
# LOAD RAG
# ============================================================

def load_rag():

    print("=" * 75)
    print("LOADING RAG DATABASE")
    print("=" * 75)

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
            "Embedding/metadata size mismatch."
        )

    print(
        f"Embeddings : {embeddings.shape}"
    )

    print(
        f"Metadata   : {len(metadata)}"
    )

    embedder = SentenceTransformer(
        EMBEDDING_MODEL
    )

    return embedder, embeddings, metadata


# ============================================================
# GROUP CHUNKS BY FILING
# ============================================================

def group_by_source(metadata):

    groups = defaultdict(list)

    for idx, doc in enumerate(metadata):

        source = doc.get(
            "source",
            ""
        )

        groups[source].append(idx)

    return groups


# ============================================================
# LOAD FILINGS
# ============================================================

def load_filings():

    filings = []

    for path in DOCUMENT_DIR.rglob("*.json"):

        try:
            data = json.loads(
                path.read_text(
                    encoding="utf-8"
                )
            )
        except Exception:
            continue

        if (
            "ticker" not in data
            or "filing_date" not in data
        ):
            continue

        text_path = path.with_suffix(
            ".txt"
        )

        if not text_path.exists():
            continue

        data["text_path"] = str(
            text_path
        )

        filings.append(data)

    filings.sort(
        key=lambda x: (
            x["ticker"],
            x["filing_date"],
        )
    )

    return filings


# ============================================================
# RETRIEVE EVIDENCE
# ============================================================

def retrieve_evidence(
    embedder,
    embeddings,
    metadata,
    source_to_indices,
    filing,
):

    source = str(
        Path(
            filing["text_path"]
        ).relative_to(
            PROJECT_ROOT
        )
    )

    indices = source_to_indices.get(
        source,
        []
    )

    if not indices:
        return []

    chunk_embeddings = embeddings[
        indices
    ]

    selected = {}

    for topic, query in QUERY_TYPES.items():

        q = embedder.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]

        scores = (
            chunk_embeddings @ q
        )

        k = min(
            TOP_CHUNKS_PER_TOPIC,
            len(indices)
        )

        best = np.argsort(
            scores
        )[::-1][:k]

        for pos in best:

            idx = indices[pos]

            score = float(
                scores[pos]
            )

            if (
                idx not in selected
                or score
                > selected[idx]["score"]
            ):

                selected[idx] = {
                    "idx": idx,
                    "topic": topic,
                    "score": score,
                }

    selected_items = sorted(
        selected.values(),
        key=lambda x: x["score"],
        reverse=True,
    )

    selected_items = selected_items[
        :MAX_TOTAL_CHUNKS
    ]

    results = []

    for item in selected_items:

        doc = metadata[
            item["idx"]
        ]

        results.append(
            {
                "idx": item["idx"],
                "topic": item["topic"],
                "score": item["score"],
                "text": doc["text"],
                "chunk_id": doc.get(
                    "chunk_id",
                    -1
                ),
            }
        )

    return results


# ============================================================
# FINBERT
# ============================================================

def load_finbert():

    print("\nLoading FinBERT...")

    tokenizer = AutoTokenizer.from_pretrained(
        FINBERT_MODEL
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        FINBERT_MODEL
    )

    model.to(DEVICE)
    model.eval()

    print(
        f"FinBERT device : {DEVICE}"
    )

    return tokenizer, model


def finbert_scores(
    tokenizer,
    model,
    texts,
):

    if not texts:

        return []

    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )

    encoded = {
        key: value.to(DEVICE)
        for key, value in encoded.items()
    }

    with torch.no_grad():

        logits = model(
            **encoded
        ).logits

        probabilities = torch.softmax(
            logits,
            dim=1
        ).cpu().numpy()

    # FinBERT labels are:
    # 0 = positive
    # 1 = negative
    # 2 = neutral

    results = []

    for probs in probabilities:

        positive = float(
            probs[0]
        )

        negative = float(
            probs[1]
        )

        neutral = float(
            probs[2]
        )

        sentiment = (
            positive
            - negative
        )

        confidence = float(
            max(
                positive,
                negative,
                neutral,
            )
        )

        results.append(
            {
                "positive": positive,
                "negative": negative,
                "neutral": neutral,
                "sentiment": sentiment,
                "confidence": confidence,
            }
        )

    return results


# ============================================================
# BUILD FEATURES
# ============================================================

def build_features(
    filing,
    evidence,
    sentiment_results,
):

    # --------------------------------------------------------
    # Weighted sentiment
    # --------------------------------------------------------

    weighted_sum = 0.0
    weight_total = 0.0

    for evidence_item, sentiment in zip(
        evidence,
        sentiment_results,
    ):

        relevance = max(
            evidence_item["score"],
            0.0
        )

        weighted_sum += (
            relevance
            * sentiment["sentiment"]
        )

        weight_total += relevance

    if weight_total > 0:

        rag_sentiment = (
            weighted_sum
            / weight_total
        )

    else:

        rag_sentiment = 0.0

    # --------------------------------------------------------
    # Topic relevance
    # --------------------------------------------------------

    topic_scores = {
        "growth": [],
        "catalyst": [],
        "risk": [],
        "outlook": [],
    }

    for item in evidence:

        topic_scores[
            item["topic"]
        ].append(
            item["score"]
        )

    def topic_mean(topic):

        values = topic_scores[topic]

        if not values:
            return 0.0

        return float(
            np.mean(values)
        )

    return {
        "ticker": filing["ticker"],

        "filing_date":
            filing["filing_date"],

        "report_date":
            filing.get(
                "report_date",
                "UNKNOWN"
            ),

        "form":
            filing.get(
                "form",
                "UNKNOWN"
            ),

        "rag_sentiment":
            float(rag_sentiment),

        "rag_confidence":
            float(
                np.mean(
                    [
                        x["confidence"]
                        for x
                        in sentiment_results
                    ]
                )
            ),

        "rag_growth_relevance":
            topic_mean("growth"),

        "rag_catalyst_relevance":
            topic_mean("catalyst"),

        "rag_risk_relevance":
            topic_mean("risk"),

        "rag_outlook_relevance":
            topic_mean("outlook"),

        "evidence_count":
            len(evidence),

        "evidence": [
            {
                "topic":
                    item["topic"],

                "similarity":
                    item["score"],

                "chunk_id":
                    item["chunk_id"],

                "sentiment":
                    sentiment_results[i][
                        "sentiment"
                    ],

                "text":
                    item["text"],
            }
            for i, item
            in enumerate(evidence)
        ],
    }


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ticker",
        default=None,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    ticker_filter = (
        args.ticker.upper()
        if args.ticker
        else None
    )

    embedder, embeddings, metadata = (
        load_rag()
    )

    source_to_indices = (
        group_by_source(
            metadata
        )
    )

    filings = load_filings()

    if ticker_filter:

        filings = [
            x
            for x in filings
            if x["ticker"].upper()
            == ticker_filter
        ]

    if args.limit is not None:

        filings = filings[
            :args.limit
        ]

    print("\n" + "=" * 75)
    print("LOCAL RAG FEATURE GENERATION")
    print("=" * 75)

    print(
        f"Ticker filter : "
        f"{ticker_filter or 'ALL'}"
    )

    print(
        f"Filings       : "
        f"{len(filings)}"
    )

    tokenizer, finbert = (
        load_finbert()
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    generated = 0
    skipped = 0
    failed = 0

    for i, filing in enumerate(
        filings,
        start=1
    ):

        ticker = filing[
            "ticker"
        ]

        filing_date = filing[
            "filing_date"
        ]

        output_path = (
            OUTPUT_DIR
            / f"{ticker}_"
              f"{filing_date}_local.json"
        )

        print(
            f"\n[{i}/{len(filings)}] "
            f"{ticker} | "
            f"{filing_date} | "
            f"{filing.get('form', '')}"
        )

        if output_path.exists():

            print(
                "Already exists -> skipping"
            )

            skipped += 1
            continue

        try:

            evidence = retrieve_evidence(
                embedder,
                embeddings,
                metadata,
                source_to_indices,
                filing,
            )

            if not evidence:

                print(
                    "No evidence found."
                )

                failed += 1
                continue

            texts = [
                item["text"]
                for item in evidence
            ]

            sentiment_results = (
                finbert_scores(
                    tokenizer,
                    finbert,
                    texts,
                )
            )

            result = build_features(
                filing,
                evidence,
                sentiment_results,
            )

            output_path.write_text(
                json.dumps(
                    result,
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            print(
                "RAG sentiment : "
                f"{result['rag_sentiment']:.4f}"
            )

            print(
                "Confidence    : "
                f"{result['rag_confidence']:.4f}"
            )

            generated += 1

        except Exception as exc:

            print(
                f"FAILED: {exc}"
            )

            failed += 1

    print("\n" + "=" * 75)
    print("DONE")
    print("=" * 75)

    print(
        f"Generated : {generated}"
    )

    print(
        f"Skipped   : {skipped}"
    )

    print(
        f"Failed    : {failed}"
    )

    print(
        f"Output    : {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()