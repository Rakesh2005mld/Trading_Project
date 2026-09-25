from pathlib import Path
import argparse
import json
import os
import time
from collections import defaultdict

import numpy as np
from google import genai
from google.genai import types
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAG_DIR = PROJECT_ROOT / "models" / "rag"

DOCUMENT_DIR = PROJECT_ROOT / "data" / "rag_documents"

OUTPUT_DIR = (
    RAG_DIR
    / "filing_features"
)

EMBEDDINGS_FILE = (
    RAG_DIR
    / "embeddings.npy"
)

METADATA_FILE = (
    RAG_DIR
    / "metadata.json"
)

EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

DEFAULT_TOP_CHUNKS = 8

REQUEST_DELAY = 1.0

QUERY_TYPES = {
    "growth": (
        "revenue growth, demand growth, "
        "business growth drivers, segment growth, "
        "customer demand"
    ),

    "catalyst": (
        "future catalysts, upcoming growth opportunities, "
        "new products, launches, expansion opportunities, "
        "positive developments"
    ),

    "risk": (
        "business risks, demand risks, supply risks, "
        "competitive risks, regulatory risks, "
        "macroeconomic risks"
    ),

    "outlook": (
        "future outlook, management expectations, "
        "forward-looking trends, guidance, "
        "expected business conditions"
    ),
}


# ============================================================
# GEMINI CLIENT
# ============================================================

def get_gemini_client():

    api_key = os.getenv(
        "GEMINI_API_KEY"
    )

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set."
        )

    return genai.Client(
        api_key=api_key
    )


def get_gemini_model():

    model_name = os.getenv(
        "RAG_LLM_MODEL"
    )

    if not model_name:
        raise RuntimeError(
            "RAG_LLM_MODEL is not set."
        )

    return model_name


# ============================================================
# LOAD DATABASE
# ============================================================

def load_rag_database():

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
            "Embeddings and metadata have "
            "different lengths."
        )

    print(
        f"Embeddings : {embeddings.shape}"
    )

    print(
        f"Metadata   : {len(metadata)}"
    )

    print(
        "Loading embedding model..."
    )

    model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    return model, embeddings, metadata


# ============================================================
# GROUP CHUNKS BY SOURCE
# ============================================================

def group_chunks_by_source(metadata):

    groups = defaultdict(list)

    for idx, doc in enumerate(metadata):

        source = doc.get(
            "source",
            ""
        )

        groups[source].append(idx)

    return groups


# ============================================================
# GET FILINGS
# ============================================================

def load_filing_metadata():

    filings = []

    metadata_files = list(
        DOCUMENT_DIR.rglob("*.json")
    )

    for path in metadata_files:

        try:

            data = json.loads(
                path.read_text(
                    encoding="utf-8"
                )
            )

        except Exception:
            continue

        if "ticker" not in data:
            continue

        if "filing_date" not in data:
            continue

        text_path = path.with_suffix(
            ".txt"
        )

        if not text_path.exists():
            continue

        data["metadata_path"] = str(path)
        data["text_path"] = str(text_path)

        filings.append(data)

    filings.sort(
        key=lambda x: (
            x.get("ticker", ""),
            x.get("filing_date", ""),
        )
    )

    return filings


# ============================================================
# RETRIEVE CHUNKS FOR ONE FILING
# ============================================================

def retrieve_filing_evidence(
    model,
    embeddings,
    metadata,
    source_to_indices,
    filing,
    top_chunks=DEFAULT_TOP_CHUNKS,
):

    source = str(
        Path(
            filing["text_path"]
        ).relative_to(
            PROJECT_ROOT
        )
    )

    candidate_indices = (
        source_to_indices.get(
            source,
            []
        )
    )

    if not candidate_indices:

        return []

    candidate_embeddings = embeddings[
        candidate_indices
    ]

    selected = {}

    for query_type, query_text in QUERY_TYPES.items():

        query_embedding = model.encode(
            [query_text],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]

        scores = (
            candidate_embeddings
            @ query_embedding
        )

        # Take more candidates for each topic,
        # then deduplicate across topics.
        local_k = min(
            3,
            len(candidate_indices)
        )

        best_positions = np.argsort(
            scores
        )[::-1][:local_k]

        for position in best_positions:

            original_idx = candidate_indices[
                position
            ]

            score = float(
                scores[position]
            )

            # Keep the highest score if the
            # same chunk matches multiple topics.
            if (
                original_idx not in selected
                or
                score
                > selected[
                    original_idx
                ]["score"]
            ):

                selected[original_idx] = {
                    "idx": original_idx,
                    "score": score,
                    "query_type": query_type,
                }

    # Sort by relevance.
    selected_items = sorted(
        selected.values(),
        key=lambda x: x["score"],
        reverse=True,
    )

    # Limit final evidence.
    selected_items = selected_items[
        :top_chunks
    ]

    results = []

    for item in selected_items:

        idx = item["idx"]
        doc = metadata[idx]

        results.append(
            {
                "score": item["score"],
                "query_type": item["query_type"],
                "chunk_id": doc.get(
                    "chunk_id",
                    -1
                ),
                "text": doc.get(
                    "text",
                    ""
                ),
            }
        )

    return results


# ============================================================
# BUILD GEMINI PROMPT
# ============================================================

def build_prompt(
    filing,
    evidence,
):

    evidence_text = []

    for i, item in enumerate(
        evidence,
        start=1,
    ):

        evidence_text.append(
            f"""
========================
EVIDENCE {i}
Topic: {item["query_type"]}
Similarity: {item["score"]:.4f}
Chunk ID: {item["chunk_id"]}
========================

{item["text"]}
"""
        )

    joined_evidence = "\n".join(
        evidence_text
    )

    ticker = filing[
        "ticker"
    ]

    filing_date = filing[
        "filing_date"
    ]

    form = filing.get(
        "form",
        "UNKNOWN"
    )

    report_date = filing.get(
        "report_date",
        "UNKNOWN"
    )

    return f"""
You are a financial research analyst
inside a quantitative trading research system.

Analyze ONLY the evidence retrieved from this
specific SEC filing.

TICKER:
{ticker}

FORM:
{form}

FILING DATE:
{filing_date}

REPORT DATE:
{report_date}

IMPORTANT:

1. Use only the supplied evidence.
2. Do not use outside knowledge.
3. Do not invent facts or numbers.
4. Do not assume information that is not supported.
5. Separate positive and negative evidence.
6. Focus on information that could affect
   future company performance.
7. Be conservative with confidence.
8. This analysis is associated with the filing date,
   so it must not use information from later periods.

Return ONLY valid JSON with exactly these fields:

{{
  "ticker": "{ticker}",
  "filing_date": "{filing_date}",
  "form": "{form}",
  "sentiment": "bullish|neutral|bearish|mixed|insufficient_evidence",
  "sentiment_score": 0.0,
  "confidence": 0.0,
  "growth_score": 0.0,
  "catalyst_score": 0.0,
  "risk_score": 0.0,
  "growth_drivers": [],
  "catalysts": [],
  "risks": [],
  "evidence_summary": [],
  "reasoning": ""
}}

SCORING:

sentiment_score:
-1.0 = strongly negative
 0.0 = neutral
+1.0 = strongly positive

confidence:
0.0 = evidence is very weak
1.0 = evidence is very strong

growth_score:
0.0 = little/no positive growth evidence
1.0 = very strong growth evidence

catalyst_score:
0.0 = little/no meaningful catalyst evidence
1.0 = strong positive catalyst evidence

risk_score:
0.0 = little/no meaningful risk evidence
1.0 = strong risk evidence

All numerical scores must be between 0 and 1,
except sentiment_score which must be between -1 and 1.

Arrays must contain concise statements supported
by the evidence.

Do not add additional fields.

RETRIEVED EVIDENCE:

{joined_evidence}
"""


# ============================================================
# GEMINI ANALYSIS
# ============================================================

def analyze_filing(
    client,
    model_name,
    filing,
    evidence,
):

    prompt = build_prompt(
        filing,
        evidence,
    )

    max_retries = 5

    for attempt in range(1, max_retries + 1):

        try:

            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0,
                    response_mime_type="application/json",
                ),
            )

            if not response.text:
                raise RuntimeError(
                    "Gemini returned empty response."
                )

            try:

                result = json.loads(
                    response.text
                )

            except json.JSONDecodeError as exc:

                print(
                    "\nRAW GEMINI RESPONSE:"
                )

                print(
                    response.text
                )

                raise RuntimeError(
                    "Gemini did not return valid JSON."
                ) from exc

            return result

        except Exception as exc:

            error_text = str(exc)

            # Retry temporary Gemini availability
            # / rate-limit type failures.
            retryable = (
                "503" in error_text
                or "UNAVAILABLE" in error_text
                or "429" in error_text
                or "RESOURCE_EXHAUSTED" in error_text
            )

            if not retryable:
                raise

            if attempt == max_retries:
                raise

            wait_seconds = 2 ** attempt

            print(
                f"\nGemini temporary failure "
                f"(attempt {attempt}/{max_retries}). "
                f"Retrying in {wait_seconds}s..."
            )

            time.sleep(
                wait_seconds
            )

    raise RuntimeError(
        "Gemini analysis failed after retries."
    )

# ============================================================
# SAVE
# ============================================================

def save_result(result):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    ticker = result[
        "ticker"
    ]

    filing_date = result[
        "filing_date"
    ]

    filename = (
        f"{ticker}_"
        f"{filing_date}_rag.json"
    )

    output_path = (
        OUTPUT_DIR
        / filename
    )

    output_path.write_text(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return output_path


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
        help="Optional maximum number of filings.",
    )

    parser.add_argument(
        "--top_chunks",
        type=int,
        default=DEFAULT_TOP_CHUNKS,
        help="Number of evidence chunks per filing.",
    )

    args = parser.parse_args()

    ticker_filter = (
        args.ticker.upper()
        if args.ticker
        else None
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    model, embeddings, metadata = (
        load_rag_database()
    )

    source_to_indices = (
        group_chunks_by_source(
            metadata
        )
    )

    filings = load_filing_metadata()

    if ticker_filter:
        filings = [
            filing
            for filing in filings
            if filing.get(
                "ticker",
                ""
            ).upper() == ticker_filter
        ]

    if args.limit is not None:

        filings = filings[
            :args.limit
        ]

    print("\n" + "=" * 75)
    print("FILING-LEVEL RAG FEATURE GENERATION")
    print("=" * 75)

    print(
        f"Ticker filter    : "
        f"{ticker_filter if ticker_filter else 'ALL'}"
    )

    print(
        f"Filings to process: {len(filings)}"
    )

    print(
        f"Evidence chunks  : {args.top_chunks}"
    )

    client = get_gemini_client()

    model_name = get_gemini_model()

    print(
        f"Gemini model     : {model_name}"
    )

    completed = 0
    skipped = 0
    failed = 0

    # --------------------------------------------------------
    # Process filings
    # --------------------------------------------------------

    for position, filing in enumerate(
        filings,
        start=1,
    ):

        ticker = filing[
            "ticker"
        ]

        filing_date = filing[
            "filing_date"
        ]

        form = filing.get(
            "form",
            "UNKNOWN"
        )

        output_path = (
            OUTPUT_DIR
            / f"{ticker}_"
              f"{filing_date}_rag.json"
        )

        print(
            "\n"
            + "-" * 75
        )

        print(
            f"[{position}/{len(filings)}] "
            f"{ticker} | "
            f"{filing_date} | "
            f"{form}"
        )

        # ----------------------------------------------------
        # Skip existing
        # ----------------------------------------------------

        if output_path.exists():

            print(
                "Already exists -> skipping"
            )

            skipped += 1
            continue

        # ----------------------------------------------------
        # Retrieval
        # ----------------------------------------------------

        try:

            evidence = retrieve_filing_evidence(
                model=model,
                embeddings=embeddings,
                metadata=metadata,
                source_to_indices=source_to_indices,
                filing=filing,
                top_chunks=args.top_chunks,
            )

            if not evidence:

                print(
                    "No evidence chunks found -> skipping"
                )

                failed += 1
                continue

            print(
                f"Retrieved {len(evidence)} chunks"
            )

            # ------------------------------------------------
            # Gemini
            # ------------------------------------------------

            result = analyze_filing(
                client=client,
                model_name=model_name,
                filing=filing,
                evidence=evidence,
            )

            # ------------------------------------------------
            # Add provenance metadata
            # ------------------------------------------------

            result[
                "source_url"
            ] = filing.get(
                "source_url",
                ""
            )

            result[
                "report_date"
            ] = filing.get(
                "report_date",
                "UNKNOWN"
            )

            result[
                "retrieved_chunks"
            ] = [
                {
                    "chunk_id":
                        item["chunk_id"],

                    "topic":
                        item["query_type"],

                    "similarity":
                        item["score"],
                }
                for item in evidence
            ]

            save_result(
                result
            )

            print(
                "Sentiment        : "
                f"{result.get('sentiment')}"
            )

            print(
                "Sentiment score  : "
                f"{result.get('sentiment_score')}"
            )

            print(
                "Confidence       : "
                f"{result.get('confidence')}"
            )

            completed += 1

        except Exception as exc:

            print(
                f"FAILED: {exc}"
            )

            failed += 1

        time.sleep(
            REQUEST_DELAY
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("DONE")
    print("=" * 75)

    print(
        f"Generated : {completed}"
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