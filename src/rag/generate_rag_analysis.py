from pathlib import Path
import argparse
import json
import os

from google import genai
from google.genai import types

from query_rag import search


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_TOP_K = 5


# ============================================================
# CLIENT
# ============================================================

def get_client():

    api_key = os.getenv(
        "GEMINI_API_KEY"
    )

    if not api_key:

        raise RuntimeError(
            "GEMINI_API_KEY is not set.\n\n"
            "PowerShell example:\n"
            '$env:GEMINI_API_KEY="YOUR_API_KEY"'
        )

    return genai.Client(
        api_key=api_key
    )


# ============================================================
# PROMPT
# ============================================================

def build_prompt(
    ticker,
    as_of_date,
    query,
    results,
):

    evidence_blocks = []

    for i, result in enumerate(
        results,
        start=1,
    ):

        evidence_blocks.append(
            f"""
========================
EVIDENCE {i}
========================

Ticker:
{result["ticker"]}

Company:
{result.get("company", "UNKNOWN")}

Form:
{result["form"]}

Filing date:
{result["filing_date"]}

Report date:
{result["report_date"]}

Source:
{result["source"]}

Text:
{result["text"]}
"""
        )

    evidence = "\n".join(
        evidence_blocks
    )

    prompt = f"""
You are a financial research assistant
inside a quantitative trading research system.

COMPANY:
{ticker}

HISTORICAL AS-OF DATE:
{as_of_date}

QUESTION:
{query}

STRICT RULES:

1. Use ONLY the supplied evidence.
2. Do not use outside knowledge.
3. Do not invent facts or numbers.
4. Do not use information published after the as-of date.
5. Treat the supplied evidence as the complete information set.
6. If the evidence is insufficient, explicitly say so.
7. Separate evidence from interpretation.
8. Be conservative.

Return a JSON object with exactly these fields:

{{
  "ticker": "{ticker}",
  "as_of_date": "{as_of_date}",
  "sentiment": "bullish|neutral|bearish|mixed|insufficient_evidence",
  "confidence": 0.0,
  "growth_drivers": [],
  "catalysts": [],
  "risks": [],
  "evidence_summary": [],
  "reasoning": ""
}}

Rules for the fields:

- confidence must be between 0 and 1.
- growth_drivers must be an array of strings.
- catalysts must be an array of strings.
- risks must be an array of strings.
- evidence_summary must be an array of strings.
- reasoning must be a concise explanation.
- Do not add extra JSON fields.

SUPPLIED EVIDENCE:

{evidence}
"""

    return prompt


# ============================================================
# GENERATE
# ============================================================

def generate_analysis(
    ticker,
    as_of_date,
    query,
    results,
):

    client = get_client()

    model_name = os.getenv(
        "RAG_LLM_MODEL",
        "gemini-3.8-flash",
    )

    prompt = build_prompt(
        ticker=ticker,
        as_of_date=as_of_date,
        query=query,
        results=results,
    )

    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0,
            response_mime_type="application/json",
        ),
    )

    text = response.text

    if not text:

        raise RuntimeError(
            "Gemini returned an empty response."
        )

    try:

        return json.loads(text)

    except json.JSONDecodeError as exc:

        print("\nRaw Gemini response:")
        print(text)

        raise RuntimeError(
            "Gemini response was not valid JSON."
        ) from exc


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ticker",
        required=True,
    )

    parser.add_argument(
        "--as_of_date",
        required=True,
    )

    parser.add_argument(
        "--query",
        required=True,
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=DEFAULT_TOP_K,
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Retrieval
    # --------------------------------------------------------

    results = search(
        query=args.query,
        ticker=args.ticker,
        as_of_date=args.as_of_date,
        top_k=args.top_k,
    )

    if not results:

        print(
            json.dumps(
                {
                    "error":
                        "No eligible evidence found."
                },
                indent=2,
            )
        )

        return

    print(
        f"\nRetrieved {len(results)} evidence chunks."
    )

    # --------------------------------------------------------
    # Gemini
    # --------------------------------------------------------

    analysis = generate_analysis(
        ticker=args.ticker,
        as_of_date=args.as_of_date,
        query=args.query,
        results=results,
    )

    print("\n" + "=" * 90)
    print("GEMINI RAG FINANCIAL ANALYSIS")
    print("=" * 90)

    print(
        json.dumps(
            analysis,
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()