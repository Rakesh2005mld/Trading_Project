from pathlib import Path
from datetime import datetime
import json
import re
import time

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "rag_documents"
)

START_DATE = "2020-01-01"

USER_AGENT = "QuantTradingRAG research rs0828772@gmail.com"

# Use verified issuer CIKs rather than relying on an ambiguous
# ticker lookup for our fixed universe.
TICKERS = {
    "AAPL": "0000320193",
    "AMZN": "0001018724",
    "AVGO": "0001730168",
    "GOOGL": "0001652044",
    "JPM": "0000019617",
    "META": "0001326801",
    "MSFT": "0000789019",
    "NVDA": "0001045810",
    "TSLA": "0001318605",
    "XOM": "0000034088",
}

FORMS = {"10-K", "10-Q"}

SEC_SUBMISSIONS = (
    "https://data.sec.gov/submissions/"
)

SEC_ARCHIVES = (
    "https://www.sec.gov/Archives/edgar/data/"
)

SESSION = requests.Session()

SESSION.headers.update(
    {
        "User-Agent": USER_AGENT,
        "Accept-Encoding": "gzip, deflate",
    }
)

REQUEST_DELAY = 0.25
MAX_RETRIES = 4


# ============================================================
# HELPERS
# ============================================================

def get_json(url):

    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):

        try:

            response = SESSION.get(
                url,
                timeout=(15, 45),
            )

            response.raise_for_status()

            return response.json()

        except Exception as exc:

            last_error = exc

            if attempt == MAX_RETRIES:
                raise

            time.sleep(
                2 ** (attempt - 1)
            )

    raise last_error


def html_to_text(content):

    soup = BeautifulSoup(
        content,
        "html.parser",
    )

    for tag in soup(
        ["script", "style", "noscript"]
    ):
        tag.decompose()

    text = soup.get_text(
        separator=" "
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def download_text(url):

    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):

        try:

            response = SESSION.get(
                url,
                timeout=(15, 60),
            )

            response.raise_for_status()

            return html_to_text(
                response.content
            )

        except Exception as exc:

            last_error = exc

            if attempt == MAX_RETRIES:
                raise

            time.sleep(
                2 ** (attempt - 1)
            )

    raise last_error


# ============================================================
# PARSE SUBMISSION ARRAYS
# ============================================================

def parse_submission_block(
    block,
    cik,
):

    forms = block.get(
        "form",
        []
    )

    filing_dates = block.get(
        "filingDate",
        []
    )

    report_dates = block.get(
        "reportDate",
        []
    )

    accession_numbers = block.get(
        "accessionNumber",
        []
    )

    primary_documents = block.get(
        "primaryDocument",
        []
    )

    results = []

    for i, form in enumerate(forms):

        if form not in FORMS:
            continue

        filing_date = filing_dates[i]

        if filing_date < START_DATE:
            continue

        results.append(
            {
                "form": form,
                "filing_date": filing_date,
                "report_date": (
                    report_dates[i]
                    if i < len(report_dates)
                    else ""
                ),
                "accession_number": (
                    accession_numbers[i]
                ),
                "primary_document": (
                    primary_documents[i]
                ),
                "cik": cik,
            }
        )

    return results


# ============================================================
# COLLECT ALL FILINGS
# ============================================================

def collect_filings(
    ticker,
    cik,
):

    print(
        f"\nCollecting submission history "
        f"for {ticker} ({cik})"
    )

    current_url = (
        f"{SEC_SUBMISSIONS}"
        f"CIK{cik}.json"
    )

    current = get_json(
        current_url
    )

    filings = []

    # --------------------------------------------------------
    # Recent filings
    # --------------------------------------------------------

    recent = (
        current
        .get("filings", {})
        .get("recent", {})
    )

    filings.extend(
        parse_submission_block(
            recent,
            cik,
        )
    )

    # --------------------------------------------------------
    # Historical continuation files
    # --------------------------------------------------------

    historical_files = (
        current
        .get("filings", {})
        .get("files", [])
    )

    for file_info in historical_files:

        filing_from = file_info.get(
            "filingFrom",
            ""
        )

        filing_to = file_info.get(
            "filingTo",
            ""
        )

        # Skip archive ranges entirely after our
        # target period.
        if (
            filing_from
            and filing_from > datetime.now().strftime(
                "%Y-%m-%d"
            )
        ):
            continue

        # Skip ranges entirely before START_DATE.
        if (
            filing_to
            and filing_to < START_DATE
        ):
            continue

        filename = file_info[
            "name"
        ]

        url = (
            f"{SEC_SUBMISSIONS}"
            f"{filename}"
        )

        historical = get_json(
            url
        )

        filings.extend(
            parse_submission_block(
                historical,
                cik,
            )
        )

        time.sleep(
            REQUEST_DELAY
        )

    # --------------------------------------------------------
    # Deduplicate
    # --------------------------------------------------------

    unique = {}

    for filing in filings:

        key = filing[
            "accession_number"
        ]

        unique[key] = filing

    filings = list(
        unique.values()
    )

    filings.sort(
        key=lambda x: (
            x["filing_date"],
            x["ticker"]
            if "ticker" in x
            else "",
        )
    )

    return filings


# ============================================================
# DOWNLOAD
# ============================================================

def download_filing(
    ticker,
    filing,
):

    cik = filing["cik"]

    accession = filing[
        "accession_number"
    ]

    accession_no_dashes = (
        accession.replace("-", "")
    )

    primary = filing[
        "primary_document"
    ]

    url = (
        f"{SEC_ARCHIVES}"
        f"{int(cik)}/"
        f"{accession_no_dashes}/"
        f"{primary}"
    )

    ticker_dir = (
        OUTPUT_DIR / ticker
    )

    ticker_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    stem = (
        f"{filing['filing_date']}_"
        f"{filing['form'].replace('-', '')}_"
        f"{accession_no_dashes}"
    )

    text_path = (
        ticker_dir
        / f"{stem}.txt"
    )

    metadata_path = (
        ticker_dir
        / f"{stem}.json"
    )

    if (
        text_path.exists()
        and metadata_path.exists()
    ):
        return "exists"

    text = download_text(
        url
    )

    if len(text) < 1000:
        return "short"

    metadata = {
        "ticker": ticker,
        "cik": cik,
        "form": filing["form"],
        "filing_date": filing["filing_date"],
        "report_date": filing["report_date"],
        "accession_number": accession,
        "primary_document": primary,
        "source_url": url,
        "downloaded_at":
            datetime.now().isoformat(),
    }

    text_path.write_text(
        text,
        encoding="utf-8",
    )

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return "downloaded"


# ============================================================
# REMOVE BAD XOM FILES
# ============================================================

def clean_bad_xom_files():

    xom_dir = (
        OUTPUT_DIR / "XOM"
    )

    if not xom_dir.exists():
        return

    removed = 0

    for metadata_path in xom_dir.glob(
        "*.json"
    ):

        try:

            metadata = json.loads(
                metadata_path.read_text(
                    encoding="utf-8"
                )
            )

            if metadata.get(
                "cik"
            ) != TICKERS["XOM"]:

                text_path = (
                    metadata_path.with_suffix(
                        ".txt"
                    )
                )

                metadata_path.unlink(
                    missing_ok=True
                )

                text_path.unlink(
                    missing_ok=True
                )

                removed += 1

        except Exception:
            continue

    print(
        f"Removed {removed} incorrect XOM files."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print("REPAIRING SEC RAG CORPUS")
    print("=" * 75)

    print(
        f"Target period : {START_DATE} -> today"
    )

    clean_bad_xom_files()

    total_candidates = 0
    downloaded = 0
    exists = 0
    failed = 0
    short = 0

    for ticker, cik in TICKERS.items():

        print(
            "\n"
            + "=" * 75
        )

        print(
            f"{ticker} | CIK {cik}"
        )

        print(
            "=" * 75
        )

        try:

            filings = collect_filings(
                ticker,
                cik,
            )

            # Add ticker to each object.
            for filing in filings:
                filing["ticker"] = ticker

            print(
                f"Eligible 10-K/10-Q: "
                f"{len(filings)}"
            )

            total_candidates += len(
                filings
            )

            for filing in tqdm(
                filings,
                desc=ticker,
            ):

                try:

                    result = download_filing(
                        ticker,
                        filing,
                    )

                    if result == "downloaded":
                        downloaded += 1

                    elif result == "exists":
                        exists += 1

                    elif result == "short":
                        short += 1

                except Exception as exc:

                    failed += 1

                    print(
                        f"\nFailed {ticker} "
                        f"{filing['filing_date']}: "
                        f"{exc}"
                    )

                time.sleep(
                    REQUEST_DELAY
                )

        except Exception as exc:

            print(
                f"\nCompany failure "
                f"{ticker}: {exc}"
            )

            failed += 1

    print(
        "\n"
        + "=" * 75
    )

    print(
        "REPAIR COMPLETE"
    )

    print(
        "=" * 75
    )

    print(
        f"Candidate filings : "
        f"{total_candidates}"
    )

    print(
        f"Downloaded        : "
        f"{downloaded}"
    )

    print(
        f"Already existed   : "
        f"{exists}"
    )

    print(
        f"Too short         : "
        f"{short}"
    )

    print(
        f"Failed            : "
        f"{failed}"
    )


if __name__ == "__main__":
    main()