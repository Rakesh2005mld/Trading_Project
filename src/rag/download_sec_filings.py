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

OUTPUT_DIR = PROJECT_ROOT / "data" / "rag_documents"

START_DATE = "2020-01-01"

FORMS = {"10-K", "10-Q"}

TICKERS = [
    "AAPL",
    "AMZN",
    "AVGO",
    "GOOGL",
    "JPM",
    "META",
    "MSFT",
    "NVDA",
    "TSLA",
    "XOM",
]

# IMPORTANT: use a descriptive User-Agent
USER_AGENT = "QuantTradingRAG research rs0828772@gmail.com"

SEC_SUBMISSIONS_URL = (
    "https://data.sec.gov/submissions/CIK{cik}.json"
)

SEC_ARCHIVES_URL = (
    "https://www.sec.gov/Archives/edgar/data"
)

# Retry configuration
MAX_RETRIES = 3
CONNECT_TIMEOUT = 15
READ_TIMEOUT = 45
REQUEST_DELAY = 0.3


# ============================================================
# SESSION
# ============================================================

session = requests.Session()

session.headers.update(
    {
        "User-Agent": USER_AGENT,
        "Accept-Encoding": "gzip, deflate",
    }
)


# ============================================================
# SEC TICKER MAP
# ============================================================

def load_company_tickers():

    url = "https://www.sec.gov/files/company_tickers.json"

    response = requests.get(
        url,
        headers={"User-Agent": USER_AGENT},
        timeout=(15, 30),
    )

    response.raise_for_status()

    data = response.json()

    ticker_map = {}

    for item in data.values():

        ticker = item["ticker"].upper()

        ticker_map[ticker] = {
            "cik": str(item["cik_str"]).zfill(10),
            "name": item["title"],
        }

    return ticker_map


# ============================================================
# SUBMISSIONS
# ============================================================

def get_submissions(cik):

    url = SEC_SUBMISSIONS_URL.format(cik=cik)

    response = session.get(
        url,
        timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
    )

    response.raise_for_status()

    return response.json()


# ============================================================
# FILINGS
# ============================================================

def get_filings(submissions):

    recent = submissions.get(
        "filings",
        {}
    ).get(
        "recent",
        {}
    )

    forms = recent.get("form", [])
    filing_dates = recent.get("filingDate", [])
    accession_numbers = recent.get(
        "accessionNumber",
        []
    )
    primary_documents = recent.get(
        "primaryDocument",
        []
    )
    report_dates = recent.get(
        "reportDate",
        []
    )

    filings = []

    for i in range(len(forms)):

        form = forms[i]

        if form not in FORMS:
            continue

        filing_date = filing_dates[i]

        if filing_date < START_DATE:
            continue

        filings.append(
            {
                "form": form,
                "filing_date": filing_date,
                "report_date": report_dates[i],
                "accession_number": accession_numbers[i],
                "primary_document": primary_documents[i],
            }
        )

    return filings


# ============================================================
# URL
# ============================================================

def build_filing_url(
    cik,
    accession_number,
    primary_document,
):

    accession_no_dashes = (
        accession_number.replace("-", "")
    )

    return (
        f"{SEC_ARCHIVES_URL}/"
        f"{int(cik)}/"
        f"{accession_no_dashes}/"
        f"{primary_document}"
    )


# ============================================================
# TEXT CLEANING
# ============================================================

def html_to_text(html):

    soup = BeautifulSoup(
        html,
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


# ============================================================
# DOWNLOAD WITH RETRIES
# ============================================================

def download_with_retry(url):

    last_error = None

    for attempt in range(
        1,
        MAX_RETRIES + 1,
    ):

        try:

            response = session.get(
                url,
                timeout=(
                    CONNECT_TIMEOUT,
                    READ_TIMEOUT,
                ),
            )

            response.raise_for_status()

            return response.content

        except Exception as exc:

            last_error = exc

            if attempt < MAX_RETRIES:

                wait_time = 2 ** (
                    attempt - 1
                )

                print(
                    f"\n  Retry {attempt}/{MAX_RETRIES - 1} "
                    f"after {wait_time}s..."
                )

                time.sleep(
                    wait_time
                )

    raise last_error


# ============================================================
# SAVE ONE FILINGS
# ============================================================

def download_filing(
    ticker,
    cik,
    company_name,
    filing,
):

    filing_date = filing[
        "filing_date"
    ]

    form = filing["form"]

    accession_number = filing[
        "accession_number"
    ]

    primary_document = filing[
        "primary_document"
    ]

    stem = (
        f"{filing_date}_"
        f"{form.replace('-', '')}_"
        f"{accession_number.replace('-', '')}"
    )

    ticker_dir = (
        OUTPUT_DIR / ticker
    )

    ticker_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        ticker_dir / f"{stem}.txt"
    )

    metadata_path = (
        ticker_dir / f"{stem}.json"
    )

    # Already complete
    if (
        output_path.exists()
        and metadata_path.exists()
    ):
        return "exists"

    url = build_filing_url(
        cik,
        accession_number,
        primary_document,
    )

    try:

        content = download_with_retry(
            url
        )

        text = html_to_text(
            content
        )

        if len(text) < 1000:

            print(
                f"\nWarning: short filing "
                f"{ticker} {filing_date}"
            )

            return "short"

        output_path.write_text(
            text,
            encoding="utf-8",
        )

        metadata = {
            "ticker": ticker,
            "cik": cik,
            "company": company_name,
            "form": form,
            "filing_date": filing_date,
            "report_date": filing[
                "report_date"
            ],
            "accession_number":
                accession_number,
            "primary_document":
                primary_document,
            "source_url": url,
            "downloaded_at":
                datetime.now().isoformat(),
        }

        metadata_path.write_text(
            json.dumps(
                metadata,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        return "downloaded"

    except Exception as exc:

        print(
            f"\nFailed after retries: "
            f"{ticker} {form} {filing_date}"
        )

        print(
            f"  URL: {url}"
        )

        print(
            f"  Error: {exc}"
        )

        return "failed"


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print("SEC FINANCIAL DOCUMENT COLLECTOR")
    print("=" * 75)

    print(
        f"Output directory : {OUTPUT_DIR}"
    )

    print(
        f"Start date       : {START_DATE}"
    )

    print(
        f"Forms            : {sorted(FORMS)}"
    )

    print(
        f"Companies        : {len(TICKERS)}"
    )

    print(
        f"Max retries      : {MAX_RETRIES}"
    )

    print("\nLoading SEC company ticker map...")

    ticker_map = (
        load_company_tickers()
    )

    stats = {
        "downloaded": 0,
        "exists": 0,
        "short": 0,
        "failed": 0,
    }

    total_filings = 0

    for ticker in TICKERS:

        if ticker not in ticker_map:

            print(
                f"\nERROR: ticker not found: "
                f"{ticker}"
            )

            continue

        cik = ticker_map[
            ticker
        ]["cik"]

        company_name = ticker_map[
            ticker
        ]["name"]

        print(
            "\n" + "=" * 75
        )

        print(
            f"{ticker} | {company_name}"
        )

        print(
            f"CIK: {cik}"
        )

        print(
            "=" * 75
        )

        try:

            submissions = (
                get_submissions(cik)
            )

            filings = get_filings(
                submissions
            )

            print(
                f"Relevant filings: "
                f"{len(filings)}"
            )

            total_filings += len(filings)

            for filing in tqdm(
                filings,
                desc=ticker,
            ):

                result = download_filing(
                    ticker=ticker,
                    cik=cik,
                    company_name=company_name,
                    filing=filing,
                )

                stats[result] += 1

                time.sleep(
                    REQUEST_DELAY
                )

        except Exception as exc:

            print(
                f"\nFailed company "
                f"{ticker}: {exc}"
            )

            stats["failed"] += 1

    print(
        "\n" + "=" * 75
    )

    print(
        "DOWNLOAD COMPLETE"
    )

    print(
        "=" * 75
    )

    print(
        f"Candidate filings : "
        f"{total_filings}"
    )

    print(
        f"Downloaded        : "
        f"{stats['downloaded']}"
    )

    print(
        f"Already existed   : "
        f"{stats['exists']}"
    )

    print(
        f"Too short         : "
        f"{stats['short']}"
    )

    print(
        f"Failed            : "
        f"{stats['failed']}"
    )

    print(
        "\nDocuments stored in:"
    )

    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()