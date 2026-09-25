from pathlib import Path
import json
import re

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DOCUMENT_DIR = (
    PROJECT_ROOT
    / "data"
    / "rag_documents"
)

RAG_DIR = (
    PROJECT_ROOT
    / "models"
    / "rag"
)

OLD_EMBEDDINGS = (
    RAG_DIR
    / "embeddings.npy"
)

OLD_METADATA = (
    RAG_DIR
    / "metadata.json"
)

NEW_EMBEDDINGS = (
    RAG_DIR
    / "embeddings_updated.npy"
)

NEW_METADATA = (
    RAG_DIR
    / "metadata_updated.json"
)

EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    text = text.replace(
        "\x00",
        " "
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def chunk_text(
    text,
    chunk_size=CHUNK_SIZE,
    overlap=CHUNK_OVERLAP,
):

    text = clean_text(text)

    if not text:
        return []

    chunks = []

    start = 0

    while start < len(text):

        end = min(
            start + chunk_size,
            len(text)
        )

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - overlap

    return chunks


# ============================================================
# CURRENT FILES
# ============================================================

def get_current_sources():

    sources = {}

    for text_path in DOCUMENT_DIR.rglob(
        "*.txt"
    ):

        metadata_path = (
            text_path.with_suffix(".json")
        )

        if not metadata_path.exists():
            continue

        try:

            metadata = json.loads(
                metadata_path.read_text(
                    encoding="utf-8"
                )
            )

        except Exception:

            continue

        source = str(
            text_path.relative_to(
                PROJECT_ROOT
            )
        )

        sources[source] = {
            "text_path": text_path,
            "metadata": metadata,
        }

    return sources


# ============================================================
# LOAD OLD DATABASE
# ============================================================

def load_old_database():

    print("=" * 75)
    print("LOADING EXISTING RAG DATABASE")
    print("=" * 75)

    embeddings = np.load(
        OLD_EMBEDDINGS
    )

    with open(
        OLD_METADATA,
        "r",
        encoding="utf-8"
    ) as f:

        metadata = json.load(f)

    if len(embeddings) != len(metadata):

        raise ValueError(
            "Existing embeddings and metadata "
            "have different lengths."
        )

    print(
        f"Old embeddings : {embeddings.shape}"
    )

    print(
        f"Old metadata   : {len(metadata)}"
    )

    return embeddings, metadata


# ============================================================
# FILTER OLD DATABASE
# ============================================================

def filter_valid_old_database(
    embeddings,
    metadata,
    current_sources,
):

    print(
        "\nRemoving stale documents..."
    )

    keep_indices = []

    for idx, doc in enumerate(metadata):

        source = doc.get(
            "source",
            ""
        )

        if source in current_sources:

            keep_indices.append(idx)

    removed = (
        len(metadata)
        - len(keep_indices)
    )

    print(
        f"Removed stale chunks : {removed}"
    )

    filtered_embeddings = (
        embeddings[keep_indices]
    )

    filtered_metadata = [
        metadata[i]
        for i in keep_indices
    ]

    print(
        f"Remaining chunks     : "
        f"{len(filtered_metadata)}"
    )

    return (
        filtered_embeddings,
        filtered_metadata,
    )


# ============================================================
# FIND NEW DOCUMENTS
# ============================================================

def find_new_sources(
    metadata,
    current_sources,
):

    existing_sources = {
        doc.get(
            "source",
            ""
        )
        for doc in metadata
    }

    new_sources = [
        source
        for source in current_sources
        if source not in existing_sources
    ]

    new_sources.sort()

    return new_sources


# ============================================================
# PROCESS NEW DOCUMENTS
# ============================================================

def process_new_documents(
    new_sources,
    current_sources,
):

    documents = []

    print(
        f"\nNew documents: "
        f"{len(new_sources)}"
    )

    for source in tqdm(
        new_sources,
        desc="Chunking new filings",
    ):

        item = current_sources[
            source
        ]

        text = item[
            "text_path"
        ].read_text(
            encoding="utf-8",
            errors="ignore",
        )

        text = clean_text(
            text
        )

        chunks = chunk_text(
            text
        )

        metadata = item[
            "metadata"
        ]

        for chunk_id, chunk in enumerate(
            chunks
        ):

            documents.append(
                {
                    "text": chunk,

                    "ticker":
                        metadata.get(
                            "ticker",
                            "UNKNOWN"
                        ),

                    "company":
                        metadata.get(
                            "company",
                            "UNKNOWN"
                        ),

                    "cik":
                        metadata.get(
                            "cik",
                            "UNKNOWN"
                        ),

                    "form":
                        metadata.get(
                            "form",
                            "UNKNOWN"
                        ),

                    "filing_date":
                        metadata.get(
                            "filing_date",
                            "UNKNOWN"
                        ),

                    "report_date":
                        metadata.get(
                            "report_date",
                            "UNKNOWN"
                        ),

                    "accession_number":
                        metadata.get(
                            "accession_number",
                            "UNKNOWN"
                        ),

                    "source_url":
                        metadata.get(
                            "source_url",
                            ""
                        ),

                    "source":
                        source,

                    "filename":
                        item[
                            "text_path"
                        ].name,

                    "chunk_id":
                        chunk_id,
                }
            )

    print(
        f"New chunks : {len(documents)}"
    )

    return documents


# ============================================================
# EMBED NEW CHUNKS
# ============================================================

def create_embeddings(
    documents
):

    if not documents:

        return np.empty(
            (0, 384),
            dtype=np.float32
        )

    print(
        "\nLoading embedding model..."
    )

    model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    texts = [
        doc["text"]
        for doc in documents
    ]

    print(
        f"Embedding {len(texts)} new chunks..."
    )

    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    return embeddings.astype(
        np.float32
    )


# ============================================================
# SAVE SAFELY
# ============================================================

def save_database(
    embeddings,
    metadata,
):

    print(
        "\nSaving updated database..."
    )

    np.save(
        NEW_EMBEDDINGS,
        embeddings,
    )

    NEW_METADATA.write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        f"Temporary embeddings : "
        f"{NEW_EMBEDDINGS}"
    )

    print(
        f"Temporary metadata   : "
        f"{NEW_METADATA}"
    )


# ============================================================
# REPLACE ORIGINAL
# ============================================================

def replace_database():

    # Backup current files first.
    backup_embeddings = (
        RAG_DIR
        / "embeddings_backup.npy"
    )

    backup_metadata = (
        RAG_DIR
        / "metadata_backup.json"
    )

    OLD_EMBEDDINGS.replace(
        backup_embeddings
    )

    OLD_METADATA.replace(
        backup_metadata
    )

    NEW_EMBEDDINGS.replace(
        OLD_EMBEDDINGS
    )

    NEW_METADATA.replace(
        OLD_METADATA
    )

    print(
        "\nOriginal database replaced."
    )

    print(
        f"Backup embeddings : "
        f"{backup_embeddings}"
    )

    print(
        f"Backup metadata   : "
        f"{backup_metadata}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print("INCREMENTAL RAG EMBEDDING UPDATE")
    print("=" * 75)

    # --------------------------------------------------------
    # Current files
    # --------------------------------------------------------

    current_sources = (
        get_current_sources()
    )

    print(
        f"Current filing documents : "
        f"{len(current_sources)}"
    )

    # --------------------------------------------------------
    # Existing database
    # --------------------------------------------------------

    old_embeddings, old_metadata = (
        load_old_database()
    )

    # --------------------------------------------------------
    # Remove stale chunks
    # --------------------------------------------------------

    (
        old_embeddings,
        old_metadata,
    ) = filter_valid_old_database(
        old_embeddings,
        old_metadata,
        current_sources,
    )

    # --------------------------------------------------------
    # Find only NEW filings
    # --------------------------------------------------------

    new_sources = find_new_sources(
        old_metadata,
        current_sources,
    )

    # --------------------------------------------------------
    # Chunk
    # --------------------------------------------------------

    new_metadata = process_new_documents(
        new_sources,
        current_sources,
    )

    # --------------------------------------------------------
    # Embed
    # --------------------------------------------------------

    new_embeddings = create_embeddings(
        new_metadata
    )

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    if len(new_metadata) > 0:

        combined_embeddings = np.vstack(
            [
                old_embeddings,
                new_embeddings,
            ]
        )

        combined_metadata = (
            old_metadata
            + new_metadata
        )

    else:

        combined_embeddings = (
            old_embeddings
        )

        combined_metadata = (
            old_metadata
        )

    print(
        "\n" + "=" * 75
    )

    print(
        "UPDATED DATABASE"
    )

    print(
        "=" * 75
    )

    print(
        f"Documents currently on disk : "
        f"{len(current_sources)}"
    )

    print(
        f"Old valid chunks             : "
        f"{len(old_metadata):,}"
    )

    print(
        f"New chunks                   : "
        f"{len(new_metadata):,}"
    )

    print(
        f"Final chunks                 : "
        f"{len(combined_metadata):,}"
    )

    print(
        f"Final embedding shape        : "
        f"{combined_embeddings.shape}"
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_database(
        combined_embeddings,
        combined_metadata,
    )

    # --------------------------------------------------------
    # Replace original
    # --------------------------------------------------------

    replace_database()

    print(
        "\n" + "=" * 75
    )

    print(
        "DONE"
    )

    print(
        "=" * 75
    )


if __name__ == "__main__":
    main()