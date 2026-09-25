from pathlib import Path
import json
import re

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DOCUMENT_DIR = (
    PROJECT_ROOT
    / "data"
    / "rag_documents"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "models"
    / "rag"
)

EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

SUPPORTED_EXTENSIONS = {
    ".txt",
}


# ============================================================
# CLEAN TEXT
# ============================================================

def clean_text(text):

    text = text.replace(
        "\x00",
        " ",
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


# ============================================================
# CHUNKING
# ============================================================

def chunk_text(
    text,
    chunk_size=CHUNK_SIZE,
    overlap=CHUNK_OVERLAP,
):

    text = clean_text(text)

    if not text:
        return []

    if overlap >= chunk_size:

        raise ValueError(
            "CHUNK_OVERLAP must be "
            "smaller than CHUNK_SIZE"
        )

    chunks = []

    start = 0

    while start < len(text):

        end = min(
            start + chunk_size,
            len(text),
        )

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - overlap

    return chunks


# ============================================================
# LOAD DOCUMENT + METADATA
# ============================================================

def load_documents():

    files = [
        path
        for path in DOCUMENT_DIR.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower()
            in SUPPORTED_EXTENSIONS
        )
    ]

    print("=" * 70)
    print("LOADING RAG DOCUMENTS")
    print("=" * 70)

    print(
        f"Document directory : "
        f"{DOCUMENT_DIR}"
    )

    print(
        f"Text files found   : "
        f"{len(files)}"
    )

    if not files:

        print(
            "\nNo documents found."
        )

        return []

    documents = []

    for path in tqdm(
        files,
        desc="Processing documents",
    ):

        # ----------------------------------------------------
        # Corresponding JSON metadata
        # ----------------------------------------------------

        metadata_path = (
            path.with_suffix(".json")
        )

        if not metadata_path.exists():

            print(
                f"\nWARNING: metadata missing:"
                f"\n{path}"
            )

            continue

        try:

            metadata = json.loads(
                metadata_path.read_text(
                    encoding="utf-8"
                )
            )

            text = path.read_text(
                encoding="utf-8",
                errors="ignore",
            )

            text = clean_text(text)

            if len(text) < 100:

                print(
                    f"\nSkipping short document:"
                    f"\n{path}"
                )

                continue

            chunks = chunk_text(
                text
            )

            for chunk_id, chunk in enumerate(
                chunks
            ):

                documents.append(
                    {
                        "text": chunk,

                        "ticker":
                            metadata.get(
                                "ticker",
                                "UNKNOWN",
                            ),

                        "company":
                            metadata.get(
                                "company",
                                "UNKNOWN",
                            ),

                        "form":
                            metadata.get(
                                "form",
                                "UNKNOWN",
                            ),

                        # IMPORTANT:
                        # authoritative date
                        "filing_date":
                            metadata.get(
                                "filing_date",
                                "UNKNOWN",
                            ),

                        "report_date":
                            metadata.get(
                                "report_date",
                                "UNKNOWN",
                            ),

                        "accession_number":
                            metadata.get(
                                "accession_number",
                                "UNKNOWN",
                            ),

                        "source_url":
                            metadata.get(
                                "source_url",
                                "",
                            ),

                        "source":
                            str(
                                path.relative_to(
                                    PROJECT_ROOT
                                )
                            ),

                        "filename":
                            path.name,

                        "chunk_id":
                            chunk_id,
                    }
                )

        except Exception as exc:

            print(
                f"\nFailed to process:"
                f"\n{path}"
                f"\nError: {exc}"
            )

    return documents


# ============================================================
# EMBEDDINGS
# ============================================================

def create_embeddings(
    documents
):

    print(
        "\n"
        + "=" * 70
    )

    print(
        "CREATING EMBEDDINGS"
    )

    print(
        "=" * 70
    )

    print(
        f"Embedding model : "
        f"{EMBEDDING_MODEL}"
    )

    print(
        f"Chunks          : "
        f"{len(documents)}"
    )

    model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    texts = [
        document["text"]
        for document in documents
    ]

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    embeddings = embeddings.astype(
        np.float32
    )

    print(
        f"Embedding shape : "
        f"{embeddings.shape}"
    )

    return embeddings


# ============================================================
# SAVE
# ============================================================

def save_database(
    documents,
    embeddings,
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    embeddings_path = (
        OUTPUT_DIR
        / "embeddings.npy"
    )

    metadata_path = (
        OUTPUT_DIR
        / "metadata.json"
    )

    np.save(
        embeddings_path,
        embeddings,
    )

    metadata_path.write_text(
        json.dumps(
            documents,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "RAG DATABASE SAVED"
    )

    print(
        "=" * 70
    )

    print(
        f"Embeddings : "
        f"{embeddings_path}"
    )

    print(
        f"Metadata   : "
        f"{metadata_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    documents = load_documents()

    if not documents:
        return

    embeddings = create_embeddings(
        documents
    )

    save_database(
        documents,
        embeddings,
    )

    print(
        "\n"
        + "=" * 70
    )

    print("DONE")

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()