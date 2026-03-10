"""
scripts/ingest_data.py
=======================
Downloads and indexes FDA DailyMed drug labels and FAERS reports
into the ChromaDB vector store.

Run this once to populate your local index before running experiments.

Usage:
    python scripts/ingest_data.py --max-labels 200 --source dailymed
    python scripts/ingest_data.py --max-labels 500 --source both
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retriever.document_loader import DailyMedLoader, FAERSLoader
from src.retriever.embedder import Embedder
from src.retriever.retriever import Retriever
from src.retriever.vector_store import VectorStore
from src.utils import get_logger, load_config

logger = get_logger(__name__)


def ingest_dailymed(max_labels: int) -> int:
    """Download and index DailyMed drug labels."""
    logger.info(f"Ingesting up to {max_labels} DailyMed drug labels...")

    loader = DailyMedLoader(data_dir="data/dailymed", max_labels=max_labels)
    docs = loader.load()

    if not docs:
        logger.warning("No DailyMed documents loaded.")
        return 0

    logger.info(f"Loaded {len(docs)} document chunks. Building index...")

    retriever = Retriever()
    retriever.index(docs, show_progress=True)

    logger.info(f"DailyMed ingestion complete. Indexed {len(docs)} chunks.")
    return len(docs)


def ingest_faers(drug_names: list[str] | None = None) -> int:
    """Download and index FAERS adverse event reports."""
    drug_names = drug_names or ["warfarin", "aspirin", "metformin", "lisinopril", "atorvastatin"]
    logger.info(f"Ingesting FAERS reports for {len(drug_names)} drugs...")

    loader = FAERSLoader(data_dir="data/faers", max_reports=200)
    all_docs = []

    for drug in drug_names:
        docs = loader.load(drug_name=drug)
        all_docs.extend(docs)
        logger.info(f"  {drug}: {len(docs)} reports")

    if not all_docs:
        logger.warning("No FAERS documents loaded.")
        return 0

    embedder = Embedder()
    store = VectorStore(collection_name="faers_reports")
    embeddings = embedder.embed([d.content for d in all_docs], show_progress=True)
    store.add_documents(all_docs, embeddings)

    logger.info(f"FAERS ingestion complete. Indexed {len(all_docs)} reports.")
    return len(all_docs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest medical data into vector store.")
    parser.add_argument(
        "--source",
        choices=["dailymed", "faers", "both"],
        default="dailymed",
        help="Data source to ingest.",
    )
    parser.add_argument("--max-labels", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config()
    max_labels = args.max_labels or cfg["data"]["max_labels"]

    total = 0
    if args.source in ("dailymed", "both"):
        total += ingest_dailymed(max_labels)
    if args.source in ("faers", "both"):
        total += ingest_faers()

    print(f"\nIngestion complete. Total documents indexed: {total}")


if __name__ == "__main__":
    main()
