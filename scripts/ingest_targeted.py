#!/usr/bin/env python3
"""
scripts/ingest_targeted.py
===========================
Ingests specific drug labels from DailyMed by searching for drug names.
This guarantees the vector store contains clinically relevant drugs
that match the benchmark question set.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import requests

from src.retriever.document_loader import DailyMedLoader, Document
from src.retriever.embedder import Embedder
from src.retriever.retriever import Retriever
from src.retriever.vector_store import VectorStore
from src.utils import get_logger

logger = get_logger(__name__)

# Drugs that cover all benchmark questions + good adversarial variety
TARGET_DRUGS = [
    # Benchmark question coverage
    "warfarin",
    "metformin",
    "sertraline",       # SSRI
    "fluoxetine",       # SSRI
    "sumatriptan",      # triptan
    "ciprofloxacin",    # fluoroquinolone
    "levofloxacin",     # fluoroquinolone
    # Broad pharmacology coverage
    "lisinopril",
    "atorvastatin",
    "metoprolol",
    "amlodipine",
    "omeprazole",
    "amoxicillin",
    "azithromycin",
    "prednisone",
    "levothyroxine",
    "gabapentin",
    "aspirin",
    "ibuprofen",
    "acetaminophen",
    "hydrochlorothiazide",
    "losartan",
    "simvastatin",
    "clopidogrel",
    "furosemide",
]

DAILYMED_SEARCH = "https://dailymed.nlm.nih.gov/dailymed/services/v2"


def search_set_id(drug_name: str) -> str | None:
    """Search DailyMed for a drug by name and return the best matching set_id."""
    try:
        resp = requests.get(
            f"{DAILYMED_SEARCH}/spls.json",
            params={"drug_name": drug_name, "pagesize": 5},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        items = data.get("data", [])
        if items:
            return items[0].get("setid")
    except Exception as e:
        logger.warning(f"Search failed for '{drug_name}': {e}")
    return None


def download_xml(set_id: str, dest: Path) -> bool:
    """Download a single SPL XML by set_id."""
    url = f"{DAILYMED_SEARCH}/spls/{set_id}.xml"
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200 and len(resp.content) > 200:
            dest.write_bytes(resp.content)
            return True
    except Exception as e:
        logger.warning(f"Download failed for {set_id}: {e}")
    return False


def main():
    parser = argparse.ArgumentParser(description="Ingest targeted drug labels from DailyMed")
    parser.add_argument("--drugs", nargs="*", default=None,
                        help="Drug names to ingest (default: built-in list)")
    parser.add_argument("--data-dir", default="data/dailymed",
                        help="Directory to store XML files")
    args = parser.parse_args()

    drug_list = args.drugs or TARGET_DRUGS
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Targeting {len(drug_list)} drugs: {', '.join(drug_list)}")

    # Download XMLs for each target drug
    downloaded = []
    for drug in drug_list:
        xml_path = data_dir / f"{drug.replace(' ', '_')}.xml"
        if xml_path.exists():
            logger.info(f"  ✓ {drug} (cached)")
            downloaded.append(xml_path)
            continue

        set_id = search_set_id(drug)
        if not set_id:
            logger.warning(f"  ✗ {drug} (not found in DailyMed)")
            continue

        if download_xml(set_id, xml_path):
            logger.info(f"  ↓ {drug} (downloaded set_id={set_id})")
            downloaded.append(xml_path)
        else:
            logger.warning(f"  ✗ {drug} (download failed)")

        time.sleep(0.4)  # be polite to the API

    logger.info(f"\nDownloaded {len(downloaded)}/{len(drug_list)} drug labels.")

    # Parse XMLs into documents
    loader = DailyMedLoader(data_dir=data_dir)
    docs = []
    for xml_path in downloaded:
        parsed = loader._parse_xml(xml_path)
        if parsed:
            docs.extend(parsed)
            logger.info(f"  Parsed {len(parsed)} chunks from {xml_path.stem}")
        else:
            logger.warning(f"  No content parsed from {xml_path.stem}")

    if not docs:
        logger.error("No documents parsed. Check XML files.")
        sys.exit(1)

    logger.info(f"\nTotal document chunks: {len(docs)}")

    # Index into ChromaDB
    embedder = Embedder()
    vector_store = VectorStore()
    retriever = Retriever(embedder=embedder, vector_store=vector_store)
    retriever.index(docs, show_progress=True)

    logger.info(f"\n✓ Ingestion complete. {len(docs)} chunks indexed.")

    # Show summary
    drugs_indexed = set(d.metadata.get("drug_name", "?") for d in docs)
    logger.info(f"Drugs indexed ({len(drugs_indexed)}):")
    for d in sorted(drugs_indexed):
        logger.info(f"  - {d}")


if __name__ == "__main__":
    main()