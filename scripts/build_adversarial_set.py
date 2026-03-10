"""
scripts/build_adversarial_set.py
=================================
Constructs the adversarial document injection set used in hallucination
detection experiments.

Strategy:
1. Load real FAERS adverse event reports that conflict with DailyMed labels.
2. For each conflict pair, create an adversarial document that contradicts
   the official label on a specific fact (severity, frequency, drug name).
3. Save labelled pairs to data/adversarial/ for use in benchmarks.

Usage:
    python scripts/build_adversarial_set.py --n-pairs 75 --output data/adversarial/
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retriever.document_loader import Document, DailyMedLoader
from src.utils import get_logger, load_config

logger = get_logger(__name__)

# Contradiction templates: each injects a specific type of hallucination
CONTRADICTION_TEMPLATES = {
    "severity_flip": (
        "Clinical studies have demonstrated that {drug} is generally well-tolerated "
        "with minimal adverse effects. Serious adverse reactions are extremely rare "
        "and occur in fewer than 0.01% of patients.",
        "severity"
    ),
    "frequency_exaggeration": (
        "Post-marketing surveillance data indicates that adverse reactions to {drug} "
        "are observed in the majority of patients (>50%), with most patients "
        "experiencing at least moderate side effects during treatment.",
        "frequency"
    ),
    "interaction_negation": (
        "No clinically significant drug-drug interactions have been identified for {drug}. "
        "It may be safely co-administered with all classes of medications without "
        "dose adjustment or monitoring.",
        "interaction"
    ),
    "contraindication_removal": (
        "{drug} has been found safe for use in patients with hepatic impairment, "
        "renal failure, and during pregnancy. No contraindications have been "
        "established in special patient populations.",
        "contraindication"
    ),
    "temporal_confusion": (
        "The adverse effects of {drug} typically resolve within 24 hours of "
        "discontinuation. Long-term or permanent effects have not been reported "
        "in clinical trials or post-marketing studies.",
        "temporal"
    ),
}


def build_adversarial_set(
    source_docs: list[Document],
    n_pairs: int,
    seed: int = 42,
) -> list[dict]:
    """
    Build adversarial document pairs from real drug label documents.

    Each pair contains:
    - ``original``: the real Document
    - ``adversarial``: a contradicting Document
    - ``contradiction_type``: what kind of hallucination was injected
    - ``drug_name``: the drug involved

    Parameters
    ----------
    source_docs : list[Document]
        Real DailyMed documents to base adversarial examples on.
    n_pairs : int
        Number of adversarial pairs to generate.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    list[dict]
        Labelled adversarial pairs.
    """
    random.seed(seed)

    # Filter to adverse reactions and drug interactions sections
    relevant_docs = [
        d for d in source_docs
        if d.metadata.get("section") in ("adverse_reactions", "drug_interactions", "warnings")
    ]

    if len(relevant_docs) < n_pairs:
        logger.warning(
            f"Only {len(relevant_docs)} relevant docs available for {n_pairs} pairs. "
            "Reducing pair count."
        )
        n_pairs = len(relevant_docs)

    selected = random.sample(relevant_docs, n_pairs)
    pairs = []

    template_names = list(CONTRADICTION_TEMPLATES.keys())

    for i, doc in enumerate(selected):
        drug_name = doc.metadata.get("drug_name", "this medication")
        template_name = template_names[i % len(template_names)]
        template_text, contradiction_type = CONTRADICTION_TEMPLATES[template_name]

        adversarial_content = template_text.format(drug=drug_name)

        adversarial_doc = Document(
            doc_id=f"adversarial_{doc.doc_id}",
            content=adversarial_content,
            source=f"[ADVERSARIAL] {doc.source}",
            metadata={
                **doc.metadata,
                "is_adversarial": True,
                "contradiction_type": contradiction_type,
                "original_doc_id": doc.doc_id,
            },
            is_adversarial=True,
        )

        pairs.append({
            "pair_id": f"pair_{i:04d}",
            "drug_name": drug_name,
            "contradiction_type": contradiction_type,
            "original": {
                "doc_id": doc.doc_id,
                "content": doc.content,
                "section": doc.metadata.get("section"),
            },
            "adversarial": {
                "doc_id": adversarial_doc.doc_id,
                "content": adversarial_doc.content,
                "contradiction_type": contradiction_type,
            },
            "ground_truth_label": "hallucination",
        })

    logger.info(f"Built {len(pairs)} adversarial pairs.")
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Build adversarial document injection set.")
    parser.add_argument("--n-pairs", type=int, default=75)
    parser.add_argument("--output", type=str, default="data/adversarial/")
    parser.add_argument("--dailymed-dir", type=str, default="data/dailymed/")
    args = parser.parse_args()

    cfg = load_config()

    logger.info("Loading DailyMed documents...")
    loader = DailyMedLoader(
        data_dir=args.dailymed_dir,
        max_labels=cfg["data"]["max_labels"],
    )
    source_docs = loader.load()

    if not source_docs:
        logger.error("No source documents loaded. Run data ingestion first.")
        sys.exit(1)

    pairs = build_adversarial_set(
        source_docs=source_docs,
        n_pairs=args.n_pairs,
        seed=cfg["experiment"]["seed"],
    )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "adversarial_pairs.json"
    output_file.write_text(json.dumps(pairs, indent=2))

    logger.info(f"Adversarial set saved to {output_file}")
    print(f"\nCreated {len(pairs)} adversarial pairs.")
    print(f"Contradiction types: {set(p['contradiction_type'] for p in pairs)}")


if __name__ == "__main__":
    main()
