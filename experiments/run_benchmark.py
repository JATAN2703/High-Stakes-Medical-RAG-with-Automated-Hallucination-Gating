"""
experiments/run_benchmark.py
==============================
Master experiment runner. Executes all three experimental conditions:

  1. CLEAN:        No adversarial documents. Tests false positive rate.
  2. ADVERSARIAL:  20% of retrieved docs are contradictions. Tests recall.
  3. LONG_CONTEXT: Queries with 2k / 4k / 8k token contexts. Tests degradation.

Results are saved to results/<timestamp>/ and a summary is printed.

Usage:
    python experiments/run_benchmark.py
    python experiments/run_benchmark.py --condition adversarial --methods llm_judge faithfulness
    python experiments/run_benchmark.py --n-samples 50
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluator.evaluator import Evaluator
from src.generator.generator import Generator
from src.retriever.document_loader import Document, DailyMedLoader
from src.retriever.retriever import Retriever
from src.utils import get_logger, load_config

logger = get_logger(__name__)


# ── Sample medical questions for benchmarking ─────────────────────────────────

BENCHMARK_QUESTIONS = [
    "What are the most common adverse reactions associated with warfarin?",
    "Are there any known drug interactions between metformin and contrast agents?",
    "What are the contraindications for lisinopril use?",
    "How does atorvastatin interact with CYP3A4 inhibitors?",
    "What are the signs of warfarin toxicity and how is it treated?",
    "Can aspirin be used in patients with renal impairment?",
    "What adverse effects are associated with long-term metformin use?",
    "What is the recommended monitoring for patients on anticoagulant therapy?",
    "Are there cardiac risks associated with fluoroquinolone antibiotics?",
    "What are the hepatotoxicity risks of acetaminophen in overdose?",
    "How do SSRIs interact with triptans for migraine treatment?",
    "What are the black box warnings for fluoroquinolone antibiotics?",
    "What are the risks of combining NSAIDs with anticoagulants?",
    "What adverse effects does prednisone have on glucose metabolism?",
    "What are the renal dosing adjustments required for metformin?",
]


def load_adversarial_pairs(path: str = "data/adversarial/adversarial_pairs.json") -> list[dict]:
    """Load pre-built adversarial document pairs."""
    p = Path(path)
    if not p.exists():
        logger.error(f"Adversarial pairs not found at {p}. Run scripts/build_adversarial_set.py first.")
        return []
    return json.loads(p.read_text())


def prepare_clean_samples(
    retriever: Retriever,
    generator: Generator,
    questions: list[str],
    n: int,
    seed: int = 42,
) -> list[dict]:
    """
    Prepare samples using the clean corpus (no adversarial injection).

    Retrieves context for each question, generates an answer, and labels
    all examples as 'grounded' (clean corpus → should not hallucinate).

    Parameters
    ----------
    retriever : Retriever
    generator : Generator
    questions : list[str]
    n : int
        Number of samples to prepare.
    seed : int

    Returns
    -------
    list[dict]
    """
    random.seed(seed)
    selected_qs = random.sample(questions, min(n, len(questions)))
    if n > len(questions):
        selected_qs = selected_qs * (n // len(questions) + 1)
    selected_qs = selected_qs[:n]

    samples = []
    for i, question in enumerate(selected_qs):
        logger.info(f"Preparing clean sample {i+1}/{len(selected_qs)}: {question[:60]}...")
        results = retriever.retrieve(question)
        if not results:
            logger.warning(f"No retrieval results for: {question}")
            continue

        context = retriever.format_context(results)
        response = generator.generate(question=question, context=context)

        samples.append({
            "question": question,
            "context": context,
            "answer": response.answer,
            "ground_truth_label": "grounded",
            "metadata": {"condition": "clean", "n_docs_retrieved": len(results)},
        })

    return samples


def prepare_adversarial_samples(
    retriever: Retriever,
    generator: Generator,
    adversarial_pairs: list[dict],
    n: int,
    injection_rate: float = 0.2,
    seed: int = 42,
) -> list[dict]:
    """
    Prepare samples with adversarial documents injected into retrieval results.

    For each question, retrieves normal context then injects adversarial
    documents at the given rate. The generator sees the corrupted context.

    Parameters
    ----------
    retriever : Retriever
    generator : Generator
    adversarial_pairs : list[dict]
        Adversarial document pairs from build_adversarial_set.py
    n : int
        Number of samples.
    injection_rate : float
        Fraction of retrieved docs to replace with adversarial ones.
    seed : int

    Returns
    -------
    list[dict]
    """
    random.seed(seed)
    questions = random.sample(BENCHMARK_QUESTIONS, min(n, len(BENCHMARK_QUESTIONS)))
    if n > len(BENCHMARK_QUESTIONS):
        questions = questions * (n // len(BENCHMARK_QUESTIONS) + 1)
    questions = questions[:n]

    # Inject adversarial docs into the retriever's index
    adversarial_docs = []
    for pair in adversarial_pairs[:int(len(adversarial_pairs) * injection_rate * 5)]:
        adv = pair["adversarial"]
        adversarial_docs.append(Document(
            doc_id=adv["doc_id"],
            content=adv["content"],
            source=f"[ADVERSARIAL] {pair['drug_name']}",
            metadata={"contradiction_type": adv["contradiction_type"]},
            is_adversarial=True,
        ))
    retriever.inject_adversarial(adversarial_docs)

    samples = []
    for i, question in enumerate(questions):
        logger.info(f"Preparing adversarial sample {i+1}/{len(questions)}: {question[:60]}...")
        results = retriever.retrieve(question)
        if not results:
            continue

        context = retriever.format_context(results)
        response = generator.generate(question=question, context=context)

        # Determine ground truth: if any adversarial doc was retrieved, label as hallucination
        retrieved_ids = {r.get("doc_id", "") for r in results}
        adv_ids = {doc.doc_id for doc in adversarial_docs}
        has_adversarial = bool(retrieved_ids & adv_ids)

        samples.append({
            "question": question,
            "context": context,
            "answer": response.answer,
            "ground_truth_label": "hallucination" if has_adversarial else "grounded",
            "metadata": {
                "condition": "adversarial",
                "has_adversarial_doc": has_adversarial,
                "n_docs_retrieved": len(results),
            },
        })

    return samples


def prepare_long_context_samples(
    retriever: Retriever,
    generator: Generator,
    questions: list[str],
    n: int,
    context_window: int,
    seed: int = 42,
) -> list[dict]:
    """
    Prepare samples with context truncated to a specific token budget.

    Used to test how each detector's performance degrades at
    different context window sizes (2k, 4k, 8k tokens).
    """
    random.seed(seed)
    selected = random.sample(questions, min(n, len(questions)))

    samples = []
    for i, question in enumerate(selected):
        logger.info(f"Long context sample {i+1}/{len(selected)} (limit={context_window}t)")
        results = retriever.retrieve(question, max_context_tokens=context_window)
        if not results:
            continue

        context = retriever.format_context(results)
        response = generator.generate(question=question, context=context)

        samples.append({
            "question": question,
            "context": context,
            "answer": response.answer,
            "ground_truth_label": "grounded",
            "metadata": {
                "condition": "long_context",
                "context_window": context_window,
            },
        })

    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Run hallucination detection benchmark.")
    parser.add_argument(
        "--condition",
        choices=["clean", "adversarial", "long_context", "all"],
        default="all",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Methods to evaluate. Defaults to all four.",
    )
    parser.add_argument("--n-samples", type=int, default=20)
    parser.add_argument("--results-dir", type=str, default="results/")
    args = parser.parse_args()

    cfg = load_config()
    seed = cfg["experiment"]["seed"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(args.results_dir) / timestamp
    results_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Initialising pipeline components...")
    retriever = Retriever()
    generator = Generator()
    evaluator = Evaluator(methods=args.methods)

    conditions_to_run = (
        ["clean", "adversarial", "long_context"]
        if args.condition == "all"
        else [args.condition]
    )

    all_reports = []

    for condition in conditions_to_run:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running condition: {condition.upper()}")
        logger.info(f"{'='*60}")

        if condition == "clean":
            samples = prepare_clean_samples(
                retriever, generator, BENCHMARK_QUESTIONS, args.n_samples, seed
            )
            if not samples:
                logger.warning("No samples prepared for clean condition. Is the index populated?")
                continue
            report = evaluator.benchmark(samples, condition="clean")
            evaluator.save_results(report, results_dir / "clean_results.json")

        elif condition == "adversarial":
            adversarial_pairs = load_adversarial_pairs()
            if not adversarial_pairs:
                logger.warning("Skipping adversarial condition (no pairs found).")
                continue
            samples = prepare_adversarial_samples(
                retriever, generator, adversarial_pairs, args.n_samples,
                injection_rate=cfg["experiment"]["adversarial_injection_rate"],
                seed=seed,
            )
            if not samples:
                continue
            report = evaluator.benchmark(samples, condition="adversarial")
            evaluator.save_results(report, results_dir / "adversarial_results.json")

        elif condition == "long_context":
            window_sizes = cfg["experiment"]["context_window_sizes"]
            for window in window_sizes:
                samples = prepare_long_context_samples(
                    retriever, generator, BENCHMARK_QUESTIONS,
                    n=args.n_samples, context_window=window, seed=seed,
                )
                if not samples:
                    continue
                report = evaluator.benchmark(
                    samples,
                    condition=f"long_context_{window}t",
                    context_window=window,
                )
                evaluator.save_results(
                    report, results_dir / f"long_context_{window}_results.json"
                )
                all_reports.append(report)
                report.print_summary()
            continue

        all_reports.append(report)
        report.print_summary()

    # Save consolidated summary
    summary = {
        "timestamp": timestamp,
        "conditions": conditions_to_run,
        "methods": args.methods or cfg["evaluator"]["methods"],
        "n_samples_per_condition": args.n_samples,
        "reports": [r.to_dict() for r in all_reports],
    }
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info(f"\nAll results saved to: {results_dir}")


if __name__ == "__main__":
    main()
