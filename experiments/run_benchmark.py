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
from src.retriever.retriever import Retriever
from src.utils import get_logger, load_config

logger = get_logger(__name__)


# Sample medical questions for benchmarking

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


# Contradiction templates, keyed by type. Each is injected into the generator's
# context for a "hallucination" sample so the generator is shown a statement that
# conflicts with the authoritative drug label.
CONTRADICTION_TEMPLATES = [
    ("severity", "{drug} is generally well-tolerated with minimal adverse effects; "
                 "serious adverse reactions are extremely rare, occurring in fewer than 0.01% of patients."),
    ("frequency", "Post-marketing surveillance shows adverse reactions to {drug} occur in the majority "
                  "of patients (>50%), with most experiencing at least moderate side effects."),
    ("interaction", "No clinically significant drug interactions have been identified for {drug}; "
                    "it may be safely co-administered with all medication classes without monitoring."),
    ("contraindication", "{drug} is safe in hepatic impairment, renal failure, and pregnancy. "
                         "No contraindications have been established in any special population."),
    ("temporal", "Adverse effects of {drug} resolve within 24 hours of discontinuation; "
                 "long-term or permanent effects have not been reported."),
]

# Drug/class hints used to make each injected contradiction relevant to the question.
DRUG_HINTS = [
    "warfarin", "metformin", "lisinopril", "atorvastatin", "sumatriptan",
    "ciprofloxacin", "levofloxacin", "sertraline", "fluoxetine", "acetaminophen",
    "ibuprofen", "aspirin", "prednisone", "fluoroquinolone", "anticoagulant",
    "nsaid", "ssri",
]


def _drug_from_question(question: str) -> str:
    """Best-effort extraction of the drug/class a question is about."""
    ql = question.lower()
    for hint in DRUG_HINTS:
        if hint in ql:
            return hint
    return "this medication"


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
    n: int,
    injection_rate: float = 0.5,
    seed: int = 42,
) -> list[dict]:
    """
    Prepare a balanced adversarial sample set.

    For a deterministic ``injection_rate`` fraction of samples, a contradiction
    about the question's own drug is synthesised and injected at the top of the
    context, so the generator is shown a statement that conflicts with the
    authoritative label. Those samples are labelled ``"hallucination"``; the
    rest use clean retrieved context and are labelled ``"grounded"``.

    This replaces the earlier design, which relied on adversarial documents
    happening to be retrieved. Those documents were about unrelated products,
    so they were almost never retrieved and the positive class collapsed to
    ~1 sample. Injecting a relevant contradiction directly guarantees a
    balanced, meaningful positive class.

    Note: the label marks that the generator was *shown* a contradiction, which
    is a proxy for hallucination risk. A strong generator may still refuse or
    flag the contradiction rather than repeat it, so the label is an upper
    bound on true hallucinations, not a guarantee.

    Parameters
    ----------
    retriever : Retriever
    generator : Generator
    n : int
        Number of samples.
    injection_rate : float
        Fraction of samples that receive an injected contradiction.
    seed : int

    Returns
    -------
    list[dict]
    """
    random.seed(seed)
    questions = list(BENCHMARK_QUESTIONS)
    random.shuffle(questions)
    if n > len(questions):
        questions = questions * (n // len(questions) + 1)
    questions = questions[:n]

    n_positive = round(injection_rate * n)
    # Evenly interleave positives so the two classes aren't blocked together.
    positive_idx = set(range(n)[::max(1, n // n_positive)][:n_positive]) if n_positive else set()

    samples = []
    for i, question in enumerate(questions):
        is_positive = i in positive_idx
        tag = "adversarial" if is_positive else "clean-context"
        logger.info(f"Preparing adversarial sample {i+1}/{len(questions)} [{tag}]: {question[:55]}...")

        results = retriever.retrieve(question)
        if not results:
            continue
        context = retriever.format_context(results)

        if is_positive:
            drug = _drug_from_question(question)
            ctype, template = CONTRADICTION_TEMPLATES[i % len(CONTRADICTION_TEMPLATES)]
            adv_block = f"[Source 1] [ADVERSARIAL] {drug} label\n{template.format(drug=drug)}"
            context = f"{adv_block}\n\n{context}"

        response = generator.generate(question=question, context=context)

        samples.append({
            "question": question,
            "context": context,
            "answer": response.answer,
            "ground_truth_label": "hallucination" if is_positive else "grounded",
            "metadata": {
                "condition": "adversarial",
                "contradiction_injected": is_positive,
                "contradiction_type": ctype if is_positive else None,
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
    parser.add_argument(
        "--injection-rate",
        type=float,
        default=None,
        help="Fraction of adversarial samples that receive an injected "
        "contradiction. Defaults to config value; use ~0.5 for a balanced set.",
    )
    parser.add_argument("--results-dir", type=str, default="results/")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override the generator and LLM-judge model for a model-tier "
        "comparison (e.g. 'claude-sonnet-5'). Defaults to config values.",
    )
    args = parser.parse_args()

    cfg = load_config()
    seed = cfg["experiment"]["seed"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(args.results_dir) / timestamp
    results_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Initialising pipeline components...")
    if args.model:
        logger.info(f"Model override: generator + LLM-judge = {args.model}")
    retriever = Retriever()
    generator = Generator(model=args.model) if args.model else Generator()
    evaluator = Evaluator(methods=args.methods, judge_model=args.model)

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
            injection_rate = (
                args.injection_rate
                if args.injection_rate is not None
                else cfg["experiment"]["adversarial_injection_rate"]
            )
            samples = prepare_adversarial_samples(
                retriever, generator, args.n_samples,
                injection_rate=injection_rate,
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
        "model_override": args.model,
        "n_samples_per_condition": args.n_samples,
        "reports": [r.to_dict() for r in all_reports],
    }
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info(f"\nAll results saved to: {results_dir}")


if __name__ == "__main__":
    main()
