# High-Stakes Medical RAG with Hallucination Detection Benchmarking

Final project for the Foundations of Generative AI course.

A RAG pipeline focused on pharmacology and drug safety, with a benchmarking suite that compares four hallucination detection methods. I wanted to avoid building yet another generic chatbot and instead dig into the part that actually blocks these systems from being used in medicine: knowing when to trust the output.

---

## Research Question

> Which hallucination detection method most reliably identifies fabricated or conflicting drug safety information in a pharmacology RAG pipeline, and under what conditions does each method fail?

---

## Architecture

Three fully decoupled modules with clean interfaces:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────────┐
│   RETRIEVER     │───▶│   GENERATOR     │───▶│       EVALUATOR         │
│                 │    │                 │    │                         │
│ • DailyMed XML  │    │ • OpenRouter    │    │ • LLM-as-Judge          │
│ • FAERS reports │    │   (gpt-4o-mini) │    │ • Self-Consistency      │
│ • BM25 + Dense  │    │ • Strict        │    │ • Faithfulness Score    │
│ • ChromaDB      │    │   grounding     │    │ • NLI Detection         │
└─────────────────┘    └─────────────────┘    └─────────────────────────┘
```

---

## Key Results

### Adversarial Condition (20 samples, 20% contradiction injection rate)

Generator and judge: GPT-4o-mini via OpenRouter.

> **Important caveat (found during a later audit):** in this run the adversarial
> documents were only injected into the corpus and had to be *retrieved* to affect
> a sample. They were built from unrelated OTC products, so they were almost never
> retrieved for these pharmacology questions. As a result only **1 of the 20
> samples** actually ended up labelled "hallucination". The recall/precision figures
> below are therefore computed over a positive class of ~1 and should be read as
> anecdotal, not robust. See [Follow-up analysis](#follow-up-analysis-balanced-injection--stronger-model) for a rebuilt, balanced experiment.

| Method | Recall | Precision | F1 | FPR | Latency |
|---|---|---|---|---|---|
| **NLI Detection** | **100%** | **33.3%** | **0.500** | **10.5%** | 165ms |
| Faithfulness Score | 100% | 12.5% | 0.222 | 36.8% | 31ms |
| Self-Consistency | 100% | 7.1% | 0.133 | 68.4% | 10,444ms |
| LLM-as-Judge | 0% | 0% | 0.000 | 10.5% | 5,152ms |

### Clean Condition (5 samples, no adversarial injection)

| Method | FPR | Latency |
|---|---|---|
| LLM-as-Judge | 20% | 4,731ms |
| NLI Detection | 20% | 363ms |
| Faithfulness Score | 20% | 22ms |
| Self-Consistency | 60% | 10,408ms |

### Long-Context Condition (FPR across context window sizes)

| Method | 2048 tokens | 4096 tokens | 8192 tokens |
|---|---|---|---|
| Self-Consistency | 77.8% | 80.0% | 70.0% |
| Faithfulness Score | 33.3% | 20.0% | 40.0% |
| NLI Detection | 11.1% | 10.0% | 10.0% |
| LLM-as-Judge | 0.0% | 0.0% | 10.0% |

### Key Findings

1. **NLI-based detection came out best overall.** Highest F1 (0.500), lowest FPR on the adversarial set (10.5%), stable behaviour across every context window size I tested, and roughly 30x faster than LLM-as-Judge.

2. **LLM-as-Judge failed completely on adversarial detection.** 0% recall across 20 adversarial samples, despite being the slowest and most expensive method (~5s/sample). GPT-4o-mini simply could not self-evaluate hallucinations that were phrased to sound grounded.

3. **Self-consistency is essentially a false-alarm generator.** 100% recall, but only 7.1% precision and 60 to 80% FPR across all conditions. Not usable in a production setting.

4. **Faithfulness scoring gave the best speed-to-accuracy tradeoff.** 31ms per sample with F1=0.222 on adversarial and a lower FPR than self-consistency, which makes it a reasonable choice for high-throughput pipelines where latency matters.

> These findings hold for the original GPT-4o-mini setup, but note the adversarial
> caveat above. The follow-up analysis below re-ran the experiment with a fixed,
> balanced injection and a stronger model, and the picture changes in instructive ways.

---

## Follow-up analysis: balanced injection + stronger model

After the first pass I audited the adversarial setup and found two problems: the
positive class had collapsed to ~1 sample (see the caveat above), and I wanted to
know how much of the LLM-as-Judge failure was model capability versus method. So I
made two changes and re-ran:

1. **Rebuilt the adversarial injection.** Instead of hoping an unrelated adversarial
   document gets retrieved, a contradiction about the *question's own drug* is now
   synthesised from a template and injected directly into the generator's context on
   a fixed 50% of samples. This gives a balanced, meaningful positive class (10
   injected / 10 clean).
2. **Swapped the generator and judge to Claude Sonnet** (via the Anthropic API) to
   test whether a frontier model changes the result.

### Results (Claude Sonnet, balanced injection)

Adversarial, n=20 (10 contradiction-injected, 10 clean):

| Method | Recall | Precision | F1 | FPR |
|---|---|---|---|---|
| LLM-as-Judge | 20% | **66.7%** | **0.308** | 10% |
| Faithfulness Score | 100% | 50% | 0.667* | 100% |
| NLI Detection | 0% | 0% | 0.000 | 10% |

Clean, n=10 (FPR): LLM-as-Judge 20%, Faithfulness 80%, NLI 10%.

\* Faithfulness's F1 looks high only because it flags *everything* (100% recall at
100% FPR) - it is not discriminating between grounded and hallucinated answers here.

### What changed, and why it matters

1. **A strong generator neutralises most injected contradictions.** Of the 10
   samples where a contradiction was injected into the context, Sonnet hedged on 1,
   answered from the real label on most, and in several cases *explicitly flagged the
   conflict* ("Sources present conflicting information"). The confident propagation
   of contradictions that the original study measured looks largely like a
   weak-model behaviour. This is the most interesting finding: with a strong model
   and a strict grounding prompt, there were few genuine hallucinations to detect.

2. **LLM-as-Judge was the only discriminating detector this time.** 66.7% precision
   and 10% FPR - when the Sonnet judge flagged an answer it was usually right. Its
   low recall mostly reflects that there was little to catch, not that it failed. This
   is the opposite of the GPT-4o-mini result (0% recall) and supports the idea that
   LLM-as-Judge is heavily model-bound.

3. **ROUGE-based faithfulness breaks down with a verbose, multi-source generator.**
   It flagged 100% of adversarial and 80% of clean answers - lexical overlap
   collapses when answers are well-written and cite several sources. It is not
   usable as a detector in this regime.

4. **NLI checks the wrong thing when the contradiction is in the context.** The
   injected contradiction lives in the retrieved context, and the answer is largely
   entailed *by that context*, so answer-vs-context entailment does not flag it (0%
   recall). NLI catches fabrications the model invents, not contradictions it faithfully
   repeats from a poisoned source.

### Honest limitations of this follow-up

- **Not a controlled head-to-head.** I could not re-run the GPT-4o-mini arm under the
  new injection scheme (its API key was no longer valid), so the GPT-4o-mini and
  Sonnet numbers use different adversarial setups and are not directly comparable.
- **Still small (n=20 / n=10)** and single-seed. Directional, not conclusive.
- **The label is a proxy.** "Hallucination" marks that the generator was *shown* a
  contradiction, which is an upper bound - a strong generator often refuses to repeat
  it, which depresses measured recall. Truly isolating detector quality would mean
  constructing hallucinated *answers* directly rather than relying on the generator.
- **Single generator and judge model** (Sonnet) in this arm.

The runner supports this out of the box: `--model claude-sonnet-5` swaps the generator
and judge, and `--injection-rate 0.5` sets the balanced positive class.

---

## Quick Start

### 1. Clone and set up environment

```bash
git clone <your-repo>
cd medical-rag
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install sentencepiece        # required for NLI tokenizer
```

### 2. Configure your API key

```bash
cp .env.example .env
# Edit .env and add your OpenRouter API key
# Get one free at https://openrouter.ai
```

### 3. Ingest targeted drug labels

```bash
# Downloads 25 specific drugs by name (warfarin, metformin, SSRIs, fluoroquinolones, etc.)
# All XMLs are cached locally, so subsequent runs are instant
python scripts/ingest_targeted.py

# Optional: ingest by random DailyMed listing (less reliable, returns mixed content)
python scripts/ingest_data.py --source dailymed --max-labels 200
```

### 4. Build adversarial injection set

```bash
python scripts/build_adversarial_set.py --n-pairs 30
```

### 5. Run the benchmark

```bash
# Clean condition (baseline false positive rate)
python experiments/run_benchmark.py --condition clean --n-samples 5

# Adversarial condition (main experiment)
python experiments/run_benchmark.py --condition adversarial --n-samples 20

# Long-context degradation (tests 2048 / 4096 / 8192 token windows)
python experiments/run_benchmark.py --condition long_context --n-samples 10
```

---

## Experimental Conditions

| Condition | Description | What it tests |
|---|---|---|
| `clean` | No adversarial documents in retrieval corpus | False positive rate of each detector |
| `adversarial` | 20% of corpus contains injected contradictions | Hallucination recall and precision |
| `long_context` | Context window swept across 2k / 4k / 8k tokens | Detector degradation under longer context |

---

## Detection Methods

| Method | Type | Speed | Notes |
|---|---|---|---|
| `llm_judge` | LLM-based (GPT-4o-mini via OpenRouter) | ~5s | Fails on adversarial, 0% recall |
| `self_consistency` | N=5 sampling + ROUGE-L agreement | ~10s | High recall, unusable FPR |
| `faithfulness` | Claim-level ROUGE-L grounding score | ~30ms | Fast, interpretable, competitive |
| `hhem` | NLI cross-encoder (DeBERTa-v3-small) | ~165ms | Best F1 and most stable across conditions |

> **Note on HHEM naming:** The original design used Vectara's HHEM model, which has dependency issues with `transformers>=5.x`. The implementation uses `cross-encoder/nli-deberta-v3-small`, a reliable entailment model with equivalent functionality. The `HHEMScorer` class name is preserved for interface consistency.

---

## Project Structure

```
medical-rag/
├── src/
│   ├── retriever/
│   │   ├── document_loader.py      # DailyMedLoader, FAERSLoader, Document dataclass
│   │   ├── embedder.py             # sentence-transformers wrapper (all-MiniLM-L6-v2)
│   │   ├── vector_store.py         # ChromaDB interface
│   │   └── retriever.py            # Hybrid BM25 + dense, BM25 auto-rebuild on startup
│   ├── generator/
│   │   └── generator.py            # Grounded generator, citation enforcement
│   └── evaluator/
│       ├── evaluator.py            # Benchmark runner, BenchmarkReport
│       └── methods/
│           ├── base.py             # BaseDetector ABC, DetectionResult dataclass
│           ├── llm_judge.py        # JSON verdict via OpenRouter
│           ├── self_consistency.py # ROUGE-L agreement across N samples
│           ├── faithfulness.py     # Claim-level grounding score
│           └── hhem.py             # NLI entailment scoring (DeBERTa)
├── scripts/
│   ├── ingest_data.py              # Random DailyMed listing ingestion
│   ├── ingest_targeted.py          # Targeted ingestion by drug name (recommended)
│   └── build_adversarial_set.py    # Build labelled contradiction pairs
├── experiments/
│   └── run_benchmark.py            # Master runner: all 3 conditions
├── tests/
│   ├── test_retriever.py
│   ├── test_generator.py
│   └── test_evaluator.py
├── configs/
│   ├── config.yaml                 # All tuneable parameters
│   └── prompts.yaml                # Versioned prompt templates
├── data/                           # Downloaded XML + adversarial set (gitignored)
└── results/                        # JSON benchmark outputs (gitignored)
```

---

## Configuration

All parameters in `configs/config.yaml`. Key settings:

```yaml
retriever:
  strategy: "hybrid"                    # "dense" | "bm25" | "hybrid"
  embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
  top_k: 5
  hybrid_alpha: 0.7                     # BM25 weight in hybrid scoring

generator:
  model: "openai/gpt-4o-mini"
  prompt_version: "v1"

evaluator:
  llm_judge:
    model: "openai/gpt-4o-mini"         # via OpenRouter
  self_consistency:
    n_samples: 5
  hhem:
    model: "cross-encoder/nli-deberta-v3-small"
    threshold: 0.5

experiment:
  seed: 42
  adversarial_injection_rate: 0.2
  context_window_sizes: [2048, 4096, 8192]
```

---

## Adversarial Contradiction Types

The `build_adversarial_set.py` script generates labelled contradiction pairs across five categories:

| Type | Example |
|---|---|
| `severity_flip` | "rare" side effect relabelled as "common" |
| `frequency_exaggeration` | "1% incidence" changed to "45% incidence" |
| `interaction_negation` | Drug interaction marked as "no known interaction" |
| `contraindication_removal` | Pregnancy contraindication deleted |
| `temporal_confusion` | Half-life or onset timing inverted |

---

## Running Tests

```bash
# Full test suite
pytest tests/ -v

# Specific module
pytest tests/test_evaluator.py -v

# With coverage
pytest tests/ --cov=src --cov-report=term-missing
```

---

## Datasets

| Dataset | Source | Usage |
|---|---|---|
| FDA DailyMed | [dailymed.nlm.nih.gov](https://dailymed.nlm.nih.gov) | Primary drug label corpus (25 targeted drugs) |
| FAERS | [FDA Adverse Event Reporting System](https://www.fda.gov) | Real-world adverse event reports |

**Targeted drugs indexed:** warfarin, metformin, sertraline, fluoxetine, sumatriptan, ciprofloxacin, levofloxacin, lisinopril, atorvastatin, metoprolol, amlodipine, omeprazole, amoxicillin, azithromycin, prednisone, levothyroxine, gabapentin, aspirin, ibuprofen, acetaminophen, hydrochlorothiazide, losartan, simvastatin, clopidogrel, furosemide

---

## Tech Stack

| Component | Technology |
|---|---|
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector store | ChromaDB |
| Lexical search | BM25Okapi (rank-bm25) |
| Generator | GPT-4o-mini via OpenRouter |
| LLM Judge | GPT-4o-mini via OpenRouter |
| NLI detector | `cross-encoder/nli-deberta-v3-small` |
| Drug label corpus | FDA DailyMed XML API |
| Language | Python 3.13 |

---

## Design Principles

- **Modular**: Retriever, Generator, and Evaluator have clean interfaces and can be swapped independently
- **Reproducible**: All experiments seeded (seed=42), prompts versioned in `configs/prompts.yaml`, all results logged to timestamped JSON
- **Testable**: Full pytest suite, no methods over 20 lines, full docstrings on every public method
- **Production-grade**: No primitive obsession, dependency injection throughout, BaseDetector ABC enforces consistent interface across all four detection methods

---

## Limitations and What I'd Do Differently

A few things I'd flag honestly about the current results:

- **Small sample sizes.** The headline numbers come from 20 adversarial and 5 clean samples. That's enough to see clear directional differences (NLI vs LLM-judge is not subtle), but the precision/FPR figures should be read as indicative, not final. A proper evaluation would run a few hundred samples per condition.
- **Single generator model.** Everything runs on GPT-4o-mini. The LLM-as-Judge failure in particular might look different with a stronger judge model, and I didn't get to test that.
- **ROUGE-L as a grounding proxy.** Both faithfulness and self-consistency lean on ROUGE-L, which rewards lexical overlap. A paraphrased-but-correct answer can be penalised, which likely inflates their false-positive rates.
- **HHEM substitution.** I used `cross-encoder/nli-deberta-v3-small` in place of Vectara's HHEM (see the note above). It works well, but it isn't the exact model the original design called for.

If I extended this, the first steps would be scaling the sample counts, adding a second (stronger) judge model, and swapping ROUGE-L for an embedding-based grounding score.