# High-Stakes Medical RAG with Hallucination Detection Benchmarking

**Foundations of Generative AI — Final Project**

A production-grade RAG pipeline focused on pharmacology and drug safety, with a rigorous benchmarking suite comparing four hallucination detection methods. Rather than building a generic chatbot, this system tackles the real industry bottleneck: *trusting the output*.

---

## Research Question

> *Which hallucination detection method most reliably identifies fabricated or conflicting drug safety information in a pharmacology RAG pipeline — and under what conditions does each method fail?*

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

### Long-Context Condition — FPR across context window sizes

| Method | 2048 tokens | 4096 tokens | 8192 tokens |
|---|---|---|---|
| Self-Consistency | 77.8% | 80.0% | 70.0% |
| Faithfulness Score | 33.3% | 20.0% | 40.0% |
| NLI Detection | 11.1% | 10.0% | 10.0% |
| LLM-as-Judge | 0.0% | 0.0% | 10.0% |

### Key Findings

1. **NLI-based detection is the best overall method.** Highest F1 (0.500), lowest FPR on adversarial (10.5%), consistent behaviour across all context window sizes, and 30× faster than LLM-as-Judge.

2. **LLM-as-Judge failed completely on adversarial detection.** 0% recall across 20 adversarial samples despite being the slowest and most expensive method (~5s/sample). GPT-4o-mini is unable to self-evaluate grounded-sounding hallucinations.

3. **Self-consistency is a false alarm generator.** 100% recall but only 7.1% precision and 60–80% FPR across all conditions — unusable in any production system.

4. **Faithfulness scoring offers the best speed–accuracy tradeoff.** 31ms per sample with F1=0.222 on adversarial and lower FPR than self-consistency. Ideal for high-throughput pipelines where latency matters.

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
# All XMLs are cached locally — subsequent runs are instant
python scripts/ingest_targeted.py

# Optional: ingest by random DailyMed listing (less reliable — returns mixed content)
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
| `llm_judge` | LLM-based (GPT-4o-mini via OpenRouter) | ~5s | Fails on adversarial — 0% recall |
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