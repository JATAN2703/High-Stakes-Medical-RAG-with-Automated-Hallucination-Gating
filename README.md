# High-Stakes Medical RAG with Automated Hallucination Gating

**Foundations of Generative AI — Final Project**

A production-grade RAG pipeline focused on pharmacology and drug safety, with a rigorous benchmarking suite for hallucination detection. Rather than building a generic chatbot, this system tackles the real industry bottleneck: *trusting the output*.

---

## Research Question

> *Which hallucination detection method most reliably identifies fabricated or conflicting drug safety information in a pharmacology RAG pipeline — and under what conditions does each method fail?*

---

## Architecture

The system is structured as three fully decoupled modules:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────────┐
│   RETRIEVER     │───▶│   GENERATOR     │───▶│       EVALUATOR         │
│                 │    │                 │    │                         │
│ • DailyMed XML  │    │ • OpenRouter    │    │ • LLM-as-Judge          │
│ • FAERS reports │    │   (gpt-4o-mini) │    │ • Self-Consistency      │
│ • BM25 + Dense  │    │ • Strict grounding    │ • Faithfulness Score    │
│ • ChromaDB      │    │ • Citation enforcement│ • HHEM (Vectara)        │
└─────────────────┘    └─────────────────┘    └─────────────────────────┘
```

---

## Quick Start

### 1. Clone and set up environment

```bash
git clone <your-repo>
cd medical-rag
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure your API key

```bash
cp .env.example .env
# Add your OpenRouter API key to .env
```

### 3. Ingest data

```bash
# Download and index FDA DailyMed drug labels
python scripts/ingest_data.py --source dailymed --max-labels 200

# Also index FAERS adverse event reports (optional)
python scripts/ingest_data.py --source faers
```

### 4. Build adversarial injection set

```bash
python scripts/build_adversarial_set.py --n-pairs 75
```

### 5. Run the benchmark

```bash
# Run all three experimental conditions
python experiments/run_benchmark.py --n-samples 20

# Run a specific condition
python experiments/run_benchmark.py --condition adversarial --methods llm_judge faithfulness

# Quick test with fewer samples
python experiments/run_benchmark.py --n-samples 5 --condition clean
```

---

## Experimental Conditions

| Condition | Description | What it tests |
|---|---|---|
| `clean` | No adversarial documents | False positive rate |
| `adversarial` | 20% of retrieved docs are contradictions | Hallucination recall |
| `long_context` | 2k / 4k / 8k token contexts | Context window degradation |

---

## Detection Methods

| Method | Type | Speed | Strength |
|---|---|---|---|
| `llm_judge` | LLM-based | Slow (~2s) | Nuanced reasoning |
| `self_consistency` | Sampling | Slow (~5s) | No extra model needed |
| `faithfulness` | ROUGE-based | Fast (<0.1s) | Transparent, interpretable |
| `hhem` | Dedicated model | Medium (~0.5s) | Specialist fine-tuned |

---

## Project Structure

```
medical-rag/
├── src/
│   ├── retriever/          # Document loading, embedding, hybrid retrieval
│   ├── generator/          # Grounded answer generation via OpenRouter
│   └── evaluator/          # Hallucination detection suite
│       └── methods/        # Four independent detection methods
├── scripts/
│   ├── ingest_data.py      # Download & index FDA data
│   └── build_adversarial_set.py  # Build adversarial injection set
├── experiments/
│   └── run_benchmark.py    # Master experiment runner
├── tests/                  # pytest test suite
├── configs/
│   ├── config.yaml         # All tuneable parameters
│   └── prompts.yaml        # Versioned prompt templates
├── data/                   # Downloaded data (gitignored)
└── results/                # Experiment outputs (gitignored)
```

---

## Running Tests

```bash
# Full test suite with coverage
pytest

# Specific test file
pytest tests/test_evaluator.py -v

# Mutation testing on evaluator logic
mutmut run --paths-to-mutate src/evaluator/
mutmut results
```

---

## Datasets

| Dataset | Source | Usage |
|---|---|---|
| FDA DailyMed | [dailymed.nlm.nih.gov](https://dailymed.nlm.nih.gov) | Primary drug label corpus |
| FAERS | [fda.gov/drugs/questions-and-answers/fda-adverse-event-reporting-system-faers](https://www.fda.gov) | Real-world adverse event reports |
| BioASQ | [bioasq.org](http://bioasq.org) | Evaluation benchmark |
| MedQA | HuggingFace `bigbio/med_qa` | USMLE-style QA benchmark |

---

## Configuration

All parameters live in `configs/config.yaml`. Key settings:

```yaml
retriever:
  strategy: "hybrid"      # "dense" | "bm25" | "hybrid"
  top_k: 5

generator:
  model: "openai/gpt-4o-mini"

evaluator:
  methods: [llm_judge, self_consistency, faithfulness, hhem]

experiment:
  adversarial_injection_rate: 0.2
  context_window_sizes: [2048, 4096, 8192]
```

---

## Key Design Principles

- **Modular**: Each module (Retriever, Generator, Evaluator) has a clean interface and can be swapped independently
- **Reproducible**: All experiments are seeded; all prompts are versioned; all results are logged to JSON
- **Testable**: Full pytest suite with coverage targets on the Evaluator's metric computation logic
- **Production-grade**: No long methods, no primitive obsession, full docstrings on every public method
