# CIS 5300 Term Project: Document-Grounded Question Answering with Evidence Citation

## Motivation

Intelligence analysts today face a difficult trade-off when investigating national security issues. They can comb through lengthy reports and documents manually—a time-consuming process—or query large language models for quick summaries that often hallucinate details and lack verifiable sourcing. What's missing is a middle ground: a tool that delivers fast, trustworthy answers with transparent citations.

This project aims to fill that gap by building a retrieval-augmented generation (RAG) system that provides concise, accurate answers while showing precisely where in the source documents the information originated. This approach provides both the efficiency and accountability that analytical work demands.

## Project Overview

This project implements a document-grounded question answering system that:
1. Retrieves relevant documents from a corpus of 787 documents (news articles + CIA World Factbook)
2. Generates answers grounded in retrieved documents using Llama 3.1 8B (via Groq API)
3. Cites specific sentence IDs as evidence

## Team Members
- Vincent Lin
- Risha Kumar
- Chih Yu Tsai
- Asher Ellis

## Acknowledgments
Thanks to Anirudh Bharadwaj for his guidance as our TA and project advisor!

## Project Structure

```
cis5300_project_unified/
├── code/                    # All Python scripts
│   ├── simple-baseline.py   # BM25 retrieval baseline
│   ├── strong-baseline.py   # LLM-only baseline (Llama 3.1 8B via Groq)
│   ├── evaluation.py        # Evaluation script for baselines
│   ├── extension1.py        # Extension 1: RAG (BM25 + LLM generation)
│   ├── extension2.py        # Extension 2: RAG + Cross-Encoder Reranking
│   ├── extension3.py        # Extension 3: RAG + LLM-based Citation Extraction
│   ├── extension4.py        # Extension 4: Hybrid Retrieval (BM25 + Dense)
│   ├── evaluate_extension*.py
│   ├── config.py            # Path configuration
│   └── README.md            # Code usage instructions
├── data/                    # Training/dev/test data
│   ├── NLP Project Data/    # Question-answer pairs (JSONL)
│   ├── corpus/              # Document corpus (787 documents)
│   └── README.md            # Data description
├── output/                  # Model predictions and evaluation results
│   └── README.md            # Output description
└── docs/                    # Documentation and reports
    ├── evaluation.md        # Evaluation metrics description
    ├── simple-baseline.md   # Simple baseline documentation
    ├── strong-baseline.md   # Strong baseline documentation
    ├── extension1_report.pdf
    ├── milestone2_report.pdf
    └── milestone3.pptx      # Presentation slides
```

## Quick Start

### Prerequisites
```bash
pip install rank-bm25 nltk groq sentence-transformers
```

### Set API Key
```bash
export GROQ_API_KEY="your-groq-api-key"
```

### Run Baselines
```bash
cd cis5300_project_unified

# Simple baseline (BM25 retrieval)
python3 code/simple-baseline.py train

# Strong baseline (LLM-only)
python3 code/strong-baseline.py train

# Evaluate baselines
python3 code/evaluation.py train
```

### Run Extensions
```bash
# Extension 1: RAG (BM25 + LLM)
python3 code/extension1.py train
python3 code/evaluate_extension1.py train

# Extension 2: RAG + Cross-Encoder Reranking
python3 code/extension2.py train
python3 code/evaluate_extension2.py train

# Extension 3: RAG + LLM-based Citation Extraction
python3 code/extension3.py train
python3 code/evaluate_extension3.py train

# Extension 4: Hybrid Retrieval (BM25 + Dense Embeddings)
python3 code/extension4.py train
python3 code/evaluate_extension4.py train
```

## Evaluation Metrics

1. **Retrieval Metrics**: Recall@1, Recall@5
2. **Citation Metrics**: Precision, Recall, F1 (exact sentence ID matching)
3. **Answer Quality**: LLM-as-Judge (1-5 rubric score)
4. **Combined Score**: λ × answer_score + (1-λ) × evidence_score (λ=0.5)

## Results Summary (Train Set - 24 Questions)

| Model | Recall@1 | Recall@5 | Citation F1 | Answer Score | Combined |
|-------|----------|----------|-------------|--------------|----------|
| Simple (BM25) | 58.33% | 79.17% | 0.222 | N/A | N/A |
| Strong (LLM) | 0% | 0% | 0% | 2.58/5 | 0.30 |
| Extension 1 (RAG) | 58.33% | 79.17% | 0.358 | 3.67/5 | 0.57 |
| Extension 2 (Rerank) | 79.17% | 87.50% | 0.358 | 3.67/5 | 0.57 |
| **Extension 3 (LLM Cite)** | 79.17% | 87.50% | **0.610** | 3.75/5 | **0.69** |
| **Extension 4 (Hybrid)** | **83.33%** | **91.67%** | 0.579 | **3.88/5** | 0.69 |

### Key Improvements

- **Extension 2** (Cross-encoder reranking): Recall@1 improved 58% → 79% (+36% relative)
- **Extension 3** (LLM citations): Citation F1 improved 0.36 → 0.61 (+70% relative)
- **Extension 4** (Hybrid retrieval): Recall@1 improved 79% → 83%, best answer score (3.88/5)

## Extensions Implemented

1. **Extension 1: RAG** - Combines BM25 retrieval with LLM answer generation
2. **Extension 2: Cross-Encoder Reranking** - Reranks BM25 results using `ms-marco-MiniLM-L-6-v2`
3. **Extension 3: LLM-based Citation Extraction** - Uses LLM to identify supporting sentences semantically
4. **Extension 4: Hybrid Retrieval** - Combines BM25 (lexical) + sentence-transformers (semantic) embeddings

## Milestones

- [x] Milestone 0: Proposal
- [x] Milestone 1: Literature Review + Data Collection
- [x] Milestone 2: Evaluation + Baselines
- [x] Milestone 3: Extension 1 (RAG)
- [x] Milestone 4: Extensions 2-4 + Final Report + Presentation
