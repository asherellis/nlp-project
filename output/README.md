# Output Directory

This directory contains model predictions and evaluation results.

## Files

### Predictions (JSON format)

| File | Model | Description |
|------|-------|-------------|
| `{split}_simple_output.json` | Simple Baseline | BM25 retrieval predictions |
| `{split}_strong_output.json` | Strong Baseline | LLM-only predictions |
| `{split}_extension1_output.json` | Extension 1 | RAG (BM25 + LLM) predictions |
| `{split}_extension2_output.json` | Extension 2 | RAG + Reranking predictions |
| `{split}_extension3_output.json` | Extension 3 | RAG + LLM Citation predictions |
| `{split}_extension4_output.json` | Extension 4 | Hybrid Retrieval predictions |
| `{split}_extension5_output.json` | Extension 5 | [PLACEHOLDER] |
| `{split}_extension6_output.json` | Extension 6 | [PLACEHOLDER] |

### Evaluation Results

| File | Description |
|------|-------------|
| `{split}_evaluation.json` | Combined evaluation for simple & strong baselines |
| `{split}_extension1_evaluation.json` | Extension 1 evaluation |
| `{split}_extension2_evaluation.json` | Extension 2 evaluation |
| `{split}_extension3_evaluation.json` | Extension 3 evaluation |
| `{split}_extension4_evaluation.json` | Extension 4 evaluation |
| `{split}_extension5_evaluation.json` | [PLACEHOLDER] |
| `{split}_extension6_evaluation.json` | [PLACEHOLDER] |

## Prediction Format

### Simple Baseline Output
```json
{
  "q001": {
    "question": "According to the article...",
    "answer": "The article states that...",
    "retrieved_docs": [
      {"doc_id": "cnn_dailymail__abc123", "score": 12.34, "rank": 1},
      ...
    ],
    "evidence_sentences": ["S13", "S14", "S15"]
  }
}
```

### Strong Baseline Output
```json
{
  "q001": {
    "question": "According to the article...",
    "answer": "The LLM-generated answer..."
  }
}
```

### Extension Outputs
Same format as Simple Baseline with both retrieval results and LLM-generated answers.

## Running Evaluation

From the project root directory:

```bash
# Evaluate baselines
python3 code/evaluation.py train

# Evaluate extensions
python3 code/evaluate_extension1.py train
python3 code/evaluate_extension2.py train
python3 code/evaluate_extension3.py train
python3 code/evaluate_extension4.py train
```

## Evaluation Metrics

### Retrieval Metrics
- **Recall@1**: % of questions where correct doc is ranked first
- **Recall@5**: % of questions where correct doc is in top 5

### Citation Metrics
- **Precision**: Correct sentences / Predicted sentences
- **Recall**: Correct sentences / Gold sentences
- **F1**: Harmonic mean of precision and recall

### Answer Quality (LLM-as-Judge)
- **Answer Score**: 1-5 rubric score (normalized to 0-1)
- **Evidence Score**: Word overlap between gold and predicted sentences
- **Combined Score**: 0.5 × answer_score + 0.5 × evidence_score

---

## Current Results

### Train Split (24 questions)

| Model | Recall@1 | Recall@5 | Citation F1 | Answer Score | Combined |
|-------|----------|----------|-------------|--------------|----------|
| Simple (BM25) | 58.33% | 79.17% | 0.222 | N/A | N/A |
| Strong (LLM) | 0% | 0% | 0% | 2.58/5 | 0.30 |
| Extension 1 (RAG) | 58.33% | 79.17% | 0.358 | 3.67/5 | 0.57 |
| Extension 2 (Rerank) | 79.17% | 87.50% | 0.358 | 3.67/5 | 0.57 |
| **Extension 3 (LLM Cite)** | 79.17% | 87.50% | **0.610** | 3.75/5 | **0.69** |
| **Extension 4 (Hybrid)** | **83.33%** | **91.67%** | 0.579 | **3.88/5** | 0.69 |
| Extension 5 | - | - | - | - | - |
| Extension 6 | - | - | - | - | - |

### Dev Split (3 questions)

| Model | Recall@1 | Recall@5 | Citation F1 | Answer Score | Combined |
|-------|----------|----------|-------------|--------------|----------|
| Simple (BM25) | 33.33% | 66.67% | 0.042 | N/A | N/A |
| Extension 1 (RAG) | 33.33% | 66.67% | 0.391 | 3.00/5 | 0.47 |
| Extension 2 (Rerank) | 66.67% | 66.67% | 0.391 | 3.00/5 | 0.49 |
| Extension 3 (LLM Cite) | 66.67% | 66.67% | 0.570 | 3.33/5 | 0.62 |
| Extension 4 (Hybrid) | 66.67% | 66.67% | 0.553 | 4.00/5 | 0.66 |
| Extension 5 | - | - | - | - | - |
| Extension 6 | - | - | - | - | - |

### Test Split (3 questions)

| Model | Recall@1 | Recall@5 | Citation F1 | Answer Score | Combined |
|-------|----------|----------|-------------|--------------|----------|
| Simple (BM25) | 33.33% | 66.67% | 0.042 | N/A | N/A |
| Extension 1 (RAG) | 33.33% | 66.67% | 0.042 | 2.33/5 | 0.29 |
| Extension 2 (Rerank) | 66.67% | 66.67% | 0.042 | 3.67/5 | 0.48 |
| Extension 3 (LLM Cite) | 66.67% | 66.67% | 0.352 | 4.33/5 | 0.65 |
| Extension 4 (Hybrid) | 66.67% | 66.67% | 0.352 | 4.00/5 | 0.62 |
| Extension 5 | - | - | - | - | - |
| Extension 6 | - | - | - | - | - |

---

## Key Findings

1. **Extension 2 (Reranking)** dramatically improves retrieval: Recall@1 jumps from 58% to 79% on train
2. **Extension 3 (LLM Citations)** dramatically improves citation quality: F1 improves from 0.36 to 0.61 (+70%)
3. **Extension 4 (Hybrid)** achieves best retrieval: 83% Recall@1, 92% Recall@5
4. Both Extensions 3 and 4 achieve the highest combined score (0.69)
