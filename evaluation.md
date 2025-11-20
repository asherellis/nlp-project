# Evaluation

The evaluation script computes metrics for document-grounded question answering with evidence citation. Each question has a gold answer, gold evidence sentences (sentence IDs), and a custom 1-5 rubric for judging answers.

We evaluate three components:

1. **Retrieval accuracy** (Recall@1, Recall@5)
2. **Citation correctness** (Precision, Recall, F1 based on sentence ID overlap)
3. **Answer quality** (LLM-as-judge with rubrics) + **Evidence quality** (word overlap) → Combined score

## What the Script Does

The `evaluation.py` script:

1. Loads questions from the input file (JSONL format)
2. Loads predictions from simple and/or strong baseline output files
3. Computes retrieval metrics (Recall@1, Recall@5)
4. Computes citation metrics (Precision, Recall, F1)
5. Runs LLM-as-Judge evaluation (if GROQ_API_KEY is set)
6. Saves results to `{split}_evaluation.json`

## Retrieval Metrics

Recall@k measures the fraction of questions where the correct gold document appears in the top-k retrieved documents:

\[
\text{Recall@k} = \frac{1}{|Q|} \sum_{q \in Q} \mathbb{1}[\text{gold\_doc}(q) \in \text{top\_k\_docs}(q)]
\]

where \(Q\) is the set of questions, \(\text{gold\_doc}(q)\) is the correct document ID for question \(q\), \(\text{top\_k\_docs}(q)\) are the top-k documents retrieved by the system, and \(\mathbb{1}[\cdot]\) is the indicator function (1 if true, 0 otherwise).

- Recall@1: Percentage of questions where the correct document is ranked first
- Recall@5: Percentage of questions where the correct document appears in the top-5 results

## Citation Metrics

Citation metrics measure how accurately the system identifies evidence sentences that support the answer. These use exact sentence ID matching - comparing predicted sentence IDs (e.g., `["S13", "S14", "S15"]`) directly against gold standard sentence IDs.

For each question with gold evidence sentences, we compute:

**Precision:**
\[
\text{Precision} = \frac{|\text{gold\_sentence\_IDs} \cap \text{pred\_sentence\_IDs}|}{|\text{pred\_sentence\_IDs}|}
\]

**Recall:**
\[
\text{Recall} = \frac{|\text{gold\_sentence\_IDs} \cap \text{pred\_sentence\_IDs}|}{|\text{gold\_sentence\_IDs}|}
\]

**F1:**
\[
\text{F1} = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
\]

Dataset-level metrics are computed by averaging Precision, Recall, and F1 across all questions with evidence sentences.

Citation metrics use exact matching on sentence IDs (e.g., `"S13"` must exactly match). Questions without gold evidence sentences are skipped from citation metric calculations.

## LLM-as-Judge Metrics

### Answer Quality Score

For each question, we ask a grading LLM (Groq `llama-3.1-8b-instant`) to score the system's answer against the gold answer and rubric.

The rubric yields a discrete score \(A_i \in \{1, 2, 3, 4, 5\}\), which we normalize to \([0,1]\):

\[
a_i = \frac{A_i}{5}
\]

Thus \(a_i = 1\) corresponds to a perfect rubric score (5/5), and \(a_i = 0.2\) corresponds to the worst score (1/5).

The LLM returns a score and rationale. This uses the "LLM-as-a-judge" approach.

**Rate Limiting:**
Groq client handles rate limits automatically. If a request fails after retries, that question gets score 0 and evaluation continues. No progress messages are shown.

### Evidence Score

For LLM-as-Judge evaluation, we compute evidence quality using word overlap between gold and predicted sentences. This is more flexible than exact sentence ID matching since it looks at actual content.

**Method:**
1. Load document structure (JSON with `sentences` array)
2. Map sentence IDs to actual text
3. Extract words from sentences (lowercase, tokenized)
4. Compute word set overlap

\[
e_i = \frac{|\text{words}(E_i) \cap \text{words}(\hat{E}_i)|}{|\text{words}(E_i)|}
\]

where \(E_i\) is the set of gold evidence sentence IDs, \(\hat{E}_i\) is the set of predicted evidence sentence IDs, and \(\text{words}(S)\) extracts all words from the sentences referenced by the sentence IDs in set \(S\).

**Special cases:**
- If there are no gold evidence sentences, the evidence score is 1.0 if there are no predicted sentences, otherwise 0.0
- If document structure is unavailable, falls back to exact sentence ID set matching: `|gold ∩ pred| / |gold|`

### Combined Score

The combined score is a weighted average of answer quality and evidence correctness:

\[
c_i = \lambda \cdot a_i + (1 - \lambda) \cdot e_i
\]

where \(\lambda \in [0,1]\) controls the relative importance of answer quality vs. evidence. We use \(\lambda = 0.5\) (fixed), meaning equal weight for answer quality and evidence quality.

Both components are normalized to \([0,1]\) range:
- \(a_i \in [0,1]\) is the normalized answer score (original rubric score / 5)
- \(e_i \in [0,1]\) is the evidence score (word overlap)

### Dataset-Level Scores

Given \(N\) examples with rubrics, we report:

- Mean raw rubric score: \(\overline{A} = \frac{1}{N}\sum_{i=1}^N A_i\)
- Mean normalized answer score: \(\overline{a} = \frac{1}{N}\sum_{i=1}^N a_i\)
- Mean evidence score: \(\overline{e} = \frac{1}{N}\sum_{i=1}^N e_i\)
- Mean combined score: \(\overline{c} = \frac{1}{N}\sum_{i=1}^N c_i\)

For retrieval metrics, we report Recall@1 and Recall@5 as percentages.

For citation metrics, we report average Precision, Recall, and F1 across all questions with evidence.

## How to Run

From the directory that contains `evaluation.py` and the `NLP Project Data/` folder:

**Basic usage (defaults to train split):**
```bash
python evaluation.py
```

**Specify dataset split:**
```bash
python evaluation.py train
python evaluation.py dev
python evaluation.py test
```

**Arguments:**
- `[split]`: Which dataset split to evaluate - `train`, `dev`, or `test` (default: `train`)

The split determines which files are used:

| split | Questions file | Simple output | Strong output | Results file |
|-------|----------------|---------------|---------------|--------------|
| `train` | `NLP Project Data/train_final_jsonl.txt` | `train_simple_output.json` | `train_strong_output.json` | `train_evaluation.json` |
| `dev` | `NLP Project Data/dev_final_jsonl.txt` | `dev_simple_output.json` | `dev_strong_output.json` | `dev_evaluation.json` |
| `test` | `NLP Project Data/test_final_jsonl.txt` | `test_simple_output.json` | `test_strong_output.json` | `test_evaluation.json` |

**Settings:**
- Document directory: `corpus` (hardcoded)
- Lambda weight: `0.5` (hardcoded)

**Environment Setup (for LLM-as-Judge):**
```bash
export GROQ_API_KEY="your-key-here"
```

**What gets evaluated:**

For **simple baseline** (if `{split}_simple_output.json` exists):
- Retrieval metrics (Recall@1, Recall@5)
- Citation metrics (Precision, Recall, F1)

For **strong baseline** (if `{split}_strong_output.json` exists):
- Retrieval metrics (Recall@1, Recall@5) - typically 0% since strong baseline doesn't output `retrieved_docs`
- Citation metrics (Precision, Recall, F1) - typically 0% since strong baseline doesn't output `evidence_sentences`
- LLM-as-Judge metrics (Answer score, Evidence score, Combined score) - only if GROQ_API_KEY is set

**Output:**
Results are automatically saved to `{split}_evaluation.json` containing separate results for `simple` and `strong` baselines.

## File Formats

**Questions file (JSONL format):** Each line is a JSON object:

```json
{
  "doc_id": "russia-ukraine-conflict_2022-08-05T12_31_57Z",
  "source_dataset": "hugginglearners/russia-ukraine-conflict-articles",
  "source": "2022-08-05T12:31:57Z",
  "question_type": "B",
  "question": "What specific EU energy measure was scheduled to take effect...",
  "answer": "The report states that an EU plan to cut gas use...",
  "evidence_sentences": ["S14", "S15"],
  "edge_case_type": "precise_fact_retrieval",
  "rubric": {
    "description": "Rubric focused on verifying exact numerical and procedural facts...",
    "scale": {
      "1": "...",
      "2": "...",
      "3": "...",
      "4": "...",
      "5": "..."
    }
  }
}
```

**Simple baseline predictions file (JSON format):** Dictionary mapping question IDs to predictions:

```json
{
  "q001": {
    "question": "...",
    "answer": "The system-generated answer...",
    "evidence_sentences": ["S13", "S14", "S15"],
    "retrieved_docs": [
      {
        "doc_id": "cnn_dailymail__abc123",
        "score": 12.34,
        "rank": 1
      },
      ...
    ]
  },
  ...
}
```

**Strong baseline predictions file (JSON format):** Dictionary mapping question IDs to predictions (only requires `question` and `answer`):

```json
{
  "q001": {
    "question": "...",
    "answer": "The system-generated answer..."
  },
  ...
}
```

Note: Strong baseline files typically don't include `retrieved_docs` or `evidence_sentences`, which is why retrieval and citation scores are usually 0 for the strong baseline.

## Example Output

Running `python evaluation.py train` produces:

```
Evaluating train split outputs...
Questions file: NLP Project Data/train_final_jsonl.txt
Loaded 24 questions

============================================================
Evaluating: train_simple_output.json
============================================================

============================================================
RETRIEVAL EVALUATION RESULTS
============================================================

Total Questions: 24

Recall@1: 14/24 = 58.33%
Recall@5: 16/24 = 66.67%

============================================================
CITATION EVALUATION RESULTS
============================================================

Total Questions: 24
Questions with Evidence: 22

--- OVERALL CITATION METRICS (all questions) ---
Average Precision: 0.3030
Average Recall: 0.2307
Average F1: 0.2224

============================================================
Evaluating: train_strong_output.json
============================================================

Loaded predictions for 24 questions

============================================================
RETRIEVAL EVALUATION RESULTS
============================================================

Total Questions: 24

Recall@1: 0/24 = 0.00%
Recall@5: 0/24 = 0.00%

============================================================
CITATION EVALUATION RESULTS
============================================================

Total Questions: 24
Questions with Evidence: 22

--- OVERALL CITATION METRICS (all questions) ---
Average Precision: 0.0000
Average Recall: 0.0000
Average F1: 0.0000

============================================================
LLM_JUDGE EVALUATION RESULTS
============================================================

Total Questions: 24
Questions with Rubrics: 24
Lambda Weight (answer vs evidence): 0.50

--- OVERALL SCORES ---
Average Answer Score (1-5): 2.58
Average Evidence Score (0-1): 0.0833
Average Combined Score (0-1): 0.30

============================================================
Evaluation results saved to: train_evaluation.json
============================================================
```

The output JSON file (`{split}_evaluation.json`) contains detailed per-question results for all metrics, organized by baseline type (simple/strong).
