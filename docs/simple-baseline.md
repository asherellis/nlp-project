# Simple Baseline: BM25 Retrieval

For the simple baseline, we use BM25 for document retrieval from a corpus of 1014 documents. Given a question, the system ranks all documents by relevance, retrieves the top-ranked document, and extracts a text snippet as the answer.

## What the Script Does

The script:

1. Installs rank-bm25 and nltk if needed
2. Loads documents from corpus
3. Builds a BM25 index
4. Loads questions from the input file (JSONL)
5. For each question, gets top-5 documents using BM25
6. Extracts a snippet from the top document as the answer
7. Finds sentence IDs from the top document for citations
8. Writes predictions to a JSON file

## Document Loading

The `load_documents()` function walks through the corpus directory recursively. It:

- Loads both structured JSON documents (with "sentences" arrays) and plain text files (which get sentence IDs computed later)
- Stores document IDs based on relative file paths (e.g., `"abisee/cnn_dailymail__123.txt"` becomes `"abisee/cnn_dailymail__123"`)

## BM25 Retrieval

We use BM25Okapi from the rank_bm25 package:

```python
def build_bm25_index(doc_texts):
    tokenized_corpus = [tokenize(doc) for doc in doc_texts]
    return BM25Okapi(tokenized_corpus)
```

Tokenization uses `nltk.word_tokenize` with lowercasing and alphanumeric filtering.

For each question, we:
1. Tokenize the question
2. Compute BM25 scores for all documents
3. Rank documents by score (descending)
4. Retrieve top-5 documents

## Snippet Extraction

Once we have the top-ranked document, we extract a relevant snippet using `extract_snippet()`:

1. Tokenize the document into sentences using a custom PunktSentenceTokenizer (handles abbreviations like "U.S.", "a.m.", "p.m.")
2. Score each sentence by counting query word matches
3. Return the top-3 highest scoring sentences as the snippet

## Sentence ID Extraction

For citation evaluation, we extract sentence IDs from the top-ranked document using `extract_sentence_ids()`:

- For structured JSON docs with "sentences" arrays, we use existing sentence IDs (e.g., `'S1'`, `'S2'`, `'S3'`)
- For plain text docs, we generate sentence IDs (e.g., `'S1'`, `'S2'`, `'S3'`) based on sentence tokenization
- We score sentences by query word overlap and return the top-3 sentence IDs

## Output Format

We store everything in a dict like:

```python
predictions[question_id] = {
    "question": question,
    "answer": snippet_text,
    "answers": [{"answer": snippet, "citation": {...}}, ...],  # for top-5 docs
    "retrieved_docs": [{"doc_id": "...", "score": 12.34, "rank": 1}, ...],
    "evidence_sentences": ["S1", "S7", "S9"]  # sentence IDs from top doc
}
```

Then we write this to the output file:

```python
with open(output_file, "w") as f:
    json.dump(predictions, f, indent=2)
```

This output file can be evaluated using `evaluation.py`.

## How to Run

From the directory that contains `simple-baseline.py` and the `NLP Project Data/` folder:

**Basic usage (defaults to train split):**
```bash
python simple-baseline.py
```

**Specify split:**
```bash
python simple-baseline.py train
python simple-baseline.py dev
python simple-baseline.py test
```

**Arguments:**
- `[split]`: Which dataset split to use - `train`, `dev`, or `test` (default: `train`)
- `[--score-threshold THRESHOLD]`: BM25 score floor (default: `0.0`)

The split determines which input file is used as well as the output:

| split | Input file | Output file |
|-------|------------|-------------|
| `train` | `NLP Project Data/train_final_jsonl.txt` | `train_simple_output.json` |
| `dev` | `NLP Project Data/dev_final_jsonl.txt` | `dev_simple_output.json` |
| `test` | `NLP Project Data/test_final_jsonl.txt` | `test_simple_output.json` |

**Input file format:**
- JSONL (one JSON object per line)
- Each line must have a `"question"` field
- Optional fields: `"question_id"`, `"split"`, `"question_type"`
- If `question_id` doesn't exist, we generate ids like `q001`, `q002`, … based on the index

**Example with threshold:**
```bash
python simple-baseline.py train --score-threshold 5.0
```

## Performance

**Test set (3 questions):**
- Recall@1: 33.33% (1/3)
- Recall@5: 66.67% (2/3)
- Citation Precision: 0.1111
- Citation Recall: 0.0256
- Citation F1: 0.0417

Full performance results for train, dev, and test sets are reported in the 1-2 page PDF report.