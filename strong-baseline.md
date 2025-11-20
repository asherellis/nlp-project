# Strong Baseline: LLM Query

For the strong baseline, we use the llama-3.1-8b-instant model developed by Meta. Rather than downloading it ourselves, we send each question to model via Groq API, and treat the response as the answer.

## What the Script Does

The script:

1. Installs pandas and groq if needed
2. Loads questions from the input file (determined by split argument)
3. Builds a prompt for each question
4. Sends prompts to Groq (llama-3.1-8b-instant)
5. Writes answers to `{split}_strong_output.json`

## API Key Handling

To set the API key, paste this into your terminal before proceeding:
```bash
export GROQ_API_KEY="your-key-here"
```


## Prompt + Model Call

For each question, we build a simple instruction-style prompt:

```python
def build_prompt(question: str):
    return (
        f"Question:\n{question}\n"
        "Answer the question in two to four complete sentences.\n"
        "Do not include any notes, disclaimers, bullet points, or follow-up offers.\n"
        "Give only the answer.\n"
        "Answer:"
    )
```

Then we call the Groq client:

```python
chat_completion = client.chat.completions.create(
    messages=[{"role": "user", "content": prompt}],
    model="llama-3.1-8b-instant",
    temperature=0.7,
    max_tokens=128,
)
answer = chat_completion.choices[0].message.content.strip()
```

## Output Format

We store everything in a dict like:

```python
predictions[question_id] = {
    "question": question,
    "answer": answer,
}
```

Then we write this to the output file:

```python
with open(output_file, "w") as f:
    json.dump(predictions, f, indent=2)
```

## How to Run

From the directory that contains `strong-baseline.py` and the `NLP Project Data/` folder:

**Basic usage (defaults to train split):**
```bash
python strong-baseline.py
```

**Specify split:**
```bash
python strong-baseline.py train
python strong-baseline.py dev
python strong-baseline.py test
```

**Arguments:**
- `[split]`: Which dataset split to use - `train`, `dev`, or `test` (default: `train`)

The split determines which input file is used as well as the output:

| split | Input file | Output file |
|-------|------------|-------------|
| `train` | `NLP Project Data/train_final_jsonl.txt` | `train_strong_output.json` |
| `dev` | `NLP Project Data/dev_final_jsonl.txt` | `dev_strong_output.json` |
| `test` | `NLP Project Data/test_final_jsonl.txt` | `test_strong_output.json` |

**Input file format:**
- JSONL (one JSON object per line)
- Each line must have a `"question"` field
- Optional fields: `"question_id"`, `"split"`, `"question_type"`
- If `question_id` doesn't exist, we generate ids like `q001`, `q002`, … based on the index

## Performance

Note that this baseline focuses on answer quality, not retrieval/citation. Retrieval and citation metrics are usually 0% since the output doesn't include `retrieved_docs` or `evidence_sentences`.

**Test set (3 questions):**
- Average Answer Score (1-5): 1.67
- Average Evidence Score (0-1): 0.0000
- Average Combined Score (0-1): 0.17

Full performance results for train, dev, and test sets are reported in the 1-2 page PDF report.

