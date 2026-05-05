# Retrieval-Augmented Generation with Financial Filings

## Summary
This project compares the assignment GPT model with a retrieval-augmented version of the same model on Form 10-K financial question answering.

## Hypothesis
Adding retrieved 10-K passages to the GPT prompt should improve factual grounding and reduce gold-answer perplexity compared with the baseline GPT that sees only the question. The tradeoff is extra retrieval latency and longer prompts.

## Model and Tokenizer
- GPT implementation: `src/gpt.py`
- trained weights: `model/model_weights.pt`
- tokenizer: `model/hftokenizer`
- all systems use the same checkpoint and tokenizer

The compared systems are:
- **Baseline GPT**: the assignment GPT receives only the question.
- **RAG-GPT TF-IDF top-3**: the same assignment GPT receives retrieved Form 10-K chunks plus the question.
- **Oracle Evidence GPT**: the same assignment GPT receives gold evidence.
- **Random Context GPT**: the same assignment GPT receives random chunks as a noise control.

## Data
- raw filings: `data/raw/`
- verified questions: `data/questions/questions_verified.jsonl`
- remapped verified questions, if available: `data/questions/questions_verified.remapped.jsonl`
- chunks: `outputs/chunks/chunks.jsonl`

## Quick Run
```bash
python main.py \
  --quick-run \
  --run-id final_verified_run \
  --report
```

This automatically uses:
- `data/questions/questions_verified.remapped.jsonl` if present
- otherwise `data/questions/questions_verified.jsonl`
- `outputs/chunks/chunks.jsonl`
- `model/model_weights.pt`
- `model/hftokenizer`

## Install Verified Questions
Install from a folder containing the uploaded files:

```bash
python scripts/install_verified_questions.py \
  --source-dir /path/to/uploaded/files \
  --out-dir data/questions
```

Or install from explicit files:

```bash
python scripts/install_verified_questions.py \
  --verified-jsonl questions_verified.jsonl \
  --verified-csv questions_verified.csv \
  --evidence-audit evidence_audit.md \
  --validation-report validation_report.md \
  --out-dir data/questions
```

The installer searches the current directory, project root, `/mnt/data`, `data/questions`, `outputs/questions/verified`, the parent project folder, and `~/Downloads`. It writes:

- `data/questions/questions_verified.jsonl`
- `data/questions/questions_verified.csv`
- `data/questions/evidence_audit.md`
- `data/questions/validation_report.md`
- `data/questions/questions.jsonl`
- `data/questions/install_summary.json`
- `data/questions/question_distribution.csv`

To inspect available candidate files:

```bash
python scripts/install_verified_questions.py \
  --list-candidates \
  --source-dir /mnt/data
```

## Build Chunks
```bash
python scripts/build_corpus.py \
  --data-dir data/raw \
  --out outputs/chunks/chunks.jsonl \
  --chunk-size 180 \
  --overlap 40
```

## Remap Verified Evidence
If chunks have been rebuilt, old verified chunk IDs may not exist. Remap evidence by gold evidence text:

```bash
python scripts/validate_verified_questions.py \
  --questions data/questions/questions_verified.jsonl \
  --chunks outputs/chunks/chunks.jsonl \
  --out data/questions/questions_verified.remapped.jsonl
```

Outputs:
- `data/questions/questions_verified.remapped.jsonl`
- `data/questions/evidence_remap_report.md`
- `data/questions/evidence_remap_summary.csv`

## Inspect Tokenizer and Checkpoint
```bash
python scripts/inspect_tokenizer.py \
  --tokenizer-dir model/hftokenizer \
  --checkpoint model/model_weights.pt
```

```bash
python scripts/inspect_checkpoint.py \
  --checkpoint model/model_weights.pt \
  --tokenizer-dir model/hftokenizer
```

Both should report vocab size `10000` and compatibility `PASS`.

## Smoke Test
```bash
python main.py \
  --quick-run \
  --limit 3 \
  --run-id smoke_test \
  --report
```

Any run using `--limit` is a smoke test and is not valid for research conclusions.

## Outputs
- `outputs/runs/{run_id}/predictions.jsonl`
- `outputs/runs/{run_id}/evaluation_summary.csv`
- `outputs/runs/{run_id}/evaluation_details.jsonl`
- `outputs/runs/{run_id}/retrieval_diagnostics.jsonl`
- `outputs/runs/{run_id}/run_metadata.json`
- `outputs/reports/{run_id}/final_report.md`
- `outputs/reports/{run_id}/comparison_table.csv`
- `outputs/reports/{run_id}/retrieval_table.csv`
- `outputs/reports/{run_id}/latency_table.csv`

## Metrics
- exact match
- token F1
- numeric accuracy
- refusal accuracy
- answer coverage@1 and answer coverage@3
- source accuracy@3
- table-row recall@3
- gold-answer loss
- gold-answer perplexity
- perplexity delta vs baseline
- generation latency
- retrieval latency
- total latency

## Additional Scripts
- `scripts/install_verified_questions.py`: installs verified benchmark artifacts.
- `scripts/validate_verified_questions.py`: remaps verified evidence IDs to current chunks.
- `scripts/inspect_tokenizer.py`: checks `model/hftokenizer`.
- `scripts/inspect_checkpoint.py`: checks `model/model_weights.pt`.
- `scripts/build_corpus.py`: preprocesses filings and builds chunks.
- `scripts/inspect_retrieval.py`: inspects retrieval results.
- `scripts/run_report.py`: rebuilds reports from saved outputs.

## Academic Honesty
Some scaffolding, debugging prompts, and documentation drafts were generated with OpenAI Codex/ChatGPT. All code was reviewed, edited, and tested by the author. The submitted experimental design, implementation choices, and results are the author's responsibility.
