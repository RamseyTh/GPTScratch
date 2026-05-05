# Retrieval-Augmented Generation with Financial Filings

## Summary
This project compares the assignment GPT model with a retrieval-augmented version of the same model on Form 10-K financial question answering.

## Hypothesis
Adding retrieved 10-K passages to the GPT prompt should improve factual grounding and reduce gold-answer perplexity compared with the baseline GPT that sees only the question. The tradeoff is additional retrieval overhead and longer prompts.

## Model and Tokenizer
- GPT implementation: `src/gpt.py`
- trained weights: `model/model_weights.pt`
- tokenizer: `model/hftokenizer`
- tokenizer/checkpoint compatibility is checked before each run
- all systems use the same checkpoint and tokenizer

## Main Systems
- `baseline_gpt`: question only
- `rag_gpt_tfidf_top3`: retrieved top-3 chunks plus question
- `oracle_gpt`: gold evidence plus question
- `random_context_gpt`: random chunks plus question

## Data
- raw filings: `data/raw/`
- verified questions: `data/questions/questions_verified.jsonl`
- remapped verified questions: `data/questions/questions_verified.remapped.jsonl`
- optional cleaned research questions: `data/questions/questions_research.validated.jsonl`
- chunks: `outputs/chunks/chunks.jsonl`

## Quick Run
```bash
python main.py --quick-run --run-id final_verified_run --report
```

This uses:
- the best available research or verified question file
- `outputs/chunks/chunks.jsonl`
- `model/model_weights.pt`
- `model/hftokenizer`
- `baseline_gpt`, `rag_gpt_tfidf_top3`, `oracle_gpt`, and `random_context_gpt`

## Smoke Test
```bash
python main.py --quick-run --limit 3 --run-id smoke_test --report
```

Any run using `--limit` is a smoke test and should not be used for final conclusions.

## Installing Verified Questions
```bash
python scripts/install_verified_questions.py \
  --source-dir /path/to/verified/files \
  --out-dir data/questions
```

## Building Chunks
```bash
python scripts/build_corpus.py \
  --data-dir data/raw \
  --out outputs/chunks/chunks.jsonl \
  --chunk-size 180 \
  --overlap 40
```

## Remapping Verified Evidence
```bash
python scripts/validate_verified_questions.py \
  --questions data/questions/questions_verified.jsonl \
  --chunks outputs/chunks/chunks.jsonl \
  --out data/questions/questions_verified.remapped.jsonl
```

## Optional: Clean Research Questions
```bash
python scripts/clean_verified_questions.py \
  --questions data/questions/questions_verified.remapped.jsonl \
  --chunks outputs/chunks/chunks.jsonl \
  --out data/questions/questions_research.jsonl
```

## Optional: Chunking Ablation
Build chunk configs:

```bash
python scripts/build_all_chunk_configs.py \
  --data-dir data/raw \
  --configs fixed_128,fixed_256,fixed_512,section_180,table_row_only,table_aware_mixed,table_aware_clean,table_aware_clean_context
```

Run ablation:

```bash
python main.py \
  --quick-run \
  --experiment chunk_ablation \
  --questions data/questions/questions_research.validated.jsonl \
  --chunk-configs fixed_128,fixed_256,fixed_512,section_180,table_row_only,table_aware_mixed,table_aware_clean,table_aware_clean_context \
  --run-id chunk_ablation_research \
  --report
```

## Outputs
Main run outputs:

`outputs/runs/{run_id}/`
- `predictions.jsonl`
- `evaluation_summary.csv`
- `evaluation_details.jsonl`
- `retrieval_diagnostics.jsonl`
- `run_metadata.json`

`outputs/reports/{run_id}/`
- `final_report.md`
- `comparison_table.csv`
- `retrieval_table.csv`
- `latency_table.csv`
- `failure_taxonomy.csv`
- `failure_examples.md`

Chunk ablation outputs:
- `outputs/chunks/{chunk_config}/`
- `outputs/runs/chunk_ablation/{run_id}/{chunk_config}/`
- `outputs/reports/chunk_ablation/{run_id}/chunk_ablation_summary.csv`
- `outputs/reports/chunk_ablation/{run_id}/retrieval_by_chunk_config.csv`
- `outputs/reports/chunk_ablation/{run_id}/generation_by_chunk_config.csv`
- `outputs/reports/chunk_ablation/{run_id}/question_type_breakdown.csv`
- `outputs/reports/chunk_ablation/{run_id}/failure_by_chunk_config.csv`
- `outputs/reports/chunk_ablation/{run_id}/latency_by_chunk_config.csv`
- `outputs/reports/chunk_ablation/{run_id}/best_config_summary.csv`

## Metrics
- exact match
- token F1
- numeric accuracy
- refusal accuracy
- answer coverage@3
- gold-answer perplexity
- retrieval latency
- generation latency
- total latency

## Additional Scripts
- `install_verified_questions.py`: copies verified benchmark files into `data/questions/`.
- `validate_verified_questions.py`: remaps verified evidence IDs to current chunk IDs.
- `clean_verified_questions.py`: removes table-of-contents and item-number artifacts from verified questions.
- `build_corpus.py`: builds the main chunk file.
- `build_all_chunk_configs.py`: builds chunk corpora for the optional ablation.
- `inspect_tokenizer.py`: checks tokenizer metadata and checkpoint compatibility.
- `inspect_checkpoint.py`: inspects the local GPT checkpoint.
- `inspect_retrieval.py`: shows retrieved chunks and diagnostics.
- `run_report.py`: rebuilds saved reports.

## Troubleshooting
- Missing model weights: confirm `model/model_weights.pt` exists.
- Missing tokenizer: confirm the Hugging Face tokenizer folder exists at `model/hftokenizer`.
- Missing chunks: run `scripts/build_corpus.py`.
- Missing verified questions: run `scripts/install_verified_questions.py`.
- Smoke test vs research run: any `--limit` run is diagnostic only.
- Weak retrieval coverage: inspect `outputs/runs/{run_id}/retrieval_diagnostics.jsonl` and consider cleaned questions or the chunking ablation.

## Academic Honesty
Some scaffolding, debugging prompts, and documentation drafts were generated with OpenAI Codex/ChatGPT. All code was reviewed, edited, and tested by the author. The submitted experimental design, implementation choices, and results are the author's responsibility.
