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

## Main Quick Run
```bash
python main.py \
  --quick-run \
  --run-id final_verified_run \
  --report
```

This runs:
- `baseline_gpt`
- `rag_gpt_tfidf_top3`
- `oracle_gpt`
- `random_context_gpt`

The default question file is selected from `data/questions/`, preferring verified remapped questions when available. Any run using `--limit` is a smoke test and should not be used for final conclusions.

## Chunking Ablation
The chunking ablation compares different ways to split 10-K filings into retrieval units. It tests whether better chunking reduces retrieval noise and improves RAG performance while the model, tokenizer, questions, retriever, top-k, prompts, and metrics stay fixed.

Build all chunk configs:

```bash
python scripts/build_all_chunk_configs.py \
  --data-dir data/raw \
  --configs fixed_128,fixed_256,fixed_512,section_180,table_row_only,table_aware_mixed,table_aware_clean,table_aware_clean_context
```

Run the research ablation:

```bash
python main.py \
  --quick-run \
  --experiment chunk_ablation \
  --questions data/questions/questions_research.validated.jsonl \
  --chunk-configs fixed_128,fixed_256,fixed_512,section_180,table_row_only,table_aware_mixed,table_aware_clean,table_aware_clean_context \
  --run-id chunk_ablation_research \
  --report
```

If `questions_research.validated.jsonl` is unavailable, run a diagnostic ablation:

```bash
python main.py \
  --quick-run \
  --experiment chunk_ablation \
  --questions data/questions/questions_verified.remapped.jsonl \
  --chunk-configs fixed_128,fixed_256,fixed_512,section_180,table_row_only,table_aware_mixed,table_aware_clean,table_aware_clean_context \
  --run-id chunk_ablation_diagnostic \
  --report
```

The verified/remapped set may contain table-of-contents or item-number artifacts, so results from that set should be interpreted as diagnostic.

## Chunk Configurations
- `fixed_128`: fixed 128-token chunks with 32-token overlap.
- `fixed_256`: fixed 256-token chunks with 64-token overlap.
- `fixed_512`: fixed 512-token chunks with 128-token overlap.
- `section_180`: section-aware narrative chunks without table rows.
- `table_row_only`: table rows and table summaries only.
- `table_aware_mixed`: narrative, table_row, table_summary, and local_context chunks.
- `table_aware_clean`: mixed chunks with stronger filtering for table of contents, cover pages, signatures, and boilerplate.
- `table_aware_clean_context`: clean mixed chunks plus local_context around MD&A and table-related evidence.

## Outputs
Main run:
- `outputs/runs/{run_id}/`
- `outputs/reports/{run_id}/`

Chunk ablation:
- `outputs/chunks/{chunk_config}/`
- `outputs/runs/chunk_ablation/{run_id}/{chunk_config}/`
- `outputs/reports/chunk_ablation/{run_id}/`

Ablation report tables:
- `chunk_ablation_summary.csv`
- `retrieval_by_chunk_config.csv`
- `generation_by_chunk_config.csv`
- `question_type_breakdown.csv`
- `failure_by_chunk_config.csv`
- `latency_by_chunk_config.csv`
- `best_config_summary.csv`
- `final_report.md`

## Metrics
- answer_coverage@1
- answer_coverage@3
- answer_coverage@5
- source_accuracy@3
- section_accuracy@3
- table_row_recall@3
- wrong_section_count
- answer_split_across_chunks_count
- token F1
- numeric accuracy
- gold-answer perplexity
- retrieval latency
- generation latency
- total latency

## Interpreting Chunking Results
- If `fixed_128` has high answer-split errors, chunks are too small.
- If `fixed_512` has high wrong-section errors, chunks are too broad.
- If `table_row_only` improves numeric questions but hurts risk or explanation questions, chunking should be question-type-aware.
- If `table_aware_clean` improves coverage, filtering table-of-contents and boilerplate helped.
- If `table_aware_clean_context` improves explanation questions, local context is useful.

## Additional Scripts
- `scripts/build_all_chunk_configs.py`: builds all chunking ablation corpora.
- `scripts/build_corpus.py`: builds one chunk corpus.
- `scripts/inspect_retrieval.py`: inspects retrieval results for questions.
- `scripts/run_report.py`: rebuilds standard or chunk-ablation reports.
- `scripts/clean_verified_questions.py`: removes table-of-contents and item-number artifacts from verified questions.
- `scripts/validate_questions.py`: validates evidence-grounded question files.
- `scripts/install_verified_questions.py`: installs verified question artifacts.
- `scripts/validate_verified_questions.py`: remaps verified evidence IDs to current chunk IDs.

## Academic Honesty
Some scaffolding, debugging prompts, and documentation drafts were generated with OpenAI Codex/ChatGPT. All code was reviewed, edited, and tested by the author. The submitted experimental design, implementation choices, and results are the author's responsibility.
