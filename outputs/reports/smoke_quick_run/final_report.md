# Final Report: Retrieval-Augmented Generation with Financial Filings

## Abstract
This run evaluates the local GPT/RAG pipeline, but the validity gates mark it as a smoke or diagnostic run. This run verifies pipeline execution. The pipeline executed successfully, but the benchmark is too small or otherwise invalid for research conclusions.

## Hypothesis
Adding retrieved 10-K context should improve factual grounding and reduce gold-answer perplexity, while increasing inference latency.

## Experiment Setup
- Baseline GPT (`baseline_gpt`): The assignment GPT receives only the question.
- RAG-GPT TF-IDF top-3 (`rag_gpt_tfidf_top3`): The same assignment GPT receives retrieved Form 10-K chunks plus the question.
- Oracle Evidence GPT (`oracle_gpt`): The same assignment GPT receives gold evidence, used as an upper bound.
- Random Context GPT (`random_context_gpt`): Noise control using random chunks.
- All systems use the same GPT architecture and the same checkpoint at model/model_weights.pt. The only difference between baseline and RAG is the inference-time retrieval context.
- retrieval method: tfidf
- chunking method: section-aware narrative chunks, table rows, table summaries, and local context chunks

## Dataset Validity
- question file: data/questions/questions_verified.remapped.jsonl
- question source: verified_remapped
- number of questions: 3
- question type counts: {'numeric_fact': 3}
- ticker counts: {'AAPL': 3}
- company counts: {'Apple': 3}
- evidence remapping success rate: 0.6667
- missing evidence rate: 0.3333
- evidence quality: weak
- smoke_test: True
- valid_for_research: False
- invalid reasons: ['num_questions<50', 'numeric_fact<5']

## Tokenizer and Checkpoint
- tokenizer dir: model/hftokenizer
- tokenizer class: GPT2Tokenizer
- tokenizer vocab size: 10000
- checkpoint: model/model_weights.pt
- checkpoint vocab size: 10000
- compatibility: PASS

## Chunking Summary
- total chunks: 5130
- narrative chunks: 3432
- table_row chunks: 1544
- table_summary chunks: 139
- local_context chunks: 15
- retrievable chunks: 3296
- non-retrievable chunks: 1834

## Research Validity Check
- num_questions: 3
- min_research_questions: 50
- is_smoke_test: True
- valid_for_research: False
- invalid_reasons: ['num_questions<50', 'numeric_fact<5']
- limit: 3
- question_source: verified_remapped
- question_type_counts: {'numeric_fact': 3}
- ticker_counts: {'AAPL': 3}
- company_counts: {'Apple': 3}
- missing_evidence_rate: 0.3333333333333333
- oracle_answer_coverage_at_3: 1.0
- rag_answer_coverage_at_3: 0.6666666666666666
- retrieval_quality: weak
- question_file: data/questions/questions_verified.remapped.jsonl
- chunks_file: outputs/chunks/chunks.jsonl
- checkpoint_path: model/model_weights.pt
- tokenizer_dir: model/hftokenizer

This run is a smoke test or diagnostic run. It verifies that the pipeline executes, but it should not be used for final research conclusions.

## Retrieval Quality
- answer_coverage@1: 0.6667
- answer_coverage@3: 0.6667
- source_accuracy@3: 1.0000
- table_row_recall@3: 1.0000
- noise reason counts: {'none': 2, 'wrong_section': 1}
- Retrieval is the main bottleneck when answer coverage is below the research threshold. Improve chunking, metadata filtering, and evidence labels.
- The benchmark is too small to determine whether RAG improves the baseline.

## Generation Quality
| system | exact_match | token_f1 | numeric_accuracy | refusal_accuracy | answer_coverage_at_3 | average_gold_answer_perplexity | perplexity_delta_vs_baseline | average_total_latency |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_gpt | 0.0000 | 0.0000 | 0.0000 |  |  | 34078.3046 | 0.0000 | 0.0335 |
| oracle_gpt | 0.0000 | 0.0000 | 0.0000 |  | 1.0000 | 23184.5516 | -10893.7530 | 0.0631 |
| rag_gpt_tfidf_top3 | 0.0000 | 0.0000 | 0.6667 |  | 0.6667 | 32119.2609 | -1959.0437 | 0.1446 |
| random_context_gpt | 0.0000 | 0.0000 | 0.0000 |  | 0.6667 | 15447.0229 | -18631.2817 | 0.1170 |

## RAG vs Baseline Conclusion
This is a smoke or diagnostic run, so it should not be used for final RAG-vs-baseline conclusions.

## Latency Tradeoff
- baseline total latency: 0.0335s
- RAG total latency: 0.1446s
- latency delta vs baseline: 0.1112s

## Ablations
No ablation tables were generated for this run.

## Discussion
The benchmark is too small to determine whether RAG improves the baseline.

## Training and Convergence
Training logs were not provided. The submitted experiment compares inference-time changes. RAG does not change training time; it adds retrieval overhead at inference.

## Limitations
- The benchmark is not research-valid because the validity gates failed.
- The local GPT checkpoint may have limited capacity for exact financial copying.
- Table extraction from flattened HTML can still miss complex row/column structure.
- The submitted experiment uses provided trained weights at model/model_weights.pt. Because training logs were not provided, training time and convergence are reported as not available. The RAG variation changes inference only, so training time is unchanged relative to the baseline GPT.

## Conclusion
This run is a smoke test or diagnostic run. It verifies that the pipeline executes, but it should not be used for final research conclusions.

The pipeline executed successfully, but the benchmark is too small for conclusions.

The benchmark is too small to determine whether RAG improves the baseline.

## Failure Analysis
See `failure_taxonomy.csv` and `failure_examples.md` for qualitative examples and failure labels.
Failure counts: {'numeric_unit_error': 6, 'correct': 2, 'wrong_section': 4}

## Reproducibility Tables
- `comparison_table.csv`
- `latency_table.csv`
- `retrieval_table.csv`
- `failure_taxonomy.csv`

## Academic Honesty
Some scaffolding, debugging prompts, and documentation drafts were generated with OpenAI Codex/ChatGPT. All code was reviewed, edited, and tested by the author. The submitted experimental design, implementation choices, and results are the author's responsibility.
