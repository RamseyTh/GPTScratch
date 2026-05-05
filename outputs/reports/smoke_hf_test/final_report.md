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
- All systems use the same GPT architecture and the same checkpoint at /private/var/folders/vn/_bjbbh9j605bzf3s4_p2bzf00000gn/T/pytest-of-sichaoliu/pytest-153/test_smoke_pipeline_runs_with_0/model_weights.pt. The only difference between baseline and RAG is the inference-time retrieval context.
- retrieval method: tfidf
- chunking method: section-aware narrative chunks, table rows, table summaries, and local context chunks

## Dataset Validity
- question file: /private/var/folders/vn/_bjbbh9j605bzf3s4_p2bzf00000gn/T/pytest-of-sichaoliu/pytest-153/test_smoke_pipeline_runs_with_0/questions.jsonl
- question source: default
- number of questions: 1
- question type counts: {'numeric_fact': 1}
- ticker counts: {'AAPL': 1}
- company counts: {'Apple': 1}
- evidence remapping success rate: 0.0000
- missing evidence rate: 1.0000
- evidence quality: weak
- smoke_test: True
- valid_for_research: False
- invalid reasons: ['num_questions<50', 'question_source_not_verified', 'numeric_fact<5']

## Tokenizer and Checkpoint
- tokenizer dir: /private/var/folders/vn/_bjbbh9j605bzf3s4_p2bzf00000gn/T/pytest-of-sichaoliu/pytest-153/test_smoke_pipeline_runs_with_0/hftokenizer
- tokenizer class: PreTrainedTokenizerFast
- tokenizer vocab size: 30
- checkpoint: /private/var/folders/vn/_bjbbh9j605bzf3s4_p2bzf00000gn/T/pytest-of-sichaoliu/pytest-153/test_smoke_pipeline_runs_with_0/model_weights.pt
- checkpoint vocab size: 30
- compatibility: PASS

## Chunking Summary
- total chunks: 1
- narrative chunks: 0
- table_row chunks: 0
- table_summary chunks: 0
- local_context chunks: 0
- retrievable chunks: 1
- non-retrievable chunks: 0

## Research Validity Check
- num_questions: 1
- min_research_questions: 50
- is_smoke_test: True
- valid_for_research: False
- invalid_reasons: ['num_questions<50', 'question_source_not_verified', 'numeric_fact<5']
- limit: 1
- question_source: default
- question_type_counts: {'numeric_fact': 1}
- ticker_counts: {'AAPL': 1}
- company_counts: {'Apple': 1}
- missing_evidence_rate: 1.0
- oracle_answer_coverage_at_3: 1.0
- rag_answer_coverage_at_3: 1.0
- retrieval_quality: acceptable
- question_file: /private/var/folders/vn/_bjbbh9j605bzf3s4_p2bzf00000gn/T/pytest-of-sichaoliu/pytest-153/test_smoke_pipeline_runs_with_0/questions.jsonl
- chunks_file: /private/var/folders/vn/_bjbbh9j605bzf3s4_p2bzf00000gn/T/pytest-of-sichaoliu/pytest-153/test_smoke_pipeline_runs_with_0/chunks.jsonl
- checkpoint_path: /private/var/folders/vn/_bjbbh9j605bzf3s4_p2bzf00000gn/T/pytest-of-sichaoliu/pytest-153/test_smoke_pipeline_runs_with_0/model_weights.pt
- tokenizer_dir: /private/var/folders/vn/_bjbbh9j605bzf3s4_p2bzf00000gn/T/pytest-of-sichaoliu/pytest-153/test_smoke_pipeline_runs_with_0/hftokenizer

This run is a smoke test or diagnostic run. It verifies that the pipeline executes, but it should not be used for final research conclusions.

## Retrieval Quality
- answer_coverage@1: 1.0000
- answer_coverage@3: 1.0000
- source_accuracy@3: 1.0000
- table_row_recall@3: 0.0000
- noise reason counts: {'none': 1}
- The benchmark is too small to determine whether RAG improves the baseline.

## Generation Quality
| system | exact_match | token_f1 | numeric_accuracy | refusal_accuracy | answer_coverage_at_3 | average_gold_answer_perplexity | perplexity_delta_vs_baseline | average_total_latency |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_gpt | 0.0000 | 0.0000 | 0.0000 |  |  | 30.0017 | 0.0000 | 0.1271 |
| oracle_gpt | 0.0000 | 0.0000 | 0.0000 |  | 1.0000 | 29.9887 | -0.0130 | 0.0884 |
| rag_gpt_tfidf_top3 | 0.0000 | 0.0000 | 1.0000 |  | 1.0000 | 29.9886 | -0.0131 | 0.1177 |
| random_context_gpt | 0.0000 | 0.0000 | 0.0000 |  | 1.0000 | 29.9887 | -0.0130 | 0.0948 |

## RAG vs Baseline Conclusion
This is a smoke or diagnostic run, so it should not be used for final RAG-vs-baseline conclusions.

## Latency Tradeoff
- baseline total latency: 0.1271s
- RAG total latency: 0.1177s
- latency delta vs baseline: -0.0094s

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
Failure counts: {'numeric_unit_error': 3, 'correct': 1}

## Reproducibility Tables
- `comparison_table.csv`
- `latency_table.csv`
- `retrieval_table.csv`
- `failure_taxonomy.csv`

## Academic Honesty
Some scaffolding, debugging prompts, and documentation drafts were generated with OpenAI Codex/ChatGPT. All code was reviewed, edited, and tested by the author. The submitted experimental design, implementation choices, and results are the author's responsibility.
