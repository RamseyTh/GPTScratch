# Retrieval-Augmented Generation with Financial Filings

## Abstract
This run evaluates the local GPT/RAG pipeline, but the validity gates mark it as a smoke or diagnostic run. This run verifies pipeline execution. The pipeline executed successfully, but the benchmark is too small or otherwise invalid for research conclusions.

## 1. Hypothesis
Adding retrieved 10-K context should improve factual grounding and reduce gold-answer perplexity, while increasing inference latency.

## 2. Experimental Design
- baseline GPT: question-only prompt
- RAG GPT: top-k TF-IDF retrieval plus context-augmented prompt
- oracle GPT: gold evidence context
- random context GPT: noise control
- all systems use the same local checkpoint and Hugging Face tokenizer
- retrieval method: tfidf
- chunking method: section-aware narrative chunks, table rows, table summaries, and local context chunks

## 3. Data Report
- run_id: smoke_report_test
- number of questions: 3
- number of chunks: 4155
- chunk types: {'table_row': 1544, 'table_summary': 139, 'narrative': 2457, 'local_context': 15}
- companies represented in retrieved chunks: Apple, Google, Meta, Microsoft, NVIDIA, Tesla
- research validity status: False

## Research Validity Check
- num_questions: 3
- min_research_questions: 50
- is_smoke_test: True
- valid_for_research: False
- invalid_reasons: ['num_questions<50', 'rag_answer_coverage_at_3<0.70']
- oracle_answer_coverage_at_3: 1.0
- rag_answer_coverage_at_3: 0.6666666666666666
- question_file: data/questions/questions.jsonl
- chunks_file: outputs/chunks/chunks.jsonl
- checkpoint_path: model/model_weights.pt
- tokenizer_dir: model/hftokenizer

This run is a smoke test or diagnostic run. It verifies that the pipeline executes, but it should not be used for final research conclusions.

## 4. Retrieval Analysis
- answer_coverage@1: 0.6667
- answer_coverage@3: 0.6667
- source_accuracy@3: 1.0000
- table_row_recall@3: 1.0000
- noise reason counts: {'none': 2, 'answer_split_across_chunks': 1}
- Retrieval is the main bottleneck when answer coverage is below the research threshold. Improve chunking, metadata filtering, and evidence labels.
- The benchmark is too small to determine whether RAG improves the baseline.

## 5. Generation Analysis
| system | exact_match | token_f1 | numeric_accuracy | refusal_accuracy | answer_coverage_at_3 | average_gold_answer_perplexity | perplexity_delta_vs_baseline | average_total_latency |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_gpt | 0.0000 | 0.0000 | 0.0000 | 0.0000 |  | 3594.9824 | 0.0000 | 0.3639 |
| oracle_gpt | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 3371.6030 | -223.3795 | 1.3882 |
| rag_gpt_tfidf_top3 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.6667 | 4101.1941 | 506.2116 | 0.6030 |
| random_context_gpt | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.3333 | 3912.1689 | 317.1865 | 0.4161 |

## 6. Latency and Computational Overhead
- baseline total latency: 0.3639s
- RAG total latency: 0.6030s
- latency delta vs baseline: 0.2391s

## 7. Ablations
No ablation tables were generated for this run.

## 8. Discussion
The benchmark is too small to determine whether RAG improves the baseline.

## 9. Limitations
- The benchmark is not research-valid because the validity gates failed.
- The local GPT checkpoint may have limited capacity for exact financial copying.
- Table extraction from flattened HTML can still miss complex row/column structure.
- The submitted experiment uses provided trained weights at model/model_weights.pt. Because training logs were not provided, training time and convergence are reported as not available. The RAG variation changes inference only, so training time is unchanged relative to the baseline GPT.

## 10. Conclusion
This run is a smoke test or diagnostic run. It verifies that the pipeline executes, but it should not be used for final research conclusions.

The pipeline executed successfully, but the benchmark is too small for conclusions.

The benchmark is too small to determine whether RAG improves the baseline.

## Failure Analysis
See `failure_taxonomy.csv` and `failure_examples.md` for qualitative examples and failure labels.
Failure counts: {'numeric_unit_error': 3, 'correct': 1, 'answer_split_across_chunks': 4, 'retrieval_noise': 3, 'generator_hallucination': 1}

## Reproducibility Tables
- `comparison_table.csv`
- `latency_table.csv`
- `retrieval_table.csv`
- `failure_taxonomy.csv`

## Academic Honesty
Some scaffolding, debugging prompts, and documentation drafts were generated with OpenAI Codex/ChatGPT. All code was reviewed, edited, and tested by the author. The submitted experimental design, implementation choices, and results are the author's responsibility.
