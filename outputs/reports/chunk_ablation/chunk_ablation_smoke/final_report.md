# Chunking Ablation Report: RAG over Financial Filings

## Abstract
This experiment compares multiple ways to split Form 10-K filings into retrieval units while keeping the same local GPT checkpoint, tokenizer, questions, TF-IDF retriever, top-k, prompts, and evaluation code. Best chunking configuration: fixed_128.

## Hypothesis
Structure-aware and table-aware chunking should improve retrieval answer coverage and reduce retrieval noise compared with fixed-size chunking.

## Experimental Design
- Same local GPT checkpoint: `model/model_weights.pt`.
- Same tokenizer: `model/hftokenizer`.
- Same systems: baseline_gpt, rag_gpt_tfidf_top3, oracle_gpt, random_context_gpt.
- Same TF-IDF retriever, top-k, context budget, prompts, and metrics.
- Only the chunking configuration changes.

## Chunking Configurations
- `fixed_128`: fixed 128-token control with 32-token overlap.
- `fixed_256`: fixed 256-token control with 64-token overlap.
- `fixed_512`: fixed 512-token control with 128-token overlap.
- `section_180`: section-aware narrative chunks without table rows.
- `table_row_only`: table_row and table_summary chunks only.
- `table_aware_mixed`: default mixed narrative, table row, table summary, and local context chunks.
- `table_aware_clean`: mixed chunks with aggressive non-retrievable filtering.
- `table_aware_clean_context`: clean mixed chunks plus MD&A/table local context.

## Dataset
The ablation uses the question file supplied to `main.py`; if it is `questions_verified.remapped.jsonl`, results should be read as diagnostic if question-quality artifacts remain.

## Retrieval Results
| chunk_config | answer_coverage_at_1 | answer_coverage_at_3 | answer_coverage_at_5 | source_accuracy_at_3 | section_accuracy_at_3 | table_row_recall_at_3 | wrong_section_count | answer_split_across_chunks_count | boilerplate_count | retrieval_miss_count | average_retrieval_latency |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_128 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |  | 0 | 0 | 0 | 0 | 0.0206 |
| table_aware_clean | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 |  | 1 | 0 | 0 | 0 | 0.0062 |

## Generation Results
| system | num_questions | exact_match | token_f1 | numeric_accuracy | numeric_generation_accuracy | numeric_extraction_accuracy | refusal_accuracy | hallucination_rate | average_generation_latency | average_retrieval_latency | average_total_latency |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_gpt | 1 | 0.0000 | 0.0000 |  |  |  |  |  | 0.1113 | 0.0000 | 0.2004 |
| oracle_gpt | 1 | 0.0000 | 0.0000 |  |  |  |  |  | 0.0335 | 0.0000 | 0.0652 |
| rag_gpt_tfidf_top3 | 1 | 0.0000 | 0.0000 |  |  |  |  |  | 0.2625 | 0.0206 | 0.6905 |
| random_context_gpt | 1 | 0.0000 | 0.0000 |  |  |  |  |  | 0.0960 | 0.0000 | 0.2074 |
| baseline_gpt | 1 | 0.0000 | 0.0000 |  |  |  |  |  | 0.0106 | 0.0000 | 0.0278 |
| oracle_gpt | 1 | 0.0000 | 0.0000 |  |  |  |  |  | 0.0097 | 0.0000 | 0.0286 |
| rag_gpt_tfidf_top3 | 1 | 0.0000 | 0.0000 |  |  |  |  |  | 0.0441 | 0.0062 | 0.1091 |
| random_context_gpt | 1 | 0.0000 | 0.0000 |  |  |  |  |  | 0.0486 | 0.0000 | 0.1169 |

## Question-Type Breakdown
| chunk_config | question_type | num_questions | token_f1 | numeric_accuracy | answer_coverage_at_3 | gold_answer_perplexity |
| --- | --- | --- | --- | --- | --- | --- |
| fixed_128 | text_fact | 1 | 0.0000 |  | 1.0000 | 4314.8413 |
| table_aware_clean | text_fact | 1 | 0.0000 |  | 0.0000 | 4706.9288 |

## Latency and Overhead
| chunk_config | system | average_generation_latency | average_retrieval_latency | average_total_latency | latency_delta_vs_baseline | average_prompt_tokens | average_completion_tokens |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_128 | baseline_gpt | 0.1113 | 0.0000 | 0.2004 | 0.0000 | 29.0000 | 1.0000 |
| fixed_128 | oracle_gpt | 0.0335 | 0.0000 | 0.0652 | -0.1352 | 66.0000 | 1.0000 |
| fixed_128 | rag_gpt_tfidf_top3 | 0.2625 | 0.0206 | 0.6905 | 0.4901 | 256.0000 | 1.0000 |
| fixed_128 | random_context_gpt | 0.0960 | 0.0000 | 0.2074 | 0.0070 | 256.0000 | 1.0000 |
| table_aware_clean | baseline_gpt | 0.0106 | 0.0000 | 0.0278 | 0.0000 | 29.0000 | 1.0000 |
| table_aware_clean | oracle_gpt | 0.0097 | 0.0000 | 0.0286 | 0.0009 | 66.0000 | 1.0000 |
| table_aware_clean | rag_gpt_tfidf_top3 | 0.0441 | 0.0062 | 0.1091 | 0.0814 | 256.0000 | 1.0000 |
| table_aware_clean | random_context_gpt | 0.0486 | 0.0000 | 0.1169 | 0.0892 | 256.0000 | 1.0000 |

## Failure Analysis
| chunk_config | failure_type | count |
| --- | --- | --- |
| fixed_128 | none | 1 |
| table_aware_clean | wrong_section | 1 |

## Best Configuration
Best chunking configuration: fixed_128

Decision rule: highest answer_coverage@3, then fewer wrong_section errors, fewer answer_split_across_chunks errors, lower gold-answer perplexity, and reasonable latency.

## Discussion
The best configuration should be interpreted by comparing retrieval coverage, generation metrics, and latency together.

## Limitations
- The local GPT is not instruction-tuned, so exact match can be harsh.
- TF-IDF is transparent and reproducible but limited compared with learned retrievers.
- Question quality and table extraction quality can still dominate results.
- Training logs are unavailable; the ablation compares inference-time retrieval units.

## Conclusion
fixed_128 is selected by the ablation decision rule. Use the CSV tables to compare whether structure-aware chunking improves answer coverage and reduces retrieval noise.

## Reproducibility Tables
- `chunk_ablation_summary.csv`
- `retrieval_by_chunk_config.csv`
- `generation_by_chunk_config.csv`
- `question_type_breakdown.csv`
- `failure_by_chunk_config.csv`
- `latency_by_chunk_config.csv`
- `best_config_summary.csv`

## Academic Honesty
Some scaffolding, debugging prompts, and documentation drafts were generated with OpenAI Codex/ChatGPT. All code was reviewed, edited, and tested by the author. The submitted experimental design, implementation choices, and results are the author's responsibility.
