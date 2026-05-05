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
| fixed_128 | 0.3611 | 0.5648 | 0.5648 | 1.0000 | 0.0000 |  | 47 | 0 | 0 | 0 | 0.0128 |
| fixed_256 | 0.2593 | 0.4444 | 0.4537 | 1.0000 | 0.0000 |  | 59 | 0 | 0 | 0 | 0.0108 |
| fixed_512 | 0.2037 | 0.2222 | 0.3611 | 1.0000 | 0.0000 |  | 69 | 0 | 0 | 0 | 0.0094 |
| section_180 | 0.1111 | 0.2407 | 0.2407 | 1.0000 | 0.2870 |  | 77 | 5 | 0 | 0 | 0.0049 |
| table_aware_clean | 0.3056 | 0.3981 | 0.3981 | 1.0000 | 0.3611 |  | 65 | 0 | 0 | 0 | 0.0103 |
| table_aware_clean_context | 0.3056 | 0.3981 | 0.3981 | 1.0000 | 0.3611 |  | 65 | 0 | 0 | 0 | 0.0095 |
| table_aware_mixed | 0.3056 | 0.3981 | 0.3981 | 1.0000 | 0.3611 |  | 65 | 0 | 0 | 0 | 0.0094 |
| table_row_only | 0.0000 | 0.0000 | 0.0000 | 0.8519 | 0.0000 |  | 92 | 0 | 0 | 16 | 0.0040 |

## Generation Results
| system | num_questions | exact_match | token_f1 | numeric_accuracy | numeric_generation_accuracy | numeric_extraction_accuracy | refusal_accuracy | hallucination_rate | average_generation_latency | average_retrieval_latency | average_total_latency |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_gpt | 108 | 0.0000 | 0.0095 | 0.0000 | 0.0000 |  |  |  | 0.4042 | 0.0000 | 0.5295 |
| oracle_gpt | 108 | 0.0000 | 0.0155 | 0.0000 | 0.0000 |  |  |  | 0.9905 | 0.0000 | 1.2482 |
| rag_gpt_tfidf_top3 | 108 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |  |  |  | 0.2191 | 0.0128 | 0.2936 |
| random_context_gpt | 108 | 0.0000 | 0.0067 | 0.0000 | 0.0000 |  |  |  | 0.3360 | 0.0000 | 0.3799 |
| baseline_gpt | 108 | 0.0000 | 0.0095 | 0.0000 | 0.0000 |  |  |  | 0.4662 | 0.0000 | 0.5307 |
| oracle_gpt | 108 | 0.0000 | 0.0155 | 0.0000 | 0.0000 |  |  |  | 0.8831 | 0.0000 | 0.9587 |
| rag_gpt_tfidf_top3 | 108 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |  |  |  | 0.2219 | 0.0108 | 0.3601 |
| random_context_gpt | 108 | 0.0000 | 0.0047 | 0.0000 | 0.0000 |  |  |  | 0.3632 | 0.0000 | 0.4442 |
| baseline_gpt | 108 | 0.0000 | 0.0095 | 0.0000 | 0.0000 |  |  |  | 0.4358 | 0.0000 | 0.5095 |
| oracle_gpt | 108 | 0.0000 | 0.0155 | 0.0000 | 0.0000 |  |  |  | 0.8495 | 0.0000 | 0.9203 |
| rag_gpt_tfidf_top3 | 108 | 0.0000 | 0.0002 | 0.0000 | 0.0000 |  |  |  | 0.2169 | 0.0094 | 0.3416 |
| random_context_gpt | 108 | 0.0000 | 0.0043 | 0.0000 | 0.0000 |  |  |  | 0.3710 | 0.0000 | 0.4458 |

## Question-Type Breakdown
| chunk_config | question_type | num_questions | token_f1 | numeric_accuracy | answer_coverage_at_3 | gold_answer_perplexity |
| --- | --- | --- | --- | --- | --- | --- |
| fixed_128 | explanation | 104 | 0.0000 | 0.0000 | 0.5673 | 2289.8142 |
| fixed_128 | text_fact | 4 | 0.0000 |  | 0.5000 | 5931.7789 |
| fixed_256 | explanation | 104 | 0.0000 | 0.0000 | 0.4423 | 2309.4642 |
| fixed_256 | text_fact | 4 | 0.0000 |  | 0.5000 | 5487.0376 |
| fixed_512 | explanation | 104 | 0.0002 | 0.0000 | 0.2308 | 2312.2678 |
| fixed_512 | text_fact | 4 | 0.0000 |  | 0.0000 | 5307.0110 |
| section_180 | explanation | 104 | 0.0000 | 0.0000 | 0.2308 | 2241.9458 |
| section_180 | text_fact | 4 | 0.0000 |  | 0.5000 | 5874.6502 |
| table_aware_clean | explanation | 104 | 0.0000 | 0.0000 | 0.4135 | 2413.9293 |
| table_aware_clean | text_fact | 4 | 0.0000 |  | 0.0000 | 6298.5509 |
| table_aware_clean_context | explanation | 104 | 0.0000 | 0.0000 | 0.4135 | 2413.9293 |
| table_aware_clean_context | text_fact | 4 | 0.0000 |  | 0.0000 | 6298.5509 |

## Latency and Overhead
| chunk_config | system | average_generation_latency | average_retrieval_latency | average_total_latency | latency_delta_vs_baseline | average_prompt_tokens | average_completion_tokens |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_128 | baseline_gpt | 0.4042 | 0.0000 | 0.5295 | 0.0000 | 40.3611 | 8.7500 |
| fixed_128 | oracle_gpt | 0.9905 | 0.0000 | 1.2482 | 0.7187 | 105.2870 | 13.6481 |
| fixed_128 | rag_gpt_tfidf_top3 | 0.2191 | 0.0128 | 0.2936 | -0.2359 | 256.0000 | 5.8611 |
| fixed_128 | random_context_gpt | 0.3360 | 0.0000 | 0.3799 | -0.1496 | 256.0000 | 7.8889 |
| fixed_256 | baseline_gpt | 0.4662 | 0.0000 | 0.5307 | 0.0000 | 40.3611 | 8.7500 |
| fixed_256 | oracle_gpt | 0.8831 | 0.0000 | 0.9587 | 0.4280 | 105.2870 | 13.6481 |
| fixed_256 | rag_gpt_tfidf_top3 | 0.2219 | 0.0108 | 0.3601 | -0.1706 | 256.0000 | 4.1296 |
| fixed_256 | random_context_gpt | 0.3632 | 0.0000 | 0.4442 | -0.0865 | 256.0000 | 6.3611 |
| fixed_512 | baseline_gpt | 0.4358 | 0.0000 | 0.5095 | 0.0000 | 40.3611 | 8.7500 |
| fixed_512 | oracle_gpt | 0.8495 | 0.0000 | 0.9203 | 0.4107 | 105.2870 | 13.6481 |
| fixed_512 | rag_gpt_tfidf_top3 | 0.2169 | 0.0094 | 0.3416 | -0.1680 | 256.0000 | 3.7130 |
| fixed_512 | random_context_gpt | 0.3710 | 0.0000 | 0.4458 | -0.0637 | 256.0000 | 6.3981 |

## Failure Analysis
| chunk_config | failure_type | count |
| --- | --- | --- |
| fixed_128 | none | 61 |
| fixed_128 | wrong_section | 47 |
| fixed_256 | none | 49 |
| fixed_256 | wrong_section | 59 |
| fixed_512 | none | 39 |
| fixed_512 | wrong_section | 69 |
| section_180 | answer_split_across_chunks | 5 |
| section_180 | none | 26 |
| section_180 | wrong_section | 77 |
| table_aware_clean | none | 43 |
| table_aware_clean | wrong_section | 65 |
| table_aware_clean_context | none | 43 |

## Best Configuration
Best chunking configuration: fixed_128

Decision rule: highest answer_coverage@3, then fewer wrong_section errors, fewer answer_split_across_chunks errors, lower gold-answer perplexity, and reasonable latency.

## Discussion
No chunk configuration reached answer_coverage@3 >= 0.70, so retrieval remains the bottleneck.

## Limitations
- The local GPT is not instruction-tuned, so exact match can be harsh.
- TF-IDF is transparent and reproducible but limited compared with learned retrievers.
- Question quality and table extraction quality can still dominate results.
- Training logs are unavailable; the ablation compares inference-time retrieval units.

## Conclusion
fixed_128 is selected by the ablation decision rule. No chunk configuration reached answer_coverage@3 >= 0.70, so retrieval remains the bottleneck.

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
