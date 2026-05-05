# Final Report: Retrieval-Augmented Generation with Financial Filings

## Abstract
This report compares Baseline GPT against RAG-GPT TF-IDF top-3 using the local assignment GPT checkpoint `model/model_weights.pt` on 271 verified questions. RAG answer coverage@3 is 0.4207.

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
- number of questions: 271
- question type counts: {'numeric_fact': 67, 'text_fact': 14, 'risk_factor': 56, 'explanation': 120, 'negative': 14}
- ticker counts: {'AAPL': 43, 'AMZN': 40, 'GOOG': 37, 'META': 37, 'MSFT': 38, 'NVDA': 38, 'TSLA': 38}
- company counts: {'Apple': 43, 'Amazon': 40, 'Google': 37, 'Unknown': 37, 'Microsoft': 38, 'Nvidia': 38, 'Tesla': 38}
- evidence remapping success rate: 0.9922
- missing evidence rate: 0.0078
- evidence quality: acceptable
- smoke_test: False
- valid_for_research: True
- invalid reasons: []

## Tokenizer and Checkpoint
- tokenizer dir: model/hftokenizer
- tokenizer class: GPT2Tokenizer
- tokenizer vocab size: 10000
- checkpoint: model/model_weights.pt
- checkpoint vocab size: 10000
- compatibility: PASS

## Chunking Summary
- total chunks: 4155
- narrative chunks: 2457
- table_row chunks: 1544
- table_summary chunks: 139
- local_context chunks: 15
- retrievable chunks: 2358
- non-retrievable chunks: 1797

## Research Validity Check
- num_questions: 271
- min_research_questions: 50
- is_smoke_test: False
- valid_for_research: True
- invalid_reasons: []
- limit: None
- question_source: verified_remapped
- question_type_counts: {'numeric_fact': 67, 'text_fact': 14, 'risk_factor': 56, 'explanation': 120, 'negative': 14}
- ticker_counts: {'AAPL': 43, 'AMZN': 40, 'GOOG': 37, 'META': 37, 'MSFT': 38, 'NVDA': 38, 'TSLA': 38}
- company_counts: {'Apple': 43, 'Amazon': 40, 'Google': 37, 'Unknown': 37, 'Microsoft': 38, 'Nvidia': 38, 'Tesla': 38}
- missing_evidence_rate: 0.007782101167315175
- oracle_answer_coverage_at_3: 1.0
- rag_answer_coverage_at_3: 0.42066420664206644
- retrieval_quality: weak
- question_file: data/questions/questions_verified.remapped.jsonl
- chunks_file: outputs/chunks/chunks.jsonl
- checkpoint_path: model/model_weights.pt
- tokenizer_dir: model/hftokenizer

## Retrieval Quality
- answer_coverage@1: 0.2583
- answer_coverage@3: 0.4207
- source_accuracy@3: 1.0000
- table_row_recall@3: 0.8507
- noise reason counts: {'none': 112, 'wrong_section': 128, 'answer_split_across_chunks': 28, 'table_label_missing': 3}
- Retrieval is the main bottleneck when answer coverage is below the research threshold. Improve chunking, metadata filtering, and evidence labels.

## Generation Quality
| system | exact_match | token_f1 | numeric_accuracy | refusal_accuracy | answer_coverage_at_3 | average_gold_answer_perplexity | perplexity_delta_vs_baseline | average_total_latency |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_gpt | 0.0000 | 0.0123 | 0.0000 | 0.0000 |  | 17527.7044 | 0.0000 | 0.7416 |
| oracle_gpt | 0.0000 | 0.0088 | 0.0000 | 0.0000 | 1.0000 | 15114.1272 | -2413.5772 | 0.8738 |
| rag_gpt_tfidf_top3 | 0.0000 | 0.0003 | 0.5052 | 0.0000 | 0.4207 | 46307.2436 | 28779.5392 | 0.2301 |
| random_context_gpt | 0.0000 | 0.0028 | 0.0000 | 0.0000 | 0.1218 | 15148.2500 | -2379.4544 | 0.2971 |

## RAG vs Baseline Conclusion
- token F1 delta: -0.0120
- gold-answer perplexity delta: 28779.5392
- numeric accuracy delta: 0.5052
- latency delta: -0.5116s
- RAG answer coverage@3: 0.4207

## Latency Tradeoff
- baseline total latency: 0.7416s
- RAG total latency: 0.2301s
- latency delta vs baseline: -0.5116s

## Ablations
No ablation tables were generated for this run.

## Discussion
Oracle context appears useful, but retrieved context is weaker; retrieval is the bottleneck.

## Training and Convergence
Training logs were not provided. The submitted experiment compares inference-time changes. RAG does not change training time; it adds retrieval overhead at inference.

## Limitations
- The local GPT checkpoint may have limited capacity for exact financial copying.
- Table extraction from flattened HTML can still miss complex row/column structure.
- The submitted experiment uses provided trained weights at model/model_weights.pt. Because training logs were not provided, training time and convergence are reported as not available. The RAG variation changes inference only, so training time is unchanged relative to the baseline GPT.

## Conclusion
RAG did not improve the primary generation metrics in this research-valid run. Token F1 delta is -0.0120 and gold-answer perplexity delta is 28779.5392.

## Failure Analysis
See `failure_taxonomy.csv` and `failure_examples.md` for qualitative examples and failure labels.
Failure counts: {'numeric_unit_error': 159, 'correct': 49, 'wrong_section': 511, 'answer_split_across_chunks': 111, 'retrieval_miss': 3, 'retrieval_noise': 186, 'generator_hallucination': 65}

## Reproducibility Tables
- `comparison_table.csv`
- `latency_table.csv`
- `retrieval_table.csv`
- `failure_taxonomy.csv`

## Academic Honesty
Some scaffolding, debugging prompts, and documentation drafts were generated with OpenAI Codex/ChatGPT. All code was reviewed, edited, and tested by the author. The submitted experimental design, implementation choices, and results are the author's responsibility.
