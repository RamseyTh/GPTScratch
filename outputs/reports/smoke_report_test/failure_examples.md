# Failure Examples

## rag_succeeds_baseline_fails

- question_id: `AAPL_2025_001`
- question: What were Apple's Services net sales in fiscal 2025?
- gold answer: $109.158 billion
- baseline prediction: , and the first @-@ century , and the first @-@ century , and the first @-@ century .
- candidate system: rag_gpt_tfidf_top3
- candidate prediction: " .
- retrieved preview: [1: AAPL 2025 Item 7 table_row] Apple (AAPL) fiscal 2025 Products and Services Performance table. Services net sales were $109.158 billion ($109,158 million) in 2025, $96.169 billion ($96,169 million) in 2024, and $85.200 billion ($85,200 million) in 2023 (dollars in millions).  [2: AAPL 2025 Item 7 table_row] Apple (AAPL) fiscal 2025 Products and Services Performance table. Products net sales were $112.887 billion ($112,887 million) in 2025, $109.633 billion ($109,633 million) in 2024, and $...
- failure type: correct

## baseline_succeeds_rag_fails

No example available in this run.

## retrieval_miss

- question_id: `AAPL_2025_002`
- question: Did Apple report a Cloud Infrastructure reportable segment in fiscal 2025?
- gold answer: not enough information
- baseline prediction: .
- candidate system: rag_gpt_tfidf_top3
- candidate prediction: " .
- retrieved preview: [1: AAPL 2025 Item 8 table_row] Apple (AAPL) fiscal 2025 (Topic 280): Improvements to Reportable Segment Disclosures table. “ASU were $2.023 billion ($2,023 million) in 2025, and $-7 million in 2023 (dollars in millions).  [2: AAPL 2025 Item 8 table_row] Apple (AAPL) fiscal 2025 (Topic 280): Improvements to Reportable Segment Disclosures table. ASU were $2.023 billion ($2,023 million) in 2025, and $-7 million in 2023 (dollars in millions).  [3: AAPL 2025 Item 8 table_row] Apple (AAPL) fiscal...
- failure type: answer_split_across_chunks

## retrieval_finds_evidence_generator_fails

- question_id: `MSFT_2025_001`
- question: What company is discussed in the Microsoft smoke filing?
- gold answer: Microsoft
- baseline prediction: .
- candidate system: rag_gpt_tfidf_top3
- candidate prediction: " .
- retrieved preview: [1: MSFT 2025 Item 8 table_row] Microsoft (MSFT) fiscal 2025 Total table. As discussed in Note were $1 million in 2025, and $2.025 billion ($2,025 million) in 2024 (dollars in millions).  [2: MSFT 2025 Item 7 table_row] Microsoft (MSFT) fiscal 2025 Productivity and Business Processes table. Microsoft were $365 million in 2025, and $10.8 million in 2024 (dollars in millions).  [3: MSFT 2025 Item 8 table_row] Microsoft (MSFT) fiscal 2025 Total table. Our Microsoft Cloud revenue, which includes...
- failure type: generator_hallucination

## random_context_fails

- question_id: `AAPL_2025_001`
- question: What were Apple's Services net sales in fiscal 2025?
- gold answer: $109.158 billion
- baseline prediction: , and the first @-@ century , and the first @-@ century , and the first @-@ century .
- candidate system: random_context_gpt
- candidate prediction: .
- retrieved preview: [1: GOOG 2025 Item 8 table_row] Google (GOOG) fiscal 2025 2025, we had no commercial paper outstanding. table. For additional information, see Note were 6 in 2025, and 8 in 2026.  [2: NVDA 2025 Item 16 narrative] 13. Code Section 409A; Tax Qualification. (a) Purchase Rights granted under the 423 Component are intended to be exempt from the application of Section 409A of the Code under Treasury Regulation Section 1.409A-1(b)(5)(ii). Purchase Rights granted under the Non-423 Component to U.S. t...
- failure type: numeric_unit_error

## oracle_succeeds

No example available in this run.

## oracle_fails

- question_id: `AAPL_2025_001`
- question: What were Apple's Services net sales in fiscal 2025?
- gold answer: $109.158 billion
- baseline prediction: , and the first @-@ century , and the first @-@ century , and the first @-@ century .
- candidate system: oracle_gpt
- candidate prediction: , and the first @-@ century , and the first time of the first time of the first time of the first time of the first time of the first time of the first time of the first time of the first time of the first time .
- retrieved preview: Services net sales of $109.158 billion
- failure type: numeric_unit_error
