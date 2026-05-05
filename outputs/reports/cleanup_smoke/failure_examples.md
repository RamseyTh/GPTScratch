# Failure Examples

## rag_succeeds_baseline_fails

No example available in this run.

## baseline_succeeds_rag_fails

No example available in this run.

## retrieval_miss

- question_id: `TSLA_2025_TEXT_FACT_9f822d5c`
- question: What reportable segments or business units did Tesla describe in fiscal 2025?
- gold answer: Segment Information We operate as two reportable segments: (i) automotive and (ii) energy generation and storage.
- baseline prediction: .
- candidate system: rag_gpt_tfidf_top3
- candidate prediction: "
- retrieved preview: [1: TSLA 2025 Item 7 table_summary] Tesla fiscal 2025 Overview and 2021 Highlights table summary. Important rows include revenue, revenue, Total revenues.  [2: TSLA 2025 Item 8 table_row] Tesla (TSLA) fiscal 2025 operating and reportable segments: (i) automotive and (ii) energy generation and storage. table. There has continued to be widespread impact from the coronavirus disease “COVID were $-19 million in 2024, and $2.021 billion ($2,021 million) in 2024 (dollars in millions).  [3: TSLA 202...
- failure type: wrong_section

## retrieval_finds_evidence_generator_fails

- question_id: `AAPL_2025_TEXT_FACT_4011dcad`
- question: What reportable segments or business units did Apple describe in fiscal 2025?
- gold answer: The Company’s reportable segments consist of the Americas, Europe, Greater China, Japan and Rest of Asia Pacific.
- baseline prediction: .
- candidate system: rag_gpt_tfidf_top3
- candidate prediction: "
- retrieved preview: [1: AAPL 2025 Item 7 table_row] Apple (AAPL) fiscal 2025 Mac. table. Apple Inc. | were $2.025 billion ($2,025 million) in 2025, and $10 million in 2024 (dollars in millions).  [2: AAPL 2025 Item 1 narrative] Digital Content The Company operates various platforms, including the App Store ® , that allow customers to discover and download applications and digital content, such as books, music, video, games and podcasts. The Company also offers digital content through subscription-based services,...
- failure type: generator_hallucination

## random_context_fails

- question_id: `AAPL_2025_TEXT_FACT_4011dcad`
- question: What reportable segments or business units did Apple describe in fiscal 2025?
- gold answer: The Company’s reportable segments consist of the Americas, Europe, Greater China, Japan and Rest of Asia Pacific.
- baseline prediction: .
- candidate system: random_context_gpt
- candidate prediction: .
- retrieved preview: [1: GOOG 2025 Item 8 table_row] Google (GOOG) fiscal 2025 2025, we had no commercial paper outstanding. table. For additional information, see Note were 6 in 2025, and 8 in 2026.  [2: NVDA 2025 Item 16 narrative] 13. Code Section 409A; Tax Qualification. (a) Purchase Rights granted under the 423 Component are intended to be exempt from the application of Section 409A of the Code under Treasury Regulation Section 1.409A-1(b)(5)(ii). Purchase Rights granted under the Non-423 Component to U.S. t...
- failure type: retrieval_noise

## oracle_succeeds

No example available in this run.

## oracle_fails

- question_id: `AAPL_2025_TEXT_FACT_4011dcad`
- question: What reportable segments or business units did Apple describe in fiscal 2025?
- gold answer: The Company’s reportable segments consist of the Americas, Europe, Greater China, Japan and Rest of Asia Pacific.
- baseline prediction: .
- candidate system: oracle_gpt
- candidate prediction: .
- retrieved preview: The Company’s reportable segments consist of the Americas, Europe, Greater China, Japan and Rest of Asia Pacific.
- failure type: retrieval_noise
