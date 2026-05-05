# Failure Examples

## rag_succeeds_baseline_fails

- question_id: `AAPL_2025_NUMERIC_FACT_87f37103`
- question: What value did Apple report for legal proceedings item mine safety disclosures part ii in fiscal 2025?
- gold answer: 4
- baseline prediction: , and
- candidate system: rag_gpt_tfidf_top3
- candidate prediction: " .
- retrieved preview: [1: AAPL 2025 Item 7 table_row] Apple (AAPL) fiscal 2025 taxes. table. Financial Statements in Part II, Item were $8 million in 2025, and $10 million in 2024 (dollars in millions).  [2: AAPL 2025 Item 7 table_row] Apple (AAPL) fiscal 2025 Mac. table. Apple Inc. | were $2.025 billion ($2,025 million) in 2025, and $10 million in 2024 (dollars in millions).  [3: AAPL 2025 Item 8 table_row] Apple (AAPL) fiscal 2025 Financial table table. Apple Inc. | were $2.025 billion ($2,025 million) in 2025,...
- failure type: correct

## baseline_succeeds_rag_fails

No example available in this run.

## retrieval_miss

- question_id: `AAPL_2025_NUMERIC_FACT_7bf60a5b`
- question: What value did Apple report for controls and procedures item other information item c in fiscal 2025?
- gold answer: 9B
- baseline prediction: , and
- candidate system: rag_gpt_tfidf_top3
- candidate prediction: " .
- retrieved preview: [1: AAPL 2025 Item 8 table_row] Apple (AAPL) fiscal 2025 Financial table table. Apple Inc. | were $2.025 billion ($2,025 million) in 2025, $10 million in 2024, and $40 million in 2023 (dollars in millions).  [2: AAPL 2025 Item 7 table_row] Apple (AAPL) fiscal 2025 Mac. table. Apple Inc. | were $2.025 billion ($2,025 million) in 2025, and $10 million in 2024 (dollars in millions).  [3: AAPL 2025 Item 7 table_row] Apple (AAPL) fiscal 2025 taxes. table. Financial Statements in Part II, Item were...
- failure type: wrong_section

## retrieval_finds_evidence_generator_fails

No example available in this run.

## random_context_fails

- question_id: `AAPL_2025_NUMERIC_FACT_87f37103`
- question: What value did Apple report for legal proceedings item mine safety disclosures part ii in fiscal 2025?
- gold answer: 4
- baseline prediction: , and
- candidate system: random_context_gpt
- candidate prediction: .
- retrieved preview: [1: GOOG 2025 Item 8 table_row] Google (GOOG) fiscal 2025 2025, we had no commercial paper outstanding. table. For additional information, see Note were 6 in 2025, and 8 in 2026.  [2: NVDA 2025 Item 16 narrative] 13. Code Section 409A; Tax Qualification. (a) Purchase Rights granted under the 423 Component are intended to be exempt from the application of Section 409A of the Code under Treasury Regulation Section 1.409A-1(b)(5)(ii). Purchase Rights granted under the Non-423 Component to U.S. t...
- failure type: numeric_unit_error

## oracle_succeeds

No example available in this run.

## oracle_fails

- question_id: `AAPL_2025_NUMERIC_FACT_87f37103`
- question: What value did Apple report for legal proceedings item mine safety disclosures part ii in fiscal 2025?
- gold answer: 4
- baseline prediction: , and
- candidate system: oracle_gpt
- candidate prediction: .
- retrieved preview: Legal Proceedings Item 4. Mine Safety Disclosures Part II Item 5. Market for Registrant’s Common Equity, Related Stockholder Matters and Issuer Purchases of Equity Item 6. [Reserved]
- failure type: numeric_unit_error
