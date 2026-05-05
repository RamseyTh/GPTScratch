# Failure Examples

## rag_succeeds_baseline_fails

- question_id: `AAPL_2025_NUMERIC_FACT_87f37103`
- question: What value did Apple report for legal proceedings item mine safety disclosures part ii in fiscal 2025?
- gold answer: 4
- baseline prediction: , and the first time of the first time of the first time of the first time of the first time of the first time of the first time of the first time .
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
- baseline prediction: , and the first time of the first time of the first time of the first time of the first time of the first time .
- candidate system: rag_gpt_tfidf_top3
- candidate prediction: " .
- retrieved preview: [1: AAPL 2025 Item 7 table_row] Apple (AAPL) fiscal 2025 taxes. table. Financial Statements in Part II, Item were $8 million in 2025, and $10 million in 2024 (dollars in millions).  [2: AAPL 2025 Item 8 table_row] Apple (AAPL) fiscal 2025 Financial table table. Apple Inc. | were $2.025 billion ($2,025 million) in 2025, $10 million in 2024, and $40 million in 2023 (dollars in millions).  [3: AAPL 2025 Item 7 table_row] Apple (AAPL) fiscal 2025 Mac. table. Apple Inc. | were $2.025 billion ($2,0...
- failure type: wrong_section

## retrieval_finds_evidence_generator_fails

- question_id: `MSFT_2025_NUMERIC_FACT_d7b97e2a`
- question: What value did Microsoft report for and analysis of financial condition and results of in fiscal 2025?
- gold answer: 7
- baseline prediction: , and the first time of the first time of the first time of the first time of the first time of the first time of the first time .
- candidate system: rag_gpt_tfidf_top3
- candidate prediction: " .
- retrieved preview: [1: MSFT 2025 Item 8 table_summary] Microsoft fiscal 2025 Revenue table summary. Important rows include revenue.  [2: MSFT 2025 Item 8 table_summary] Microsoft fiscal 2025 Revenue table summary. Important rows include revenue, revenue.  [3: MSFT 2025 Item 7 table_row] Microsoft (MSFT) fiscal 2025 Microsoft Cloud revenue and revenue growth table. Revenue from Microsoft were 365% in 2025, and 365% in 2024 (percent).
- failure type: wrong_section

## random_context_fails

- question_id: `AAPL_2025_NUMERIC_FACT_87f37103`
- question: What value did Apple report for legal proceedings item mine safety disclosures part ii in fiscal 2025?
- gold answer: 4
- baseline prediction: , and the first time of the first time of the first time of the first time of the first time of the first time of the first time of the first time .
- candidate system: random_context_gpt
- candidate prediction: .
- retrieved preview: [1: AMZN 2025 Item 8 narrative] filed a complaint against Amazon Web Services, Inc. in the United States District Court for the Northern District of Illinois. The complaint alleged, among other things, that Amazon S3 and DynamoDB infringe U.S. Patent Nos. 7,814,170; 7,103,640; and 7,233,978. The complaint sought an unspecified amount of damages, enhanced damages, attorneys’ fees, costs, interest, and injunctive relief. In April 2024, a jury found that Amazon infringed the asserted patents and...
- failure type: numeric_unit_error

## oracle_succeeds

No example available in this run.

## oracle_fails

- question_id: `AAPL_2025_NUMERIC_FACT_87f37103`
- question: What value did Apple report for legal proceedings item mine safety disclosures part ii in fiscal 2025?
- gold answer: 4
- baseline prediction: , and the first time of the first time of the first time of the first time of the first time of the first time of the first time of the first time .
- candidate system: oracle_gpt
- candidate prediction: .
- retrieved preview: Legal Proceedings Item 4. Mine Safety Disclosures Part II Item 5. Market for Registrant’s Common Equity, Related Stockholder Matters and Issuer Purchases of Equity Item 6. [Reserved]
- failure type: numeric_unit_error
