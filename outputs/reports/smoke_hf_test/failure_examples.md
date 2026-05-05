# Failure Examples

## rag_succeeds_baseline_fails

- question_id: `SMOKE_001`
- question: What was Apple revenue?
- gold answer: 10 billion
- baseline prediction: 10 Services
- candidate system: rag_gpt_tfidf_top3
- candidate prediction: the the
- retrieved preview: [1: AAPL 2025] Apple revenue was 10 billion in 2025.
- failure type: correct

## baseline_succeeds_rag_fails

No example available in this run.

## retrieval_miss

No example available in this run.

## retrieval_finds_evidence_generator_fails

No example available in this run.

## random_context_fails

- question_id: `SMOKE_001`
- question: What was Apple revenue?
- gold answer: 10 billion
- baseline prediction: 10 Services
- candidate system: random_context_gpt
- candidate prediction: the the
- retrieved preview: [1: AAPL 2025] Apple revenue was 10 billion in 2025.
- failure type: numeric_unit_error

## oracle_succeeds

No example available in this run.

## oracle_fails

- question_id: `SMOKE_001`
- question: What was Apple revenue?
- gold answer: 10 billion
- baseline prediction: 10 Services
- candidate system: oracle_gpt
- candidate prediction: the the
- retrieved preview: Apple revenue was 10 billion
- failure type: numeric_unit_error
