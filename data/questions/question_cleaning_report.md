# Question Cleaning Report

- input question file: `data/questions/questions_verified.remapped.jsonl`
- chunks file: `outputs/chunks/chunks.jsonl`
- input question count: 271
- accepted research_valid count: 108
- rejected count: 163

## Rejected Count By Reason
- vague_question: 63
- toc_item_number_answer: 35
- table_of_contents_question: 30
- malformed_question: 14
- not_financial_or_operational: 11
- boilerplate_evidence: 5
- incidental_year_answer: 4
- unsupported_answer: 1

## Accepted Count By Question Type
- explanation: 104
- text_fact: 4

## Accepted Count By Ticker
- AAPL: 19
- TSLA: 19
- GOOG: 17
- NVDA: 16
- AMZN: 14
- MSFT: 13
- META: 10

## Accepted Count By Company
- Apple: 19
- Tesla: 19
- Google: 17
- Nvidia: 16
- Amazon: 14
- Microsoft: 13
- Meta: 10

## Examples Of Rejected TOC/Item-Number Questions
- `AAPL_2025_NUMERIC_FACT_87f37103` (toc_item_number_answer): What value did Apple report for legal proceedings item mine safety disclosures part ii in fiscal 2025?
  answer: 4
- `AAPL_2025_NUMERIC_FACT_01b17ed7` (toc_item_number_answer): What value did Apple report for management s discussion and analysis of financial condition in fiscal 2025?
  answer: 7
- `AAPL_2025_NUMERIC_FACT_7bf60a5b` (toc_item_number_answer): What value did Apple report for controls and procedures item other information item c in fiscal 2025?
  answer: 9B
- `AAPL_2025_NUMERIC_FACT_798af295` (toc_item_number_answer): What value did Apple report for exhibit and financial statement schedules item form k in fiscal 2025?
  answer: 16
- `AAPL_2025_NUMERIC_FACT_1d8568a6` (toc_item_number_answer): What value did Apple report for that might cause such differences include but are in fiscal 2025?
  answer: 1
- `AAPL_2025_NUMERIC_FACT_e8393a7f` (toc_item_number_answer): What value did Apple report for based on its visionos operating system home includes in fiscal 2025?
  answer: 4
- `AAPL_2025_NUMERIC_FACT_1fd4ea85` (incidental_year_answer): What value did Apple report for the consumer small and mid sized business education in fiscal 2025?
  answer: 2025,
- `AAPL_2025_NUMERIC_FACT_080ba473` (incidental_year_answer): What value did Apple report for very low cost structures and by imitating the in fiscal 2025?
  answer: 2025

## Examples Of Accepted High-Quality Questions
- `AAPL_2025_TEXT_FACT_4011dcad`: What reportable segments or business units did Apple describe in fiscal 2025?
  answer: The Company’s reportable segments consist of the Americas, Europe, Greater China, Japan and Rest of Asia Pacific.
- `TSLA_2025_TEXT_FACT_9f822d5c`: What reportable segments or business units did Tesla describe in fiscal 2025?
  answer: Segment Information We operate as two reportable segments: (i) automotive and (ii) energy generation and storage.
- `AAPL_2025_EXPLANATION_ac3df4d4`: According to Apple's fiscal 2025 10-K, what drove or explained research and development because the industries in which?
  answer: Research and Development Because the industries in which the Company competes are characterized by rapid technologica...
- `AAPL_2025_EXPLANATION_74449cd9`: According to Apple's fiscal 2025 10-K, what drove or explained major public health issues including pandemics such as?
  answer: Major public health issues, including pandemics such as the COVID-19 pandemic, have adversely affected, and could in...
- `AAPL_2025_EXPLANATION_a38bb9a4`: According to Apple's fiscal 2025 10-K, what drove or explained because the company relies on single or limited?
  answer: Because the Company relies on single or limited sources for the supply and manufacture of many critical components, a...
- `AAPL_2025_EXPLANATION_1b1734db`: According to Apple's fiscal 2025 10-K, what drove or explained due to the highly volatile and competitive nature?
  answer: Due to the highly volatile and competitive nature of the markets and industries in which the Company competes, the Co...
- `AAPL_2025_EXPLANATION_1c53cfe5`: According to Apple's fiscal 2025 10-K, what drove or explained because the company currently obtains certain components from?
  answer: Because the Company currently obtains certain components from single or limited sources, the Company is subject to si...
- `AAPL_2025_EXPLANATION_7c4d05ee`: According to Apple's fiscal 2025 10-K, what drove or explained because the company s markets are volatile competitive?
  answer: Because the Company’s markets are volatile, competitive and subject to rapid technology and price changes, there is a...
