from __future__ import annotations

import os
import time
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .evaluation import answer_coverage_for_results, source_accuracy_for_results, table_row_recall_for_results


class Retriever:
    def __init__(self, chunks: list[dict], method: str = "tfidf"):
        self.all_chunks = chunks
        self.chunks = [chunk for chunk in chunks if chunk.get("retrievable", True) is not False]
        self.method = method
        self._dense_model = None
        self._dense_embeddings = None
        if method == "tfidf":
            self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
            texts = [chunk.get("text", "") for chunk in self.chunks] or [""]
            self.matrix = self.vectorizer.fit_transform(texts)
        elif method == "dense":
            self._init_dense()
        else:
            raise ValueError(f"Unsupported retrieval method: {method}")

    def _init_dense(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError:
            print("sentence-transformers is not installed. Falling back to TF-IDF retrieval.")
            self.method = "tfidf"
            self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
            texts = [chunk.get("text", "") for chunk in self.chunks] or [""]
            self.matrix = self.vectorizer.fit_transform(texts)
            return
        self._dense_model = SentenceTransformer("all-MiniLM-L6-v2")
        texts = [chunk.get("text", "") for chunk in self.chunks]
        self._dense_embeddings = self._dense_model.encode(texts, normalize_embeddings=True) if texts else np.empty((0, 384))

    def retrieve(self, query: str, top_k: int = 3, filters: dict | None = None, question: dict | None = None) -> list[dict]:
        if not self.chunks or top_k <= 0:
            return []
        candidate_indices = self._candidate_indices(filters)
        if not candidate_indices:
            return []
        if self.method == "dense":
            scores = self._dense_scores(query, candidate_indices)
        else:
            scores = self._tfidf_scores(query, candidate_indices)
        scores = self._apply_boosts(scores, candidate_indices, question)
        ranked = sorted(zip(candidate_indices, scores), key=lambda item: item[1], reverse=True)[:top_k]
        return [self._result(rank, idx, score) for rank, (idx, score) in enumerate(ranked, start=1)]

    def retrieve_with_latency(
        self,
        query: str,
        top_k: int = 3,
        filters: dict | None = None,
        question: dict | None = None,
    ) -> tuple[list[dict], float]:
        start = time.perf_counter()
        results = self.retrieve(query, top_k=top_k, filters=filters, question=question)
        return results, time.perf_counter() - start

    def _candidate_indices(self, filters: dict | None) -> list[int]:
        if not filters:
            return list(range(len(self.chunks)))
        return [idx for idx, chunk in enumerate(self.chunks) if _matches_filters(chunk, filters)]

    def _tfidf_scores(self, query: str, candidate_indices: list[int]) -> list[float]:
        query_vector = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vector, self.matrix[candidate_indices]).ravel()
        return [float(score) for score in sims]

    def _dense_scores(self, query: str, candidate_indices: list[int]) -> list[float]:
        if self._dense_model is None or self._dense_embeddings is None:
            return self._tfidf_scores(query, candidate_indices)
        query_embedding = self._dense_model.encode([query], normalize_embeddings=True)[0]
        candidate_embeddings = self._dense_embeddings[candidate_indices]
        return [float(score) for score in np.dot(candidate_embeddings, query_embedding)]

    def _apply_boosts(self, scores: list[float], candidate_indices: list[int], question: dict | None) -> list[float]:
        if not question:
            return scores
        boosted = []
        qtype = question.get("question_type")
        for score, idx in zip(scores, candidate_indices):
            chunk = self.chunks[idx]
            multiplier = 1.0
            if qtype == "numeric_fact" and chunk.get("source_type") in {"table_row", "table_summary"}:
                multiplier *= 1.75
                if chunk.get("section_id") in {"Item 7", "Item 8"}:
                    multiplier *= 1.35
                terms = _numeric_row_terms(question)
                text = str(chunk.get("text", "")).lower()
                if terms and all(term in text for term in terms):
                    multiplier *= 2.50
            if qtype == "risk_factor" and chunk.get("section_id") == "Item 1A" and chunk.get("source_type") == "narrative":
                multiplier *= 1.75
            if qtype == "explanation" and chunk.get("section_id") == "Item 7" and chunk.get("source_type") in {"narrative", "local_context"}:
                multiplier *= 1.50
            boosted.append(score * multiplier)
        return boosted

    def _result(self, rank: int, idx: int, score: float) -> dict:
        chunk = self.chunks[idx]
        metadata_keys = (
            "source_file",
            "ticker",
            "company",
            "year",
            "section_id",
            "section_name",
            "source_type",
            "table_id",
            "table_title",
            "row_label",
            "chunk_index",
            "retrievable",
        )
        metadata = {key: chunk.get(key, "") for key in metadata_keys}
        return {
            "rank": rank,
            "score": float(score),
            "chunk_id": chunk.get("chunk_id", str(idx)),
            "text": chunk.get("text", ""),
            "metadata": metadata,
        }


def build_retrieval_query(question: dict) -> str:
    parts = [str(question.get("question", ""))]
    for key in ("ticker", "company", "year"):
        if question.get(key):
            parts.append(str(question[key]))
    qtype = question.get("question_type")
    if qtype == "numeric_fact":
        parts.extend(str(value) for value in (question.get("row_label"), question.get("source_section"), question.get("source_type")) if value)
        parts.extend(str(alias) for alias in question.get("answer_aliases", [])[:3] if alias)
        parts.extend(_numeric_row_terms(question))
        parts.append("net sales revenue table")
    elif qtype == "risk_factor":
        parts.append("risk factors Item 1A")
    elif qtype == "explanation":
        parts.append("MD&A Item 7 driven by increase decrease")
    elif qtype == "text_fact":
        parts.append("business segment products services")
    return " ".join(part for part in parts if part)


def _numeric_row_terms(question: dict) -> list[str]:
    explicit = str(question.get("row_label") or "")
    text = explicit or str(question.get("question", ""))
    match = re_search_row_label(text)
    terms_text = match or explicit
    terms = [term.lower() for term in terms_text.replace("'s", "").split() if len(term) > 2]
    return [term for term in terms if term not in {"what", "were", "was", "fiscal", "year", "apple", "microsoft", "company"}]


def re_search_row_label(text: str) -> str:
    import re

    match = re.search(r"([A-Za-z][A-Za-z &'/-]{2,80}?\s(?:net sales|revenue|gross margin|income|expenses|cash flow|sales))", text, re.I)
    return match.group(1) if match else ""


def retrieval_filters(question: dict, open_corpus: bool = False) -> dict:
    if open_corpus:
        return {}
    return {key: question.get(key) for key in ("ticker", "year") if question.get(key)}


def retrieval_context(results: list[dict], max_chars: int | None = None, token_budget: int = 700) -> str:
    pieces = []
    used_tokens = 0
    for result in results:
        meta = result.get("metadata", {})
        label_bits = [str(meta.get(key, "")).strip() for key in ("ticker", "year", "section_id", "source_type") if meta.get(key)]
        prefix = f"[{result['rank']}: {' '.join(label_bits)}] "
        text = result.get("text", "").replace("\n", " ")
        piece = prefix + text
        piece_tokens = len(piece.split())
        if token_budget and used_tokens + piece_tokens > token_budget:
            remaining = token_budget - used_tokens
            if remaining <= 8:
                break
            piece = " ".join(piece.split()[:remaining])
            piece_tokens = remaining
        if max_chars is not None and len(piece) > max_chars:
            piece = piece[: max(0, max_chars - 3)].rstrip() + "..."
        pieces.append(piece)
        used_tokens += piece_tokens
        if token_budget and used_tokens >= token_budget:
            break
    return "\n\n".join(pieces)


def retrieval_diagnostic(question: dict, query: str, filters: dict, results: list[dict], latency: float) -> dict:
    coverage_at_1 = answer_coverage_for_results(results, question, k=1)
    coverage = answer_coverage_for_results(results, question, k=3)
    source_ok = source_accuracy_for_results(results, question, k=3)
    return {
        "question_id": question.get("question_id", ""),
        "query": query,
        "filters": filters,
        "retrieved_chunk_ids": [row.get("chunk_id") for row in results],
        "retrieved_scores": [row.get("score") for row in results],
        "retrieved_sections": [row.get("metadata", {}).get("section_id") for row in results],
        "retrieved_source_types": [row.get("metadata", {}).get("source_type") for row in results],
        "answer_coverage_at_1": coverage_at_1,
        "answer_coverage_at_3": coverage,
        "source_accuracy": source_ok,
        "source_accuracy_at_3": source_ok,
        "table_row_recall_at_3": table_row_recall_for_results(results, question, k=3),
        "retrieval_latency_seconds": latency,
        "noise_reason": retrieval_noise_reason(question, results, coverage, source_ok),
    }


def retrieval_noise_reason(question: dict, results: list[dict], coverage: bool, source_ok: bool | None) -> str:
    if coverage and (source_ok is not False):
        return "none"
    if any("forward-looking" in str(result.get("text", "")).lower() for result in results):
        return "boilerplate"
    if source_ok is False:
        return "wrong_company"
    if results and question.get("source_section"):
        expected = str(question.get("source_section"))
        if not any(str(result.get("metadata", {}).get("section_id")) == expected for result in results):
            return "wrong_section"
    if question.get("question_type") == "numeric_fact" and not any(
        result.get("metadata", {}).get("source_type") == "table_row" for result in results
    ):
        return "table_label_missing"
    if results and not coverage:
        return "answer_split_across_chunks"
    return "none"


def _matches_filters(chunk: dict, filters: dict[str, Any]) -> bool:
    for key in ("ticker", "year", "company"):
        expected = filters.get(key)
        if not expected:
            continue
        actual = str(chunk.get(key, ""))
        if key == "company":
            if str(expected).lower() not in actual.lower():
                return False
        elif str(expected).upper() != actual.upper():
            return False
    return True
