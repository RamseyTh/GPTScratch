"""Build retrieval chunks from Form 10-K filing text.

The default project corpus uses section-aware narrative chunks plus
natural-language table rows.  The chunking ablation reuses the same helpers
with named configurations so only the retrieval units change across runs.
"""

from __future__ import annotations

import csv
import hashlib
import re
from collections import Counter, defaultdict
from pathlib import Path

from .utils import ensure_dir, normalize_whitespace, simple_word_count


SECTION_NAMES = {
    "Item 1": "Business",
    "Item 1A": "Risk Factors",
    "Item 7": "Management's Discussion and Analysis",
    "Item 8": "Financial Statements and Supplementary Data",
}
BOILERPLATE_PATTERNS = (
    "table of contents",
    "forward-looking statements",
    "forward looking statements",
    "signatures",
    "exhibit index",
    "exhibits",
)

CHUNK_CONFIGS = {
    "fixed_128": {"kind": "fixed", "chunk_size": 128, "overlap": 32},
    "fixed_256": {"kind": "fixed", "chunk_size": 256, "overlap": 64},
    "fixed_512": {"kind": "fixed", "chunk_size": 512, "overlap": 128},
    "section_180": {"kind": "section", "chunk_size": 180, "overlap": 40},
    "table_row_only": {"kind": "table_row_only", "chunk_size": 180, "overlap": 40},
    "table_aware_mixed": {"kind": "table_aware_mixed", "chunk_size": 180, "overlap": 40},
    "table_aware_clean": {"kind": "table_aware_clean", "chunk_size": 180, "overlap": 40},
    "table_aware_clean_context": {"kind": "table_aware_clean_context", "chunk_size": 180, "overlap": 40},
}
DEFAULT_CHUNK_CONFIG = "table_aware_mixed"
TARGET_SECTION_IDS = {"Item 1", "Item 1A", "Item 7", "Item 8"}


def parse_chunk_configs(value: str | list[str] | tuple[str, ...] | None) -> list[str]:
    """Parse and validate a comma-separated chunk config list."""
    if value is None:
        return [DEFAULT_CHUNK_CONFIG]
    configs = value if isinstance(value, (list, tuple)) else [item.strip() for item in str(value).split(",")]
    out = [config for config in configs if config]
    unknown = [config for config in out if config not in CHUNK_CONFIGS]
    if unknown:
        raise ValueError(f"Unsupported chunk config(s): {', '.join(unknown)}. Supported configs: {', '.join(CHUNK_CONFIGS)}")
    return out or [DEFAULT_CHUNK_CONFIG]


def chunk_documents_for_config(
    documents: list[dict],
    chunk_config: str = DEFAULT_CHUNK_CONFIG,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list[dict]:
    """Build chunks for one named ablation configuration."""
    if chunk_config not in CHUNK_CONFIGS:
        raise ValueError(f"Unsupported chunk config: {chunk_config}")
    spec = CHUNK_CONFIGS[chunk_config]
    size = int(chunk_size or spec["chunk_size"])
    ov = int(overlap if overlap is not None else spec["overlap"])
    chunks: list[dict] = []
    for document in documents:
        if spec["kind"] == "fixed":
            chunks.extend(_fixed_chunks(document, chunk_config, size, ov))
        elif spec["kind"] == "section":
            chunks.extend(_section_narrative_chunks(document, chunk_config, size, ov))
        else:
            mixed = chunk_document(document, chunk_size=size, overlap=ov)
            if spec["kind"] == "table_row_only":
                mixed = [chunk for chunk in mixed if chunk.get("source_type") in {"table_row", "table_summary"}]
            elif spec["kind"] == "table_aware_clean_context":
                mixed = _add_local_context_chunks_if_needed(document, mixed, size, ov)
            chunks.extend(mixed)
    return _finalize_chunk_config(chunks, chunk_config, aggressive_clean=chunk_config in {"table_aware_clean", "table_aware_clean_context"})


def chunk_document(document: dict, chunk_size: int = 180, overlap: int = 40) -> list[dict]:
    """Build the default mixed table-aware chunks for one filing."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size.")

    chunks: list[dict] = []
    base = _chunk_base_id(document)
    chunk_index = 0
    for section in split_sections(document.get("text", "")):
        section_chunks = _table_chunks(document, section, base, chunk_index)
        chunks.extend(section_chunks)
        chunk_index += len(section_chunks)
        narrative_chunks = _narrative_chunks(document, section, base, chunk_index, chunk_size, overlap)
        chunks.extend(narrative_chunks)
        chunk_index += len(narrative_chunks)
    return chunks


def chunk_documents(documents: list[dict], chunk_size: int = 180, overlap: int = 40) -> list[dict]:
    """Build the default mixed table-aware corpus."""
    chunks: list[dict] = []
    for document in documents:
        chunks.extend(chunk_document(document, chunk_size=chunk_size, overlap=overlap))
    return chunks


def split_sections(text: str) -> list[dict]:
    text = normalize_whitespace(text)
    if not text:
        return []
    pattern = re.compile(r"(?im)^\s*(Item\s+\d+[A-Z]?)\.?\s+(.{0,100})$")
    matches = list(pattern.finditer(text))
    if not matches:
        return [{"section_id": "Unknown", "section_name": "Unknown", "text": text}]
    sections: list[dict] = []
    if matches[0].start() > 0:
        sections.append({"section_id": "Front Matter", "section_name": "Front Matter", "text": text[: matches[0].start()]})
    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        section_id = _canonical_item(match.group(1))
        heading_tail = normalize_whitespace(match.group(2))
        section_name = SECTION_NAMES.get(section_id) or heading_tail or section_id
        sections.append({"section_id": section_id, "section_name": section_name, "text": text[start:end]})
    return sections


def chunk_stats(chunks: list[dict]) -> dict:
    lengths = [int(chunk.get("token_count") or simple_word_count(chunk.get("text", ""))) for chunk in chunks]
    by_source_type = Counter(chunk.get("source_type", "unknown") for chunk in chunks)
    by_section = Counter(chunk.get("section_id", "unknown") for chunk in chunks)
    by_config = Counter(chunk.get("chunk_config", DEFAULT_CHUNK_CONFIG) for chunk in chunks)
    retrievable = Counter(bool(chunk.get("retrievable", True)) for chunk in chunks)
    return {
        "num_chunks": len(chunks),
        "average_chunk_length": sum(lengths) / len(lengths) if lengths else 0.0,
        "by_source_type": dict(by_source_type),
        "by_section": dict(by_section),
        "by_chunk_config": dict(by_config),
        "retrievable_count": retrievable.get(True, 0),
        "non_retrievable_count": retrievable.get(False, 0),
        "table_row_count": by_source_type.get("table_row", 0),
        "table_summary_count": by_source_type.get("table_summary", 0),
        "local_context_count": by_source_type.get("local_context", 0),
    }


def write_chunk_audit(chunks: list[dict], out_dir: str | Path = "outputs/chunks") -> None:
    """Write compact corpus audit files for a chunk set."""
    out_dir = ensure_dir(out_dir)
    rows = _audit_rows(chunks)
    with (out_dir / "chunk_audit.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["category", "value", "count"])
        writer.writeheader()
        writer.writerows(rows)
    _write_chunk_samples(chunks, out_dir / "chunk_samples.md")
    _write_non_retrievable_audit(chunks, out_dir / "non_retrievable_audit.csv")


def _fixed_chunks(document: dict, chunk_config: str, chunk_size: int, overlap: int) -> list[dict]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size.")
    words = normalize_whitespace(document.get("text", "")).replace("\n", " ").split()
    base = f"{_chunk_base_id(document)}_{chunk_config}"
    step = max(1, chunk_size - overlap)
    section = {"section_id": "Mixed", "section_name": "Fixed-size chunk"}
    chunks: list[dict] = []
    for idx, start in enumerate(range(0, len(words), step)):
        piece = " ".join(words[start : start + chunk_size])
        if not piece:
            continue
        chunks.append(_make_chunk(document, section, base, idx, piece, "fixed"))
        if start + chunk_size >= len(words):
            break
    return chunks


def _section_narrative_chunks(document: dict, chunk_config: str, chunk_size: int, overlap: int) -> list[dict]:
    base = f"{_chunk_base_id(document)}_{chunk_config}"
    chunks: list[dict] = []
    chunk_index = 0
    for section in split_sections(document.get("text", "")):
        if section.get("section_id") not in TARGET_SECTION_IDS:
            continue
        if _retrievability_reason(section.get("text", ""), section, "narrative", aggressive=True) in {"table_of_contents", "cover_page", "signatures", "exhibit_list", "navigation_only"}:
            continue
        narrative = _narrative_chunks(document, section, base, chunk_index, chunk_size, overlap)
        chunks.extend(narrative)
        chunk_index += len(narrative)
    return chunks


def _add_local_context_chunks_if_needed(document: dict, chunks: list[dict], chunk_size: int, overlap: int) -> list[dict]:
    if any(chunk.get("source_type") == "local_context" for chunk in chunks):
        return chunks
    base = f"{_chunk_base_id(document)}_table_aware_clean_context"
    next_index = max((int(chunk.get("chunk_index") or 0) for chunk in chunks), default=-1) + 1
    additions: list[dict] = []
    for section in split_sections(document.get("text", "")):
        if section.get("section_id") != "Item 7":
            continue
        text = _local_context_text(section.get("text", ""))
        if text:
            additions.append(_make_chunk(document, section, base, next_index, text, "local_context"))
            next_index += 1
    return chunks + additions


def _finalize_chunk_config(chunks: list[dict], chunk_config: str, aggressive_clean: bool = False) -> list[dict]:
    finalized: list[dict] = []
    for index, chunk in enumerate(chunks):
        row = dict(chunk)
        row["chunk_config"] = chunk_config
        row.setdefault("value_aliases", _value_aliases(row))
        row.setdefault("columns", [])
        row.setdefault("values", [])
        row.setdefault("unit", None)
        reason = _retrievability_reason(row.get("text", ""), row, row.get("source_type", ""), aggressive=aggressive_clean)
        if reason:
            row["retrievable"] = False
            row["non_retrievable_reason"] = reason
        else:
            row["retrievable"] = bool(row.get("retrievable", True))
            row["non_retrievable_reason"] = None
        if chunk_config not in str(row.get("chunk_id", "")):
            row["chunk_id"] = f"{row.get('chunk_id', 'chunk')}_{chunk_config}"
        row["chunk_index"] = int(row.get("chunk_index") or index)
        finalized.append(row)
    return finalized


def _table_chunks(document: dict, section: dict, base: str, start_index: int) -> list[dict]:
    rows = detect_table_rows(section.get("text", ""), document)
    if not rows:
        return []
    chunks: list[dict] = []
    table_id = f"{base}_table_{start_index:04d}"
    for offset, row in enumerate(rows):
        chunks.append(
            _make_chunk(
                document,
                section,
                base,
                start_index + offset,
                row["text"],
                "table_row",
                table_id=table_id,
                table_title=row["table_title"],
                row_label=row["row_label"],
                columns=row.get("years", []),
                values=row.get("values", []),
                unit=row.get("units", ""),
                retrievable=None,
            )
        )
    summary_text = summarize_table_rows(document, rows)
    if summary_text:
        chunks.append(
            _make_chunk(
                document,
                section,
                base,
                start_index + len(rows),
                summary_text,
                "table_summary",
                table_id=table_id,
                table_title=rows[0]["table_title"],
                columns=rows[0].get("years", []),
                values=[row.get("row_label", "") for row in rows[:5]],
                unit=rows[0].get("units", ""),
                retrievable=None,
            )
        )
    local_context = _local_context_text(section.get("text", ""))
    if local_context and section.get("section_id") == "Item 7":
        chunks.append(
            _make_chunk(
                document,
                section,
                base,
                start_index + len(rows) + 1,
                local_context,
                "local_context",
                table_id=table_id,
                table_title=rows[0]["table_title"],
                retrievable=None,
            )
        )
    return chunks


def detect_table_rows(text: str, document: dict) -> list[dict]:
    lines = [normalize_whitespace(line) for line in text.splitlines()]
    rows: list[dict] = []
    recent_title = ""
    recent_years: list[str] = []
    units = _infer_units(text)
    for idx, line in enumerate(lines):
        if not line:
            continue
        years = re.findall(r"\b(20\d{2}|19\d{2})\b", line)
        if len(years) >= 2:
            recent_years = years[:5]
            if idx > 0:
                recent_title = _nearest_title(lines, idx)
            continue
        numbers = re.findall(r"\$?\(?-?\d[\d,]*(?:\.\d+)?\)?\s*%?", line)
        if len(numbers) < 2 or not re.search(r"[A-Za-z]", line):
            if len(line.split()) <= 16 and re.search(r"[A-Za-z]", line):
                recent_title = line
            continue
        label = _row_label(line, numbers[0])
        if not label or len(label) > 90:
            continue
        values = numbers[: len(recent_years) or min(3, len(numbers))]
        years_for_values = recent_years[: len(values)] or _default_years(document, len(values))
        table_title = recent_title or _nearest_title(lines, idx) or "Financial table"
        rows.append(
            {
                "row_label": label,
                "table_title": table_title,
                "values": values,
                "years": years_for_values,
                "units": units,
                "text": naturalize_table_row(document, table_title, label, values, years_for_values, units),
            }
        )
    rows.extend(_detect_inline_table_rows(text, document, units))
    return _dedupe_rows(rows)


def _detect_inline_table_rows(text: str, document: dict, units: str) -> list[dict]:
    rows: list[dict] = []
    compact = normalize_whitespace(text).replace("\n", " ")
    years = re.findall(r"\b(20\d{2}|19\d{2})\b", compact)
    header_years = _likely_header_years(years, document)
    title = _inline_table_title(compact)
    labels = [
        "iPhone",
        "Mac",
        "iPad",
        "Wearables, Home and Accessories",
        "Services",
        "Products",
        "Total net sales",
        "Total gross margin",
        "Total operating expenses",
        "Net sales",
        "Revenue",
        "Total revenues",
    ]
    for label in labels:
        pattern = re.compile(
            rf"(?<![A-Za-z])({re.escape(label)})(?:\s*\(\d+\))?\s+\$?\s*([\d,]+(?:\.\d+)?)"
            rf"(?:\s+\(?-?\d+\)?\s*%|\s+—\s*%)?\s+\$?\s*([\d,]+(?:\.\d+)?)"
            rf"(?:\s+\(?-?\d+\)?\s*%|\s+—\s*%)?\s+\$?\s*([\d,]+(?:\.\d+)?)",
            re.I,
        )
        for match in pattern.finditer(compact):
            row_label = match.group(1)
            values = [match.group(2), match.group(3), match.group(4)]
            rows.append(
                {
                    "row_label": row_label,
                    "table_title": title,
                    "values": values,
                    "years": header_years[:3],
                    "units": units,
                    "text": naturalize_table_row(document, title, row_label, values, header_years[:3], units),
                }
            )
    return rows


def _likely_header_years(years: list[str], document: dict) -> list[str]:
    doc_year = str(document.get("year") or "")
    for idx in range(0, max(0, len(years) - 2)):
        trio = years[idx : idx + 3]
        if doc_year and trio == [doc_year, str(int(doc_year) - 1), str(int(doc_year) - 2)]:
            return trio
        if len(set(trio)) == 3 and all(int(trio[pos]) - 1 == int(trio[pos + 1]) for pos in range(2)):
            return trio
    if doc_year:
        year = int(doc_year)
        return [str(year), str(year - 1), str(year - 2)]
    return years[:3] or ["2025", "2024", "2023"]


def _inline_table_title(text: str) -> str:
    candidates = [
        "Products and Services Performance",
        "Consolidated Statements of Operations",
        "disaggregated net sales",
        "Revenue",
        "Gross Margin",
        "Operating Expenses",
    ]
    lowered = text.lower()
    for candidate in candidates:
        if candidate.lower() in lowered:
            return candidate
    return "Financial table"


def _dedupe_rows(rows: list[dict]) -> list[dict]:
    seen: set[tuple[str, tuple[str, ...]]] = set()
    out: list[dict] = []
    for row in rows:
        key = (str(row.get("row_label", "")).lower(), tuple(row.get("values", [])))
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def naturalize_table_row(document: dict, table_title: str, row_label: str, values: list[str], years: list[str], units: str) -> str:
    company = document.get("company") or document.get("ticker") or "Company"
    ticker = document.get("ticker") or ""
    year = document.get("year") or (years[0] if years else "")
    label = _natural_row_label(table_title, row_label)
    pieces = []
    for value, value_year in zip(values, years):
        formatted = _format_financial_value(value, units)
        pieces.append(f"{formatted} in {value_year}")
    joined = _join_series(pieces)
    units_suffix = f" ({units})" if units else ""
    ticker_text = f" ({ticker})" if ticker else ""
    return f"{company}{ticker_text} fiscal {year} {table_title} table. {label} were {joined}{units_suffix}."


def summarize_table_rows(document: dict, rows: list[dict]) -> str:
    if not rows:
        return ""
    company = document.get("company") or document.get("ticker") or "Company"
    year = document.get("year") or ""
    title = rows[0]["table_title"]
    important = [row for row in rows if re.search(r"total|net sales|revenue|income|cash", row["row_label"], re.I)]
    selected = (important or rows)[:5]
    labels = ", ".join(row["row_label"] for row in selected)
    text = f"{company} fiscal {year} {title} table summary. Important rows include {labels}."
    return _limit_words(text, 220)


def _narrative_chunks(document: dict, section: dict, base: str, start_index: int, chunk_size: int, overlap: int) -> list[dict]:
    units = _split_narrative_units(section.get("text", ""))
    chunks: list[dict] = []
    current: list[str] = []
    current_len = 0
    chunk_index = start_index
    for unit in units:
        token_count = simple_word_count(unit)
        if current and current_len + token_count > chunk_size:
            text = " ".join(current)
            chunks.append(_make_chunk(document, section, base, chunk_index, text, "narrative"))
            chunk_index += 1
            overlap_units = _tail_units(current, overlap)
            current = overlap_units
            current_len = simple_word_count(" ".join(current))
        current.append(unit)
        current_len += token_count
    if current:
        chunks.append(_make_chunk(document, section, base, chunk_index, " ".join(current), "narrative"))
    return chunks


def _make_chunk(
    document: dict,
    section: dict,
    base: str,
    chunk_index: int,
    text: str,
    source_type: str,
    table_id: str | None = None,
    table_title: str | None = None,
    row_label: str | None = None,
    columns: list[str] | None = None,
    values: list[str] | None = None,
    unit: str | None = None,
    retrievable: bool | None = None,
) -> dict:
    text = normalize_whitespace(text).replace("\n", " ")
    token_count = simple_word_count(text)
    if retrievable is None:
        retrievable = _is_retrievable(text, section, source_type)
    reason = None if retrievable else _retrievability_reason(text, section, source_type)
    return {
        "chunk_id": f"{base}_{source_type}_{chunk_index:04d}",
        "chunk_config": DEFAULT_CHUNK_CONFIG,
        "text": text,
        "source_file": document.get("source_file", ""),
        "ticker": document.get("ticker", ""),
        "company": document.get("company", ""),
        "year": document.get("year", ""),
        "section_id": section.get("section_id", "Unknown"),
        "section_name": section.get("section_name", "Unknown"),
        "source_type": source_type,
        "table_id": table_id,
        "table_title": table_title,
        "row_label": row_label,
        "columns": columns or [],
        "values": values or [],
        "unit": unit or "",
        "value_aliases": _value_aliases({"values": values or [], "unit": unit or ""}),
        "chunk_index": chunk_index,
        "token_count": token_count,
        "retrievable": bool(retrievable),
        "non_retrievable_reason": reason,
    }


def _is_retrievable(text: str, section: dict, source_type: str) -> bool:
    return _retrievability_reason(text, section, source_type) is None


def _retrievability_reason(text: str, section: dict, source_type: str, aggressive: bool = False) -> str | None:
    normalized = normalize_whitespace(text).lower()
    if simple_word_count(text) < 8:
        return "too_short"
    if section.get("section_id") == "Front Matter":
        return "cover_page"
    if source_type.startswith("table") and section.get("section_id") not in {"Item 7", "Item 8"}:
        return "navigation_only"
    if "/mnt/data/" in normalized:
        return "header_footer"
    if _looks_like_table_of_contents(normalized):
        return "table_of_contents"
    if "signature" in normalized and simple_word_count(text) < 120:
        return "signatures"
    if "exhibit index" in normalized or "exhibit and financial statement schedules" in normalized:
        return "exhibit_list"
    if "forward-looking statements" in normalized or "forward looking statements" in normalized:
        return "forward_looking_boilerplate"
    if aggressive and any(pattern in normalized for pattern in BOILERPLATE_PATTERNS):
        return "navigation_only"
    return None


def _split_narrative_units(text: str) -> list[str]:
    paragraphs = [para.strip() for para in re.split(r"\n\s*\n+", text) if para.strip()]
    units: list[str] = []
    for paragraph in paragraphs:
        if simple_word_count(paragraph) <= 70:
            units.append(paragraph)
            continue
        sentences = re.split(r"(?<=[.!?])\s+", paragraph)
        units.extend(sentence.strip() for sentence in sentences if sentence.strip())
    return units


def _tail_units(units: list[str], max_tokens: int) -> list[str]:
    out: list[str] = []
    count = 0
    for unit in reversed(units):
        unit_count = simple_word_count(unit)
        if out and count + unit_count > max_tokens:
            break
        out.insert(0, unit)
        count += unit_count
    return out


def _audit_rows(chunks: list[dict]) -> list[dict]:
    rows: list[dict] = [{"category": "total", "value": "all", "count": len(chunks)}]
    for category, key in (
        ("chunk_config", "chunk_config"),
        ("company", "company"),
        ("section", "section_id"),
        ("source_type", "source_type"),
        ("non_retrievable_reason", "non_retrievable_reason"),
    ):
        counts = Counter(str(chunk.get(key) or "unknown") for chunk in chunks)
        rows.extend({"category": category, "value": value, "count": count} for value, count in sorted(counts.items()))
    retrievable = Counter(str(bool(chunk.get("retrievable", True))) for chunk in chunks)
    rows.extend({"category": "retrievable", "value": value, "count": count} for value, count in sorted(retrievable.items()))
    rows.append({"category": "average_token_count", "value": "all", "count": f"{chunk_stats(chunks)['average_chunk_length']:.2f}"})
    return rows


def _write_non_retrievable_audit(chunks: list[dict], path: Path) -> None:
    rows = [
        {
            "chunk_id": chunk.get("chunk_id"),
            "chunk_config": chunk.get("chunk_config"),
            "ticker": chunk.get("ticker"),
            "year": chunk.get("year"),
            "section_id": chunk.get("section_id"),
            "source_type": chunk.get("source_type"),
            "non_retrievable_reason": chunk.get("non_retrievable_reason"),
            "text_preview": str(chunk.get("text", ""))[:240],
        }
        for chunk in chunks
        if chunk.get("retrievable") is False
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "chunk_id",
            "chunk_config",
            "ticker",
            "year",
            "section_id",
            "source_type",
            "non_retrievable_reason",
            "text_preview",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_chunk_samples(chunks: list[dict], path: Path) -> None:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for chunk in chunks:
        grouped[str(chunk.get("source_type", "unknown"))].append(chunk)
    lines = ["# Chunk Samples", ""]
    for source_type in ("table_row", "table_summary", "local_context", "narrative"):
        lines.extend([f"## {source_type}", ""])
        for chunk in grouped.get(source_type, [])[:5]:
            lines.append(f"- `{chunk.get('chunk_id')}` ({chunk.get('section_id')}, retrievable={chunk.get('retrievable')})")
            lines.append(f"  {chunk.get('text', '')[:500]}")
        lines.append("")
    lines.extend(["## non_retrievable", ""])
    for chunk in [chunk for chunk in chunks if chunk.get("retrievable") is False][:5]:
        lines.append(f"- `{chunk.get('chunk_id')}` ({chunk.get('section_id')}, {chunk.get('source_type')})")
        lines.append(f"  {chunk.get('text', '')[:500]}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _looks_like_table_of_contents(normalized_text: str) -> bool:
    item_hits = len(re.findall(r"\bitem\s+\d+[a-z]?\b", normalized_text, re.I))
    navigation_terms = sum(
        1
        for term in (
            "table of contents",
            "part i",
            "part ii",
            "part iii",
            "part iv",
            "form 10-k summary",
            "management's discussion and analysis",
            "financial statements",
        )
        if term in normalized_text
    )
    return "table of contents" in normalized_text or (item_hits >= 4 and navigation_terms >= 2)


def _value_aliases(chunk: dict) -> list[str]:
    aliases: list[str] = []
    unit = str(chunk.get("unit") or "")
    for value in chunk.get("values") or []:
        formatted = _format_financial_value(str(value), unit)
        aliases.append(formatted)
        cleaned = str(value).replace("$", "").replace(",", "").strip()
        if cleaned:
            aliases.append(f"{cleaned} {unit}".strip())
    seen: set[str] = set()
    out: list[str] = []
    for alias in aliases:
        if alias and alias not in seen:
            seen.add(alias)
            out.append(alias)
    return out


def _canonical_item(raw: str) -> str:
    match = re.search(r"Item\s+(\d+[A-Z]?)", raw, re.I)
    return f"Item {match.group(1).upper()}" if match else raw.strip()


def _row_label(line: str, first_number: str) -> str:
    label = line.split(first_number, 1)[0]
    label = re.sub(r"[$()]", "", label)
    return normalize_whitespace(label).strip(" .:-")


def _nearest_title(lines: list[str], idx: int) -> str:
    for prior in range(idx - 1, max(-1, idx - 6), -1):
        candidate = lines[prior].strip()
        if candidate and re.search(r"[A-Za-z]", candidate) and len(candidate.split()) <= 14:
            return candidate
    return "Financial table"


def _infer_units(text: str) -> str:
    lowered = text.lower()
    if "dollars in millions" in lowered or "in millions" in lowered:
        return "dollars in millions"
    if "dollars in billions" in lowered or "in billions" in lowered:
        return "dollars in billions"
    if "percent" in lowered:
        return "percent"
    return ""


def _default_years(document: dict, count: int) -> list[str]:
    year = int(document.get("year") or 0)
    if year:
        return [str(year - idx) for idx in range(count)]
    return [str(idx + 1) for idx in range(count)]


def _format_financial_value(value: str, units: str) -> str:
    raw = value.strip()
    is_percent = "%" in raw or "percent" in units
    cleaned = raw.replace("$", "").replace(",", "").replace("(", "-").replace(")", "").replace("%", "")
    try:
        number = float(cleaned)
    except ValueError:
        return raw
    if is_percent:
        return f"{number:g}%"
    if "millions" in units and abs(number) >= 1000:
        return f"${number / 1000:.3f} billion (${number:,.0f} million)"
    if "millions" in units:
        return f"${number:g} million"
    if "billions" in units:
        return f"${number:g} billion"
    return raw


def _join_series(items: list[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + ", and " + items[-1]


def _limit_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


def _natural_row_label(table_title: str, row_label: str) -> str:
    if row_label.lower() == "services" and "products and services" in table_title.lower():
        return "Services net sales"
    if row_label.lower() == "products" and "gross margin" not in table_title.lower():
        return "Products net sales"
    return row_label


def _local_context_text(text: str) -> str:
    sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", normalize_whitespace(text).replace("\n", " ")) if sentence.strip()]
    selected = [
        sentence
        for sentence in sentences
        if re.search(r"increased|decreased|primarily|due to|driven by|compared to", sentence, re.I)
    ][:4]
    return _limit_words(" ".join(selected), 220)


def _chunk_base_id(document: dict) -> str:
    source_name = Path(document.get("source_file", "document")).stem
    ticker = document.get("ticker") or "UNK"
    year = document.get("year") or "YYYY"
    digest = hashlib.md5(str(document.get("source_file", source_name)).encode("utf-8")).hexdigest()[:8]
    safe_source = re.sub(r"[^A-Za-z0-9]+", "_", source_name).strip("_")[:30]
    return f"{ticker}_{year}_{safe_source}_{digest}"
