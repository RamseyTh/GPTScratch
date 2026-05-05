"""Load and lightly normalize raw Form 10-K filings from local files."""

from __future__ import annotations

import re
from pathlib import Path

from bs4 import BeautifulSoup

from .utils import normalize_whitespace


SUPPORTED_EXTENSIONS = {".txt", ".html", ".htm", ".pdf"}


def discover_raw_files(data_dir: str | Path) -> list[Path]:
    """Find supported raw filing files under a data directory."""
    root = Path(data_dir)
    if not root.exists():
        return []
    files = [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS]
    return sorted(files)


def load_raw_documents(data_dir: str | Path) -> list[dict]:
    """Read local filings into dictionaries with inferred company metadata."""
    documents = []
    for path in discover_raw_files(data_dir):
        text = read_filing_file(path)
        if not text.strip():
            continue
        metadata = infer_metadata_from_filename(path)
        documents.append(
            {
                "text": preprocess_text(text, path.suffix.lower()),
                "source_file": str(path),
                **metadata,
            }
        )
    return documents


def read_filing_file(path: str | Path) -> str:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return path.read_text(encoding="utf-8", errors="ignore")
    if suffix in {".html", ".htm"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".pdf":
        try:
            import fitz  # type: ignore
        except ImportError:
            print(f"PDF support requires PyMuPDF. Skipping {path}.")
            return ""
        text_parts = []
        with fitz.open(path) as doc:
            for page in doc:
                text_parts.append(page.get_text("text"))
        return "\n".join(text_parts)
    return ""


def preprocess_text(text: str, extension: str | None = None) -> str:
    if extension in {".html", ".htm"} or looks_like_html(text):
        soup = BeautifulSoup(text, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = soup.get_text("\n")
    text = text.replace("\xa0", " ")
    text = re.sub(r"-\n(?=[a-z])", "", text)
    text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)
    return normalize_whitespace(text)


def looks_like_html(text: str) -> bool:
    sample = text[:1000].lower()
    return "<html" in sample or "<table" in sample or "<body" in sample


def infer_metadata_from_filename(path: str | Path) -> dict[str, str]:
    stem = Path(path).stem
    parts = [part for part in re.split(r"[_\-\s.]+", stem) if part]

    year = ""
    for part in parts:
        match = re.search(r"(20\d{2}|19\d{2})", part)
        if match:
            year = match.group(1)
            break

    ticker = ""
    for part in parts:
        clean = re.sub(r"[^A-Za-z]", "", part)
        if 1 <= len(clean) <= 6 and clean.upper() == clean and not clean.isdigit() and clean.lower() not in {"form", "10k"}:
            ticker = clean
            break

    company_parts = []
    for part in parts:
        clean_lower = part.lower()
        if part == year or part == ticker or clean_lower in {"10k", "form", "annual", "report"}:
            continue
        if re.fullmatch(r"20\d{2}|19\d{2}", part):
            continue
        company_parts.append(part)
    company = " ".join(company_parts).strip()

    return {"ticker": ticker, "company": company, "year": year}


def smoke_documents() -> list[dict]:
    return [
        {
            "text": (
                "Apple reported Services net sales of $109.158 billion in fiscal 2025. "
                "The filing also discusses product sales, risks, liquidity, and operating expenses."
            ),
            "source_file": "synthetic_smoke_apple_2025.txt",
            "ticker": "AAPL",
            "company": "Apple",
            "year": "2025",
        },
        {
            "text": (
                "Microsoft reported revenue from cloud and productivity offerings. "
                "This synthetic smoke filing exists only to exercise retrieval code."
            ),
            "source_file": "synthetic_smoke_msft_2025.txt",
            "ticker": "MSFT",
            "company": "Microsoft",
            "year": "2025",
        },
    ]
