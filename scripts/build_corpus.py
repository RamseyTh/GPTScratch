from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.chunking import chunk_documents, chunk_stats, write_chunk_audit
from src.preprocess import load_raw_documents
from src.utils import write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess 10-K files and write retrieval chunks.")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--out", default="outputs/chunks/chunks.jsonl")
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=64)
    args = parser.parse_args()

    documents = load_raw_documents(args.data_dir)
    chunks = chunk_documents(documents, chunk_size=args.chunk_size, overlap=args.overlap)
    write_jsonl(args.out, chunks)
    write_chunk_audit(chunks, Path(args.out).parent)
    stats = chunk_stats(chunks)
    print(f"number of files processed: {len(documents)}")
    print(f"number of chunks: {stats['num_chunks']}")
    print(f"average chunk length: {stats['average_chunk_length']:.1f} words")
    print(f"table_row chunks: {stats['table_row_count']}")
    print(f"table_summary chunks: {stats['table_summary_count']}")
    print(f"chunk audit: {Path(args.out).parent / 'chunk_audit.csv'}")


if __name__ == "__main__":
    main()
