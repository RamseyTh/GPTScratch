from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.chunking import CHUNK_CONFIGS, chunk_documents_for_config, chunk_stats, parse_chunk_configs, write_chunk_audit
from src.preprocess import load_raw_documents
from src.utils import write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Build retrieval chunks for every configured chunking ablation strategy.")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--configs", default=",".join(CHUNK_CONFIGS.keys()))
    parser.add_argument("--out-dir", default="outputs/chunks")
    args = parser.parse_args()

    documents = load_raw_documents(args.data_dir)
    configs = parse_chunk_configs(args.configs)
    for config in configs:
        out_dir = Path(args.out_dir) / config
        out_path = out_dir / "chunks.jsonl"
        spec = CHUNK_CONFIGS[config]
        chunks = chunk_documents_for_config(
            documents,
            chunk_config=config,
            chunk_size=int(spec["chunk_size"]),
            overlap=int(spec["overlap"]),
        )
        write_jsonl(out_path, chunks)
        write_chunk_audit(chunks, out_dir)
        stats = chunk_stats(chunks)
        print(
            f"{config}: files={len(documents)}, chunks={stats['num_chunks']}, "
            f"retrievable={stats['retrievable_count']}, average_tokens={stats['average_chunk_length']:.1f}"
        )


if __name__ == "__main__":
    main()
