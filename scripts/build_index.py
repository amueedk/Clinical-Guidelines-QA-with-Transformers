"""Build a FAISS index from chunked JSONL produced by prepare_corpus.py

Usage:
    python scripts/build_index.py --chunks data/processed/chunks.jsonl --out_dir indexes --model models/all-MiniLM-L6-v2

Produces:
- indexes/faiss.index (FAISS binary)
- indexes/metadata.json (list of metadata for each vector id)
"""
import argparse
import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def load_chunks(path: Path):
    items = []
    with path.open("r", encoding="utf-8") as fin:
        for line in fin:
            items.append(json.loads(line))
    return items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", type=str, default="data/processed/chunks.jsonl")
    parser.add_argument("--out_dir", type=str, default="indexes")
    parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--batch", type=int, default=64)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = SentenceTransformer(args.model)

    chunks = load_chunks(Path(args.chunks))
    texts = [c["text"] for c in chunks]

    embeddings = model.encode(texts, show_progress_bar=True, batch_size=args.batch)
    embeddings = np.array(embeddings).astype("float32")

    # normalize for cosine similarity
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, str(out_dir / "faiss.index"))

    # write metadata mapping
    with (out_dir / "metadata.json").open("w", encoding="utf-8") as fout:
        json.dump(chunks, fout, ensure_ascii=False, indent=2)

    print(f"Wrote index and metadata to {out_dir}")


if __name__ == "__main__":
    main()
