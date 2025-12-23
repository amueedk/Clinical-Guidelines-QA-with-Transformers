"""Prepare WHO text files into chunked JSONL for indexing.

Usage:
    python scripts/prepare_corpus.py --input_dir data/raw/who --out data/processed/chunks.jsonl

This script:
- Reads .txt files under input_dir
- Parses metadata headers (SOURCE, TITLE, URL, DATE_ACCESSED, TOPIC)
- Chunks text into token-length chunks using a tokenizer (default: flan-t5-base tokenizer)
- Writes one JSON line per chunk with metadata and chunk id

Notes:
- Requires `transformers` installed. Tokenizer is used only for chunking size estimation.
"""
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

from transformers import AutoTokenizer


def parse_file(path: Path) -> Dict:
    text = path.read_text(encoding="utf-8")
    # split headers and body by the dashed separator
    if "--------------------------------------------------" in text:
        header, body = text.split("--------------------------------------------------", 1)
    else:
        # fallback: treat entire file as body
        header, body = "", text

    meta = {}
    for line in header.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip().upper()] = v.strip()

    return {
        "file": str(path.name),
        "meta": meta,
        "text": body
    }


def preprocess_body(body: str) -> str:
    """Normalize whitespace, attach detected headings to following paragraph, and
    return a cleaned string where paragraphs are separated by double newlines.

    Heuristics for headings:
    - A short line (<= 6 words, length < 80) without sentence punctuation is
      likely a heading and will be prefixed with '## ' and attached to the next paragraph.
    """
    # normalize newlines
    body = body.replace('\r\n', '\n').replace('\r', '\n')

    lines = [l.rstrip() for l in body.split('\n')]
    paragraphs = []
    current = []

    def flush_current():
        if current:
            # join with single space to avoid accidental token splits
            paragraphs.append(' '.join(current).strip())
            current.clear()

    for i, line in enumerate(lines):
        s = line.strip()
        if s == "":
            # paragraph break
            flush_current()
            continue

        # simple heading detection
        words = s.split()
        has_punct = any(ch in s for ch in '.:;,-—()[]/')
        is_heading = (len(words) <= 6 and len(s) < 80 and not has_punct and s[0].isupper())

        if is_heading:
            # flush any previous paragraph, then start a new paragraph with heading marker
            flush_current()
            current.append('## ' + s)
        else:
            current.append(s)

    flush_current()
    # Collapse multiple empty paragraphs and join with double newline
    return '\n\n'.join(p for p in paragraphs if p)



def chunk_text(text: str, tokenizer, chunk_size: int = 300, overlap: int = 60) -> List[Dict]:
    """Token-based chunking with overlap. Returns list of dicts with chunk_id and text."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    start = 0
    chunk_id = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append({
            "chunk_id": f"chunk_{chunk_id:04d}",
            "text": chunk_text
        })
        chunk_id += 1
        if end == len(tokens):
            break
        start += chunk_size - overlap
    return chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="data/raw/who")
    parser.add_argument("--out", type=str, default="data/processed/chunks.jsonl")
    parser.add_argument("--model", type=str, default="google/flan-t5-base")
    parser.add_argument("--chunk_size", type=int, default=400)
    parser.add_argument("--overlap", type=int, default=100)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    input_path = Path(args.input_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    files = list(input_path.rglob("*.txt"))
    with out_path.open("w", encoding="utf-8") as fout:
        for f in files:
            parsed = parse_file(f)
            chunks = chunk_text(parsed["text"], tokenizer, args.chunk_size, args.overlap)
            for c in chunks:
                record = {
                    "id": f"{parsed['file']}::{c['chunk_id']}",
                    "file": parsed["file"],
                    "title": parsed["meta"].get("TITLE", ""),
                    "url": parsed["meta"].get("URL", ""),
                    "topic": parsed["meta"].get("TOPIC", ""),
                    "text": c["text"],
                    "paragraph_indices": c.get("paragraph_indices", [])
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
