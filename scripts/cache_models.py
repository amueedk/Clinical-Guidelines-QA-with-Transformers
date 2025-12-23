"""Cache models locally into a `models/` folder to avoid repeated downloads.

Usage:
    python scripts/cache_models.py --models "sentence-transformers/all-MiniLM-L6-v2" "google/flan-t5-base"

This script downloads and saves the specified models into `models/<safe-name>`.
It handles SentenceTransformer models and standard Hugging Face transformer models.
"""
import argparse
import os
from pathlib import Path

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from sentence_transformers import SentenceTransformer


def safe_name(name: str) -> str:
    return name.replace('/', '__').replace(':', '__')


def cache_sentence_transformer(model_name: str, out_dir: Path):
    print(f"Caching sentence-transformer {model_name} -> {out_dir}")
    model = SentenceTransformer(model_name)
    model.save(str(out_dir))


def cache_transformer_seq2seq(model_name: str, out_dir: Path):
    print(f"Caching seq2seq model {model_name} -> {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    AutoTokenizer.from_pretrained(model_name, cache_dir=str(out_dir))
    AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=str(out_dir))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs='+', required=True)
    parser.add_argument("--out", type=str, default="models")
    args = parser.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    for m in args.models:
        target = out_root / safe_name(m)
        if 'sentence-transformers' in m or 'all-' in m and 'MiniLM' in m:
            cache_sentence_transformer(m, target)
        else:
            cache_transformer_seq2seq(m, target)

    print("Done caching models.")


if __name__ == '__main__':
    main()
