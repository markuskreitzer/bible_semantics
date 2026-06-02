#!/usr/bin/env python3
"""Upload the curated dataset artifacts to Hugging Face Hub."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi


ALLOW_PATTERNS = [
    "README.md",
    "data/lets_church_queries.yaml",
    "data/lets_church_eval_results.json",
    "data/parquet/*.parquet",
]

IGNORE_PATTERNS = [
    ".git/*",
    ".venv/*",
    "**/.env*",
    "**/*secret*",
    "**/*token*",
    "**/*key*",
    "**/credentials*",
    "data/*_flattened.jsonl",
    "data/*_flattened_with_embeddings.jsonl",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo_id", help="Dataset repo id, for example 'username/bible-semantics'")
    parser.add_argument("--private", action="store_true", default=True)
    parser.add_argument("--public", action="store_false", dest="private")
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()

    api = HfApi()
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=args.private,
        exist_ok=True,
    )
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="dataset",
        folder_path=args.root,
        allow_patterns=ALLOW_PATTERNS,
        ignore_patterns=IGNORE_PATTERNS,
        commit_message="Upload Bible semantics Parquet embeddings and benchmark results",
    )
    print(f"Uploaded dataset artifacts to https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
