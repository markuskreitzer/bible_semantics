#!/usr/bin/env python3
"""Evaluate local Bible embeddings against LetsChurch retrieval queries."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any
from urllib import request

import numpy as np
import pyarrow.parquet as pq
import yaml


BOOK_ALIASES = {
    "Psalms": "Psalm",
}


def normalize_book(book: str) -> str:
    return BOOK_ALIASES.get(book, book)


def parse_ref(reference: str) -> str:
    match = re.match(r"^(?P<book>.+)\s+(?P<chapter>\d+):(?P<verse>\d+)$", reference.strip())
    if not match:
        raise ValueError(f"Unsupported reference format: {reference!r}")
    book = normalize_book(match.group("book"))
    return f"{book} {int(match.group('chapter'))}:{int(match.group('verse'))}"


def load_queries(path: Path) -> list[dict[str, Any]]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [
        {
            "query": item["query"],
            "expected": {parse_ref(reference) for reference in item["expected"]},
        }
        for item in raw
    ]


def embed_queries(
    queries: list[str],
    model: str,
    endpoint: str,
    batch_size: int,
) -> np.ndarray:
    vectors: list[list[float]] = []
    url = endpoint.rstrip("/") + "/v1/embeddings"

    for start in range(0, len(queries), batch_size):
        batch = queries[start : start + batch_size]
        payload = json.dumps({"model": model, "input": batch}).encode("utf-8")
        req = request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(req, timeout=120) as response:
            data = json.loads(response.read().decode("utf-8"))
        vectors.extend(item["embedding"] for item in data["data"])

    return normalize(np.asarray(vectors, dtype=np.float32))


def normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def load_translation(path: Path) -> tuple[list[str], np.ndarray]:
    table = pq.read_table(
        path,
        columns=["book", "chapter", "verse", "embedding"],
    )
    books = table.column("book").to_pylist()
    chapters = table.column("chapter").to_pylist()
    verses = table.column("verse").to_pylist()
    references = [
        f"{book} {int(chapter)}:{int(verse)}"
        for book, chapter, verse in zip(books, chapters, verses)
    ]

    embedding_column = table.column("embedding").combine_chunks()
    embedding_dim = embedding_column.type.list_size
    embeddings = embedding_column.values.to_numpy(zero_copy_only=False).reshape(
        len(embedding_column),
        embedding_dim,
    )
    embeddings = normalize(np.asarray(embeddings, dtype=np.float32))
    return references, embeddings


def evaluate_translation(
    parquet_path: Path,
    queries: list[dict[str, Any]],
    query_vectors: np.ndarray,
) -> dict[str, Any]:
    references, embeddings = load_translation(parquet_path)
    reference_array = np.asarray(references)
    scores = query_vectors @ embeddings.T
    top3_indices = np.argpartition(-scores, kth=2, axis=1)[:, :3]

    total_points = 0
    hits_at_1 = 0
    hits_at_3 = 0
    misses: list[dict[str, Any]] = []

    for query_index, row_indices in enumerate(top3_indices):
        ordered = row_indices[np.argsort(-scores[query_index, row_indices])]
        top_refs = reference_array[ordered].tolist()
        expected = queries[query_index]["expected"]
        points = 0

        for rank, reference in enumerate(top_refs, start=1):
            if reference in expected:
                points = 4 - rank
                break

        total_points += points
        if points == 3:
            hits_at_1 += 1
        if points:
            hits_at_3 += 1
        else:
            misses.append(
                {
                    "query": queries[query_index]["query"],
                    "expected": sorted(expected),
                    "top3": top_refs,
                }
            )

    max_points = len(queries) * 3
    return {
        "translation": parquet_path.name.replace("_flattened_with_embeddings.parquet", ""),
        "queries": len(queries),
        "points": total_points,
        "max_points": max_points,
        "score_percent": round(total_points / max_points * 100, 2),
        "hits_at_1": hits_at_1,
        "hits_at_3": hits_at_3,
        "misses": misses,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, default=Path("data/lets_church_queries.yaml"))
    parser.add_argument("--parquet-dir", type=Path, default=Path("data/parquet"))
    parser.add_argument("--output", type=Path, default=Path("data/lets_church_eval_results.json"))
    parser.add_argument("--model", default="nomic-embed-text:latest")
    parser.add_argument("--endpoint", default="http://localhost:11434")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    queries = load_queries(args.queries)
    query_vectors = embed_queries(
        [query["query"] for query in queries],
        model=args.model,
        endpoint=args.endpoint,
        batch_size=args.batch_size,
    )

    parquet_files = sorted(args.parquet_dir.glob("*_flattened_with_embeddings.parquet"))
    if not parquet_files:
        raise SystemExit(f"No Parquet files found in {args.parquet_dir}")

    results = []
    for path in parquet_files:
        print(f"Evaluating {path.name}...", flush=True)
        results.append(evaluate_translation(path, queries, query_vectors))
    results.sort(key=lambda result: result["points"], reverse=True)
    summary = {
        "query_suite": str(args.queries),
        "model": args.model,
        "metric": "Top-3 points: 3 for rank 1, 2 for rank 2, 1 for rank 3, 0 otherwise",
        "results": results,
    }
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    for result in results:
        print(
            f"{result['translation']:8} "
            f"{result['points']:4}/{result['max_points']} "
            f"{result['score_percent']:5.2f}% "
            f"hit@1={result['hits_at_1']:3} hit@3={result['hits_at_3']:3}"
        )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
