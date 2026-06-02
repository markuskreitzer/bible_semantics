#!/usr/bin/env python3
"""Convert embedded Bible JSONL files into Parquet shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_PATTERN = "*_flattened_with_embeddings.jsonl"


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_number}") from exc


def build_schema(embedding_dim: int) -> pa.Schema:
    return pa.schema(
        [
            pa.field("source", pa.string()),
            pa.field("book", pa.string()),
            pa.field("chapter", pa.string()),
            pa.field("verse", pa.string()),
            pa.field("content", pa.string()),
            pa.field("id", pa.string()),
            pa.field("embedding_model", pa.string()),
            pa.field("embedding", pa.list_(pa.float32(), list_size=embedding_dim)),
        ]
    )


def rows_to_table(rows: list[dict], schema: pa.Schema, embedding_dim: int) -> pa.Table:
    embeddings: list[list[float]] = []
    columns: dict[str, list[str]] = {
        "source": [],
        "book": [],
        "chapter": [],
        "verse": [],
        "content": [],
        "id": [],
        "embedding_model": [],
    }

    for row in rows:
        embedding = row.get("embedding")
        if not isinstance(embedding, list) or len(embedding) != embedding_dim:
            row_id = row.get("id", "<missing>")
            raise ValueError(
                f"Row {row_id} has embedding dimension "
                f"{len(embedding) if isinstance(embedding, list) else 'missing'}, "
                f"expected {embedding_dim}"
            )

        for column in columns:
            value = row.get(column, "")
            columns[column].append("" if value is None else str(value))
        embeddings.append(embedding)

    arrays = [pa.array(columns[column], type=pa.string()) for column in columns]
    arrays.append(pa.array(embeddings, type=schema.field("embedding").type))
    return pa.Table.from_arrays(arrays, schema=schema)


def convert_file(input_path: Path, output_path: Path, batch_size: int, force: bool) -> None:
    if output_path.exists() and not force:
        print(f"Skipping {output_path.name}; already exists")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer: pq.ParquetWriter | None = None
    schema: pa.Schema | None = None
    batch: list[dict] = []
    embedding_dim: int | None = None
    row_count = 0

    try:
        for row in iter_jsonl(input_path):
            if embedding_dim is None:
                embedding = row.get("embedding")
                if not isinstance(embedding, list) or not embedding:
                    raise ValueError(f"First row in {input_path} does not contain an embedding")
                embedding_dim = len(embedding)
                schema = build_schema(embedding_dim)
                writer = pq.ParquetWriter(
                    output_path,
                    schema,
                    compression="zstd",
                    use_dictionary=["source", "book", "chapter", "verse", "embedding_model"],
                )

            batch.append(row)
            if len(batch) >= batch_size:
                assert writer is not None and schema is not None
                writer.write_table(rows_to_table(batch, schema, embedding_dim))
                row_count += len(batch)
                batch.clear()

        if batch:
            assert writer is not None and schema is not None and embedding_dim is not None
            writer.write_table(rows_to_table(batch, schema, embedding_dim))
            row_count += len(batch)

        print(f"Wrote {row_count:,} rows -> {output_path}")
    finally:
        if writer is not None:
            writer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/parquet"))
    parser.add_argument("--pattern", default=DEFAULT_PATTERN)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    input_files = sorted(args.data_dir.glob(args.pattern))
    if not input_files:
        raise SystemExit(f"No files matching {args.pattern!r} in {args.data_dir}")

    for input_path in input_files:
        output_path = args.output_dir / f"{input_path.stem}.parquet"
        convert_file(input_path, output_path, args.batch_size, args.force)


if __name__ == "__main__":
    main()
