---
pretty_name: Bible Semantics Embeddings
language:
  - en
license: other
task_categories:
  - sentence-similarity
  - feature-extraction
tags:
  - bible
  - embeddings
  - semantic-search
  - retrieval
  - parquet
size_categories:
  - 100K<n<1M
---

# bible_semantics

A framework for semantic research and RAG over Bible texts.

This repository contains flattened Bible translation JSONL files, verse embeddings generated with `nomic-embed-text:latest`, Parquet exports for efficient analysis, and benchmark tooling for fuzzy verse retrieval.

## Setup

```bash
uv venv
uv pip install -e .
```

The embedding generator expects an Ollama-compatible embeddings endpoint. By default it uses:

```text
http://localhost:11434
```

## Generate Embeddings

```bash
uv run embed --data_dir data
```

The script processes files matching `*_flattened.jsonl` and writes sibling files named `*_flattened_with_embeddings.jsonl`.

## Convert to Parquet

```bash
uv run --no-sync python scripts/jsonl_to_parquet.py \
  --data-dir data \
  --output-dir data/parquet
```

The Parquet files store embeddings as `fixed_size_list<float>[768]`, which is much more compact and faster to scan than the source JSONL.

## Batch Evaluation

The benchmark uses the LetsChurch `queries.yaml` retrieval suite:

- `532` natural-language and paraphrased Bible verse queries
- `1596` maximum points
- `3` points if an expected verse is ranked first
- `2` points if ranked second
- `1` point if ranked third
- `0` points if no expected verse appears in the top 3

```bash
uv run --no-sync python scripts/evaluate_lets_church_queries.py \
  --queries data/lets_church_queries.yaml \
  --parquet-dir data/parquet \
  --output data/lets_church_eval_results.json
```

## Model Performance

For comparison, the LetsChurch `bible-embeddings` dataset reports `70.0%` (`1117/1596`) for `huggingface/nomic-ai-nomic-embed-text-v1.5` on BSB, and `89.3%` (`1426/1596`) for `openai/text-embedding-3-large`.

This project uses `nomic-embed-text:latest` through Ollama and evaluates the same query suite across all local translations.

## Results

### nomic-embed-text:latest

| Translation | Accuracy | Points | Hit@1 | Hit@3 |
| --- | ---: | ---: | ---: | ---: |
| ESV | 69.49% | 1109/1596 | 322 | 405 |
| ESVUK | 69.17% | 1104/1596 | 322 | 402 |
| NIVUK | 68.48% | 1093/1596 | 314 | 404 |
| EHV | 68.36% | 1091/1596 | 316 | 399 |
| NIV | 67.98% | 1085/1596 | 309 | 400 |
| NKJV | 67.48% | 1077/1596 | 313 | 393 |
| MEV | 67.42% | 1076/1596 | 307 | 399 |
| NASB1995 | 66.73% | 1065/1596 | 308 | 391 |
| NRSVUE | 66.29% | 1058/1596 | 310 | 386 |
| NRSV | 66.23% | 1057/1596 | 310 | 385 |
| NASB | 66.17% | 1056/1596 | 305 | 394 |
| WEB | 65.16% | 1040/1596 | 302 | 388 |
| NET | 64.47% | 1029/1596 | 299 | 382 |
| LEB | 64.41% | 1028/1596 | 298 | 381 |
| KJ21 | 64.35% | 1027/1596 | 294 | 379 |
| KJV | 63.60% | 1015/1596 | 283 | 377 |
| BRG | 63.41% | 1012/1596 | 282 | 376 |
| JUB | 63.35% | 1011/1596 | 289 | 370 |
| AKJV | 63.28% | 1010/1596 | 281 | 376 |
| ASV | 61.22% | 977/1596 | 283 | 357 |
| GNV | 60.40% | 964/1596 | 278 | 356 |
| ISV | 60.15% | 960/1596 | 268 | 360 |
| NLT | 59.77% | 954/1596 | 267 | 362 |
| GW | 54.26% | 866/1596 | 242 | 328 |
| YLT | 52.32% | 835/1596 | 232 | 318 |
| NOG | 51.25% | 818/1596 | 228 | 313 |
| NLV | 49.94% | 797/1596 | 222 | 305 |

Full benchmark output is written to `data/lets_church_eval_results.json`.

## File Structure

```text
data/
├── *_flattened.jsonl
├── *_flattened_with_embeddings.jsonl
├── lets_church_queries.yaml
├── lets_church_eval_results.json
└── parquet/
    └── *_flattened_with_embeddings.parquet
scripts/
├── evaluate_lets_church_queries.py
└── jsonl_to_parquet.py
src/
└── bible_semantics/
    └── main.py
```
