# Bible datasets

Generated dataset files are kept outside Git.

The Parquet files are stored in the private Hugging Face dataset
[`elec3647/bible-semantics`](https://huggingface.co/datasets/elec3647/bible-semantics).
Each Parquet row includes the Bible text in the `content` column together with
its source, book, chapter, verse, ID, embedding model, and embedding.

After signing in with the Hugging Face CLI, restore all Parquet files into a
checkout with:

```sh
hf download elec3647/bible-semantics \
  --repo-type dataset \
  --include "data/parquet/*.parquet" \
  --local-dir .
```

JSONL and embedding JSONL files are generated locally and are intentionally
ignored by Git.
