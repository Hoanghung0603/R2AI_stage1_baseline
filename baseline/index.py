from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from .chunking import make_chunks, validate_articles
from .core import (
    Config,
    legal_tokenize,
    read_jsonl,
    sha256_file,
    write_json,
    write_jsonl,
)
from .models import DenseEncoder


def build_index(
    corpus_path: str | Path,
    index_directory: str | Path,
    config: Config,
    encoder: Any | None = None,
) -> dict[str, Any]:
    directory = Path(index_directory)
    directory.mkdir(parents=True, exist_ok=True)

    articles = validate_articles(read_jsonl(corpus_path))
    chunks = make_chunks(articles, config)
    tokenized = [legal_tokenize(chunk["text"]) for chunk in chunks]
    bm25 = BM25Okapi(
        tokenized,
        k1=config.bm25_k1,
        b=config.bm25_b,
        epsilon=config.bm25_epsilon,
    )

    dense_encoder = encoder or DenseEncoder(config)
    vectors = dense_encoder.encode([chunk["text"] for chunk in chunks])

    write_jsonl(directory / "articles.jsonl", articles)
    write_jsonl(directory / "chunks.jsonl", chunks)
    with (directory / "bm25.pkl").open("wb") as stream:
        pickle.dump(bm25, stream, protocol=pickle.HIGHEST_PROTOCOL)
    np.save(directory / "vectors.npy", vectors.astype(np.float32, copy=False))

    files = ("articles.jsonl", "chunks.jsonl", "bm25.pkl", "vectors.npy")
    manifest = {
        "version": 1,
        "config": config.as_dict(),
        "articles": len(articles),
        "chunks": len(chunks),
        "files": {name: sha256_file(directory / name) for name in files},
    }
    write_json(directory / "manifest.json", manifest)
    return manifest
