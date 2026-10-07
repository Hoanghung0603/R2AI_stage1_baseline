from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from .core import Config, legal_tokenize, read_jsonl, rrf, sha256_file
from .models import CrossEncoderReranker, DenseEncoder


def _top_indices(scores: np.ndarray, k: int, positive_only: bool = False) -> list[int]:
    if scores.ndim != 1:
        raise ValueError("Scores must be a one-dimensional array")
    if not np.isfinite(scores).all():
        raise ValueError("Scores must be finite")
    available = np.flatnonzero(scores > 0) if positive_only else np.arange(len(scores))
    k = min(k, len(available))
    if k == 0:
        return []
    available_scores = scores[available]
    threshold = np.partition(available_scores, len(available_scores) - k)[
        len(available_scores) - k
    ]
    candidates = available[available_scores >= threshold]
    return sorted(
        candidates.tolist(), key=lambda index: (-float(scores[index]), index)
    )[:k]


class RetrievalPipeline:
    def __init__(
        self,
        index_directory: str | Path,
        config: Config,
        encoder: Any | None = None,
        reranker: Any | None = None,
    ):
        directory = Path(index_directory)
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("config") != config.as_dict():
            raise ValueError("The configuration does not match the index")
        for name, digest in manifest["files"].items():
            if sha256_file(directory / name) != digest:
                raise ValueError(f"Index file changed: {name}")

        self.config = config
        self.chunk_list = read_jsonl(directory / "chunks.jsonl")
        self.chunks = {chunk["chunk_id"]: chunk for chunk in self.chunk_list}
        self.articles = {
            article["article_id"]: article
            for article in read_jsonl(directory / "articles.jsonl")
        }
        with (directory / "bm25.pkl").open("rb") as stream:
            self.bm25 = pickle.load(stream)
        self.vectors = np.load(
            directory / "vectors.npy", mmap_mode="r", allow_pickle=False
        )
        if len(self.vectors) != len(self.chunk_list):
            raise ValueError("The vector index is not aligned with the chunks")
        self.encoder = encoder or DenseEncoder(config)
        self.reranker = reranker or CrossEncoderReranker(config)

    def _bm25_search(self, query: str) -> list[tuple[str, float]]:
        scores = np.asarray(
            self.bm25.get_scores(legal_tokenize(query)), dtype=np.float64
        )
        return [
            (self.chunk_list[index]["chunk_id"], float(scores[index]))
            for index in _top_indices(
                scores, self.config.top_per_retriever, positive_only=True
            )
        ]

    def _dense_search(self, query: str) -> list[tuple[str, float]]:
        query_vector = self.encoder.encode([query])[0]
        if query_vector.shape != (self.vectors.shape[1],):
            raise ValueError("The query vector does not match the index")
        scores = np.asarray(self.vectors @ query_vector, dtype=np.float64)
        return [
            (self.chunk_list[index]["chunk_id"], float(scores[index]))
            for index in _top_indices(scores, self.config.top_per_retriever)
        ]

    def rank_articles(self, query: str) -> list[dict[str, Any]]:
        fused = rrf(
            [self._bm25_search(query), self._dense_search(query)],
            [self.config.bm25_weight, self.config.dense_weight],
            self.config.rrf_c,
        )
        candidates = fused[: self.config.rerank_candidates]
        reranked = self.reranker.rank(
            query, [self.chunks[chunk_id] for chunk_id, _ in candidates]
        )

        best: dict[str, dict[str, Any]] = {}
        for chunk_id, score in reranked:
            article_id = self.chunks[chunk_id]["article_id"]
            current = best.get(article_id)
            if current is None or score > current["score"]:
                best[article_id] = {
                    "article_id": article_id,
                    "chunk_id": chunk_id,
                    "score": float(score),
                }
        return sorted(
            best.values(), key=lambda item: (-item["score"], item["article_id"])
        )

    def predict(self, question: dict[str, Any]) -> dict[str, Any]:
        ranked = self.rank_articles(question["question"])[: self.config.top_k]
        sources = [self.articles[item["article_id"]] for item in ranked]
        documents = list(dict.fromkeys(source["doc_id"] for source in sources))
        return {
            "id": question["id"],
            "question": question["question"],
            "answer": "",
            "relevant_docs": documents,
            "relevant_articles": [source["article_reference"] for source in sources],
        }
