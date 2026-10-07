from __future__ import annotations

import inspect
from typing import Any

import numpy as np

from .core import Config


class DenseEncoder:
    def __init__(self, config: Config):
        import torch
        from sentence_transformers import SentenceTransformer

        self.config = config
        dtype = getattr(torch, config.embedding_dtype)
        self.model = SentenceTransformer(
            config.embedding_model,
            revision=config.embedding_revision,
            device=config.device,
            trust_remote_code=False,
            model_kwargs={"torch_dtype": dtype},
        )
        self.model.max_seq_length = config.embedding_max_tokens

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(
            texts,
            batch_size=self.config.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or len(vectors) != len(texts):
            raise ValueError("The embedding model returned an invalid array")
        if not np.isfinite(vectors).all():
            raise ValueError("The embedding model returned non-finite values")
        return vectors


class CrossEncoderReranker:
    def __init__(self, config: Config):
        import torch
        from sentence_transformers import CrossEncoder

        self.config = config
        dtype = getattr(torch, config.embedding_dtype)
        dtype_parameter = (
            "model_kwargs"
            if "model_kwargs" in inspect.signature(CrossEncoder.__init__).parameters
            else "automodel_args"
        )
        self.model = CrossEncoder(
            config.reranker_model,
            revision=config.reranker_revision,
            device=config.device,
            max_length=config.reranker_max_tokens,
            trust_remote_code=False,
            **{dtype_parameter: {"torch_dtype": dtype}},
        )
        self.tokenizer = self.model.tokenizer

    def _windows(self, query: str, passage: str) -> list[str]:
        query_ids = self.tokenizer(query, add_special_tokens=False, truncation=False)[
            "input_ids"
        ]
        pair_special_tokens = self.tokenizer.num_special_tokens_to_add(pair=True)
        budget = self.config.reranker_max_tokens - len(query_ids) - pair_special_tokens
        if budget <= 0:
            raise ValueError("The question exceeds the reranker input limit")

        encoded = self.tokenizer(
            passage,
            add_special_tokens=False,
            truncation=False,
            return_offsets_mapping=True,
        )
        offsets = encoded.get("offset_mapping")
        if offsets is None:
            token_ids = encoded["input_ids"]
            return [
                self.tokenizer.decode(token_ids[start : start + budget])
                for start in range(0, len(token_ids), budget)
            ] or [""]

        windows: list[str] = []
        for start in range(0, len(offsets), budget):
            window_offsets = offsets[start : start + budget]
            if window_offsets:
                windows.append(passage[window_offsets[0][0] : window_offsets[-1][1]])
        return windows or [""]

    def rank(
        self, query: str, candidates: list[dict[str, Any]]
    ) -> list[tuple[str, float]]:
        pairs: list[tuple[str, str]] = []
        owners: list[str] = []
        for candidate in candidates:
            for window in self._windows(query, candidate["text"]):
                pairs.append((query, window))
                owners.append(candidate["chunk_id"])
        if not pairs:
            return []

        scores = np.asarray(
            self.model.predict(
                pairs,
                batch_size=self.config.batch_size,
                show_progress_bar=False,
            )
        ).reshape(-1)
        if len(scores) != len(pairs) or not np.isfinite(scores).all():
            raise ValueError("The reranker returned invalid scores")

        best: dict[str, float] = {}
        for chunk_id, score in zip(owners, scores):
            best[chunk_id] = max(best.get(chunk_id, -float("inf")), float(score))
        return sorted(best.items(), key=lambda item: (-item[1], item[0]))
