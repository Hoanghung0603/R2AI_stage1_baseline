from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

LEGAL_TOKEN_PATTERN = re.compile(r"\d+/[\d/\w.\-]+|[^\W\d_]+|\d+", flags=re.UNICODE)


@dataclass(frozen=True)
class Config:
    article_max_chars: int = 4000
    chunk_target_chars: int = 1800
    structural_unit_max_chars: int = 3200

    embedding_model: str = "AITeamVN/Vietnamese_Embedding"
    embedding_revision: str = "main"
    embedding_max_tokens: int = 2048
    embedding_dtype: str = "float16"

    reranker_model: str = "AITeamVN/Vietnamese_Reranker"
    reranker_revision: str = "main"
    reranker_max_tokens: int = 512

    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    bm25_epsilon: float = 0.25
    top_per_retriever: int = 100
    rerank_candidates: int = 100
    rrf_c: float = 60.0
    bm25_weight: float = 1.0
    dense_weight: float = 1.0

    top_k: int = 3
    batch_size: int = 8
    device: str = "cuda"

    def __post_init__(self) -> None:
        positive_ints = (
            "article_max_chars",
            "chunk_target_chars",
            "structural_unit_max_chars",
            "embedding_max_tokens",
            "reranker_max_tokens",
            "top_per_retriever",
            "rerank_candidates",
            "top_k",
            "batch_size",
        )
        for name in positive_ints:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.embedding_dtype not in {"float16", "float32"}:
            raise ValueError("embedding_dtype must be float16 or float32")
        if not (self.bm25_k1 > 0 and 0 <= self.bm25_b <= 1):
            raise ValueError("Invalid BM25 parameters")
        if any(
            not math.isfinite(value) or value <= 0
            for value in (self.rrf_c, self.bm25_weight, self.dense_weight)
        ):
            raise ValueError("RRF parameters must be finite and positive")

    @classmethod
    def load(cls, path: str | Path) -> Config:
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def legal_tokenize(text: str) -> list[str]:
    return LEGAL_TOKEN_PATTERN.findall(normalize_text(text).lower())


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise TypeError(f"Expected an object at {path}:{line_number}")
            rows.append(row)
    return rows


def write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(destination)


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(destination)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rrf(
    rankings: list[list[tuple[str, float]]],
    weights: list[float],
    c: float,
) -> list[tuple[str, float]]:
    if len(rankings) != len(weights):
        raise ValueError("Each ranking must have one weight")
    scores: defaultdict[str, float] = defaultdict(float)
    for ranking, weight in zip(rankings, weights):
        seen: set[str] = set()
        for rank, (item_id, _) in enumerate(ranking, 1):
            if item_id in seen:
                raise ValueError("A ranking contains duplicate IDs")
            seen.add(item_id)
            scores[item_id] += weight / (c + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))
