from __future__ import annotations

import hashlib
import re
from typing import Any

from .core import Config, normalize_text

CLAUSE_PATTERN = re.compile(r"(?<![\w/])(\d{1,3})[.)][ \t]+")
POINT_PATTERN = re.compile(r"(?<!\w)([a-zđ])\)[ \t]+", flags=re.IGNORECASE)
SENTENCE_END_PATTERN = re.compile(r"[.!?;](?:\s+|$)")
REFERENCE_PREFIX_PATTERN = re.compile(r"(?:điều|khoản|điểm)\s*$", flags=re.IGNORECASE)
REFERENCE_CONTEXT_CHARS = 20
CHUNK_ID_HEX_LENGTH = 24
CLAUSE_LABELS = [str(number) for number in range(1, 1000)]
POINT_LABELS = [
    "a",
    "b",
    "c",
    "d",
    "đ",
    "e",
    "g",
    "h",
    "i",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "q",
    "r",
    "s",
    "t",
    "u",
    "ư",
    "v",
    "x",
    "y",
]


def _ordered_spans(
    text: str, pattern: re.Pattern[str], labels: list[str]
) -> list[tuple[int, int]]:
    matches = [
        match
        for match in pattern.finditer(text)
        if not REFERENCE_PREFIX_PATTERN.search(
            text[max(0, match.start() - REFERENCE_CONTEXT_CHARS) : match.start()]
        )
    ]
    best: list[int] = []
    for first_index, match in enumerate(matches):
        if match.group(1).lower() != labels[0]:
            continue
        starts = [match.start()]
        expected = 1
        for candidate in matches[first_index + 1 :]:
            if expected == len(labels):
                break
            if candidate.group(1).lower() == labels[expected]:
                starts.append(candidate.start())
                expected += 1
        if len(starts) > len(best):
            best = starts
    starts = best if len(best) >= 2 else []
    return [
        (start, starts[index + 1] if index + 1 < len(starts) else len(text))
        for index, start in enumerate(starts)
    ]


def _hard_split(text: str, limit: int) -> list[str]:
    parts: list[str] = []
    remaining = text.strip()
    while len(remaining) > limit:
        split_at = remaining.rfind(" ", 0, limit + 1)
        if split_at <= 0:
            split_at = limit
        parts.append(remaining[:split_at].strip())
        remaining = remaining[split_at:].strip()
    if remaining:
        parts.append(remaining)
    return parts


def _split_sentences(text: str, limit: int) -> list[str]:
    boundaries = [match.end() for match in SENTENCE_END_PATTERN.finditer(text)]
    if not boundaries or boundaries[-1] != len(text):
        boundaries.append(len(text))

    sentences: list[str] = []
    start = 0
    for end in boundaries:
        sentence = text[start:end].strip()
        if sentence:
            sentences.extend(_hard_split(sentence, limit))
        start = end

    groups: list[str] = []
    current = ""
    for sentence in sentences:
        candidate = f"{current} {sentence}".strip()
        if current and len(candidate) > limit:
            groups.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        groups.append(current)
    return groups


def _split_oversized_unit(text: str, limit: int) -> list[str]:
    if len(text) <= limit:
        return [text.strip()]

    point_spans = _ordered_spans(text, POINT_PATTERN, POINT_LABELS)
    if len(point_spans) >= 2:
        prefix = text[: point_spans[0][0]].strip()
        parts: list[str] = []
        for index, (start, end) in enumerate(point_spans):
            point = text[start:end].strip()
            if index == 0 and prefix:
                point = f"{prefix}\n{point}"
            parts.extend(_split_sentences(point, limit))
        return parts
    return _split_sentences(text, limit)


def _group_units(units: list[str], target: int) -> list[str]:
    chunks: list[str] = []
    current = ""
    for unit in units:
        unit = unit.strip()
        if not unit:
            continue
        candidate = f"{current}\n{unit}".strip()
        if current and len(candidate) > target:
            chunks.append(current)
            current = unit
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def validate_articles(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    required = ("article_id", "doc_id", "article_number", "article_reference", "text")
    articles: list[dict[str, Any]] = []
    seen: set[str] = set()
    for position, source in enumerate(rows):
        if not isinstance(source, dict):
            raise TypeError(f"Article {position} is not an object")
        article = dict(source)
        for field in required:
            if not isinstance(article.get(field), str) or not article[field].strip():
                raise ValueError(f"Article {position} has no valid {field}")
        article_id = article["article_id"]
        if article_id in seen:
            raise ValueError(f"Duplicate article_id: {article_id}")
        seen.add(article_id)
        article["text"] = normalize_text(article["text"]).strip()
        articles.append(article)
    if not articles:
        raise ValueError("The corpus is empty")
    return articles


def split_article(article: dict[str, Any], config: Config) -> list[dict[str, Any]]:
    text = article["text"]
    if len(text) <= config.article_max_chars:
        bodies = [text]
    else:
        clause_spans = _ordered_spans(text, CLAUSE_PATTERN, CLAUSE_LABELS)
        if clause_spans:
            introduction = text[: clause_spans[0][0]].strip()
            units: list[str] = []
            for start, end in clause_spans:
                units.extend(
                    _split_oversized_unit(
                        text[start:end].strip(), config.structural_unit_max_chars
                    )
                )
            grouped = _group_units(units, config.chunk_target_chars)
            bodies = [f"{introduction}\n{body}".strip() for body in grouped]
        else:
            bodies = _split_sentences(text, config.structural_unit_max_chars)

    chunks: list[dict[str, Any]] = []
    for position, body in enumerate(bodies):
        digest = hashlib.sha256(
            f"{article['article_id']}\0{position}\0{body}".encode()
        ).hexdigest()[:CHUNK_ID_HEX_LENGTH]
        chunks.append(
            {
                "chunk_id": digest,
                "article_id": article["article_id"],
                "position": position,
                "text": body,
            }
        )
    return chunks


def make_chunks(articles: list[dict[str, Any]], config: Config) -> list[dict[str, Any]]:
    return [chunk for article in articles for chunk in split_article(article, config)]
