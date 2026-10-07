from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .core import Config, read_json, write_json
from .index import build_index
from .pipeline import RetrievalPipeline


def load_questions(path: str | Path) -> list[dict[str, Any]]:
    value = read_json(path)
    if not isinstance(value, list) or not value:
        raise ValueError("Questions must be a non-empty JSON array")
    questions: list[dict[str, Any]] = []
    seen: set[tuple[type[Any], Any]] = set()
    for position, row in enumerate(value):
        if not isinstance(row, dict) or type(row.get("id")) not in {int, str}:
            raise ValueError(f"Question {position} has no valid id")
        if not isinstance(row.get("question"), str) or not row["question"].strip():
            raise ValueError(f"Question {position} has no text")
        key = (type(row["id"]), row["id"])
        if key in seen:
            raise ValueError(f"Duplicate question id: {row['id']}")
        seen.add(key)
        questions.append({"id": row["id"], "question": row["question"]})
    return questions


def main() -> None:
    parser = argparse.ArgumentParser(description="Vietnamese legal retrieval")
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build", help="Build the BM25 and dense indexes")
    build.add_argument("--corpus", required=True)
    build.add_argument("--index", required=True)
    build.add_argument("--config", default="config.json")

    run = commands.add_parser("run", help="Retrieve legal articles for questions")
    run.add_argument("--questions", required=True)
    run.add_argument("--index", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--config", default="config.json")

    args = parser.parse_args()
    config = Config.load(args.config)
    if args.command == "build":
        manifest = build_index(args.corpus, args.index, config)
        print(f"Built {manifest['chunks']} chunks from {manifest['articles']} articles")
        return

    questions = load_questions(args.questions)
    pipeline = RetrievalPipeline(args.index, config)
    results = []
    for position, question in enumerate(questions, 1):
        results.append(pipeline.predict(question))
        print(f"Processed {position}/{len(questions)}", flush=True)
    write_json(args.output, results)


if __name__ == "__main__":
    main()
