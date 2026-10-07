from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from baseline.chunking import split_article
from baseline.core import Config, legal_tokenize, rrf, write_jsonl
from baseline.index import build_index
from baseline.pipeline import RetrievalPipeline, _top_indices


class FakeEncoder:
    def encode(self, texts):
        vectors = []
        for text in texts:
            row = np.asarray(
                [
                    text.lower().count("hồ sơ") + 0.1,
                    text.lower().count("doanh nghiệp") + 0.1,
                    text.lower().count("thuế") + 0.1,
                ],
                dtype=np.float32,
            )
            vectors.append(row / np.linalg.norm(row))
        return np.asarray(vectors)


class FakeReranker:
    def rank(self, query, candidates):
        return sorted(
            [
                (candidate["chunk_id"], float(candidate["text"].lower().count("hồ sơ")))
                for candidate in candidates
            ],
            key=lambda item: (-item[1], item[0]),
        )


def article(article_id, text, number="1"):
    doc_id = f"{article_id}/2026/QH|Luật {article_id}"
    return {
        "article_id": article_id,
        "doc_id": doc_id,
        "article_number": number,
        "article_reference": f"{doc_id}|Điều {number}",
        "text": text,
    }


class RetrievalTests(unittest.TestCase):
    def test_legal_identifier_is_one_token(self):
        self.assertIn("22/2023/qh15", legal_tokenize("Luật số 22/2023/QH15"))

    def test_rrf_rewards_shared_results(self):
        ranking = rrf(
            [[("a", 2.0), ("b", 1.0)], [("b", 9.0), ("c", 8.0)]],
            [1.0, 1.0],
            60.0,
        )
        self.assertEqual(ranking[0][0], "b")

    def test_top_k_ties_are_deterministic_and_zero_bm25_scores_are_ignored(self):
        scores = np.asarray([1.0, 1.0, 1.0, 0.0])
        self.assertEqual(_top_indices(scores, 2), [0, 1])
        self.assertEqual(_top_indices(np.zeros(4), 2, positive_only=True), [])

    def test_short_article_stays_whole(self):
        config = Config(article_max_chars=200)
        source = article("A", "Điều 1. Nội dung\n1. Quy định thứ nhất.")
        chunks = split_article(source, config)
        self.assertEqual([chunk["text"] for chunk in chunks], [source["text"]])

    def test_long_article_keeps_heading_and_introduction(self):
        config = Config(
            article_max_chars=80,
            chunk_target_chars=55,
            structural_unit_max_chars=70,
        )
        prefix = "Điều 1. Nghĩa vụ\nDoanh nghiệp có các nghĩa vụ sau:"
        text = prefix + "\n1. " + "Nghĩa vụ một. " * 4 + "\n2. " + "Nghĩa vụ hai. " * 4
        chunks = split_article(article("A", text), config)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk["text"].startswith(prefix) for chunk in chunks))

    def test_inline_clauses_are_detected_but_references_are_not_boundaries(self):
        config = Config(
            article_max_chars=80,
            chunk_target_chars=60,
            structural_unit_max_chars=100,
        )
        prefix = "Điều 1. Nghĩa vụ Doanh nghiệp thực hiện như sau:"
        text = (
            prefix
            + " 1. Nghĩa vụ thứ nhất được quy định theo khoản 2 Điều 5."
            + " 2. Nghĩa vụ thứ hai áp dụng cho doanh nghiệp."
            + " 3. Nghĩa vụ thứ ba áp dụng khi có yêu cầu."
        )
        chunks = split_article(article("A", text), config)
        self.assertEqual(len(chunks), 3)
        self.assertTrue(all(chunk["text"].startswith(prefix) for chunk in chunks))

    def test_maxp_keeps_one_chunk_per_article_and_top_three(self):
        rows = [
            article("A", "Điều 1. Hồ sơ\n1. Hồ sơ A.\n2. Hồ sơ bổ sung."),
            article("B", "Điều 1. Hồ sơ B."),
            article("C", "Điều 1. Nghĩa vụ doanh nghiệp."),
            article("D", "Điều 1. Thuế."),
        ]
        config = Config(
            article_max_chars=25,
            chunk_target_chars=15,
            structural_unit_max_chars=20,
            top_per_retriever=4,
            rerank_candidates=4,
            top_k=3,
            device="cpu",
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_jsonl(root / "articles.jsonl", rows)
            build_index(root / "articles.jsonl", root / "index", config, FakeEncoder())
            pipeline = RetrievalPipeline(
                root / "index", config, encoder=FakeEncoder(), reranker=FakeReranker()
            )
            result = pipeline.predict({"id": 1, "question": "hồ sơ doanh nghiệp"})
            self.assertLessEqual(len(result["relevant_articles"]), 3)
            self.assertEqual(
                len(result["relevant_articles"]), len(set(result["relevant_articles"]))
            )
            self.assertEqual(result["answer"], "")


if __name__ == "__main__":
    unittest.main()
