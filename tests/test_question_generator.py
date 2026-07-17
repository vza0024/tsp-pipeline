"""Tests for deterministic question construction and schema validation."""

from __future__ import annotations

import random
import unittest

from config import TOPICS, validate_question
from question_generator import generate

EXPECTED_TYPES = {
    "fractions": {"simplify", "equivalent", "compare", "add", "of"},
    "ratios": {
        "ratio_simplify",
        "ratio_equiv",
        "scale_up",
        "part_whole",
        "unit_rate",
    },
    "decimals": {"to_fraction", "to_percent", "dec_compare", "dec_add", "dec_of"},
    "percentages": {
        "of_number",
        "pct_fraction",
        "to_decimal",
        "pct_change",
        "find_whole",
    },
    "probability": {"simple", "complement", "express", "expected", "prob_compare"},
}


class QuestionGeneratorTests(unittest.TestCase):
    def setUp(self) -> None:
        random.seed(2026)

    def test_all_topics_generate_valid_questions(self) -> None:
        for topic in TOPICS:
            with self.subTest(topic=topic):
                for _ in range(500):
                    question = generate(topic)
                    validate_question(question)
                    self.assertEqual(question["topic"], topic)
                    self.assertNotIn(None, question["working"].values())

    def test_all_question_types_are_reachable(self) -> None:
        for topic in TOPICS:
            observed = {generate(topic)["qtype"] for _ in range(1_000)}
            self.assertEqual(observed, EXPECTED_TYPES[topic])

    def test_unknown_topic_has_clear_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported topic"):
            generate("algebra")


if __name__ == "__main__":
    unittest.main()
