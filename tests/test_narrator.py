"""Unit tests for narration parsing, sanitization, and fallback behavior."""

from __future__ import annotations

import random
import unittest
from unittest.mock import patch

import llm_narrator
from question_generator import generate


class NarratorTests(unittest.TestCase):
    def setUp(self) -> None:
        random.seed(7)
        self.question = generate("fractions")

    def test_json_extraction_handles_fenced_output(self) -> None:
        raw = '```json\n{"instruction":"Think","explanation":"Explain"}\n```'
        parsed = llm_narrator._extract_json_object(raw)
        self.assertEqual(parsed["instruction"], "Think")

    def test_sanitizer_preserves_strong_only(self) -> None:
        value = '<strong>4</strong><script>alert("x")</script>'
        cleaned = llm_narrator._sanitize_rich_text(value)
        self.assertIn("<strong>4</strong>", cleaned)
        self.assertNotIn("<script>", cleaned)
        self.assertIn("&lt;script&gt;", cleaned)

    def test_fallback_generates_all_three_steps_without_model(self) -> None:
        previous_steps: list[dict[str, object]] = []
        with patch.object(llm_narrator, "_get_pipeline", return_value=None):
            for step_num in (1, 2, 3):
                step = llm_narrator.generate_step(
                    step_num,
                    self.question,
                    "tile",
                    "fill",
                    previous_steps,
                )
                self.assertEqual(step["step_index"], step_num)
                self.assertTrue(step["instruction"])
                self.assertTrue(step["explanation"])
                previous_steps.append(step)

    def test_invalid_step_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "step_num"):
            llm_narrator.generate_step(4, self.question, "tile", "fill")


if __name__ == "__main__":
    unittest.main()
