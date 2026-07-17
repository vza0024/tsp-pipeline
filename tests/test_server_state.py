"""Tests for thread-safe lesson state and object/action selection."""

from __future__ import annotations

import unittest

from config import BEST_ACTIONS, OBJECT_ACTIONS
from server import LessonState, _pick_action


class LessonStateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.state = LessonState()
        self.question = {
            "topic": "fractions",
            "qtype": "compare",
            "question": "Which is bigger?",
            "answer": "4/5",
            "object_num": 4,
            "object_den": 5,
            "group_a": 4,
            "group_b": 1,
            "working": {},
        }

    def test_begin_resets_state_and_cancels_previous_lesson(self) -> None:
        first_id = self.state.begin(self.question, "tile", "fill")
        second_id = self.state.begin(self.question, "sphere", "bounce")

        self.assertNotEqual(first_id, second_id)
        self.assertTrue(self.state.is_cancelled(first_id))
        self.assertFalse(self.state.is_cancelled(second_id))
        self.assertEqual(self.state.snapshot()["obj"], "sphere")

    def test_append_step_ignores_cancelled_worker(self) -> None:
        active_id = self.state.begin(self.question, "tile", "fill")
        stale_id = active_id - 1
        step = {"step_index": 1, "instruction": "Think", "explanation": "Explain"}

        self.assertFalse(self.state.append_step(stale_id, step))
        self.assertTrue(self.state.append_step(active_id, step))
        self.assertEqual(len(self.state.snapshot()["steps"]), 1)

    def test_advance_is_capped_at_three(self) -> None:
        self.state.begin(self.question, "tile", "fill")
        for _ in range(10):
            self.state.advance()
        self.assertEqual(self.state.snapshot()["advance_to"], 3)


class ActionSelectionTests(unittest.TestCase):
    def test_selected_action_is_supported_and_preferred(self) -> None:
        for question_type, preferred_actions in BEST_ACTIONS.items():
            obj, action = _pick_action(question_type)
            self.assertIn(obj, OBJECT_ACTIONS)
            self.assertIn(action, OBJECT_ACTIONS[obj])
            self.assertIn(action, preferred_actions)


if __name__ == "__main__":
    unittest.main()
