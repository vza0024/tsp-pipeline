"""Shared configuration and question-schema validation.

This module is the single source of truth for supported topics, visual objects,
actions, and colors. Runtime modules should import these values rather than
redefining them.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final, TypeAlias

RGB: TypeAlias = tuple[float, float, float]
Question: TypeAlias = dict[str, Any]

MAX_OBJECTS: Final = 21

TOPICS: Final = (
    "fractions",
    "ratios",
    "decimals",
    "percentages",
    "probability",
)

# Each object has four supported actions: 4 objects x 4 actions = 16 total.
OBJECT_ACTIONS: Final = {
    "sphere": ("bounce", "grouping", "split", "push_apart"),
    "tile": ("fill", "stamp", "rotate", "grow"),
    "cylinder": ("lid_drop", "spin", "knock_over", "shrink"),
    "book": ("open_book", "stack_books", "change_cover", "stand_up"),
}

# Actions that are conceptually appropriate for each question type.
BEST_ACTIONS: Final = {
    # Fractions
    "equivalent": {"grouping", "fill", "change_cover", "shrink"},
    "simplify": {"grouping", "knock_over", "shrink", "split"},
    "compare": {"fill", "stamp", "grow"},
    "add": {"fill", "stamp", "bounce", "stack_books"},
    "of": {"spin", "lid_drop", "open_book", "rotate"},
    # Ratios
    "ratio_simplify": {"grouping"},
    "ratio_equiv": {"grouping"},
    "scale_up": {"grow", "stamp", "fill", "bounce"},
    "part_whole": {"fill", "stamp", "lid_drop", "spin"},
    "unit_rate": {"open_book", "rotate", "spin", "stand_up"},
    # Decimals
    "to_fraction": {"fill", "stamp", "shrink", "knock_over"},
    "to_percent": {"grow", "spin", "change_cover", "rotate"},
    "dec_compare": {"grow", "stand_up", "fill", "stamp"},
    "dec_add": {"fill", "stamp"},
    "dec_of": {"spin", "lid_drop", "open_book"},
    # Percentages
    "of_number": {"spin", "lid_drop", "open_book", "rotate"},
    "pct_fraction": {"fill", "stamp", "shrink", "change_cover"},
    "to_decimal": {"shrink", "change_cover", "rotate", "spin"},
    "pct_change": {"push_apart", "knock_over", "split", "shrink"},
    "find_whole": {"fill", "stamp", "stack_books", "stand_up"},
    # Probability
    "simple": {"stamp", "fill", "lid_drop", "spin"},
    "complement": {"push_apart", "knock_over", "shrink", "split"},
    "express": {"change_cover", "rotate", "grow", "spin"},
    "expected": {"fill", "stamp", "bounce", "stack_books"},
    "prob_compare": {"grow", "stand_up", "fill", "stamp"},
}

COLORS: Final[dict[str, RGB]] = {
    "blue": (0.18, 0.52, 0.92),
    "orange": (0.95, 0.50, 0.05),
    "green": (0.15, 0.78, 0.38),
    "purple": (0.62, 0.20, 0.85),
    "red": (0.88, 0.18, 0.18),
    "teal": (0.05, 0.72, 0.68),
    "yellow": (0.92, 0.80, 0.10),
    "neutral": (0.82, 0.80, 0.76),
    "white": (0.95, 0.95, 0.95),
}

# (highlight color, secondary-group color, neutral color)
TYPE_COLORS: Final = {
    "ratio_simplify": ("blue", "orange", "neutral"),
    "ratio_equiv": ("blue", "orange", "neutral"),
    "scale_up": ("blue", "orange", "neutral"),
    "part_whole": ("green", "orange", "neutral"),
    "simplify": ("purple", "neutral", "neutral"),
    "equivalent": ("blue", "neutral", "neutral"),
    "compare": ("teal", "neutral", "neutral"),
    "dec_compare": ("teal", "neutral", "neutral"),
    "prob_compare": ("teal", "neutral", "neutral"),
    "complement": ("red", "neutral", "neutral"),
    "pct_change": ("red", "neutral", "neutral"),
    "simple": ("yellow", "neutral", "neutral"),
    "expected": ("yellow", "neutral", "neutral"),
}
DEFAULT_COLORS: Final = ("blue", "neutral", "neutral")

REQUIRED_KEYS: Final = (
    "topic",
    "qtype",
    "question",
    "answer",
    "object_num",
    "object_den",
    "group_a",
    "group_b",
    "working",
)


def get_colors(question_type: str) -> tuple[RGB, RGB, RGB]:
    """Return highlight, secondary, and neutral colors for a question type."""
    highlight, secondary, neutral = TYPE_COLORS.get(question_type, DEFAULT_COLORS)
    return COLORS[highlight], COLORS[secondary], COLORS[neutral]


def validate_question(question: Mapping[str, Any]) -> None:
    """Validate the shared question schema.

    Raises:
        ValueError: If a required field is absent or contains an invalid value.
    """
    missing = [key for key in REQUIRED_KEYS if key not in question]
    if missing:
        raise ValueError(f"Question is missing required keys: {', '.join(missing)}")

    topic = question["topic"]
    if topic not in TOPICS:
        raise ValueError(f"Unsupported topic: {topic!r}")

    question_type = question["qtype"]
    if question_type not in BEST_ACTIONS:
        raise ValueError(f"Unsupported question type: {question_type!r}")

    object_num = question["object_num"]
    object_den = question["object_den"]
    if not isinstance(object_num, int) or not isinstance(object_den, int):
        raise ValueError("Object counts must be integers")
    if not (1 <= object_num <= object_den <= MAX_OBJECTS):
        raise ValueError(
            f"Invalid object counts {object_num}/{object_den} for "
            f"{question['question']!r}; expected 1 <= numerator <= denominator "
            f"<= {MAX_OBJECTS}."
        )

    for group_key in ("group_a", "group_b"):
        group_value = question[group_key]
        if not isinstance(group_value, int) or group_value < 0:
            raise ValueError(f"{group_key} must be a non-negative integer")

    if question["group_a"] + question["group_b"] > object_den:
        raise ValueError("group_a + group_b cannot exceed object_den")

    if not isinstance(question["working"], Mapping):
        raise ValueError("working must be a mapping")
