"""Flask orchestration server for the Think-See-Prove tutor.

Request flow:
1. ``/question`` creates a question and starts a lesson worker.
2. The worker generates narration for each step and optionally drives CoppeliaSim.
3. ``/step`` returns a thread-safe snapshot for the browser.
4. ``/advance`` lets the learner move from Think to See to Prove.
5. Starting another question cancels the previous lesson by changing its ID.
"""

from __future__ import annotations

import logging
import os
import random
import socket
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request, send_from_directory

from config import BEST_ACTIONS, OBJECT_ACTIONS, TOPICS, get_colors
from llm_narrator import _get_pipeline, generate_step
from question_generator import generate
from sim import coppeliasim_renderer as renderer

LOGGER = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR.parent / "frontend"
TOTAL_STEPS = 3

SIM_HOST = os.getenv("COPPELIASIM_HOST", "localhost")
SIM_PORT = int(os.getenv("COPPELIASIM_ZMQ_PORT", "23000"))
SERVER_HOST = os.getenv("TSP_HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("TSP_PORT", "5000"))


@dataclass
class LessonState:
    """Mutable lesson state protected by a re-entrant lock."""

    question: dict[str, Any] | None = None
    obj: str | None = None
    action: str | None = None
    steps: list[dict[str, Any]] = field(default_factory=list)
    running: bool = False
    step_index: int = 0
    previous_steps: list[dict[str, Any]] = field(default_factory=list)
    sim_connected: bool = False
    lesson_id: int = 0
    advance_to: int = 0
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def begin(self, question: dict[str, Any], obj: str, action: str) -> int:
        """Cancel any previous lesson and initialize a new one."""
        with self._lock:
            self.lesson_id += 1
            self.question = question
            self.obj = obj
            self.action = action
            self.steps = []
            self.previous_steps = []
            self.running = True
            self.step_index = 0
            self.advance_to = 1  # Think starts automatically.
            self.sim_connected = False
            return self.lesson_id

    def is_cancelled(self, lesson_id: int) -> bool:
        with self._lock:
            return self.lesson_id != lesson_id

    def previous_for(self, lesson_id: int) -> list[dict[str, Any]]:
        with self._lock:
            if self.lesson_id != lesson_id:
                return []
            return list(self.previous_steps)

    def append_step(self, lesson_id: int, step: dict[str, Any]) -> bool:
        """Append a step only if the worker still owns the active lesson."""
        with self._lock:
            if self.lesson_id != lesson_id:
                return False
            self.steps.append(step)
            self.previous_steps.append(step)
            self.step_index = int(step["step_index"])
            return True

    def set_sim_connected(self, lesson_id: int, connected: bool) -> None:
        with self._lock:
            if self.lesson_id == lesson_id:
                self.sim_connected = connected

    def advance(self) -> int:
        with self._lock:
            self.advance_to = min(TOTAL_STEPS, self.advance_to + 1)
            return self.advance_to

    def can_advance_to(self, lesson_id: int, step_num: int) -> bool:
        with self._lock:
            return self.lesson_id == lesson_id and self.advance_to >= step_num

    def finish(self, lesson_id: int) -> None:
        with self._lock:
            if self.lesson_id == lesson_id:
                self.running = False
                # Preserve the original UI behavior: a failed worker still allows
                # the learner to move to the next question.
                self.step_index = max(self.step_index, TOTAL_STEPS)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            question = self.question
            return {
                "step_index": self.step_index,
                "total_steps": TOTAL_STEPS,
                "running": self.running,
                "steps": list(self.steps),
                "obj": self.obj,
                "action": self.action,
                "sim_connected": self.sim_connected,
                "object_num": question["object_num"] if question else 0,
                "object_den": question["object_den"] if question else 0,
                "group_a": question["group_a"] if question else 0,
                "advance_to": self.advance_to,
            }


STATE = LessonState()


def _pick_action(question_type: str) -> tuple[str, str]:
    """Choose an object/action pair that fits the question type."""
    preferred = BEST_ACTIONS.get(question_type, set())
    candidates = [
        (obj, action)
        for obj, actions in OBJECT_ACTIONS.items()
        for action in actions
        if action in preferred
    ]
    if not candidates:
        candidates = [
            (obj, action) for obj, actions in OBJECT_ACTIONS.items() for action in actions
        ]
    return random.choice(candidates)


def _check_sim() -> bool:
    """Return whether the CoppeliaSim ZMQ endpoint is reachable."""
    try:
        with socket.create_connection((SIM_HOST, SIM_PORT), timeout=2):
            return True
    except OSError:
        return False


def _wait_for_advance(step_num: int, lesson_id: int) -> bool:
    """Block until the learner advances or the lesson is cancelled."""
    while not STATE.can_advance_to(lesson_id, step_num):
        if STATE.is_cancelled(lesson_id):
            return False
        time.sleep(0.25)
    return True


def _set_step(
    step_num: int,
    lesson_id: int,
    question: dict[str, Any],
    obj: str,
    action: str,
) -> None:
    """Generate and store one narration step for the active lesson."""
    if STATE.is_cancelled(lesson_id):
        return

    step = generate_step(
        step_num,
        question,
        obj,
        action,
        previous_steps=STATE.previous_for(lesson_id),
    )
    STATE.append_step(lesson_id, step)


def _label_values(question: dict[str, Any]) -> tuple[Any, Any, Any, Any]:
    """Return optional left/right values for the simulator's math label."""
    question_type = question["qtype"]
    working = question["working"]
    object_num = question["object_num"]
    object_den = question["object_den"]

    if question_type in {"simplify", "equivalent"}:
        return (
            working.get("ne"),
            working.get("de"),
            working.get("sn"),
            working.get("sd"),
        )
    if question_type in {"ratio_simplify", "ratio_equiv"}:
        return (
            working.get("a"),
            working.get("b"),
            working.get("sa"),
            working.get("sb"),
        )
    if question_type == "add":
        return (
            working.get("total_n"),
            working.get("den"),
            working.get("sn"),
            working.get("sd"),
        )
    if question_type in {"simple", "complement"}:
        return object_num, object_den, working.get("sn"), working.get("sd")
    if question_type in {"pct_fraction", "to_fraction"}:
        return working.get("sn"), working.get("sd"), None, None
    return None, None, None, None


def _run_without_sim(question: dict[str, Any], obj: str, action: str, lesson_id: int) -> None:
    for step_num in range(1, TOTAL_STEPS + 1):
        if not _wait_for_advance(step_num, lesson_id):
            return
        _set_step(step_num, lesson_id, question, obj, action)


def _run_lesson(question: dict[str, Any], obj: str, action: str, lesson_id: int) -> None:
    """Run one lesson with learner-controlled pacing."""
    sim_available = _check_sim()
    STATE.set_sim_connected(lesson_id, sim_available)
    LOGGER.info("Lesson %s: simulator available=%s", lesson_id, sim_available)

    try:
        if not sim_available:
            _run_without_sim(question, obj, action, lesson_id)
            return

        color_a, color_b, color_neutral = get_colors(question["qtype"])
        lhs_num, lhs_den, rhs_num, rhs_den = _label_values(question)

        renderer.run(
            obj=obj,
            action=action,
            numerator=question["object_num"],
            denominator=question["object_den"],
            set_step=lambda n: _set_step(n, lesson_id, question, obj, action),
            col_a=color_a,
            col_b=color_b,
            col_n=color_neutral,
            group_a=question["group_a"],
            group_b=question["group_b"],
            lhs_num=lhs_num,
            lhs_den=lhs_den,
            rhs_num=rhs_num,
            rhs_den=rhs_den,
            cancel_check=lambda: STATE.is_cancelled(lesson_id),
            qtype=question["qtype"],
            wait_for_advance=lambda n: _wait_for_advance(n, lesson_id),
        )
    except Exception:
        LOGGER.exception("Lesson %s failed; finishing with narration-only mode", lesson_id)
        snapshot = STATE.snapshot()
        for step_num in range(snapshot["step_index"] + 1, TOTAL_STEPS + 1):
            if STATE.is_cancelled(lesson_id):
                break
            try:
                _set_step(step_num, lesson_id, question, obj, action)
            except Exception:
                LOGGER.exception(
                    "Lesson %s: failed to generate fallback step %s",
                    lesson_id,
                    step_num,
                )
    finally:
        if not STATE.is_cancelled(lesson_id):
            STATE.finish(lesson_id)


def create_app() -> Flask:
    """Create the Flask application."""
    app = Flask(__name__, static_folder=str(FRONTEND_DIR))
    app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

    @app.get("/")
    def index():
        response = send_from_directory(FRONTEND_DIR, "topics.html")
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        return response

    @app.get("/learn")
    def learn():
        response = send_from_directory(FRONTEND_DIR, "index.html")
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        return response

    @app.get("/question")
    def question_route():
        topic = request.args.get("topic", "fractions")
        if topic not in TOPICS:
            return (
                jsonify(
                    {
                        "error": f"Unsupported topic: {topic}",
                        "supported_topics": TOPICS,
                    }
                ),
                400,
            )

        question = generate(topic)
        obj, action = _pick_action(question["qtype"])
        lesson_id = STATE.begin(question, obj, action)

        worker = threading.Thread(
            target=_run_lesson,
            args=(question, obj, action, lesson_id),
            name=f"lesson-{lesson_id}",
            daemon=True,
        )
        worker.start()

        return jsonify(
            {
                "question": question["question"],
                "topic": question["topic"],
                "qtype": question["qtype"],
                "obj": obj,
                "action": action,
            }
        )

    @app.route("/advance", methods=["GET", "POST"])
    def advance_route():
        return jsonify({"advance_to": STATE.advance()})

    @app.get("/step")
    def step_route():
        return jsonify(STATE.snapshot())

    @app.get("/health")
    def health_route():
        return jsonify(
            {
                "status": "ok",
                "sim_reachable": _check_sim(),
                "active_lesson": STATE.snapshot()["running"],
            }
        )

    return app


app = create_app()


def main() -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if os.getenv("PRELOAD_MODEL", "1") == "1":
        LOGGER.info("Pre-loading the language model")
        _get_pipeline()

    LOGGER.info("Starting Think-See-Prove server at http://%s:%s", SERVER_HOST, SERVER_PORT)
    app.run(host=SERVER_HOST, port=SERVER_PORT, debug=False, threaded=True)


if __name__ == "__main__":
    main()
