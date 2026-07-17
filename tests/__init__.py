"""Test package bootstrap.

Adds the ``backend`` directory to ``sys.path`` so tests can import the
application modules (``config``, ``question_generator``, ``llm_narrator``,
``server``, ``sim``) by their top-level names, matching how the server runs.
"""

import sys
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))
