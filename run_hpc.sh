#!/usr/bin/env bash
# Launch the Think-See-Prove Flask server in a tmux session.

set -euo pipefail

SESSION="${TSP_TMUX_SESSION:-think-see-prove}"
PROJECT_DIR="${TSP_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is required but was not found." >&2
  exit 1
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  tmux kill-session -t "$SESSION"
fi

tmux new-session -d -s "$SESSION" \
  "cd '$PROJECT_DIR' && '$PYTHON_BIN' backend/server.py"

echo "Started tmux session: $SESSION"
echo "Project directory: $PROJECT_DIR"
echo "Attach with: tmux attach -t $SESSION"
echo "Stop with:   tmux kill-session -t $SESSION"
