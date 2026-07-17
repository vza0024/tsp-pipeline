# Think, See, Prove

A three-step embodied math-tutoring demo that combines:

- Python-generated, verified math questions
- LLaMA 3.1 8B narration
- A Flask web interface
- CoppeliaSim visualizations

The runtime sequence is:

1. **Think** — introduce the concept and strategy
2. **See** — show the quantities and calculation
3. **Prove** — animate and confirm the result

> This repository contains the interactive demonstration pipeline. The separate
> baseline-versus-constrained experiment runner and paper-evaluation scripts are
> not included here.

## Project structure

```text
pipeline/
├── backend/                      Python application (run and imported from here)
│   ├── config.py                 Shared topics, actions, colors, and validation
│   ├── question_generator.py     Verified question generation for 25 subtypes
│   ├── llm_narrator.py           Prompt construction and model inference
│   ├── server.py                 Thread-safe Flask orchestration server
│   └── sim/
│       ├── coppeliasim_renderer.py   Object placement and animation
│       └── display_utils.py          Fraction and ratio labels
├── frontend/                     Web interface served by the backend
│   ├── index.html                Lesson interface
│   └── topics.html               Topic picker
├── tests/
│   ├── test_question_generator.py
│   ├── test_narrator.py
│   └── test_server_state.py
├── requirements.txt
├── requirements-dev.txt
├── .env.example
└── run_hpc.sh
```

`objects.zip` is retained only as the original object-animation reference archive.
It is not imported at runtime.

## Requirements

- Python 3.10 or newer
- NVIDIA GPU recommended for LLaMA 3.1 8B
- Access to `meta-llama/Meta-Llama-3.1-8B-Instruct`
- CoppeliaSim for the 3D visualization layer

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Authenticate with Hugging Face if the model is not already stored locally.

## Configuration

Copy the example file and edit values as needed:

```bash
cp .env.example .env
```

The application reads environment variables directly. Common settings are:

| Variable | Default | Purpose |
|---|---:|---|
| `TSP_MODEL_ID` | `meta-llama/Meta-Llama-3.1-8B-Instruct` | Hugging Face model ID |
| `TSP_MODEL_PATH` | unset | Explicit local model directory |
| `TSP_MAX_NEW_TOKENS` | `350` | Generation limit per step |
| `TSP_TEMPERATURE` | `0.3` | Runtime demo sampling temperature |
| `TSP_TOP_P` | `0.9` | Nucleus sampling threshold |
| `PRELOAD_MODEL` | `1` | Load model when the server starts |
| `COPPELIASIM_HOST` | `localhost` | CoppeliaSim ZMQ host |
| `COPPELIASIM_ZMQ_PORT` | `23000` | CoppeliaSim ZMQ port |
| `TSP_HOST` | `0.0.0.0` | Flask bind host |
| `TSP_PORT` | `5000` | Flask port |
| `LOG_LEVEL` | `INFO` | Python logging level |

Load `.env` through your shell, process manager, or scheduler. For example:

```bash
set -a
source .env
set +a
python backend/server.py
```

## Run locally

1. Start CoppeliaSim and its ZMQ remote API service.
2. Start the application:

```bash
python backend/server.py
```

3. Open `http://localhost:5000`.

The app automatically falls back to narration-only mode when CoppeliaSim is not
reachable. It also returns deterministic fallback narration when the language
model cannot be loaded.

## Run in tmux or on an HPC node

```bash
./run_hpc.sh
```

Optional overrides:

```bash
TSP_PROJECT_DIR=/path/to/pipeline \
TSP_TMUX_SESSION=tsp-demo \
./run_hpc.sh
```

## Tests

Install development dependencies and run:

```bash
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -t . -v
ruff check .
ruff format --check .
```

The unit tests do not load the LLaMA model or require CoppeliaSim.

## Adding a question type

1. Add the question type and appropriate actions to `BEST_ACTIONS` in `backend/config.py`.
2. Generate its verified values in `backend/question_generator.py`.
3. Add Think, See, and Prove facts in `backend/llm_narrator.py`.
4. Add or reuse a compatible visualization action in
   `backend/sim/coppeliasim_renderer.py`.
5. Extend the tests.

## Safety and reliability notes

- Mathematical values are computed in Python before narration.
- Model output is escaped before it is returned to the browser; only
  `<strong>` formatting is preserved.
- Shared lesson state is protected by a lock so concurrent polling and lesson
  workers do not mutate the same data unsafely.
- Starting a new question cancels the previous worker through a lesson ID.
