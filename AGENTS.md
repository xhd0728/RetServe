# AGENTS.md

## Project Overview

RetServe is a compact retrieval service. It builds embeddings from a JSONL corpus,
stores vectors in `.npy` files, builds a FAISS index, and serves retrieval through
FastAPI.

Primary entry points:

- `embed.py`: generate embeddings from the configured JSONL corpus.
- `index.py`: build a FAISS index from saved embeddings.
- `ret_serve.py`: run the FastAPI retrieval service.

## Setup

Use uv by default:

```bash
uv sync
```

Use pip only as a fallback:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

FAISS is selected by platform. Linux x86_64 installs `faiss-gpu-cu12`; macOS,
Windows, and non-x86 Linux install `faiss-cpu`.

## Code Style

- Use Ruff for linting, import sorting, and formatting.
- Keep Python comments and docstrings in English.
- Prefer short comments that explain compatibility, failure handling, or
  non-obvious behavior.
- Do not add comments that simply restate the next line of code.
- Keep public API fields, CLI commands, and YAML config keys backward compatible.

## Validation

Run these checks after code changes:

```bash
uvx ruff check .
uvx ruff format --check --diff .
python -m unittest discover -s test -p 'test_service_units.py'
python -m compileall -q src test
git diff --check
```

For dependency changes, also run:

```bash
uv lock
```

## Data And Secrets

- Do not commit private endpoints, API keys, real service IPs, or private corpus
  content.
- Use placeholders such as `http://localhost:8000/v1`,
  `<your-openai-compatible-endpoint>/v1`, and `<api-key>`.
- Qwen3-Embedding model names may remain visible.
- MIT license metadata and the existing author name may remain visible.
