<p align="center">
  <img src="assets/logo.png" alt="RetServe logo" width="180">
</p>

# RetServe

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Python 3.10-3.12](https://img.shields.io/badge/python-3.10--3.12-blue.svg)
![FastAPI](https://img.shields.io/badge/fastapi-0.100-green.svg)
![FAISS](https://img.shields.io/badge/FAISS-vector_search-informational.svg)
![Embeddings](https://img.shields.io/badge/embeddings-OpenAI--compatible-purple.svg)
![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)

**RetServe** is a compact retrieval service that turns a JSONL corpus into a FAISS-backed HTTP search API using any OpenAI-compatible embeddings endpoint.

[English](README.md) | [简体中文](README.zh.md)

```text
JSONL corpus -> /v1/embeddings -> .npy vectors -> FAISS index -> /search
```

## Why RetServe

- **Simple pipeline**: one embedding protocol for both offline encoding and online search.
- **Large-corpus friendly**: streaming JSONL reads, batched embedding requests, and memory-mapped index builds.
- **Fast serving path**: FastAPI API surface, FAISS search, optional GPU index loading, and configurable concurrency.
- **Typed configuration**: YAML config files are parsed into Pydantic models instead of ad hoc dictionaries.
- **Portable endpoint support**: works with vLLM, One API, New API, or any `/v1/embeddings` compatible service.

## Installation

```bash
uv sync
```

Or with regular pip as a fallback:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

Keep real API keys in environment variables:

```bash
export RET_SERVE_EMBED_API_KEY="sk-..."
```

## Corpus Format

RetServe expects JSONL, one document per line. The `contents` field is embedded and later split into `title` and `text`.

```jsonl
{"id":"1","contents":"Liu Bei\nLiu Bei was the founding emperor of Shu Han."}
{"id":"2","contents":"Zhuge Liang\nZhuge Liang assisted Liu Bei and is known for strategy."}
```

The JSONL row order must stay unchanged after embedding generation because FAISS IDs map to row positions.

## Quick Start

### 1. Configure embeddings

Edit `config/embed.yaml` and `config/serve.yaml`:

```yaml
embedding:
  url: "<your-openai-compatible-endpoint>/v1"
  model: "your-embedding-model"
  api_key_env: "RET_SERVE_EMBED_API_KEY"
  batch_size: 128
  concurrency_limit: 16
  normalize: true
```

Use the same `url`, `model`, and `normalize` settings for encoding and serving.

### 2. Generate vectors

```bash
.venv/bin/python embed.py --config embed
```

This reads the configured JSONL corpus and writes a `.npy` embedding matrix.

### 3. Build the FAISS index

```bash
.venv/bin/python index.py --config index
```

The default index is `IndexFlatIP + IndexIDMap2`. With normalized embeddings, inner product behaves like cosine similarity.

### 4. Serve search

```bash
.venv/bin/python ret_serve.py --config serve
```

Open:

- `GET /health`
- `POST /search`
- `GET /docs`
- `GET /static/index.html`

## API

Search request:

```bash
curl http://localhost:8088/search \
  -H "Content-Type: application/json" \
  -d '{"queries":["relationship between Liu Bei and Zhuge Liang"],"topk":3}'
```

Response shape:

```json
{
  "contents": [
    [
      {
        "id": "2",
        "title": "Zhuge Liang",
        "text": "Zhuge Liang assisted Liu Bei and is known for strategy.",
        "contents": "Zhuge Liang\nZhuge Liang assisted Liu Bei and is known for strategy."
      }
    ]
  ],
  "scores": [[0.65]]
}
```

Health check:

```bash
curl http://localhost:8088/health
```

## Configuration Files

| File | Purpose |
| --- | --- |
| `config/embed.yaml` | JSONL corpus -> `.npy` vectors |
| `config/index.yaml` | `.npy` vectors -> FAISS `.index` |
| `config/serve.yaml` | HTTP retrieval service settings |
| `config/log.yaml` | Logging level and rotating file output |

Important consistency rules:

- Encoding and serving must use the same embedding model.
- Encoding and serving should use the same normalization behavior.
- Avoid double normalization unless the stored `.npy` was intentionally raw.
- Do not reorder the corpus after vectors are generated.

## Project Layout

```text
src/
  corpus.py             JSONL corpus loader
  embedding_client.py   OpenAI-compatible embedding client
  embed.py              JSONL -> .npy pipeline
  index.py              .npy -> FAISS index pipeline
  ret_serve.py          FastAPI app and CLI entry point
  service_container.py  retrieval lifecycle and search orchestration
  vector_index.py       FAISS-backed vector index
config/
  embed.yaml
  index.yaml
  serve.yaml
  log.yaml
```

## Notes

- `server.max_topk` limits expensive requests.
- `index.use_gpu` moves the FAISS index to GPU at service startup when possible.
- `index.search_concurrency_limit` controls CPU search concurrency; GPU search is serialized for safety.
- For large corpora, keep `index.mmap: true` and tune `embedding.batch_size`, `embedding.concurrency_limit`, and `index.chunk_size`.

## Contributing

Issues and pull requests are welcome. For retrieval behavior changes, include the embedding model, normalization settings, corpus assumptions, and index configuration used for testing.

## License

RetServe is released under the [MIT License](LICENSE).

Copyright (c) 2026 Haidong Xin.
