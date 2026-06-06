#!/usr/bin/env bash
set -euo pipefail

vllm serve /path/to/Qwen3-Embedding-0.6B \
  --served-model-name qwen3-emb \
  --data-parallel-size 1 \
  --host 0.0.0.0 \
  --port 8000 \
  --runner pooling \
  --dtype auto \
  --gpu-memory-utilization 0.3 \
  --trust-remote-code \
  --log-error-stack
