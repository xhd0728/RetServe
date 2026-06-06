#!/usr/bin/env bash
set -euo pipefail

docker run -d --gpus all \
  -e OMP_NUM_THREADS=1 \
  -v /path/to/Qwen3-Embedding-0.6B:/workspace/Qwen3-Embedding-0.6B \
  -p 8000:8000 \
  --ipc=host \
  --name qwen3-emb \
  vllm/vllm-openai:latest \
  --model /workspace/Qwen3-Embedding-0.6B \
  --served-model-name qwen3-emb \
  --data-parallel-size 1 \
  --host 0.0.0.0 \
  --port 8000 \
  --runner pooling \
  --dtype auto \
  --gpu-memory-utilization 0.3 \
  --trust-remote-code \
  --log-error-stack
