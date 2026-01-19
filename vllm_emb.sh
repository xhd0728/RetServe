CUDA_VISIBLE_DEVICES=7 nohup python -m vllm.entrypoints.openai.api_server \
    --served-model-name qwen3-emb \
    --model ./models/Qwen3-Embedding-0.6B \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 65501 \
    --task embed \
    --gpu-memory-utilization 0.6 \
    >./logs/vllm.log 2>&1 &
