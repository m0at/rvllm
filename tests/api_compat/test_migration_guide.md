# Migrating from Python vLLM to rvllm

## For API Consumers (no code changes needed)

If you're calling vLLM's OpenAI-compatible API, rvllm is a drop-in replacement.
Just change the server URL:

```python
# Before (Python vLLM)
client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")

# After (Rust rvllm) -- same URL, same API
client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
```

## For Server Operators

Replace:
```bash
# Python vLLM
python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-3-8B

# Rust rvllm
rvllm serve --model meta-llama/Llama-3-8B
```

Same CLI flags: --model, --port, --host, --gpu-memory-utilization, --max-model-len, --tensor-parallel-size

## Supported Features
- [x] /v1/completions (streaming + non-streaming)
- [x] /v1/chat/completions (streaming + non-streaming)
- [x] /v1/models
- [x] /health
- [x] /metrics (Prometheus)
- [x] Temperature, top_p, top_k, presence/frequency penalty
- [x] Stop strings
- [x] Logprobs
- [ ] Tool/function calling (planned)
- [ ] Vision models (planned)
