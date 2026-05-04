# rvllm-vision-sidecar

HTTP service that runs the vision tower + projector for either Qwen 3.6
(`Qwen3_5MoeForConditionalGeneration`) or Gemma 4 31B
(`Gemma4ForConditionalGeneration`) and returns f16 token-level embeddings
ready to be spliced into rvllm-serve's post-embed hidden buffer.

The language-model weights are skipped (meta-init only); only vision +
projector are materialized on GPU. Memory cost ≈ 1.2 GiB (Qwen ViT) or
800 MiB (Gemma SigLIP) depending on which model is selected.

## Run

```bash
# Qwen 3.6 vision
./run.sh                       # default
./run.sh qwen3_vl

# Gemma 4 vision
./run.sh gemma4_mm

# Pin to a separate GPU (avoid SM-contention with rvllm-serve worker)
VISION_GPU=1 ./run.sh qwen3_vl

# Override paths
QWEN36_PATH=/some/dir GEMMA4_PATH=/other/dir ./run.sh qwen3_vl
```

## API

`GET /health` → `{model, cuda, hidden_dim}`

`POST /embed` body `{image_b64: str, mime?: str}`
  → `{num_tokens, hidden_dim, dtype: "float16", embeddings_b64, grid_thw?}`

Embeddings are raw little-endian f16 bytes, layout `[num_tokens, hidden_dim]`,
base64-encoded.

## Smoke

```bash
python -c "from PIL import Image; Image.new('RGB',(224,224),'white').save('/tmp/w.png')"
B64=$(base64 -w0 /tmp/w.png)
curl -s -X POST http://127.0.0.1:8765/embed \
  -H 'content-type: application/json' \
  -d "{\"image_b64\":\"$B64\",\"mime\":\"image/png\"}" | jq '{num_tokens, hidden_dim, grid_thw}'
```
