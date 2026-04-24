#!/usr/bin/env bash
# Pull cached python env + JAX XLA compile cache from a private HF dataset.
# Runs ON a fresh TPU VM. If both caches hit, skips pip install AND JIT compile.
#
# Usage (called by deploy_m2_tpu.sh):
#   SHA=<git-sha> ACCEL=v6e-8 bash pull_cache_from_hf.sh
#
# Env vars set on success (caller should export them):
#   JAX_COMPILATION_CACHE_DIR — if compile cache was pulled
#   PATH — ~/.local/bin prefixed (the py-env's hf/python scripts)
set -euo pipefail

: "${SHA:?Set SHA env var}"
: "${ACCEL:=v6e-8}"
: "${REPO:=and-y/rvllm-m2-build}"

export PATH="$HOME/.local/bin:$PATH"

# Ensure hf CLI exists. If not, pip install minimal hf.
if ! command -v hf >/dev/null 2>&1; then
  echo ">> bootstrapping hf CLI"
  pip3 install --quiet --user --upgrade huggingface_hub
  export PATH="$HOME/.local/bin:$PATH"
fi

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

ENV_KEY="py-env/python3.10-${ACCEL}.tar.gz"
CACHE_KEY="jax-cache/${SHA}-${ACCEL}.tar.gz"

echo ">> (1/2) pulling py-env $ENV_KEY"
if hf download "$REPO" "$ENV_KEY" --local-dir "$TMP" --repo-type dataset 2>&1 | tail -3; then
  if [[ -f "$TMP/$ENV_KEY" ]]; then
    echo "   extracting into \$HOME"
    tar xzf "$TMP/$ENV_KEY" -C "$HOME"
    echo "   py-env restored"
    ENV_HIT=1
  else
    ENV_HIT=0
  fi
else
  echo "   py-env MISS"
  ENV_HIT=0
fi

echo ">> (2/2) pulling jax-cache $CACHE_KEY"
if hf download "$REPO" "$CACHE_KEY" --local-dir "$TMP" --repo-type dataset 2>&1 | tail -3; then
  if [[ -f "$TMP/$CACHE_KEY" ]]; then
    echo "   extracting into \$HOME"
    mkdir -p "$HOME/.jax_cache"
    tar xzf "$TMP/$CACHE_KEY" -C "$HOME"
    echo "   jax-cache restored"
    CACHE_HIT=1
  else
    CACHE_HIT=0
  fi
else
  echo "   jax-cache MISS (may be first run for this SHA)"
  CACHE_HIT=0
fi

# Emit env exports for the caller to source
cat > "$HOME/.rvllm_hf_cache_env" <<EOF
export PATH="\$HOME/.local/bin:\$PATH"
export JAX_COMPILATION_CACHE_DIR="\$HOME/.jax_cache"
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
EOF

echo ""
echo ">> summary: env-hit=${ENV_HIT} cache-hit=${CACHE_HIT}"
echo ">> source \$HOME/.rvllm_hf_cache_env before running python"
