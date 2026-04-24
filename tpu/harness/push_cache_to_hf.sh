#!/usr/bin/env bash
# Push JAX compile cache + installed python env + repo source to a private
# HuggingFace dataset so a fresh TPU VM can boot with zero recompile / reinstall.
#
# Runs ON the TPU VM. Invoked after a successful bench compile.
#
# Uploads to dataset `and-y/rvllm-m2-build` with keys:
#   jax-cache/<git-sha>-<accelerator>.tar.gz  — XLA compile cache
#   py-env/python3.10-<accelerator>.tar.gz    — site-packages + .local/bin
#   revision/<git-sha>.txt                    — marker with build info
#
# Usage:
#   SHA=$(git rev-parse HEAD) ACCEL=v6e-8 bash push_cache_to_hf.sh
set -euo pipefail

: "${SHA:?Set SHA env var (git sha of the run)}"
: "${ACCEL:=v6e-8}"
: "${REPO:=and-y/rvllm-m2-build}"
: "${JAX_CACHE_DIR:=$HOME/.jax_cache}"

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

export PATH="$HOME/.local/bin:$PATH"
command -v hf >/dev/null || { echo "ERROR: hf CLI missing"; exit 1; }

echo ">> (1/3) packing JAX compile cache from $JAX_CACHE_DIR"
if [[ ! -d "$JAX_CACHE_DIR" || -z "$(ls -A "$JAX_CACHE_DIR" 2>/dev/null)" ]]; then
  echo "WARN: no JAX cache found at $JAX_CACHE_DIR — did you run with JAX_COMPILATION_CACHE_DIR set? Skipping compile-cache upload."
else
  CACHE_TGZ="$TMP/jax-cache.tar.gz"
  tar czf "$CACHE_TGZ" -C "$(dirname "$JAX_CACHE_DIR")" "$(basename "$JAX_CACHE_DIR")"
  du -sh "$CACHE_TGZ"
  hf upload "$REPO" "$CACHE_TGZ" "jax-cache/${SHA}-${ACCEL}.tar.gz" --repo-type dataset
fi

echo ">> (2/3) packing python site-packages + .local/bin"
ENV_TGZ="$TMP/py-env.tar.gz"
tar czf "$ENV_TGZ" \
  -C "$HOME" \
  --exclude='.local/lib/python3.10/site-packages/__pycache__' \
  --exclude='.local/lib/python3.10/site-packages/*/__pycache__' \
  --exclude='.local/lib/python3.10/site-packages/*.dist-info/RECORD' \
  .local/lib/python3.10/site-packages \
  .local/bin
du -sh "$ENV_TGZ"
hf upload "$REPO" "$ENV_TGZ" "py-env/python3.10-${ACCEL}.tar.gz" --repo-type dataset

echo ">> (3/3) writing revision marker"
MARK="$TMP/revision.txt"
cat > "$MARK" <<EOF
sha: ${SHA}
accelerator: ${ACCEL}
date: $(date -u +%Y-%m-%dT%H:%M:%SZ)
jax: $(python3 -c 'import jax; print(jax.__version__)' 2>/dev/null || echo "?")
jaxlib: $(python3 -c 'import jaxlib; print(jaxlib.__version__)' 2>/dev/null || echo "?")
libtpu: $(python3 -c 'import libtpu; print(getattr(libtpu, "__version__", "?"))' 2>/dev/null || echo "?")
python: $(python3 --version)
EOF
cat "$MARK"
hf upload "$REPO" "$MARK" "revision/${SHA}.txt" --repo-type dataset

echo ">> done. artifacts at https://huggingface.co/datasets/${REPO}"
