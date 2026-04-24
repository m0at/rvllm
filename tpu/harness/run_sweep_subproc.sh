#!/usr/bin/env bash
# Drive the MiniMax-M2.7 bench: one subprocess per batch size so each batch's
# XLA compile buffers get released on python exit.
#
# Usage:
#   MODEL_DIR=/dev/shm/m2-nvfp4 BATCHES="1 8 16 32 64 128" bash run_sweep_subproc.sh
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/dev/shm/m2-nvfp4}"
BATCHES="${BATCHES:-1 8 16 32 64 128}"
CTX="${CTX:-2048}"
ITERS="${ITERS:-10}"
WARMUP="${WARMUP:-3}"
WORKERS="${WORKERS:-32}"
OUT_DIR="${OUT_DIR:-/tmp/m2_sweep}"
LOG_DIR="${LOG_DIR:-/tmp/m2_sweep_logs}"

export PATH="$HOME/.local/bin:$PATH"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$HOME/.jax_cache}"
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0

mkdir -p "$OUT_DIR" "$LOG_DIR"

echo "=== MiniMax-M2.7 batch sweep (subprocess-per-batch) ==="
echo "model:   $MODEL_DIR"
echo "batches: $BATCHES"
echo "ctx:     $CTX"
echo "cache:   $JAX_COMPILATION_CACHE_DIR"
echo ""

for B in $BATCHES; do
  echo ">> B=$B starting $(date +%H:%M:%S)"
  t0=$(date +%s)
  if python3 -u /tmp/m2_full_bench.py \
      --model-dir "$MODEL_DIR" \
      --batches "$B" --single-batch "$B" \
      --ctx "$CTX" --iters "$ITERS" --warmup "$WARMUP" --workers "$WORKERS" \
      --skip-ppl --skip-gen \
      --out "$OUT_DIR/b${B}.json" \
      > "$LOG_DIR/b${B}.log" 2>&1
  then
    elapsed=$(( $(date +%s) - t0 ))
    tail -5 "$LOG_DIR/b${B}.log" | grep -E "ms/step|FAILED" || true
    echo ">> B=$B done (${elapsed}s)"
  else
    elapsed=$(( $(date +%s) - t0 ))
    echo ">> B=$B FAILED (${elapsed}s):"
    tail -5 "$LOG_DIR/b${B}.log"
  fi
  echo ""
done

echo "=== Running PPL + generation (fresh subprocess) ==="
python3 -u /tmp/m2_full_bench.py \
  --model-dir "$MODEL_DIR" \
  --batches 1 \
  --ctx "$CTX" --iters 2 --warmup 1 --workers "$WORKERS" \
  --skip-sweep \
  --prompt 'Explain angular momentum.' \
  --gen-tokens 2048 \
  --ppl-text /tmp/wiki.txt \
  --out "$OUT_DIR/ppl_gen.json" \
  > "$LOG_DIR/ppl_gen.log" 2>&1 || tail -5 "$LOG_DIR/ppl_gen.log"

echo ""
echo "=== Aggregating ==="
python3 -c "
import json, glob
results = {'batch_sweep': [], 'ppl': None, 'generation': None}
for f in sorted(glob.glob('$OUT_DIR/b*.json')):
    d = json.load(open(f))
    results['batch_sweep'].extend(d.get('sweep') or [])
ppl_f = '$OUT_DIR/ppl_gen.json'
try:
    d = json.load(open(ppl_f))
    results['ppl'] = d.get('ppl')
    results['generation'] = d.get('generation')
except Exception as e:
    print('ppl_gen.json missing:', e)
with open('$OUT_DIR/final.json', 'w') as f:
    json.dump(results, f, indent=2)
print('final:', '$OUT_DIR/final.json')
import json as _j; print(_j.dumps(results['batch_sweep'], indent=2))
"
echo "done."
