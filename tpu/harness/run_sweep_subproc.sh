#!/usr/bin/env bash
# Drive the MiniMax-M2.7 bench: one subprocess per batch size so each batch's
# XLA compile buffers get released on python exit.
#
# Usage:
#   MODEL_DIR=/dev/shm/m2-nvfp4 BATCHES="1 8 16 32 64 128" bash run_sweep_subproc.sh
#   RUN_CORRECTNESS_GATE=1 OPT_LIBTPU_INIT_ARGS="$LIBTPU_INIT_ARGS" bash run_sweep_subproc.sh
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/dev/shm/m2-nvfp4}"
BATCHES="${BATCHES:-1 8 16 32 64 128}"
CTX="${CTX:-2048}"
ITERS="${ITERS:-10}"
WARMUP="${WARMUP:-3}"
WORKERS="${WORKERS:-32}"
OUT_DIR="${OUT_DIR:-/tmp/m2_sweep}"
LOG_DIR="${LOG_DIR:-/tmp/m2_sweep_logs}"
BENCH_PY="${BENCH_PY:-/tmp/m2_full_bench.py}"
COMPARE_PY="${COMPARE_PY:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/compare_m2_correctness.py}"
PPL_TEXT="${PPL_TEXT:-/tmp/wiki.txt}"
PROMPT="${PROMPT:-Explain angular momentum.}"
RUN_CORRECTNESS_GATE="${RUN_CORRECTNESS_GATE:-1}"
GATE_GEN_TOKENS="${GATE_GEN_TOKENS:-64}"
GATE_ITERS="${GATE_ITERS:-2}"
GATE_WARMUP="${GATE_WARMUP:-1}"
BASELINE_M2_MOE="${BASELINE_M2_MOE:-shardmap}"
OPT_M2_MOE="${OPT_M2_MOE:-${M2_MOE:-shardmap}}"
BASELINE_LIBTPU_INIT_ARGS="${BASELINE_LIBTPU_INIT_ARGS:-}"
OPT_LIBTPU_INIT_ARGS="${OPT_LIBTPU_INIT_ARGS:-${LIBTPU_INIT_ARGS:-}}"
PPL_REL_TOL="${PPL_REL_TOL:-0.03}"
PPL_ABS_TOL="${PPL_ABS_TOL:-0.10}"
MIN_PREFIX_CHARS="${MIN_PREFIX_CHARS:-80}"

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
  if M2_MOE="$OPT_M2_MOE" LIBTPU_INIT_ARGS="$OPT_LIBTPU_INIT_ARGS" python3 -u "$BENCH_PY" \
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

if [[ "$RUN_CORRECTNESS_GATE" == "1" ]]; then
  echo "=== Correctness gate: baseline PPL + generation ==="
  M2_MOE="$BASELINE_M2_MOE" LIBTPU_INIT_ARGS="$BASELINE_LIBTPU_INIT_ARGS" \
    python3 -u "$BENCH_PY" \
      --model-dir "$MODEL_DIR" \
      --batches 1 \
      --ctx "$CTX" --iters "$GATE_ITERS" --warmup "$GATE_WARMUP" --workers "$WORKERS" \
      --skip-sweep \
      --prompt "$PROMPT" \
      --gen-tokens "$GATE_GEN_TOKENS" \
      --ppl-text "$PPL_TEXT" \
      --out "$OUT_DIR/baseline_ppl_gen.json" \
      > "$LOG_DIR/baseline_ppl_gen.log" 2>&1 || {
        tail -20 "$LOG_DIR/baseline_ppl_gen.log"
        exit 1
      }

  echo "=== Correctness gate: optimized PPL + generation ==="
  M2_MOE="$OPT_M2_MOE" LIBTPU_INIT_ARGS="$OPT_LIBTPU_INIT_ARGS" \
    python3 -u "$BENCH_PY" \
      --model-dir "$MODEL_DIR" \
      --batches 1 \
      --ctx "$CTX" --iters "$GATE_ITERS" --warmup "$GATE_WARMUP" --workers "$WORKERS" \
      --skip-sweep \
      --prompt "$PROMPT" \
      --gen-tokens "$GATE_GEN_TOKENS" \
      --ppl-text "$PPL_TEXT" \
      --out "$OUT_DIR/optimized_ppl_gen.json" \
      > "$LOG_DIR/optimized_ppl_gen.log" 2>&1 || {
        tail -20 "$LOG_DIR/optimized_ppl_gen.log"
        exit 1
      }

  echo "=== Correctness gate: compare ==="
  python3 "$COMPARE_PY" \
    --baseline "$OUT_DIR/baseline_ppl_gen.json" \
    --candidate "$OUT_DIR/optimized_ppl_gen.json" \
    --ppl-rel-tol "$PPL_REL_TOL" \
    --ppl-abs-tol "$PPL_ABS_TOL" \
    --min-prefix-chars "$MIN_PREFIX_CHARS" \
    | tee "$OUT_DIR/correctness_gate.json"
else
  echo "=== Running PPL + generation (fresh subprocess) ==="
  M2_MOE="$OPT_M2_MOE" LIBTPU_INIT_ARGS="$OPT_LIBTPU_INIT_ARGS" python3 -u "$BENCH_PY" \
    --model-dir "$MODEL_DIR" \
    --batches 1 \
    --ctx "$CTX" --iters 2 --warmup 1 --workers "$WORKERS" \
    --skip-sweep \
    --prompt "$PROMPT" \
    --gen-tokens 2048 \
    --ppl-text "$PPL_TEXT" \
    --out "$OUT_DIR/ppl_gen.json" \
    > "$LOG_DIR/ppl_gen.log" 2>&1 || tail -5 "$LOG_DIR/ppl_gen.log"
fi

echo ""
echo "=== Aggregating ==="
python3 -c "
import json, glob
results = {'batch_sweep': [], 'ppl': None, 'generation': None, 'correctness_gate': None}
for f in sorted(glob.glob('$OUT_DIR/b*.json')):
    if f.endswith('/baseline_ppl_gen.json'):
        continue
    d = json.load(open(f))
    results['batch_sweep'].extend(d.get('sweep') or [])
if '$RUN_CORRECTNESS_GATE' == '1':
    ppl_f = '$OUT_DIR/optimized_ppl_gen.json'
else:
    ppl_f = '$OUT_DIR/ppl_gen.json'
try:
    d = json.load(open(ppl_f))
    results['ppl'] = d.get('ppl')
    results['generation'] = d.get('generation')
except Exception as e:
    print('ppl_gen.json missing:', e)
try:
    results['correctness_gate'] = json.load(open('$OUT_DIR/correctness_gate.json'))
except Exception:
    pass
with open('$OUT_DIR/final.json', 'w') as f:
    json.dump(results, f, indent=2)
print('final:', '$OUT_DIR/final.json')
import json as _j; print(_j.dumps(results['batch_sweep'], indent=2))
"
echo "done."
