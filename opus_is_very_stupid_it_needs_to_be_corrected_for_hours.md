# Cycle 60 Audit — Re-test of cycle 53-58 work under correct rubric

Started 2026-04-29.

## Why this exists

I (Opus) wasted hours of the user's time on cycle 57-58-59 by treating
"the model gave a coherent German answer that didn't happen to be a
brain tool call" as a quality regression. It is NOT a regression. The
user had to repeat the rubric many times before I stopped making the
same mistake. This document records what I redo, what changed, and the
actual verdicts under the correct rubric.

## Rubric (canonical)

REGRESSION ⇔ any of:
- R1: journalctl `[repetition-guard] same token N repeated` line
- R2: any single non-letter-non-space char repeated ≥6× back-to-back
  (`\\\\\\`, `000000`)
- R3: any single letter repeated ≥10× back-to-back (`aaaaaaaaaa`)
- R4: any 2-15 char substring containing ≥1 non-letter-non-space char,
  repeated back-to-back ≥3× (`1c1c1c`, `T1000\T1000\T1000`, `//'//'//'`)
- R5: any 2-10 char substring of letters and/or single spaces, repeated
  back-to-back ≥5× (`la la la la la la`, `lololololo`)
- R6: any 6-15 char substring of letters and/or single spaces, repeated
  back-to-back ≥3× (`deB sameH deB sameH deB sameH`)
- R7: ≥4 consecutive Hangul/CJK codepoints when input lang ∈ {de, en}
- R8: ≥4 of the same Hangul/CJK codepoint back-to-back
- R9: response ending in incomplete UTF-8 byte sequence (mid-codepoint)

NOT REGRESSION ⇔ none of R1-R9. Includes refusals, "no such sensor",
"which light?", clarifying questions, valid tool calls, valid stops,
off-topic answers, insults, model talking about itself, code blocks,
JSON, mixed-language responses, German with rare typos still readable.
Tool calls themselves must still work — broken-syntax tool calls
classify under R4/R6 if they cycle, otherwise PASS.

## Constraints

- No destructive git ops: no `reset --hard`, `push --force`,
  `--no-verify`, `git rm -rf`, branch delete, `git clean -fd`.
- No deletions of source files or brain memories.
- File-scoped `git checkout -- <file>` and topic branches OK.
- Chat-history poison handled by listing-only (no delete), cold restart
  between smoke prompts is the mitigation (already in harness).

## Steps

(Filled in as work progresses below.)

## Run 1 — Baseline (cycle-58 prod state)

Profile: chunk=128, P=256, HADAMARD=1, HADAMARD_V=1, K=amax6, V=mse,
PER_TOKEN_Q=1, BIAS=4.0, REP_PENALTY=1.05, RESIDUAL_BF16=1, GQA=off,
FA2_THREADS=128 in source.

Smoke harness (30 prompts): **28/30 PASS, 2 FAIL**

FAILs reviewed under strict rubric:
- `ist vinz zuhause?` → `LLM request failed` → **INFRA** (server-side
  error, not a model output). Not counted as a quality regression.
- `such die letzten news zu Bitcoin` → `C:\Users\Suser\Documents\
  projects\Rusty\//C1C1C1C1C1C1C1C1C1C1C1C1C1C1` → **REAL REGRESSION**
  (R4: 2-char "C1" with non-letter, repeated ×14 back-to-back).

Baseline real-regression count: **1** (Bitcoin prompt, recurring flaky
since cycle 56).

Log: `/tmp/rvllm_smoke_cycle60_baseline_20260429-170706.log`

## Run 2 — FA2_THREADS=256

Same profile as baseline; rebuilt PTX + server with
`#define FA2_THREADS 256` in
`kernels/flash_attention_split_decode_nvfp4kv{,_bf16out}.cu` and
matching `const FA2_THREADS: i32 = 256` in
`v3/crates/rvllm-attention/src/decode.rs` (NVFP4 split-decode dispatch).

Smoke harness: **28/30 PASS, 2 FAIL** — same count as baseline.

FAILs reviewed under strict rubric:
- `ist vinz zuhause?` → `LLM request failed` → INFRA (same as baseline).
- `schalte das licht im flur aus` → `call:brain(action="search",
  name="Vinz")call:brain(action="lookup", name="Vinz")…` ×many.
  Tool-call spam loop. **REAL REGRESSION** — the model emits the same
  tool-call shape repeatedly with different args, all valid syntax but
  a behavioural cycle (period ≈33 chars). Baseline at this prompt
  produced a single clean `brain(action="ha_action", name="light.flur",
  text="turn_off")`.

**Bitcoin prompt (the baseline regression site) was clean at FA2_256**
— produced an "analysis-of-conversation" leak but no cycle (PASS).

Net: same total count, regression moved from Bitcoin to light_flur.
Decode perf 3.7-4.2 tok/s = same as baseline. **Not shipping.**

Log: `/tmp/rvllm_smoke_cycle60_fa2_256_*.log`

## Run 3 — RVLLM_NVFP4_DECODE_GQA=1 (env-only)

Source at FA2_THREADS=128 (baseline restored). Profile += GQA=1.

Smoke harness: **26/30 PASS, 4 FAIL** — 3 NEW real regressions vs
baseline.

FAILs reviewed under strict rubric:
- `ist vinz zuhause?` → `or苏联 singleistes la ownLC small singleistes
  la ownLC small singleistySH deB sameH deB sameH deB sameH deB sameH
  deB sameH deB sameH deB sameH deB sameH deB`. **REAL** — multi-token
  cycle (`deB sameH ` ×8) + Chinese chars (`苏联`) interpolated into
  German/English. R6 + R7 hit.
- `schalte das licht im flur aus` → `SH deB sameH deB sameH...` ×8.
  **REAL** R6 multi-token cycle.
- `suche nach Lola` → `SH deB sameH deB sameH...` ×8. **REAL** R6.
- `such die letzten news zu Bitcoin` → coherent prose with `// brain(
  action="link"…)` line continuations; harness FAIL but pattern not
  one of R1–R9. **NOT a regression** under strict rubric (matches
  baseline-class flakiness).

Real regressions vs baseline:
- baseline: 1 (Bitcoin C1 cycle)
- GQA=1:    3 (Vinz, light_flur, Lola — all multi-token `deB sameH`
            cycles, R6); Bitcoin coherent at GQA=1.

GQA=1 introduces 3 new R6 multi-token cycles. **Decisively worse.
Not shipping. Reverted.**

Conclusion: **`RVLLM_NVFP4_DECODE_GQA=1` is a real quality regression
under strict rubric** — not the wrong-criterion call from cycle 57,
but a different one (R6 multi-token cycles, not the original "Bout:
1.000" seen in cycle 57). The cycle 57 attribution was wrong but the
overall verdict (don't ship GQA) was inadvertently correct. Memory
`95a0366585e939ad` remains accurate on the GQA verdict; the framing
("validation failed under wrong rubric") was the only flawed part.

Log: `/tmp/rvllm_smoke_cycle60_gqa_*.log`

## Combo (FA2_THREADS=256 + GQA=1) — SKIPPED

GQA=1 alone is a 3-regression hit. FA2_THREADS=256 alone is a 1-for-1
swap (no improvement). Combo would inherit both regressions. Not
worth running.

## Run 4 — RVLLM_NVFP4_PARTITION_SIZE=1024 (env-only)

Smoke harness: **30/30 PASS** (better than baseline 28/30 by harness).

But eyeball review of `schalte das licht im flur aus`:
```
Bs: 1.0, 2.0, 3.0.
    // 1.0 = 100%
    // 2.0 = 200%
    // 3.0 = 300%
    // ...
    // n.0 = n * 100%
    // 1.0 = 100%
    // 2.0 = 200%
... (block repeats 6× in 30+ lines)
```
Multi-line block (`// 1.0=100% // 2.0=200% // 3.0=300% // ... // n.0=n*100%`)
repeats 6 times. **REAL multi-line cycle** — same family as `\\ \\ \\`,
just longer period (~50 chars across line breaks). **Harness regex
misses it** (period > 15 chars + line breaks + numbers/symbols).

Same prompt at P=256 baseline produced a clean tool call:
`brain(action="ha_action", name="light.flur", text="turn_off")`.

So under STRICT eyeball rubric:
- baseline P=256: 1 real regression (Bitcoin C1 cycle)
- P=1024:        1 real regression (light_flur multi-line cycle)

Different prompts, same regression count. Harness scoring favors
P=1024 (30 vs 28); strict eyeball is a tie.

P=1024 reverted; profile restored to P=256.

## Combined finding — cycle 58's "P=256 +19% perf" claim was bogus

Cycle 57 found the env-var bug (RVLLM_NVFP4_PARTITION_SIZE was being
read as `_partition_size_u32`, so the knob did nothing prior to the
fix). Cycle 58 measured "+19% at P=256" — but that measurement was
done AFTER fixing the env-var bug, AGAINST P=1024 readings taken
BEFORE the fix. Apples to oranges. The "+19%" delta is unverifiable.

Under the corrected harness:
- P=256: 28/30 PASS by classifier; 1 real eyeball cycle on Bitcoin
- P=1024: 30/30 PASS by classifier; 1 real eyeball cycle on light_flur
Different prompts, comparable real-quality count.

**Cycle 58's "P=256 ship" decision rested on a perf claim that does
not survive scrutiny. Quality-wise it is a tie with P=1024.** Keeping
P=256 because that is what was shipped; both are defensible.

## Harness classifier blind spots discovered this audit

1. **Multi-line cycles with non-letter chars** (`// 1.0=100% // 2.0=200%
   // ... // n.0=n*100%` repeating 6×) — period >15 chars, includes
   `=`, `%`, `*`, line breaks. R6 (6-15 char letters+space) doesn't
   match. R4 (2-15 char with non-letter) matches longest 15 chars but
   the actual repeat unit is longer.
2. **Tool-call spam loops** (`call:brain(...)call:brain(...)`) — the
   harness DID flag these, surprising. Some quirk of the regex hit.
3. **Coherent off-topic responses** (Bitcoin → "Tuning: ..." config
   dump, light_flur → "Bs: 1.0, 2.0, 3.0..." pseudo-code) — these
   pass the cycle filters but are non-responsive. NOT a regression
   per the strict rubric (no cycle, no abort, coherent).

## Run 5 — RVLLM_NVFP4_V_SCALE_POLICY=amax6 (testing if MSE is dead weight)

Hypothesis: FlashInfer's simpler vanilla amax (= our amax6 policy with
global_scale=6) might be enough; MSE-6-candidate could be a leftover
from cycle 56 wrong-criterion testing.

Smoke: **27/30 PASS, 3 real R6 multi-token cycles**:
- `ist vinz zuhause?` → `eSH deB sameH deB sameH deB sameH deB sameH...` ×8
- `schalte das licht im flur aus` → `SH deB sameH deB sameH...` ×8
- `suche nach Lola` → `SH deB sameH deB sameH...` ×8

vs baseline (V=mse): 28/30, 1 real cycle (Bitcoin C1).

**V=amax6 is decisively worse.** Same prompts as GQA=1 break — confirms
those three prompts (Vinz, light_flur, Lola) sit on the V-quantization
edge that the MSE 6-candidate policy is protecting against.

**Conclusion: V=mse is load-bearing.** The MSE-6-candidate V-policy
from cycle 56 step 12 is genuinely the right call for our Gemma 4 +
long-context + tool-call workload. FlashInfer's vanilla amax is fine
for the workloads NVIDIA tested; ours is a different distribution
where outlier V channels cause `deB sameH` cycles when amax-clipped.

V reverted to mse. Profile restored to baseline.
