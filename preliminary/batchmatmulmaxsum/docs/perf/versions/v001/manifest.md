# Performance version v001

## Status

`MEASURED`

## Implementation identity

| Field | Value |
|---|---|
| Performance version | `v001` |
| Core implementation commit | `7caae52da159ccb48b656edbbad3b0f076373075` |
| Core change | Split kernel sources without changing the intended algorithm; declare MIX task type for profiler replay. |
| Measurement date | 2026-09-16 |
| Measurement workflow | `docs/perf/benchmark_workflow.md`, revision 2026-09-16 |

This version was assigned retrospectively when versioned performance storage was introduced. Later documentation-only commits do not invalidate it because they do not change the compiled operator.

## Environment

| Field | Value |
|---|---|
| Device | Ascend 910 |
| CANN | 9.0.0 |
| ASC architecture | `dav-2201` |
| Cube cores | 20 |
| Device state | Healthy, no competing NPU process during measurement |

## Archived profiling rounds

| Round | Shape `(B,M,N,K)` | DType/layout | Default key | Purpose |
|---|---|---|---:|---|
| [round_001](round_001/summary.txt) | `(4,128,128,128)` | FP16, TX1/TX2 `00` | 121 | Normal 8-core async double-buffer path |
| [round_002](round_002/summary.txt) | `(1,33,513,256)` | FP16, TX1/TX2 `00` | 121 | Single-core under-occupancy path |
| [round_003](round_003/summary.txt) | `(1,1,8192,8192)` | FP16, TX1/TX2 `00` | 130 | 20-way split-N path |

Host-side steady-state and candidate-key results are summarized in [../../analysis.md](../../analysis.md). Future regenerable JSON reports for this version should be written under the ignored `tuning_results/v001/` area; they are not the persistent source of truth.

## Coverage and limitations

- Representative hardware matrix: 48/48 passed.
- Candidate-key comparison: 7/7 passed.
- Same-machine comparison against core commit `5e38cba`: 3/3 passed, with no observable regression.
- The official separated `BatchMatMul + ReduceMax + ReduceSum` baseline and hidden competition cases were not measured; this version has no official competition speedup or score.

## Revision record

| Date | Change |
|---|---|
| 2026-09-16 | Created `v001` and moved the three measurements formerly named root rounds 008-010 into version-local rounds 001-003. |
