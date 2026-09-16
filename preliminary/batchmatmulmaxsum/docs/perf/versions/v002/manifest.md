# Performance version v002

## Status

`MEASURED`

## Implementation identity

| Field | Value |
|---|---|
| Performance version | `v002` |
| Core implementation identity | Base commit `97cf68e54c296e98a854f7b63c3b420c902c8ca2` plus the documented v002 worktree change |
| Core diff SHA-256 | `063718eaaac45626d6e794a5e3cc80b0b46aa19ae6819e99e5baf5a5a1363c35` |
| Core change | Add tuning-only key 122 (`TileM=32`) to test c20 M-axis parallelism against key 121 (`TileM=64`). |
| Measurement date | 2026-09-16 |
| Measurement workflow | `docs/perf/benchmark_workflow.md`, revision 2026-09-16 |

The submission policy remains unchanged and does not select key 122. The key
122 dispatch is compiled only when `BMMS_ENABLE_TUNING` is defined. No v001
profiling round was copied into this version.

## Environment

| Field | Value |
|---|---|
| Device | Ascend 910 |
| CANN | 9.0.0 |
| ASC architecture | `dav-2201` |
| Cube cores | 20 |
| Device state | Healthy; no competing NPU process before measurement |

## Archived profiling rounds

| Round | Shape / layout / key | Purpose |
|---|---|---|
| [round_001](round_001/summary.txt) | `(1,33,513,256)`, FP16 `00`, key 121 | `TileM=64` control |
| [round_002](round_002/summary.txt) | `(1,33,513,256)`, FP16 `00`, key 122 | `TileM=32` experiment |

## Validation

- Static catalog/unit tests: 23/23 passed.
- Fresh `dav-2201` build: passed.
- Key 122 c20 correctness: 8/8 across FP16/BF16 and all four layouts.
- Four 200-repeat FP16 `00` comparisons: key 122 p50 was 0.4%-2.1%
  lower, but its median p95 was 1.1% higher.
- Standard `msprof`: key 122 Task Duration was 35.860 us versus 35.320 us
  for key 121, so the apparent p50 gain is not a kernel improvement.

## Conclusion and limitations

Key 122 is not eligible for the submission policy. It doubles the c20 launch
grid from one to two Cube blocks, but the fixed M groups contain 32 and 1 rows.
The resulting 25.49% core imbalance and 5.370 us task head overhead erase the
shorter longest-core time. Both candidates remain Scalar Bound.

This is a scoped c20 experiment, not a full v002 release qualification. Key
122 has not passed the repository-wide smoke/correctness/fuzz/stress matrix,
and no official separated-operator baseline was measured.

## Revision record

| Date | Change |
|---|---|
| 2026-09-16 | Created v002 as `UNMEASURED`; no historical rounds copied. |
| 2026-09-16 | Archived c20 key 121/122 rounds and marked the scoped experiment `MEASURED`. |
