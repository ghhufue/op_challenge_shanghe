# BatchMatmulMaxSum performance status and history

性能数据按 [README.md](README.md) 规定的实现版本归档。当前版本由 [CURRENT_VERSION](CURRENT_VERSION) 指向；完整测评命令见 [benchmark_workflow.md](benchmark_workflow.md)。本节只把当前版本作为有效现状，后续章节保留的早期数据均视为历史参考。

## Current status: performance version [v001](versions/v001/manifest.md)

Core implementation commit: `7caae52`

Version status: `MEASURED`

Measurement date: 2026-09-16

Platform: Ascend 910, CANN 9.0.0, `dav-2201`, 20 Cube cores

Test state: `main` and `origin/main` both pointed to `7caae52`; the device was healthy and had no competing NPU process. The executable was rebuilt from this commit in a fresh `build_perf_latest` directory. Tracked source files were not modified during measurement.

### Validation scope

| Check | Result |
|---|---:|
| Python source/unit tests | 22/22 |
| Representative hardware matrix: 6 base cases x 2 dtypes x 4 layouts | 48/48 |
| Candidate-key comparison rows | 7/7 |
| Same-machine previous-commit comparison | 3/3 |

All repeated outputs remained within tolerance. The `c21_square` atomic accumulation path was not bitwise identical across every repeat, but its maximum repeat-to-repeat absolute difference was only `3.81469727e-6`.

### Stable host-side latency

The table uses FP16 and non-transposed storage. `c20`, `c21`, and `t03` are p50/p95 values from 10 warm-ups plus 200 measured launch-and-synchronize calls. The remaining representative rows use 100 measured calls.

| Case | Shape `(B,M,N,K)` | Default key | Blocks / split-N | p50 | p95 |
|---|---|---:|---:|---:|---:|
| `s00_dot` | `(1,1,1,32)` | 0 | 1 / 1 | 78.023 us | 92.593 us |
| `c16_tail63` | `(2,63,65,128)` | 121 | 2 / 1 | 90.683 us | 98.393 us |
| `c20_rect_n` | `(1,33,513,256)` | 121 | 1 / 1 | 97.934 us | 106.844 us |
| `c21_square` | `(4,128,128,128)` | 121 | 8 / 1 | 87.714 us | 95.424 us |
| `c22_large_k` | `(1,17,19,8192)` | 121 | 1 / 1 | 178.927 us | 195.088 us |
| `t03_max_nk` | `(1,1,8192,8192)` | 130 | 20 / 20 | 460.787 us | 501.359 us |

These are absolute custom-kernel timings and internal candidate comparisons. The official separated `BatchMatMul + ReduceMax + ReduceSum` benchmark was not run in this round, so these results must not be presented as an official competition speedup or score.

### Candidate effectiveness

| Case | Baseline | Candidate | p50 change | Speedup | Conclusion |
|---|---:|---:|---:|---:|---|
| `c21_square_float16_00` | key 100: 95.583 us | key 121: 87.714 us | -7.869 us | 1.090x | Async double buffering is beneficial. |
| `c20_rect_n_float16_00` | key 100: 126.234 us | key 121: 97.934 us | -28.300 us | 1.289x | Async execution hides a substantial part of the single-core pipeline. |
| `c20_rect_n_float16_00` | key 121: 97.934 us | key 130: 144.846 us | +46.912 us | 0.676x | Split-N is not amortized below the current threshold. |
| `t03_max_nk_float16_00` | key 121: 1731.443 us | key 130: 460.787 us | -1270.656 us | 3.757x | 20-way split-N is essential at maximum N/K. |

The submission thresholds choose the correct strategy for these three representative regions.

### Current `msprof` results

The following rounds were captured from commit `7caae52` with the standard `ops-profiling` flow: 3 warm-ups, seven `aic-metrics` groups, and one sample-based per-core collection.

| Version round | Shape / key | Task Duration | Dominant AIC pipes | Cube utilization | Core imbalance | Strict bound |
|---|---|---:|---|---:|---:|---|
| [v001/round_001](versions/v001/round_001/summary.txt) | `(4,128,128,128)`, key 121 | 16.100 us | Scalar 79.9%, FIXP 26.8%, MTE2 24.5% | 35.99% | 16.08% | No strict bound; borderline Scalar |
| [v001/round_002](versions/v001/round_002/summary.txt) | `(1,33,513,256)`, key 121 | 34.260 us | Scalar 93.8%, MTE2 75.6%, FIXP 74.3% | 4.94% | n/a, one core | **Scalar Bound** |
| [v001/round_003](versions/v001/round_003/summary.txt) | `(1,1,8192,8192)`, key 130 | 452.940 us | MTE2 66.1%, Scalar 53.6%, FIXP 4.4% | 99.08% | 0.97% | No strict bound |

Interpretation:

1. `c21_square`: the kernel itself is only 16.1 us while host p50 is 87.7 us. Launch and synchronization overhead dominate end-to-end latency. Eight blocks on a 20-core device explain the approximately 36% global Cube utilization. Per-core imbalance remains above the 10% warning threshold.
2. `c20_rect_n`: only one Cube block is launched, so global utilization is 4.94%. AIC Scalar busy reaches 93.8%, satisfying the strict Scalar Bound rule. The next material gain needs more useful inter-core work or less scalar control, not GM bandwidth tuning alone.
3. `t03_max_nk`: split-N fills all 20 Cube cores and balances them well. Kernel Task Duration accounts for almost all host p50. However, MAC busy is only 3.6% and reported Cube FLOPs are about 4.64G versus roughly 0.134G useful mathematical FLOPs. Fixed 32-row per-lane work for `M=1` causes substantial padded computation, so 99% global Cube utilization does not mean high useful-compute efficiency.

No AIV vector bank, bank-group, or resource conflict was reported in these three rounds.

### Layout sensitivity

The current submission policy ignores `transposeX1` and `transposeX2` when selecting a key. The measured FP16 p50 values show that the same policy and tile shape do not perform uniformly across layouts:

| Case | TX1/TX2 `00` | `01` | `10` | `11` |
|---|---:|---:|---:|---:|
| `c22_large_k` | 178.927 us | 91.223 us | 262.329 us | 181.246 us |
| `t03_max_nk` | 493.949 us | 275.380 us | 4997.235 us | 4099.421 us |

`TX1=true` is the largest current performance risk for maximum-K workloads. For `t03_max_nk`, the slowest layout is about 18.1x slower than the fastest layout even though both select key 130. Layout-aware policy or tiling is therefore a higher-priority investigation than small changes to the already well-balanced non-transposed split-N path.

### Same-machine regression check

Commit `5e38cba` was independently archived to `/tmp`, rebuilt with the same CANN toolchain, and measured in the same session with 10 warm-ups and 200 repeats.

| Case | Previous `5e38cba` p50 | Current `7caae52` p50 | Relative change |
|---|---:|---:|---:|
| `c20_rect_n_float16_00` | 97.813 us | 97.934 us | +0.12% |
| `c21_square_float16_00` | 88.923 us | 87.714 us | -1.36% |
| `t03_max_nk_float16_00` | 463.526 us | 460.787 us | -0.59% |

All changes are within normal run-to-run variation. The source-layout refactor in `7caae52` introduces no observable performance regression. The new `c21_square` Task Duration is also exactly 16.100 us, matching the archived pre-refactor measurement.

### Current priorities

1. Treat the `TX1=true`, large-K layout path as the highest performance risk.
2. Improve useful parallelism for small-M, medium-N work such as `c20_rect_n`; it is both under-occupied and Scalar Bound.
3. Investigate a smaller M tile for very small M to reduce padded Cube work, while preserving the good 20-core balance of key 130.
4. Avoid prioritizing further kernel-only optimization for tiny and ordinary short kernels until the roughly 60-70 us host launch/sync floor is accounted for in the target scoring path.
5. Add the official separated-operator baseline before claiming competition acceleration ratios.

## Revision history

| Date | Commit | Change |
|---|---|---|
| 2026-09-16 | `v001` | Introduced implementation-versioned storage; current measurements are now v001 rounds 001-003 and pre-version measurements are explicitly legacy. |
| 2026-09-16 | `7caae52` | Added the current 48-case hardware sample, candidate comparisons, the three measurements now stored as v001 rounds 001-003, layout sensitivity, and same-machine regression result. |
| 2026-09-14 | `5e38cba` and earlier | Recorded correctness fixes, async pipeline development, and historical split-N measurements below. |

---

## Historical analysis: 2026-09-14

Date: 2026-09-14

Platform: Ascend, CANN 9.0.0, `dav-2201`, 20 Cube cores

## Correctness result

The fused Matmul output produced by sequential `GetTensorC` is compact in the
current N tile width. The old reduction used the aligned Matmul `baseN` as its
row stride, so rows were skipped whenever `actualN != baseN`.

The fix uses `actualN` for aligned compact rows. For an unaligned compact row,
it peels at most seven FP32 elements with scalar loads, then invokes
`WholeReduceMax` only from 32-byte-aligned source and destination addresses.

Validation after the fix:

| Test | Result |
|---|---:|
| CPU golden/layout self-check | 152/152 |
| Python source/unit tests | 21/21 |
| Final default-policy correctness, FP16/BF16 and all transpose layouts | 216/216 |
| FP16 random shapes, forced key 100 | 64/64 |
| FP16 random shapes, forced key 121 | 64/64 |
| Final submission bundle, split-N target cases | 24/24 |
| Final submission bundle, maximum N/K stress (FP16/BF16) | 2/2 |

The final 216-case submission-bundle run selected key 0 for 80 cases, key 100
for 40, and key 121 for 96. The maximum N/K stress cases selected key 130
with 20 N shards. All repeated outputs stayed within tolerance.

## Repeated host-side timing

The following values are p50 launch-and-synchronize latency after 5 warm-ups
and 30 measured iterations. Both keys passed all 27 tested shapes.

| Case | Shape | key 100 | key 121 | key 121 speedup |
|---|---|---:|---:|---:|
| `c04_tie` | B2 M17 N65 K40 | 87.043 us | 89.214 us | 0.976x |
| `c16_tail63` | B2 M63 N65 K128 | 99.814 us | 90.654 us | 1.101x |
| `c20_rect_n` | B1 M33 N513 K256 | 129.564 us | 107.283 us | 1.208x |
| `c21_square` | B4 M128 N128 K128 | 94.313 us | 86.783 us | 1.087x |
| `c22_large_k` | B1 M17 N19 K8192 | 193.017 us | 181.656 us | 1.063x |

Across all 27 shapes, key 121 was faster on 19 and slower on 8. Its p50
geometric-mean speedup was 1.055x and median speedup was 1.016x. The observed
range was 0.971x to 1.242x. Regressions were concentrated in small workloads.

### Split-N crossover measurements

These are final-code host launch-and-synchronize p50 values after 5 warm-ups.
The correctness-suite rows use 100 repeats; the threshold sweep and maximum
N/K row use 50 repeats.

| Shape | N*K per M task | key 121 | key 130 | key 130 speedup |
|---|---:|---:|---:|---:|
| B1 M33 N513 K256 | 131,328 | 99.154 us | 97.633 us | 1.016x |
| B2 M17 N513 K64 | 32,832 | 85.683 us | 93.113 us | 0.920x |
| B2 M17 N1025 K64 | 65,600 | 87.493 us | 94.494 us | 0.926x |
| B1 M1 N2048 K512 | 1,048,576 | 106.744 us | 94.054 us | 1.135x |
| B1 M1 N2048 K1024 | 2,097,152 | 127.044 us | 94.914 us | 1.338x |
| B1 M1 N4096 K256 | 1,048,576 | 108.894 us | 96.384 us | 1.130x |
| B1 M1 N8192 K8192 | 67,108,864 | 1,753.035 us | 376.654 us | 4.654x |

The policy therefore requires `N*K >= 2^20`, `N >= 512`, and at least two
available shards before choosing key 130. Smaller shapes stay on key 121;
their second-stage launch is not reliably amortized.

## msprof pipeline results

### B4 M128 N128 K128

| Metric | key 100 sync/scalar | key 110 sync/vector | key 121 async/vector/DB |
|---|---:|---:|---:|
| Task Duration | 22.599 us | 19.579 us | 16.100 us |
| Longest AIC time | 21.429 us | 18.086 us | 14.674 us |
| AIV time | 17.400 us | 15.107 us | 13.985 us |
| AIC scalar busy | 83.8% | 78.4% | 73.9% |
| AIC MTE2 busy | 14.7% | 17.4% | 23.4% |
| AIC FIXP busy | 19.6% | 23.6% | 25.6% |
| Cube utilization | 37.93% | 36.95% | 36.46% |
| Per-core imbalance | 8.68% | 11.54% | 16.07% |

Key 100 is Scalar Bound by the profiling rule (`scalar > 80%`). Vectorizing
the final lane reduction and enabling asynchronous double buffering reduce
Task Duration by 28.8%. Key 121 has no single pipe above the strict bound
threshold, although scalar control remains the largest component.

The launch has 8 AIC blocks on a 20-core device. The approximately 37% global
Cube utilization is therefore primarily a multicore occupancy limit rather
than a single-core Cube throughput limit.

Raw summaries:

- `versions/legacy_unversioned/round_001`: key 100
- `versions/legacy_unversioned/round_002`: key 110
- `versions/legacy_unversioned/round_004`: key 121

### B1 M33 N513 K256

| Metric | key 100 sync/scalar | key 120 async/vector |
|---|---:|---:|
| Task Duration | 62.399 us | 31.979 us |
| Longest AIC time | 62.045 us | 31.636 us |
| AIV time | 51.844 us | 28.288 us |
| AIC scalar busy | 95.1% | 92.3% |
| AIC MTE2 busy | 41.0% | 75.1% |
| AIC FIXP busy | 40.9% | 74.1% |
| Cube utilization | 4.97% | 4.95% |
| BlockDim | 1 | 1 |

Async execution overlaps Matmul result production and vector consumption,
cutting kernel time by 48.8%. The case remains Scalar Bound and uses only one
of 20 Cube cores. The next major improvement requires multicore decomposition,
not only more single-core buffering.

Raw summaries:

- `versions/legacy_unversioned/round_003`: key 100
- `versions/legacy_unversioned/round_005`: key 120

### Historical two-launch split-N pipeline for B1 M33 N513 K256

| Metric | key 121 single stage | key 130 first stage | key 130 finalize |
|---|---:|---:|---:|
| Task Duration | 35.680 us | 34.640 us | 5.680 us |
| Longest AIC time | 35.296 us | 30.956 us | n/a |
| AIV time | 29.559 us | 29.415 us | 5.068 us |
| BlockDim | 1 | 5 | vector stage |
| Cube utilization | 4.95% | 22.34% | n/a |
| Per-core imbalance | 0.00% | 16.30% | n/a |

The measured implementation raised global Cube utilization from 4.95% to
22.34%, but its first stage still took almost as long as the single-core kernel
and its separate 5.680 us finalizer raised total device time to 40.320 us. These
numbers are retained as historical evidence. The current implementation runs
both stages in one MIX-kernel launch with an AIV software barrier. Its current
maximum-N/K behavior is recorded in `versions/v001/round_003` and the
current-status section.

Raw summaries:

- `versions/legacy_unversioned/round_006`: key 130, including both stages
- `versions/legacy_unversioned/round_007`: key 121 comparison baseline

## Tiling observations

| Shape | baseM | baseN | baseK | depthA1 | depthB1 | stepN |
|---|---:|---:|---:|---:|---:|---:|
| M128 N128 K128 | 32 | 128 | 128 | 1 | 1 | 1 |
| M33 N513 K256 | 32 | 128 | 128 | 2 | 10 | 5 |
| M17 N19 K8192 | 32 | 32 | 512 | 8 | 2 | 1 |

## Four-step optimization conclusion

1. **Tiling model:** this is a fused batch Matmul plus row reduction. The
   current implementation owns full N per B/M tile, with 32 rows per AIV and
   up to 128 columns per N tile. L1/L0 tiling is generated by the Matmul API.
2. **Inter-card pipeline:** skipped; this is not an MC2 communication operator.
3. **Inter-core pipeline:** key 130 now uses one AIV software barrier between
   partial-max production and distributed cross-shard reduction. Its GM/UB
   synchronization workspace is separate from the Matmul workspace.
4. **Single-core pipeline:** key 100 is Scalar Bound. Key 121 is the best
   existing candidate for medium/large shapes, while small cases do not
   consistently amortize its async workspace and control overhead.

Implemented optimization results and next order:

1. Implemented: route workloads with `B*M*N*K >= 300000` to key 121 after
   complete dtype/layout validation.
2. Implemented: hoist invariant `SetOrgShape` and bias state out of the
   per-task loop.
3. Implemented: key 130 writes per-row partial maxima for N shards, then all
   AIVs cross the same in-kernel barrier and distribute the shard/max plus M-sum
   work. The final conservative gate is
   `N*K >= 2^20`, `N >= 512`, and an underfilled M-task grid.
4. Evaluate a smaller TileM candidate for small M tails. It may create more M
   tasks, but must be balanced against lower Cube efficiency.
5. Do not prioritize Stream-K for the current `K <= 8192` problem range; the
   referenced Matmul model gates Stream-K at substantially larger K.
