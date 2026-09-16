# Performance version v003

## Status

`MEASURED` (scoped c20, t03, and balanced-large msOpProf visualization rounds)

## Implementation identity

| Field | Value |
|---|---|
| Performance version | `v003` |
| Core implementation commit | `97d36dd` |
| Core diff SHA-256 | `b37c9878f24103a2907aaf62d7694c7289cdbb6e99e48ac32cc756da06f003d2` |
| Core change | Specialize the compile-time Matmul configuration by disabling unused Bias, QuantVector, and SelfDefineData support and selecting `ITERATE_MODE_NORMAL`; remove the runtime `DisableBias()` call. |
| Created | 2026-09-17 |
| Measurement date | 2026-09-17 |
| Device / CANN | Ascend 910 / CANN 9.0.0 / `dav-2201` |

No earlier performance rounds were copied into this version.

## Archived profiling rounds

| Round | Shape / layout / key | Purpose |
|---|---|---|
| [round_001](round_001/summary.txt) | `(1,33,513,256)`, FP16 `00`, key 121 | Complete msOpProf visualization using the existing v003 binary, without rebuilding |
| [round_002](round_002/summary.txt) | `(1,1,8192,8192)`, FP16 `00`, key 130 | Complete msOpProf visualization of the 20-way split-N large case, without rebuilding |
| [round_003](round_003/summary.txt) | `(4,512,512,512)`, FP16 `00`, key 121 | Complete msOpProf visualization of a dimension-balanced large case, without rebuilding |

## Validation and scope

The current implementation was built successfully from fresh sources for
`dav-2201`. Static tests passed 23/23. Targeted hardware checks passed for:

- c20 FP16/BF16 and all four input layouts: 8/8;
- c21 multi-core async, c22 large-K TX1, and t03 20-way split-N: 3/3.

Four interleaved 200-repeat c20 FP16 `00` host-side rounds did not show a
robust latency change. The v002/v003 mean p50 values were 99.226/99.108 us
(0.12% lower), which is smaller than run-to-run variation, while the upper
median p95 changed from 114.931 to 116.241 us. The full seven-group profiler
gate was therefore not opened.

Rounds 001 through 003 are scoped visualization collections, not the standard
seven-group regression workflow. They produced real Details, Roofline,
Cache, and Raw Data pages. Timeline, Source, Warp Stall, On-Chip Memory, and
Instruction Timeline remain diagnostic pages because the installed CLI or
existing non-debug build did not provide their required semantic payloads.

## Revision record

| Date | Change |
|---|---|
| 2026-09-17 | Created v003 as `UNMEASURED`; no historical rounds copied. |
| 2026-09-17 | Recorded fresh build, 11 targeted hardware checks, and the inconclusive c20 host gate. |
| 2026-09-17 | Archived scoped c20 msOpProf visualization round 001 and marked v003 `MEASURED`. |
| 2026-09-17 | Archived large t03 20-way split-N visualization round 002. |
| 2026-09-17 | Archived balanced `(4,512,512,512)` visualization round 003. |
