# Tiling development workflow

The runtime and offline tuner share `configs/tiling_candidates.json` as the
single source of candidate identities and static tile parameters. Run:

```bash
python scripts/generate_tiling_catalog.py
python scripts/generate_tiling_catalog.py --check
```

after editing the catalog. Commit the generated
`tiling/tiling_catalog_generated.h` together with the JSON source.

## Inspect candidates without an NPU

```bash
python -m tuning.cli --B 1 --M 4096 --N 8192 --K 1024
python -m tuning.cli --B 1 --M 4096 --N 8192 --K 1024 --json
```

Capacity checks reject impossible tiles. `estimated_score` is only a cheap
ordering hint and must not be reported as measured performance.
The default offline hardware profile is the 20-AIC/40-AIV competition target;
pass explicit `Hardware` values when exploring another device.

## Benchmark implemented candidates

The local CMake target defines `BMMS_ENABLE_TUNING`. It accepts an optional
final positional `tiling_key`, while the competition `run_kernel` ABI remains
unchanged. Benchmark all implemented keys with:

```bash
python scripts/tune_cases.py \
  --exe build/batch_matmul_max_sum_custom \
  --cann-root /path/to/cann \
  --suite smoke \
  --repeats 100
```

Candidates with `implemented: false` can be inspected by the planner but are
rejected by both the tuning runner and the C++ dispatcher. Change the flag only
after the corresponding kernel path exists and passes forced-path tests.

The BM path uses compile-time tile instances selected by tiling key. Keys
100 through 103 use `16 x 128 x 64`, `32 x 128 x 64`, `32 x 256 x 64`, and
`64 x 128 x 128`, respectively; all share the same Matmul, row-max and final
atomic-reduction implementation. Each AIC computes one
compact FP32 `tileM x tileN` result, and an AIV kernel folds it into the persistent
task-major row maxima. Every BM task owns one padded `tileM` segment, which the
AIV kernel maintains in UB and transfers between N tiles with aligned `DataCopy`
operations. Tiles are processed in core-sized waves, so the workspace is the
aligned task-padded row-max buffer, one staging tile per active AIC, and a padded
atomic output accumulator, rather than a full `B x M x N` matrix. Force keys
100 through 103 locally to validate and benchmark them; the production policy
intentionally remains on key 0 for now.

The BMN path adds N-group parallelism for shapes where `B * ceil(M / tileM)`
does not fill the AICs. Key 200 uses `16 x 256 x 64` with at most two N groups,
and key 201 uses the same tile with at most four N groups. N tiles are assigned
round-robin to groups. Each `(batch, M group, N group)` task maintains an aligned
partial row-max tile in GM; after all groups finish, a separate AIV kernel takes
the maximum across N groups, sums the valid M rows, and atomically accumulates
the batch result. Keys 200 and 201 are available only through forced-key testing
until measured device results justify a production-policy branch.

The production policy is kept in `tiling/generated_policy.inc`. It must select
only implemented keys and always retain a legal general fallback. Derive its
branches from measured device results, then test both sides of every threshold.

Once at least two implemented keys have complete measurements, a shallow
regret-minimizing tree can be generated with:

```bash
python -m tuning.fit_policy tuning_results/benchmark.json --max-depth 3
```

Review the generated conditions before committing them. Exact case IDs and
input values are never used as policy features.
