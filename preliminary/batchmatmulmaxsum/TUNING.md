# BatchMatmulMaxSum tuning workflow

The catalog keeps the correctness path, the stable automatic-API baseline and
three ordered single-core experiments:

- key `0`, `VECTOR_REFERENCE`: correctness and tiny-shape fallback;
- key `100`, `AUTO_MATMUL_FUSED`: the fixed high-level Matmul API baseline.
- key `110`, `AUTO_MATMUL_FUSED_VECTOR`: synchronous Matmul with vector-only
  per-tile max accumulation and a vector lane sum;
- key `120`, `AUTO_MATMUL_FUSED_ASYNC`: key 110 plus asynchronous Iterate and
  one VECIN UB slot;
- key `121`, `AUTO_MATMUL_FUSED_ASYNC_DB`: key 120 plus two VECIN UB slots and
  a prologue/steady-state/epilogue drain.
- key `122`, `AUTO_MATMUL_FUSED_ASYNC_DB_M32`: tuning-only key 121 variant
  with `tile_m=32` and 16 rows per AIV lane. It tests whether a second M task
  improves small-M under-occupancy without paying split-N synchronization.
- key `130`, `AUTO_MATMUL_FUSED_SPLIT_N`: dynamically partitions N to fill
  otherwise idle AIC cores. One MIX kernel first stores padded per-row partial
  maxima, synchronizes all of its AIV subblocks with a dedicated software
  barrier, then takes max across N shards before summing M.

Key 100 uses `matmul::Matmul` with C in `VECIN`. It does not use a user-written
cross-core flag protocol or a GM C-tile staging buffer. Keep key 100 stable so
future automatic-API tilings and manual-flag implementations have a reproducible
baseline. Each AIV accumulates its shard sums per batch in UB and performs one
aligned FP32 AtomicAdd before leaving the single MIX kernel.

Keys 120 and 121 follow the CANN asynchronous Iterate contract: each AIV owns a
private GM result cache, calls `SetWorkspace` and `Iterate<false>` once per M
task, then drains exactly `ceil(singleCoreN/baseN)` tiles with
`GetTensorC<false>`. Their extra GM traffic is intentional and must be measured
against keys 100 and 110; asynchronous candidates are not selected by the
submission policy until passing measurements have been fitted.

For key 100, `tile_m=64` is the two-lane shard size, `vec_m=32` is the row count
owned by each AIV, and `tile_n=128` is the maximum VECIN tile capacity. The Host
tiler chooses `baseN=min(128, align_up(N, 16))` so small-N shapes remain legal;
the `TCubeTiling` returned by CANN is authoritative for the actual base sizes.

## Catalog

`configs/tiling_candidates.json` remains the source of candidate identities.
After changing it, run:

```bash
python scripts/generate_tiling_catalog.py
python scripts/generate_tiling_catalog.py --check
```

For ordinary candidates `split_n=1`. A value greater than one marks a dynamic
split-N candidate: the host chooses up to `ceil(N / tile_n)` balanced shards,
capped so the expanded task grid does not oversubscribe available AIC cores.
If only one shard is useful, key 130 reuses the key 121 execution path. N
shards cannot use the regular atomic-sum output because their partial results
require max before the final M reduction.

The submission policy selects key 130 only when the device is underfilled and
each M task has at least `N*K >= 2^20` work. Measurements below that crossover
showed that the historical separate finalizer and workspace reduction could
outweigh the extra parallelism. The finalizer is now fused into the same kernel
launch; the conservative crossover remains until the fused version is profiled.

Suggested key ranges:

```text
100        fixed automatic-fusion baseline
110-119    automatic API vector or baseM/baseN/baseK variants
120-129    automatic API async and UB-pipeline variants
130-139    automatic API split-N and in-kernel second-stage reduction variants
200-299    manual flag-controlled fusion variants
```

## Inspect and benchmark

Inspect legal candidates without an NPU:

```bash
python -m tuning.cli --B 1 --M 4096 --N 8192 --K 1024
```

Benchmark every implemented key on the target device:

```bash
python scripts/tune_cases.py \
  --exe build/batch_matmul_max_sum_custom \
  --cann-root /path/to/cann \
  --suite correctness \
  --warmups 10 \
  --repeats 100
```

Every new key must pass smoke, correctness, fuzz, repeated-run and transpose
coverage under forced-key execution before it is eligible for policy fitting.
Do not overwrite key 100 with an experiment.

Run the single-core sequence in dependency order so a regression can be
attributed to one change:

```bash
python scripts/tune_cases.py --exe build/batch_matmul_max_sum_custom \
  --cann-root /path/to/cann --suite correctness --key 100 --key 110
python scripts/tune_cases.py --exe build/batch_matmul_max_sum_custom \
  --cann-root /path/to/cann --suite correctness --key 120
python scripts/tune_cases.py --exe build/batch_matmul_max_sum_custom \
  --cann-root /path/to/cann --suite correctness --key 121
python scripts/tune_cases.py --exe build/batch_matmul_max_sum_custom \
  --cann-root /path/to/cann --suite all --repeats 100
```

The final benchmark must cover one-, two- and many-N-tile shapes and the full K
range. `tuning.fit_policy` already exposes N and K as independent features, so
the generated policy can retain synchronous candidates for small N/K and choose
an asynchronous candidate only where measurements show a benefit.

Once complete measurements exist for at least two candidates, generate a
policy with:

```bash
python -m tuning.fit_policy tuning_results/benchmark.json --max-depth 3
```

The generated policy may use only shape, dtype and transpose metadata. Review
the thresholds before committing it.

## Submission bundle

Generate the self-contained source and policy header with either:

```bash
python scripts/bundle_submission.py
```

or:

```powershell
./scripts/bundle_submission.ps1
```

Always compile the regenerated bundle before submission. Build artifacts,
test data and large benchmark result files remain local and are ignored by Git.
