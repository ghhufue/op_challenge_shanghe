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

The optimized paths use one `KERNEL_TYPE_MIX_AIC_1_2` launch. Work is divided
over `(batch, M group, N group)`, with enough round-robin N groups to fill the
available Cube cores. Each logical AIC streams its assigned N tiles through two
compact FP32 GM staging buffers. Its paired AIV sub-blocks consume half of the M
rows each and use `WholeReduceMax` to maintain online row maxima in UB. The AIV
workers first merge N-group partial maxima, then publish one padded batch-sum
vector per worker for the final reduction to `y`. Cross-core flag 5 announces a
completed Matmul tile; flags 6 and 7 return the two ping-pong buffers only after
the AIV online reduction has consumed them. There are no host
stream synchronizations, auxiliary compute launches, or per-call workspace
frees in this path. Matmul tiling is uploaded synchronously to persistent GM
once per cached stream/shape plan, so it does not add a task to the execution
stream; subsequent calls reuse the same tiling buffer.

Keys 100 through 103 retain the `16 x 128 x 64`, `32 x 128 x 64`,
`32 x 256 x 64`, and `64 x 128 x 128` tile choices. Keys 200 and 201 both use
`16 x 256 x 64`; their old fixed N-split values are retained only as policy
identities because the MIX planner now chooses the useful N-group count from
the shape and available Cube cores.
The checked-in production policy preserves the previously measured key choices,
but changing from multiple launches to one fused launch changes their relative
costs, so benchmark all keys again before treating that policy as newly tuned.

The production policy is kept in `tiling/submission_policy.h`. It must select
only implemented keys and always retain a legal general fallback. Derive its
branches from measured device results, then test both sides of every threshold.

Once at least two implemented keys have complete measurements, a shallow
regret-minimizing tree can be generated with:

```bash
python -m tuning.fit_policy tuning_results/benchmark.json --max-depth 3
```

Review the generated conditions before committing them. Exact case IDs and
input values are never used as policy features.

For competition submission, run `scripts/bundle_submission.ps1` and create only
`kernel.asc` and `submission_policy.h` in the submission UI. Frequent policy
changes require replacing only `submission_policy.h`; keep both files in the
same virtual directory so the quoted include resolves.
