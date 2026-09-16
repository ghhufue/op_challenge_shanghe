# Repository agent instructions

## BatchMatmulMaxSum performance-version gate

These instructions apply to every AI agent that changes files under `preliminary/batchmatmulmaxsum`.

Before changing operator runtime behavior, read:

- `preliminary/batchmatmulmaxsum/docs/perf/README.md`
- `preliminary/batchmatmulmaxsum/docs/perf/CURRENT_VERSION`
- `preliminary/batchmatmulmaxsum/docs/perf/benchmark_workflow.md`

A core-logic change MUST increment `docs/perf/CURRENT_VERSION` from `vNNN` to the next unused number in the same change. Create `docs/perf/versions/vNNN/manifest.md` for the new version and initially mark it `UNMEASURED`. Never copy an older version's performance rounds into the new version.

Core logic includes any change that can affect numerical behavior, selected execution path, generated device code, workspace, launch shape, or measured latency, including:

- `kernel.asc`, `main.asc`, and `CMakeLists.txt` when build or launch behavior changes;
- `kernels/**`, `host/**`, `tiling/**`, and `configs/tiling_candidates.json`;
- `submission/kernel.asc` and `submission/submission_policy.h`;
- generators or bundlers when their output changes the submitted implementation.

Increment once for each independently reviewable core implementation change. If core logic changes again before the previous version is measured, increment again and retain the prior version as `UNMEASURED`; version numbers are never reused.

After measuring a version:

1. Put new profiler archives only under `docs/perf/versions/<CURRENT_VERSION>/round_NNN/`.
2. Update that version's `manifest.md` with commit, environment, cases, round mapping, and status `MEASURED`.
3. Update `docs/perf/analysis.md` so its current section points to `CURRENT_VERSION` only.
4. Keep older versions immutable except for explicit metadata corrections; never describe their measurements as current.

Documentation-only, test-only, or host tooling-only changes do not require a performance-version increment unless they alter compiled operator behavior or the measurement semantics. When uncertain, increment the version and mark it `UNMEASURED`.
