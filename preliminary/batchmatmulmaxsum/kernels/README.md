# Kernel source layout

`auto_matmul_fused.asc` and `kernel_dispatch.asc` are stable aggregation entry
points. They contain includes only; edit the fragments below instead.

## `matmul_api/`

- `class_header.inc`: template declaration and class opening
- `init.inc`: buffer and tensor initialization
- `process.inc`: task traversal and split-N stage sequencing
- `synchronous_tiles.inc`: synchronous Matmul result consumption
- `asynchronous_tiles.inc`: asynchronous/double-buffered result consumption
- `row_max.inc`, `tile_reduction.inc`, `lane_sum.inc`: reduction helpers
- `partial_max_and_state.inc`: split-N partial output plus class state
- `split_n_reduce.asc`: in-kernel second-stage split-N reduction
- `launch.inc`: global kernel and host launch wrappers

The `.inc` files form one template class and must remain included in the order
listed by `auto_matmul_fused.asc`.

## `dispatch/`

- `resources.inc`: persistent workspace and tiling resource cache
- `reference_plan.inc`: reference-path executor
- `auto_plan.inc`: ordinary auto-fused executor
- `split_n_plan.inc`: split-N executor
- `execute_plan.inc`: tiling-key dispatch

## `reference/`

- `kernel.asc`: small-shape reference kernel

When files are added, removed, or reordered, update both submission bundlers in
`scripts/` and regenerate `submission/kernel.asc`.
