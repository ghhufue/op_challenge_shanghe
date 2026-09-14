"""Jointly materialize tile, algorithm-path and cross-core candidate plans."""

from __future__ import annotations

import math

from .catalog import load_catalog
from .constraints import memory_usage
from .model import BufferPolicy, CandidatePlan, Hardware, Problem, TilingConfig, ceil_div


def _estimate(
    problem: Problem,
    config: TilingConfig,
    launch_blocks: int,
    available_aic: int,
) -> float:
    if config.path == "reference":
        return 0.0
    useful_m = min(problem.m, config.tile_m) / config.tile_m
    useful_n = min(problem.n, config.tile_n) / config.tile_n
    useful_k = min(problem.k, config.tile_k) / config.tile_k
    macs = config.tile_m * config.tile_n * config.tile_k
    traffic = 2 * config.tile_k * (config.tile_m + config.tile_n)
    parallelism = launch_blocks / available_aic
    return macs / max(traffic, 1) * math.sqrt(
        useful_m * useful_n * useful_k,
    ) * parallelism


def _materialize(
    problem: Problem,
    hardware: Hardware,
    policy: BufferPolicy,
    config: TilingConfig,
) -> CandidatePlan | None:
    memory = memory_usage(config, hardware, policy)
    if not memory.legal:
        return None

    if config.path == "reference":
        task_count = problem.b * problem.m
        split_m = min(problem.m, hardware.aiv)
        launch_blocks = min(task_count, hardware.aiv)
        workspace_bytes = ceil_div(problem.b, 8) * 8 * 4
        return CandidatePlan(
            config, split_m, 1, task_count, launch_blocks,
            1, problem.n, workspace_bytes, memory, 0.0,
        )

    max_m_groups = ceil_div(problem.m, config.tile_m)
    m_tasks = problem.b * max_m_groups
    if config.split_n > 1:
        split_n = min(
            ceil_div(problem.n, config.tile_n),
            max(1, hardware.aic // m_tasks),
        )
        split_n = max(split_n, 1)
        n_shard_columns = 0
        for shard in range(split_n):
            begin = (problem.n * shard // split_n // 16) * 16
            end = (
                problem.n
                if shard + 1 == split_n
                else (problem.n * (shard + 1) // split_n // 16) * 16
            )
            n_shard_columns = max(n_shard_columns, end - begin)
    else:
        split_n = 1
        n_shard_columns = problem.n
    split_m = min(max_m_groups, hardware.aic)
    task_count = m_tasks * split_n
    launch_blocks = min(task_count, hardware.aic)
    single_core_m = min(problem.m, config.vec_m)
    single_core_n = min(problem.n, n_shard_columns)
    workspace_cursor = ceil_div(
        hardware.system_workspace_bytes, 512,
    ) * 512
    if config.schedule == "async":
        padded_n = ceil_div(n_shard_columns, config.tile_n) * config.tile_n
        client_stride = ceil_div(
            config.vec_m * padded_n * 4, 512,
        ) * 512
        workspace_cursor += launch_blocks * 2 * client_stride
    if config.split_n > 1:
        partial_max_offset = ceil_div(workspace_cursor, 512) * 512
        workspace_cursor = (
            partial_max_offset
            + task_count * 2 * config.vec_m * 4
        )
    atomic_output_offset = ceil_div(workspace_cursor, 512) * 512
    atomic_output_bytes = ceil_div(problem.b, 8) * 8 * 4
    workspace_bytes = ceil_div(
        atomic_output_offset + atomic_output_bytes, 512,
    ) * 512
    score = _estimate(problem, config, launch_blocks, hardware.aic)
    return CandidatePlan(
        config, split_m, split_n, task_count, launch_blocks,
        single_core_m, single_core_n, workspace_bytes, memory, score,
    )


def enumerate_plans(
    problem: Problem,
    hardware: Hardware = Hardware(),
    policy: BufferPolicy = BufferPolicy(),
    *,
    implemented_only: bool = False,
) -> list[CandidatePlan]:
    problem.validate()
    hardware.validate()
    plans = []
    for config in load_catalog():
        if implemented_only and not config.implemented:
            continue
        plan = _materialize(problem, hardware, policy, config)
        if plan is not None:
            plans.append(plan)
    return sorted(plans, key=lambda item: item.estimated_score, reverse=True)
