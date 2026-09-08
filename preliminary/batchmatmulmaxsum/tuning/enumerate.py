"""Jointly materialize tile, algorithm-path and cross-core candidate plans."""

from __future__ import annotations

import math

from .catalog import load_catalog
from .constraints import memory_usage
from .model import BufferPolicy, CandidatePlan, Hardware, Problem, TilingConfig, ceil_div


def _estimate(problem: Problem, config: TilingConfig, launch_blocks: int) -> float:
    if config.path == "reference":
        return 0.0
    useful_m = min(problem.m, config.tile_m) / config.tile_m
    useful_n = min(problem.n, config.tile_n) / config.tile_n
    useful_k = min(problem.k, config.tile_k) / config.tile_k
    macs = config.tile_m * config.tile_n * config.tile_k
    traffic = 2 * config.tile_k * (config.tile_m + config.tile_n)
    parallelism = launch_blocks / 24.0
    n_penalty = 1.0 / config.split_n
    return macs / max(traffic, 1) * math.sqrt(useful_m * useful_n * useful_k) * parallelism * n_penalty


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
        workspace_bytes = task_count * 4
        return CandidatePlan(
            config, split_m, 1, task_count, launch_blocks,
            1, problem.n, workspace_bytes, memory, 0.0,
        )

    max_m_groups = ceil_div(problem.m, config.tile_m)
    split_n = min(config.split_n, max(1, ceil_div(problem.n, config.tile_n)))
    available_per_batch_n = max(1, hardware.aic // max(1, problem.b * split_n))
    split_m = min(max_m_groups, available_per_batch_n)
    task_count = problem.b * split_m * split_n
    launch_blocks = min(task_count, hardware.aic)
    single_core_m = ceil_div(problem.m, split_m)
    single_core_n = ceil_div(problem.n, split_n)
    if split_n == 1:
        workspace_bytes = problem.b * split_m * 4
    else:
        workspace_bytes = problem.b * split_m * split_n * single_core_m * 4
        workspace_bytes += problem.b * split_m * 4
    score = _estimate(problem, config, launch_blocks)
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
