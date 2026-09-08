"""Capacity checks used to reject impossible offline candidates."""

from __future__ import annotations

from .model import BufferPolicy, Hardware, MemoryUsage, TilingConfig


def memory_usage(
    config: TilingConfig,
    hardware: Hardware,
    policy: BufferPolicy,
) -> MemoryUsage:
    if config.path == "reference":
        atomic_local = 64 * 4
        return MemoryUsage(0, 0, 0, 0, atomic_local,
                           atomic_local <= hardware.ub_bytes - policy.ub_reserve_bytes)

    input_bytes = 2
    accumulator_bytes = 4
    a_tile = config.tile_m * config.tile_k * input_bytes
    b_tile = config.tile_k * config.tile_n * input_bytes
    c_tile = config.tile_m * config.tile_n * accumulator_bytes
    vec_tile = config.vec_m * config.vec_n * accumulator_bytes

    l1_used = policy.l1_a_buffers * a_tile + policy.l1_b_buffers * b_tile
    l0a_used = policy.l0a_buffers * a_tile
    l0b_used = policy.l0b_buffers * b_tile
    l0c_used = policy.l0c_buffers * c_tile
    ub_used = (
        policy.ub_input_buffers * vec_tile
        + config.vec_m * accumulator_bytes
        + policy.ub_row_buffers * config.vec_m * accumulator_bytes
        + 64 * accumulator_bytes
        + policy.ub_extra_scratch_bytes
    )

    legal = (
        l1_used <= hardware.l1_bytes - policy.l1_reserve_bytes
        and l0a_used <= hardware.l0a_bytes - policy.l0a_reserve_bytes
        and l0b_used <= hardware.l0b_bytes - policy.l0b_reserve_bytes
        and l0c_used <= hardware.l0c_bytes - policy.l0c_reserve_bytes
        and ub_used <= hardware.ub_bytes - policy.ub_reserve_bytes
    )
    return MemoryUsage(l1_used, l0a_used, l0b_used, l0c_used, ub_used, legal)
