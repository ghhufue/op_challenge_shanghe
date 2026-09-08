"""Shared data model for offline tiling enumeration."""

from __future__ import annotations

from dataclasses import dataclass


def ceil_div(value: int, divisor: int) -> int:
    if divisor <= 0:
        raise ValueError("divisor must be positive")
    return (value + divisor - 1) // divisor


@dataclass(frozen=True)
class Problem:
    b: int
    m: int
    n: int
    k: int
    dtype: str = "float16"
    transpose_x1: bool = False
    transpose_x2: bool = False

    def validate(self) -> None:
        if not 1 <= self.b <= 64:
            raise ValueError("B must be in [1, 64]")
        if not 1 <= self.m <= 8192 or not 1 <= self.n <= 8192:
            raise ValueError("M and N must be in [1, 8192]")
        if not 32 <= self.k <= 8192 or self.k % 8:
            raise ValueError("K must be in [32, 8192] and divisible by 8")
        if self.b * self.m * self.k > 2**26:
            raise ValueError("B*M*K exceeds 2^26")
        if self.b * self.n * self.k > 2**26:
            raise ValueError("B*N*K exceeds 2^26")
        if self.dtype not in {"float16", "bfloat16"}:
            raise ValueError("dtype must be float16 or bfloat16")


@dataclass(frozen=True)
class Hardware:
    aic: int = 24
    aiv: int = 48
    ub_bytes: int = 192 * 1024
    l1_bytes: int = 512 * 1024
    l0a_bytes: int = 64 * 1024
    l0b_bytes: int = 64 * 1024
    l0c_bytes: int = 128 * 1024

    def validate(self) -> None:
        if self.aic <= 0 or self.aiv <= 0:
            raise ValueError("core counts must be positive")
        for value in (
            self.ub_bytes,
            self.l1_bytes,
            self.l0a_bytes,
            self.l0b_bytes,
            self.l0c_bytes,
        ):
            if value <= 0:
                raise ValueError("memory capacities must be positive")


@dataclass(frozen=True)
class BufferPolicy:
    l1_a_buffers: int = 2
    l1_b_buffers: int = 2
    l0a_buffers: int = 2
    l0b_buffers: int = 2
    l0c_buffers: int = 1
    ub_input_buffers: int = 2
    ub_row_buffers: int = 2
    ub_reserve_bytes: int = 0
    l1_reserve_bytes: int = 0
    l0a_reserve_bytes: int = 0
    l0b_reserve_bytes: int = 0
    l0c_reserve_bytes: int = 0
    ub_extra_scratch_bytes: int = 0


@dataclass(frozen=True)
class TilingConfig:
    key: int
    symbol: str
    name: str
    path: str
    implemented: bool
    tile_m: int
    tile_n: int
    tile_k: int
    vec_m: int
    vec_n: int
    split_n: int


@dataclass(frozen=True)
class MemoryUsage:
    l1_used: int
    l0a_used: int
    l0b_used: int
    l0c_used: int
    ub_used: int
    legal: bool


@dataclass(frozen=True)
class CandidatePlan:
    config: TilingConfig
    split_m: int
    split_n: int
    task_count: int
    launch_blocks: int
    single_core_m: int
    single_core_n: int
    workspace_bytes: int
    memory: MemoryUsage
    estimated_score: float
