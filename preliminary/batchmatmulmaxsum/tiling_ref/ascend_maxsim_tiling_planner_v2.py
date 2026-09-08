#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ascend 910B3 / Atlas A2
MaxSim + SumReduction Tiling & Buffer Planner v2
================================================

目标算子
--------
逻辑输入：
    x1: [B, M, K]
    x2: [B, K, N]

计算：
    sim[b,m,n] = sum_k x1[b,m,k] * x2[b,k,n]    # FP32 accumulate
    maxsim[b,m] = max_n sim[b,m,n]              # FP32
    y[b] = sum_m maxsim[b,m]                    # FP32

============================================================
一、核心切分逻辑
============================================================

跨核优先级：

    B -> M -> N -> K

具体规则：

1) B
   不同 batch 完全独立，天然适合并行。

2) M
   是首选的跨 AIC 切分维。
   原因：
       每个 M-group 扫描完整 N 后，可以直接得到自己的 MaxSim；
       最后跨 M-group 只需要 Sum 若干 FP32 partialSum，代价很低。

3) N
   只有 B/M 并行度不足以填满 AIC 时，才跨核切 N。
   原因：
       跨 N 后每个核只能得到 partialMax[M_chunk]；
       必须再对不同 N-group 做 Max reduction，
       workspace 和 GM traffic 都比只切 M 更大。

4) K
   默认绝不跨 AIC 切。
   原因：
       K 是点积 reduction 维；
       若跨核切 K，必须先对完整 [M_tile, N_tile] partial matrix 求和，
       然后才能做 Max(N)，代价很大。

   但是：
       K 必须在单个 AIC 内按 baseK 分块：
           C_fp32 += A[M, Ktile] @ B[Ktile, N]
       直到完整 K reduction 完成。

============================================================
二、为什么 v2 要把 Cube tile 和 Vector tile 分开
============================================================

AIC 侧主要关心：
    L1 / L0A / L0B / L0C

AIV 侧主要关心：
    UB

对于 AIC/AIV 分离架构，不应该简单假设：

    一个 [baseM, baseN] 的 C tile
    必须同时完整存在于 L0C 和 UB。

更合理的规划模型是：

    AIC:
        [baseM, baseK] x [baseK, baseN]
              -> FP32 C cube tile [baseM, baseN]

    AIC -> GM/workspace bridge（规划模型）
              -> AIV

    AIV:
        再把 C cube tile 切成
        [vecM, vecN]
        的 Vector reduction tile放入 UB，
        做 Max(N) / runningMax 更新。

因此：

    Cube tile:
        baseM, baseN, baseK

    Vector tile:
        vecM, vecN

是两套不同粒度。

============================================================
三、Buffer 模型
============================================================

AIC / L1
--------
A tile bytes:
    baseM * baseK * sizeof(input)

B tile bytes:
    baseK * baseN * sizeof(input)

L1 occupancy:
    l1_a_buffers * A_tile
  + l1_b_buffers * B_tile

默认：
    l1_a_buffers = 2
    l1_b_buffers = 2

即 A/B 都使用 Ping-Pong。

AIC / L0A
---------
L0A occupancy:
    l0a_buffers * A_tile

默认：
    l0a_buffers = 2

AIC / L0B
---------
L0B occupancy:
    l0b_buffers * B_tile

默认：
    l0b_buffers = 2

AIC / L0C
---------
C tile 为 FP32：

    C_tile_bytes = baseM * baseN * 4

L0C occupancy:
    l0c_buffers * C_tile_bytes

默认：
    l0c_buffers = 1

是否能/应该对 L0C 做双缓冲取决于具体实现，
因此脚本把它做成可配置参数，而不是写死。

AIV / UB
--------
AIV 并不强制一次装下整个 Cube C tile。

Vector input tile:
    vecM * vecN * sizeof(FP32)

UB input buffering:
    ub_input_buffers * vector_input_tile

默认：
    ub_input_buffers = 2

此外 UB 还估算：
    runningMax[vecM]         FP32
    rowMax scratch           FP32
    UB extra scratch         可配置
    UB reserve               可配置

============================================================
四、Tail 规则
============================================================

M tail:
    最终 Sum 只累计 actualM。

K tail:
    无效 K 可以补 0，因为 K 是加法 reduction。

N tail:
    绝不能简单补 0 后参加 Max。
    因为真实相似度可能全部是负数。

正确方案：
    - 只对 actualN 做 Max；或
    - 无效 lane 填 -INF。

runningMax:
    必须初始化为 -INF，不能初始化为 0。

============================================================
五、默认硬件参数（可全部覆盖）
============================================================

Atlas A2 / 910B3 常用规划默认值：

    AIC  = 24
    AIV  = 48

    UB   = 192 KiB
    L1   = 512 KiB
    L0A  = 64 KiB
    L0B  = 64 KiB
    L0C  = 128 KiB

生产 Host Tiling 中应优先通过 PlatformAscendC 查询真实平台信息：
    GetCoreNumAic()
    GetCoreNumAiv()
    GetCoreMemSize(...)

============================================================
六、使用方法
============================================================

交互模式：
    python ascend_maxsim_tiling_planner_v2.py

直接输入 shape：
    python ascend_maxsim_tiling_planner_v2.py \
        --B 1 --M 4096 --N 8192 --K 1024

修改硬件：
    python ascend_maxsim_tiling_planner_v2.py \
        --B 1 --M 128 --N 32768 --K 1024 \
        --aic 20 --aiv 40 \
        --ub-kib 192 --l1-kib 512 \
        --l0a-kib 64 --l0b-kib 64 --l0c-kib 128

修改 Buffer 深度：
    python ascend_maxsim_tiling_planner_v2.py \
        --B 1 --M 4096 --N 8192 --K 1024 \
        --l1-a-buffers 2 --l1-b-buffers 2 \
        --l0a-buffers 2 --l0b-buffers 2 \
        --l0c-buffers 1 \
        --ub-input-buffers 2 --ub-row-buffers 2

给各级存储留安全余量：
    python ascend_maxsim_tiling_planner_v2.py \
        --B 1 --M 4096 --N 8192 --K 1024 \
        --ub-reserve-kib 8 \
        --l1-reserve-kib 32

JSON 输出：
    python ascend_maxsim_tiling_planner_v2.py \
        --B 1 --M 4096 --N 8192 --K 1024 --json
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def align_up(x: int, align: int) -> int:
    return ceil_div(x, align) * align


def kib(x: int) -> int:
    return x * 1024


def human_bytes(x: int) -> str:
    if x >= 1024 * 1024:
        return f"{x / (1024 * 1024):.2f} MiB"
    if x >= 1024:
        return f"{x / 1024:.2f} KiB"
    return f"{x} B"


def pct(used: int, cap: int) -> float:
    return 100.0 * used / cap if cap > 0 else 0.0


def prompt_int(name: str, default: int) -> int:
    raw = input(f"{name} [{default}]: ").strip()
    return default if raw == "" else int(raw)


@dataclass
class Hardware:
    aic: int = 24
    aiv: int = 48
    ub: int = kib(192)
    l1: int = kib(512)
    l0a: int = kib(64)
    l0b: int = kib(64)
    l0c: int = kib(128)


@dataclass
class BufferPolicy:
    # L1：A/B staging buffer 深度
    l1_a_buffers: int = 2
    l1_b_buffers: int = 2

    # L0：Cube operand/output buffer 深度
    l0a_buffers: int = 2
    l0b_buffers: int = 2
    l0c_buffers: int = 1

    # UB：Vector reduction 输入和行归约 scratch
    ub_input_buffers: int = 2
    ub_row_buffers: int = 2

    # 每级存储的安全余量
    ub_reserve: int = 0
    l1_reserve: int = 0
    l0a_reserve: int = 0
    l0b_reserve: int = 0
    l0c_reserve: int = 0

    # UB 额外 scratch
    ub_extra_scratch: int = 0


@dataclass
class Problem:
    B: int
    M: int
    N: int
    K: int
    dtype: str = "fp16"
    transpose_x1: bool = False
    transpose_x2: bool = False


@dataclass
class MemoryUsage:
    a_tile: int
    b_tile: int
    c_tile: int
    vec_input_tile: int

    l1_used: int
    l0a_used: int
    l0b_used: int
    l0c_used: int
    ub_used: int

    l1_available: int
    l0a_available: int
    l0b_available: int
    l0c_available: int
    ub_available: int

    ub_input_used: int
    ub_running_max: int
    ub_row_scratch: int
    ub_extra_scratch: int


@dataclass
class TilePlan:
    baseM: int
    baseN: int
    baseK: int

    vecM: int
    vecN: int

    memory: MemoryUsage

    cube_score: float
    vector_score: float
    total_score: float


@dataclass
class SplitPlan:
    splitM: int
    splitN: int

    total_tasks: int
    active_aic: int
    aic_utilization: float

    singleCoreM: int
    singleCoreN: int

    needs_n_reduction: bool
    workspace_bytes: int

    bm_parallel_capacity: int
    decision_reason: str


def available(capacity: int, reserve: int) -> int:
    return max(0, capacity - reserve)


def validate_buffer_policy(p: BufferPolicy) -> None:
    for f in (
        "l1_a_buffers", "l1_b_buffers",
        "l0a_buffers", "l0b_buffers", "l0c_buffers",
        "ub_input_buffers", "ub_row_buffers",
    ):
        if getattr(p, f) <= 0:
            raise ValueError(f"{f} must be > 0")

    for f in (
        "ub_reserve", "l1_reserve",
        "l0a_reserve", "l0b_reserve", "l0c_reserve",
        "ub_extra_scratch",
    ):
        if getattr(p, f) < 0:
            raise ValueError(f"{f} must be >= 0")


def calculate_memory_usage(
    baseM: int,
    baseN: int,
    baseK: int,
    vecM: int,
    vecN: int,
    hw: Hardware,
    policy: BufferPolicy,
    input_bytes: int = 2,
    acc_bytes: int = 4,
) -> MemoryUsage:
    """
    详细容量模型：

    AIC:
        A_tile = baseM * baseK * input_bytes
        B_tile = baseK * baseN * input_bytes
        C_tile = baseM * baseN * acc_bytes

        L1  = l1_a_buffers*A_tile + l1_b_buffers*B_tile
        L0A = l0a_buffers*A_tile
        L0B = l0b_buffers*B_tile
        L0C = l0c_buffers*C_tile

    AIV:
        vec_input = vecM * vecN * acc_bytes

        UB =
            ub_input_buffers * vec_input
          + runningMax[vecM]
          + ub_row_buffers * rowScratch[vecM]
          + ub_extra_scratch

    注意：
    这是容量/规划模型，不是 CANN 内部 cycle-accurate allocator。
    """

    a_tile = baseM * baseK * input_bytes
    b_tile = baseK * baseN * input_bytes
    c_tile = baseM * baseN * acc_bytes

    vec_input_tile = vecM * vecN * acc_bytes

    l1_used = (
        policy.l1_a_buffers * a_tile
        + policy.l1_b_buffers * b_tile
    )

    l0a_used = policy.l0a_buffers * a_tile
    l0b_used = policy.l0b_buffers * b_tile
    l0c_used = policy.l0c_buffers * c_tile

    ub_input_used = policy.ub_input_buffers * vec_input_tile
    ub_running_max = vecM * acc_bytes
    ub_row_scratch = policy.ub_row_buffers * vecM * acc_bytes

    ub_used = (
        ub_input_used
        + ub_running_max
        + ub_row_scratch
        + policy.ub_extra_scratch
    )

    return MemoryUsage(
        a_tile=a_tile,
        b_tile=b_tile,
        c_tile=c_tile,
        vec_input_tile=vec_input_tile,

        l1_used=l1_used,
        l0a_used=l0a_used,
        l0b_used=l0b_used,
        l0c_used=l0c_used,
        ub_used=ub_used,

        l1_available=available(hw.l1, policy.l1_reserve),
        l0a_available=available(hw.l0a, policy.l0a_reserve),
        l0b_available=available(hw.l0b, policy.l0b_reserve),
        l0c_available=available(hw.l0c, policy.l0c_reserve),
        ub_available=available(hw.ub, policy.ub_reserve),

        ub_input_used=ub_input_used,
        ub_running_max=ub_running_max,
        ub_row_scratch=ub_row_scratch,
        ub_extra_scratch=policy.ub_extra_scratch,
    )


def memory_fits(mem: MemoryUsage) -> bool:
    return (
        mem.l1_used <= mem.l1_available
        and mem.l0a_used <= mem.l0a_available
        and mem.l0b_used <= mem.l0b_available
        and mem.l0c_used <= mem.l0c_available
        and mem.ub_used <= mem.ub_available
    )


def cube_score(
    baseM: int,
    baseN: int,
    baseK: int,
    problem: Problem,
) -> float:
    """
    启发式评分，不是假装成真实性能模型。

    奖励：
    - 较大的 M/N tile
    - 较好的 compute / A+B tile traffic 比
    - b16 下 baseK=64 的官方样例先验

    惩罚：
    - tile 明显大于实际 M/N/K 时的 tail 浪费
    """
    input_bytes = 2

    macs = baseM * baseN * baseK
    ab_bytes = (
        baseM * baseK
        + baseK * baseN
    ) * input_bytes

    intensity = macs / max(ab_bytes, 1)

    k_bonus = {
        16: 0.92,
        32: 1.00,
        64: 1.15,
        128: 1.05,
        256: 0.95,
    }.get(baseK, 0.90)

    area_bonus = math.sqrt(baseM * baseN)

    util_m = min(problem.M, baseM) / baseM
    util_n = min(problem.N, baseN) / baseN
    util_k = min(problem.K, baseK) / baseK
    shape_util = util_m * util_n * util_k

    return intensity * area_bonus * k_bonus * math.sqrt(shape_util)


def vector_score(
    vecM: int,
    vecN: int,
    baseM: int,
    baseN: int,
    mem: MemoryUsage,
) -> float:
    """
    Vector tile 优先：
    - tile 较大
    - 一个 Cube C tile 被切成的 Vector subtile 数较少
    - UB 保留一定余量
    """
    area = vecM * vecN

    subtiles = (
        ceil_div(baseM, vecM)
        * ceil_div(baseN, vecN)
    )

    if mem.ub_available <= 0:
        return 0.0

    pressure = mem.ub_used / mem.ub_available
    headroom_factor = max(0.65, 1.0 - 0.20 * pressure)

    return (area / subtiles) * headroom_factor


def enumerate_tile_plans(
    problem: Problem,
    hw: Hardware,
    policy: BufferPolicy,
) -> List[TilePlan]:

    cube_m_candidates = [16, 32, 64, 128, 256]
    cube_n_candidates = [16, 32, 64, 128, 256, 512]
    cube_k_candidates = [16, 32, 64, 128, 256]

    vector_m_candidates = [16, 32, 64, 128, 256]
    vector_n_candidates = [16, 32, 64, 128, 256, 512]

    plans: List[TilePlan] = []

    for bm in cube_m_candidates:
        for bn in cube_n_candidates:
            for bk in cube_k_candidates:

                c_score = cube_score(
                    bm, bn, bk, problem
                )

                for vm in vector_m_candidates:
                    if vm > bm:
                        continue

                    for vn in vector_n_candidates:
                        if vn > bn:
                            continue

                        mem = calculate_memory_usage(
                            bm, bn, bk,
                            vm, vn,
                            hw, policy,
                        )

                        if not memory_fits(mem):
                            continue

                        v_score = vector_score(
                            vm, vn,
                            bm, bn,
                            mem,
                        )

                        # Cube 为主，Vector 用于同级候选的二次排序。
                        total = c_score * (
                            1.0 + 0.0005 * v_score
                        )

                        plans.append(
                            TilePlan(
                                baseM=bm,
                                baseN=bn,
                                baseK=bk,
                                vecM=vm,
                                vecN=vn,
                                memory=mem,
                                cube_score=c_score,
                                vector_score=v_score,
                                total_score=total,
                            )
                        )

    # 同一 Cube tile 只保留 UB 下最优的 Vector tile。
    best_by_cube: Dict[
        Tuple[int, int, int],
        TilePlan
    ] = {}

    for p in plans:
        key = (
            p.baseM,
            p.baseN,
            p.baseK,
        )

        old = best_by_cube.get(key)

        if (
            old is None
            or p.total_score > old.total_score
        ):
            best_by_cube[key] = p

    result = list(
        best_by_cube.values()
    )

    result.sort(
        key=lambda x: x.total_score,
        reverse=True,
    )

    return result


def choose_cross_core_split(
    problem: Problem,
    hw: Hardware,
    tile: TilePlan,
) -> SplitPlan:
    """
    不用拍脑袋加权，而用字典序决策：

    1. 最大化 AIC 利用率
    2. 同利用率下，最小化 splitN
    3. 减少超出 AIC 数的额外 task
    4. 优先更多 M 并行，而不是 N 并行

    对应：
        B -> M -> N -> K
    """

    max_m_groups = max(
        1,
        ceil_div(problem.M, tile.baseM),
    )

    max_n_groups = max(
        1,
        ceil_div(problem.N, tile.baseN),
    )

    candidates = []

    max_split_m = min(
        max_m_groups,
        hw.aic,
    )

    max_split_n = min(
        max_n_groups,
        hw.aic,
    )

    for sm in range(
        1,
        max_split_m + 1,
    ):
        for sn in range(
            1,
            max_split_n + 1,
        ):
            tasks = (
                problem.B
                * sm
                * sn
            )

            active = min(
                tasks,
                hw.aic,
            )

            util = (
                active / hw.aic
            )

            excess_tasks = max(
                0,
                tasks - hw.aic,
            )

            key = (
                util,           # 越大越好
                -sn,            # N 切得越少越好
                -excess_tasks,  # 少产生多余 task
                sm,             # 更偏向 M 并行
            )

            candidates.append(
                (
                    key,
                    sm,
                    sn,
                    tasks,
                    active,
                )
            )

    candidates.sort(
        reverse=True,
        key=lambda x: x[0],
    )

    (
        _,
        splitM,
        splitN,
        total_tasks,
        active_aic,
    ) = candidates[0]

    single_core_m = align_up(
        ceil_div(
            problem.M,
            splitM,
        ),
        16,
    )

    single_core_n = align_up(
        ceil_div(
            problem.N,
            splitN,
        ),
        16,
    )

    needs_n_reduction = (
        splitN > 1
    )

    if not needs_n_reduction:
        workspace = (
            problem.B
            * splitM
            * 4
        )

        reason = (
            "B/M parallelism is sufficient, "
            "so splitN=1. Each M group scans "
            "the complete N dimension."
        )

    else:
        partial_max_workspace = (
            problem.B
            * splitM
            * splitN
            * single_core_m
            * 4
        )

        scalar_workspace = (
            problem.B
            * splitM
            * 4
        )

        workspace = (
            partial_max_workspace
            + scalar_workspace
        )

        reason = (
            "B/M parallelism cannot fill all AICs, "
            "so N is additionally split. A second-stage "
            "Max across N groups is required."
        )

    return SplitPlan(
        splitM=splitM,
        splitN=splitN,

        total_tasks=total_tasks,
        active_aic=active_aic,
        aic_utilization=(
            active_aic / hw.aic
        ),

        singleCoreM=single_core_m,
        singleCoreN=single_core_n,

        needs_n_reduction=(
            needs_n_reduction
        ),

        workspace_bytes=workspace,

        bm_parallel_capacity=(
            problem.B
            * max_m_groups
        ),

        decision_reason=reason,
    )


def print_problem(problem: Problem) -> None:
    print("\n=== Problem ===")
    print(
        f"x1 logical shape : "
        f"[{problem.B}, {problem.M}, {problem.K}]"
    )
    print(
        f"x2 logical shape : "
        f"[{problem.B}, {problem.K}, {problem.N}]"
    )
    print(f"dtype             : {problem.dtype}")
    print(f"transposeX1       : {problem.transpose_x1}")
    print(f"transposeX2       : {problem.transpose_x2}")
    print("K accumulation    : FP32")
    print("Max(N)            : FP32")
    print("Sum(M)            : FP32")


def print_hardware(
    hw: Hardware,
    policy: BufferPolicy,
) -> None:
    print("\n=== Hardware / Effective Capacity ===")
    print(f"AIC               : {hw.aic}")
    print(f"AIV               : {hw.aiv}")

    for name, cap, reserve in (
        ("UB ", hw.ub, policy.ub_reserve),
        ("L1 ", hw.l1, policy.l1_reserve),
        ("L0A", hw.l0a, policy.l0a_reserve),
        ("L0B", hw.l0b, policy.l0b_reserve),
        ("L0C", hw.l0c, policy.l0c_reserve),
    ):
        print(
            f"{name:18}: "
            f"{human_bytes(cap):>10}  "
            f"reserve={human_bytes(reserve):>10}  "
            f"usable={human_bytes(available(cap, reserve)):>10}"
        )


def print_policy(
    policy: BufferPolicy,
) -> None:
    print("\n=== Buffer Policy ===")
    print(f"L1  A buffers      : {policy.l1_a_buffers}")
    print(f"L1  B buffers      : {policy.l1_b_buffers}")
    print(f"L0A buffers        : {policy.l0a_buffers}")
    print(f"L0B buffers        : {policy.l0b_buffers}")
    print(f"L0C buffers        : {policy.l0c_buffers}")
    print(f"UB input buffers   : {policy.ub_input_buffers}")
    print(f"UB row buffers     : {policy.ub_row_buffers}")
    print(f"UB extra scratch   : {human_bytes(policy.ub_extra_scratch)}")

    print("\nInterpretation:")
    print("  buffers=1 -> single buffer")
    print("  buffers=2 -> Ping-Pong / double buffer")
    print("  buffers=3 -> triple buffer")


def print_memory(
    plan: TilePlan,
) -> None:
    m = plan.memory

    print("\n=== Recommended Tile ===")
    print(
        f"Cube baseM/N/K     : "
        f"{plan.baseM} / {plan.baseN} / {plan.baseK}"
    )
    print(
        f"Vector vecM/vecN   : "
        f"{plan.vecM} / {plan.vecN}"
    )

    print("\nAIC raw tile sizes:")
    print(
        f"  A tile           : "
        f"{human_bytes(m.a_tile)}"
    )
    print(
        f"  B tile           : "
        f"{human_bytes(m.b_tile)}"
    )
    print(
        f"  C FP32 tile      : "
        f"{human_bytes(m.c_tile)}"
    )

    print("\nAIC buffer occupancy:")

    for name, used, cap in (
        ("L1 ", m.l1_used, m.l1_available),
        ("L0A", m.l0a_used, m.l0a_available),
        ("L0B", m.l0b_used, m.l0b_available),
        ("L0C", m.l0c_used, m.l0c_available),
    ):
        print(
            f"  {name}: "
            f"{human_bytes(used):>10} / "
            f"{human_bytes(cap):>10} "
            f"({pct(used, cap):5.1f}%)"
        )

    print("\nAIV / UB occupancy:")
    print(
        f"  Vector input tile: "
        f"{human_bytes(m.vec_input_tile)}"
    )
    print(
        f"  Input buffers    : "
        f"{human_bytes(m.ub_input_used)}"
    )
    print(
        f"  runningMax       : "
        f"{human_bytes(m.ub_running_max)}"
    )
    print(
        f"  row scratch      : "
        f"{human_bytes(m.ub_row_scratch)}"
    )
    print(
        f"  extra scratch    : "
        f"{human_bytes(m.ub_extra_scratch)}"
    )
    print(
        f"  UB total         : "
        f"{human_bytes(m.ub_used):>10} / "
        f"{human_bytes(m.ub_available):>10} "
        f"({pct(m.ub_used, m.ub_available):5.1f}%)"
    )

    if (
        plan.vecM != plan.baseM
        or plan.vecN != plan.baseN
    ):
        vec_tiles = (
            ceil_div(
                plan.baseM,
                plan.vecM,
            )
            * ceil_div(
                plan.baseN,
                plan.vecN,
            )
        )

        print(
            f"\n  NOTE: one Cube C tile is consumed as "
            f"~{vec_tiles} Vector subtile(s)."
        )
        print(
            "        UB does not need to hold "
            "the whole Cube output tile."
        )


def print_split(
    problem: Problem,
    hw: Hardware,
    plan: TilePlan,
    split: SplitPlan,
) -> None:
    print("\n=== Cross-core Split ===")
    print(f"B                  : {problem.B}")
    print(f"splitM             : {split.splitM}")
    print(f"splitN             : {split.splitN}")
    print("splitK             : 1")
    print(f"total logical tasks: {split.total_tasks}")
    print(f"active AIC         : {split.active_aic}/{hw.aic}")
    print(
        f"AIC utilization    : "
        f"{100.0 * split.aic_utilization:.1f}%"
    )
    print(
        f"singleCoreM <=     : "
        f"{split.singleCoreM}"
    )
    print(
        f"singleCoreN <=     : "
        f"{split.singleCoreN}"
    )
    print(
        f"workspace estimate : "
        f"{human_bytes(split.workspace_bytes)}"
    )
    print(
        f"BM parallel capacity: "
        f"{split.bm_parallel_capacity}"
    )
    print(
        f"reason             : "
        f"{split.decision_reason}"
    )

    print("\nPer-core logical loop:")
    print("  local M chunk")
    print("    -> assigned N group")
    print(
        f"      -> Cube N tile baseN="
        f"{plan.baseN}"
    )
    print(
        f"        -> K loop baseK="
        f"{plan.baseK}"
    )
    print(
        f"      -> Vector reduction "
        f"vecM={plan.vecM}, vecN={plan.vecN}"
    )
    print("    -> update runningMax")
    print("  -> Sum valid M rows")


def print_tail(
    problem: Problem,
    plan: TilePlan,
) -> None:
    print("\n=== Tail Rules ===")
    print(
        f"M % baseM          : "
        f"{problem.M % plan.baseM or 'none'}"
    )
    print(
        f"N % baseN          : "
        f"{problem.N % plan.baseN or 'none'}"
    )
    print(
        f"K % baseK          : "
        f"{problem.K % plan.baseK or 'none'}"
    )

    print("\nCorrectness:")
    print("  K invalid lanes  -> zero")
    print("  M invalid rows   -> exclude from final Sum")
    print("  N invalid lanes  -> mask or set to -INF")
    print("  runningMax       -> initialize to -INF")


def print_pipeline(
    split: SplitPlan,
) -> None:
    print("\n=== Pipeline Pseudocode ===")

    if not split.needs_n_reduction:
        print(r"""
STAGE 1: each (b, mGroup)

runningMax[actualM] = -INF

for nCubeTile in FULL N:

    # AIC / Cube
    C_fp32[baseM, baseN] = 0

    for kTile in FULL K:
        # Ping/Pong can overlap:
        #   load next A/B tile
        #   compute current A/B tile
        C_fp32 += A[Mtile, Ktile] @ B[Ktile, Ntile]

    # AIC -> AIV bridge
    # Planning model allows GM/workspace staging.

    # AIV / Vector
    for vecSubTile in C_fp32:
        # UB input can also Ping/Pong:
        #   process current vec tile
        #   load next vec tile
        rowMax = Max(valid N lanes)
        runningMax = Max(runningMax, rowMax)

partialSum[b, mGroup] =
    Sum(runningMax over valid M)

STAGE 2:
y[b] = Sum_mGroup(partialSum[b, mGroup])
""".strip())

    else:
        print(r"""
STAGE 1: each (b, mGroup, nGroup)

partialMax[actualM] = -INF

for nCubeTile in THIS N GROUP:

    C_fp32[baseM, baseN] = 0

    for kTile in FULL K:
        C_fp32 += A[Mtile, Ktile] @ B[Ktile, Ntile]

    for vecSubTile in C_fp32:
        rowMax = Max(valid N lanes)
        partialMax = Max(partialMax, rowMax)

write partialMax[b, mGroup, nGroup, :]

STAGE 2:
maxAcrossN[m] =
    Max_nGroup(partialMax[..., nGroup, m])

partialSum[b, mGroup] =
    Sum_m(maxAcrossN[m])

STAGE 3:
y[b] =
    Sum_mGroup(partialSum[b, mGroup])
""".strip())


def print_top_candidates(
    plans: List[TilePlan],
    topk: int = 8,
) -> None:
    print("\n=== Top Legal Tile Candidates ===")
    print(
        " rank | cube(M,N,K)   | vector(M,N) | "
        "L0A%  L0B%  L0C%   L1%   UB%"
    )

    for i, p in enumerate(
        plans[:topk],
        1,
    ):
        m = p.memory

        print(
            f"{i:>5} | "
            f"({p.baseM:>3},{p.baseN:>3},{p.baseK:>3}) | "
            f"({p.vecM:>3},{p.vecN:>3})   | "
            f"{pct(m.l0a_used,m.l0a_available):5.1f} "
            f"{pct(m.l0b_used,m.l0b_available):5.1f} "
            f"{pct(m.l0c_used,m.l0c_available):5.1f} "
            f"{pct(m.l1_used,m.l1_available):5.1f} "
            f"{pct(m.ub_used,m.ub_available):5.1f}"
        )


def make_json(
    problem: Problem,
    hw: Hardware,
    policy: BufferPolicy,
    plan: TilePlan,
    split: SplitPlan,
    plans: List[TilePlan],
) -> Dict[str, Any]:
    return {
        "problem": asdict(problem),
        "hardware": asdict(hw),
        "buffer_policy": asdict(policy),

        "recommended": {
            "cube_tile": {
                "baseM": plan.baseM,
                "baseN": plan.baseN,
                "baseK": plan.baseK,
            },
            "vector_tile": {
                "vecM": plan.vecM,
                "vecN": plan.vecN,
            },
            "memory": asdict(plan.memory),
            "split": asdict(split),
        },

        "top_candidates": [
            {
                "baseM": p.baseM,
                "baseN": p.baseN,
                "baseK": p.baseK,
                "vecM": p.vecM,
                "vecN": p.vecN,
                "memory": asdict(p.memory),
                "total_score": p.total_score,
            }
            for p in plans[:8]
        ],

        "core_rules": [
            "cross-core priority: B -> M -> N -> K",
            "K is not split across AICs by default",
            "K is locally tiled and accumulated in FP32",
            "N cross-core split is used only when B/M parallelism is insufficient",
            "Cube tile and Vector UB tile are planned independently",
            "N tail invalid lanes are masked or set to -INF",
            "runningMax starts at -INF",
        ],
    }


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Ascend MaxSim tiling + "
            "detailed buffer planner v2"
        )
    )

    p.add_argument("--B", type=int)
    p.add_argument("--M", type=int)
    p.add_argument("--N", type=int)
    p.add_argument("--K", type=int)

    p.add_argument(
        "--dtype",
        choices=["fp16", "bf16"],
        default="fp16",
    )

    p.add_argument(
        "--transpose-x1",
        action="store_true",
    )

    p.add_argument(
        "--transpose-x2",
        action="store_true",
    )

    p.add_argument("--aic", type=int, default=24)
    p.add_argument("--aiv", type=int, default=48)

    p.add_argument("--ub-kib", type=int, default=192)
    p.add_argument("--l1-kib", type=int, default=512)
    p.add_argument("--l0a-kib", type=int, default=64)
    p.add_argument("--l0b-kib", type=int, default=64)
    p.add_argument("--l0c-kib", type=int, default=128)

    p.add_argument("--l1-a-buffers", type=int, default=2)
    p.add_argument("--l1-b-buffers", type=int, default=2)
    p.add_argument("--l0a-buffers", type=int, default=2)
    p.add_argument("--l0b-buffers", type=int, default=2)
    p.add_argument("--l0c-buffers", type=int, default=1)

    p.add_argument("--ub-input-buffers", type=int, default=2)
    p.add_argument("--ub-row-buffers", type=int, default=2)

    p.add_argument("--ub-reserve-kib", type=int, default=0)
    p.add_argument("--l1-reserve-kib", type=int, default=0)
    p.add_argument("--l0a-reserve-kib", type=int, default=0)
    p.add_argument("--l0b-reserve-kib", type=int, default=0)
    p.add_argument("--l0c-reserve-kib", type=int, default=0)
    p.add_argument("--ub-extra-scratch-kib", type=int, default=0)

    p.add_argument(
        "--interactive",
        action="store_true",
    )

    p.add_argument(
        "--json",
        action="store_true",
    )

    return p


def validate_problem(
    p: Problem,
) -> None:
    for name in (
        "B", "M", "N", "K"
    ):
        if getattr(p, name) <= 0:
            raise ValueError(
                f"{name} must be > 0"
            )


def validate_hardware(
    h: Hardware,
) -> None:
    if h.aic <= 0:
        raise ValueError("aic must be > 0")

    if h.aiv <= 0:
        raise ValueError("aiv must be > 0")

    for name in (
        "ub", "l1", "l0a", "l0b", "l0c"
    ):
        if getattr(h, name) <= 0:
            raise ValueError(
                f"{name} must be > 0"
            )


def main() -> None:
    args = make_parser().parse_args()

    missing_shape = any(
        x is None
        for x in (
            args.B,
            args.M,
            args.N,
            args.K,
        )
    )

    interactive = (
        args.interactive
        or missing_shape
    )

    if interactive:
        print(
            "Enter logical shape; "
            "press Enter to accept defaults."
        )

        B = prompt_int(
            "B",
            args.B if args.B is not None else 1,
        )
        M = prompt_int(
            "M",
            args.M if args.M is not None else 4096,
        )
        N = prompt_int(
            "N",
            args.N if args.N is not None else 4096,
        )
        K = prompt_int(
            "K",
            args.K if args.K is not None else 1024,
        )

        print(
            "\nHardware; press Enter "
            "to keep defaults."
        )

        aic = prompt_int(
            "AIC count",
            args.aic,
        )
        aiv = prompt_int(
            "AIV count",
            args.aiv,
        )

        ub_kib = prompt_int(
            "UB KiB",
            args.ub_kib,
        )
        l1_kib = prompt_int(
            "L1 KiB",
            args.l1_kib,
        )
        l0a_kib = prompt_int(
            "L0A KiB",
            args.l0a_kib,
        )
        l0b_kib = prompt_int(
            "L0B KiB",
            args.l0b_kib,
        )
        l0c_kib = prompt_int(
            "L0C KiB",
            args.l0c_kib,
        )

        print(
            "\nBuffer depth; "
            "2 means Ping-Pong."
        )

        l1_a_buffers = prompt_int(
            "L1 A buffers",
            args.l1_a_buffers,
        )
        l1_b_buffers = prompt_int(
            "L1 B buffers",
            args.l1_b_buffers,
        )
        l0a_buffers = prompt_int(
            "L0A buffers",
            args.l0a_buffers,
        )
        l0b_buffers = prompt_int(
            "L0B buffers",
            args.l0b_buffers,
        )
        l0c_buffers = prompt_int(
            "L0C buffers",
            args.l0c_buffers,
        )
        ub_input_buffers = prompt_int(
            "UB input buffers",
            args.ub_input_buffers,
        )
        ub_row_buffers = prompt_int(
            "UB row buffers",
            args.ub_row_buffers,
        )

    else:
        B, M, N, K = (
            args.B,
            args.M,
            args.N,
            args.K,
        )

        aic = args.aic
        aiv = args.aiv

        ub_kib = args.ub_kib
        l1_kib = args.l1_kib
        l0a_kib = args.l0a_kib
        l0b_kib = args.l0b_kib
        l0c_kib = args.l0c_kib

        l1_a_buffers = args.l1_a_buffers
        l1_b_buffers = args.l1_b_buffers
        l0a_buffers = args.l0a_buffers
        l0b_buffers = args.l0b_buffers
        l0c_buffers = args.l0c_buffers
        ub_input_buffers = args.ub_input_buffers
        ub_row_buffers = args.ub_row_buffers

    problem = Problem(
        B=B,
        M=M,
        N=N,
        K=K,
        dtype=args.dtype,
        transpose_x1=args.transpose_x1,
        transpose_x2=args.transpose_x2,
    )

    hw = Hardware(
        aic=aic,
        aiv=aiv,
        ub=kib(ub_kib),
        l1=kib(l1_kib),
        l0a=kib(l0a_kib),
        l0b=kib(l0b_kib),
        l0c=kib(l0c_kib),
    )

    policy = BufferPolicy(
        l1_a_buffers=l1_a_buffers,
        l1_b_buffers=l1_b_buffers,

        l0a_buffers=l0a_buffers,
        l0b_buffers=l0b_buffers,
        l0c_buffers=l0c_buffers,

        ub_input_buffers=ub_input_buffers,
        ub_row_buffers=ub_row_buffers,

        ub_reserve=kib(args.ub_reserve_kib),
        l1_reserve=kib(args.l1_reserve_kib),
        l0a_reserve=kib(args.l0a_reserve_kib),
        l0b_reserve=kib(args.l0b_reserve_kib),
        l0c_reserve=kib(args.l0c_reserve_kib),

        ub_extra_scratch=kib(
            args.ub_extra_scratch_kib
        ),
    )

    validate_problem(problem)
    validate_hardware(hw)
    validate_buffer_policy(policy)

    plans = enumerate_tile_plans(
        problem,
        hw,
        policy,
    )

    if not plans:
        raise RuntimeError(
            "No legal Cube+Vector tile found. "
            "Reduce buffer depth/reserve, or "
            "extend candidate tile sizes."
        )

    best = plans[0]

    split = choose_cross_core_split(
        problem,
        hw,
        best,
    )

    if args.json:
        print(
            json.dumps(
                make_json(
                    problem,
                    hw,
                    policy,
                    best,
                    split,
                    plans,
                ),
                indent=2,
            )
        )
        return

    print_problem(problem)
    print_hardware(hw, policy)
    print_policy(policy)
    print_memory(best)
    print_split(
        problem,
        hw,
        best,
        split,
    )
    print_tail(problem, best)
    print_pipeline(split)
    print_top_candidates(plans)

    print("\n=== Core Logic Summary ===")
    print(
        "1. Search legal Cube tiles under "
        "L1/L0A/L0B/L0C buffer constraints."
    )
    print(
        "2. Independently search legal AIV "
        "Vector tiles under UB constraints."
    )
    print(
        "3. Prefer B/M cross-core parallelism; "
        "introduce N only when needed."
    )
    print(
        "4. Never cross-core split K "
        "in the default plan."
    )
    print(
        "5. K is locally tiled and "
        "accumulated in FP32."
    )
    print(
        "6. splitN>1 requires partialMax "
        "workspace and second-stage Max."
    )
    print(
        "7. N tail uses mask/-INF; "
        "runningMax starts at -INF."
    )

    print(
        "\n=== Production Host-Tiling Reminder ==="
    )

    print(r"""
auto platform =
    platform_ascendc::PlatformAscendC(
        context->GetPlatformInfo()
    );

uint32_t aicNum =
    platform.GetCoreNumAic();

uint32_t aivNum =
    platform.GetCoreNumAiv();

uint64_t ubSize;
uint64_t l1Size;
uint64_t l0aSize;
uint64_t l0bSize;
uint64_t l0cSize;

platform.GetCoreMemSize(
    platform_ascendc::CoreMemType::UB,
    ubSize
);

platform.GetCoreMemSize(
    platform_ascendc::CoreMemType::L1,
    l1Size
);

platform.GetCoreMemSize(
    platform_ascendc::CoreMemType::L0_A,
    l0aSize
);

platform.GetCoreMemSize(
    platform_ascendc::CoreMemType::L0_B,
    l0bSize
);

platform.GetCoreMemSize(
    platform_ascendc::CoreMemType::L0_C,
    l0cSize
);
""".strip())

    print("\nNOTE:")
    print(
        "This is a capacity + tiling heuristic planner, "
        "not a cycle-accurate simulator."
    )
    print(
        "Final choices should be benchmarked/msProf'd "
        "on representative shapes."
    )


if __name__ == "__main__":
    main()
