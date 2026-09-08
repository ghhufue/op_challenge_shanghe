"""Inspect legal tiling candidates; estimates are only pre-benchmark ordering."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json

from .enumerate import enumerate_plans
from .model import Hardware, Problem


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--B", type=int, required=True)
    result.add_argument("--M", type=int, required=True)
    result.add_argument("--N", type=int, required=True)
    result.add_argument("--K", type=int, required=True)
    result.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    result.add_argument("--transpose-x1", action="store_true")
    result.add_argument("--transpose-x2", action="store_true")
    result.add_argument("--aic", type=int, default=24)
    result.add_argument("--aiv", type=int, default=48)
    result.add_argument("--implemented-only", action="store_true")
    result.add_argument("--json", action="store_true")
    return result


def main() -> int:
    args = parser().parse_args()
    problem = Problem(
        args.B, args.M, args.N, args.K, args.dtype,
        args.transpose_x1, args.transpose_x2,
    )
    plans = enumerate_plans(
        problem,
        Hardware(aic=args.aic, aiv=args.aiv),
        implemented_only=args.implemented_only,
    )
    if args.json:
        print(json.dumps([asdict(plan) for plan in plans], indent=2))
    else:
        print("key  implemented path       tile(M,N,K) split(M,N) blocks workspace score")
        for plan in plans:
            c = plan.config
            print(
                f"{c.key:3d}  {str(c.implemented):11s} {c.path:9s} "
                f"({c.tile_m:3d},{c.tile_n:3d},{c.tile_k:3d}) "
                f"({plan.split_m:2d},{plan.split_n:2d}) {plan.launch_blocks:6d} "
                f"{plan.workspace_bytes:9d} {plan.estimated_score:.4f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
