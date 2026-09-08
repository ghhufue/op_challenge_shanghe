"""Fit a shallow timing-regret tree and emit a simple C++ tiling policy."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path

from .catalog import load_catalog


@dataclass
class Sample:
    features: dict[str, int]
    times: dict[int, float]


@dataclass
class Leaf:
    key: int
    loss: float


@dataclass
class Branch:
    feature: str
    threshold: int
    left: "Node"
    right: "Node"
    loss: float


Node = Leaf | Branch


FEATURE_CPP = {
    "B": "shape.b",
    "M": "shape.m",
    "N": "shape.n",
    "K": "shape.k",
    "work": "static_cast<uint64_t>(shape.b) * shape.m * shape.n * shape.k",
    "bm_rows": "static_cast<uint64_t>(shape.b) * shape.m",
    "dtype": "inputDtype",
    "tx1": "static_cast<int>(transposeX1)",
    "tx2": "static_cast<int>(transposeX2)",
}


def _leaf(samples: list[Sample], keys: set[int]) -> Leaf:
    best_key = min(
        keys,
        key=lambda key: sum(sample.times[key] / min(sample.times.values()) for sample in samples),
    )
    loss = sum(sample.times[best_key] / min(sample.times.values()) - 1.0 for sample in samples)
    return Leaf(best_key, loss)


def fit_tree(
    samples: list[Sample],
    keys: set[int],
    max_depth: int,
    min_samples_leaf: int,
) -> Node:
    base = _leaf(samples, keys)
    if max_depth == 0 or len(samples) < 2 * min_samples_leaf or base.loss <= 1e-12:
        return base

    best = None
    for feature in FEATURE_CPP:
        values = sorted({sample.features[feature] for sample in samples})
        for threshold in values[:-1]:
            left = [sample for sample in samples if sample.features[feature] <= threshold]
            right = [sample for sample in samples if sample.features[feature] > threshold]
            if len(left) < min_samples_leaf or len(right) < min_samples_leaf:
                continue
            left_leaf = _leaf(left, keys)
            right_leaf = _leaf(right, keys)
            split_loss = left_leaf.loss + right_leaf.loss
            candidate = (split_loss, feature, threshold, left, right)
            if best is None or candidate[:3] < best[:3]:
                best = candidate
    if best is None or best[0] >= base.loss - 1e-12:
        return base

    _, feature, threshold, left, right = best
    left_node = fit_tree(left, keys, max_depth - 1, min_samples_leaf)
    right_node = fit_tree(right, keys, max_depth - 1, min_samples_leaf)
    return Branch(feature, threshold, left_node, right_node, left_node.loss + right_node.loss)


def render_policy(node: Node, symbols: dict[int, str], indent: str = "") -> str:
    if isinstance(node, Leaf):
        return f"{indent}return TilingKey::{symbols[node.key]};\n"
    expression = FEATURE_CPP[node.feature]
    return (
        f"{indent}if ({expression} <= {node.threshold}ULL) {{\n"
        + render_policy(node.left, symbols, indent + "    ")
        + f"{indent}}} else {{\n"
        + render_policy(node.right, symbols, indent + "    ")
        + f"{indent}}}\n"
    )


def load_samples(path: Path, metric: str) -> tuple[list[Sample], set[int]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("benchmark input must be a JSON list")
    implemented = {item.key for item in load_catalog() if item.implemented}
    grouped: dict[tuple, Sample] = {}
    for row in rows:
        if not row.get("passed"):
            continue
        key = int(row.get("candidate_key", row.get("tiling_key", -1)))
        value = float(row.get(metric, math.nan))
        if key not in implemented or not math.isfinite(value) or value <= 0:
            continue
        identity = (
            row.get("id"), row["B"], row["M"], row["N"], row["K"], row["dtype"],
            bool(row.get("transpose_x1", row.get("transposeX1", False))),
            bool(row.get("transpose_x2", row.get("transposeX2", False))),
        )
        if identity not in grouped:
            b, m, n, k = (int(row[name]) for name in ("B", "M", "N", "K"))
            dtype = row["dtype"]
            grouped[identity] = Sample(
                {
                    "B": b, "M": m, "N": n, "K": k,
                    "work": b * m * n * k,
                    "bm_rows": b * m,
                    "dtype": 2 if dtype in {2, "bfloat16"} else 1,
                    "tx1": int(identity[-2]), "tx2": int(identity[-1]),
                },
                {},
            )
        previous = grouped[identity].times.get(key)
        grouped[identity].times[key] = value if previous is None else min(previous, value)

    samples = list(grouped.values())
    if not samples:
        raise ValueError("no passing, finite benchmark rows for implemented keys")
    common_keys = set.intersection(*(set(sample.times) for sample in samples))
    if not common_keys:
        raise ValueError("no tiling key was measured for every training case")
    for sample in samples:
        sample.times = {key: sample.times[key] for key in common_keys}
    return samples, common_keys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", type=Path)
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).resolve().parents[1] / "tiling" / "generated_policy.inc",
    )
    parser.add_argument("--metric", default="mean_call_ms")
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    args = parser.parse_args()
    if args.max_depth < 0 or args.min_samples_leaf < 1:
        parser.error("max-depth must be non-negative and min-samples-leaf must be positive")

    samples, keys = load_samples(args.benchmark, args.metric)
    tree = fit_tree(samples, keys, args.max_depth, args.min_samples_leaf)
    symbols = {item.key: item.symbol for item in load_catalog()}
    body = (
        "// Generated by tuning.fit_policy from measured passing runs.\n"
        f"// samples={len(samples)} metric={args.metric} max_depth={args.max_depth}\n"
        + render_policy(tree, symbols)
    )
    args.output.write_text(body, encoding="utf-8", newline="\n")
    print(f"Wrote {args.output}; training regret={tree.loss:.6f}; keys={sorted(keys)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
