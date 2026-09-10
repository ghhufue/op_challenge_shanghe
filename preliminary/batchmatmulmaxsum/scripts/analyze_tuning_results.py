"""Summarize benchmark results without printing or embedding raw input rows."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import statistics
import sys


OP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OP_ROOT))

from tuning.fit_policy import Branch, Leaf, fit_tree, load_samples  # noqa: E402


def candidate_key(row: dict) -> int:
    return int(row.get("candidate_key", row.get("tiling_key", -1)))


def identity(row: dict) -> tuple:
    return (
        row["id"], int(row["B"]), int(row["M"]), int(row["N"]), int(row["K"]),
        str(row["dtype"]),
        bool(row.get("transpose_x1", row.get("transposeX1", False))),
        bool(row.get("transpose_x2", row.get("transposeX2", False))),
    )


def quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    index = (len(ordered) - 1) * fraction
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - index) + ordered[upper] * (index - lower)


def geometric_mean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def predict(node: Leaf | Branch, features: dict[str, int]) -> int:
    while isinstance(node, Branch):
        node = node.left if features[node.feature] <= node.threshold else node.right
    return node.key


def tree_stats(node: Leaf | Branch) -> tuple[int, int, set[int]]:
    if isinstance(node, Leaf):
        return 1, 1, {node.key}
    left_nodes, left_leaves, left_keys = tree_stats(node.left)
    right_nodes, right_leaves, right_keys = tree_stats(node.right)
    return 1 + left_nodes + right_nodes, left_leaves + right_leaves, left_keys | right_keys


def fmt_percent(value: float) -> str:
    return f"{100.0 * value:.3f}%"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", type=Path)
    parser.add_argument("--metric", default="p50_call_ms")
    parser.add_argument("--close", type=float, default=0.01,
                        help="Very-close relative gap (default: 0.01 = 1%%)")
    parser.add_argument("--detail-limit", type=int, default=24)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rows = json.loads(args.benchmark.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("benchmark must be a non-empty JSON list")
    failed = [row for row in rows if not row.get("passed")]
    if failed:
        raise ValueError(f"benchmark contains {len(failed)} failed rows")

    names = {candidate_key(row): str(row.get("candidate_name", row.get("tiling_name", "")))
             for row in rows}
    groups: dict[tuple, dict[int, dict]] = defaultdict(dict)
    duplicate_count = 0
    for row in rows:
        group = groups[identity(row)]
        key = candidate_key(row)
        if key in group:
            duplicate_count += 1
        group[key] = row
    all_keys = sorted(set(names))
    incomplete = sum(set(group) != set(all_keys) for group in groups.values())
    if incomplete:
        raise ValueError(f"{incomplete} cases do not contain every candidate key")

    metrics = [name for name in ("mean_call_ms", "p50_call_ms", "p95_call_ms")
               if all(name in row and math.isfinite(float(row[name])) for row in rows)]
    if args.metric not in metrics:
        raise ValueError(f"metric {args.metric!r} is unavailable; complete metrics={metrics}")

    wins = Counter()
    near_best = {threshold: Counter() for threshold in (0.005, 0.01, 0.03, 0.05)}
    regrets: dict[int, list[float]] = defaultdict(list)
    runner_gaps: list[float] = []
    closest: list[tuple[float, tuple, int, int, float, float]] = []
    metric_winners: dict[str, dict[tuple, int]] = {metric: {} for metric in metrics}

    for case_id, candidates in groups.items():
        for metric in metrics:
            metric_winners[metric][case_id] = min(
                candidates, key=lambda key: float(candidates[key][metric]))
        ordered = sorted(
            ((float(row[args.metric]), key) for key, row in candidates.items()),
            key=lambda item: (item[0], item[1]),
        )
        best_time, best_key = ordered[0]
        second_time, second_key = ordered[1]
        wins[best_key] += 1
        gap = second_time / best_time - 1.0
        runner_gaps.append(gap)
        closest.append((gap, case_id, best_key, second_key, best_time, second_time))
        for key, row in candidates.items():
            relative = float(row[args.metric]) / best_time - 1.0
            regrets[key].append(relative)
            for threshold in near_best:
                if relative <= threshold:
                    near_best[threshold][key] += 1

    samples, common_keys = load_samples(args.benchmark, args.metric)
    tree_rows = []
    for depth in range(1, 6):
        for min_leaf in (2, 4, 8, 12):
            if len(samples) < 2 * min_leaf:
                continue
            tree = fit_tree(samples, common_keys, depth, min_leaf)
            selected_ratios = []
            for sample in samples:
                selected = predict(tree, sample.features)
                selected_ratios.append(sample.times[selected] / min(sample.times.values()))
            nodes, leaves, selected_keys = tree_stats(tree)
            tree_rows.append((
                depth, min_leaf, nodes, leaves, selected_keys,
                geometric_mean(selected_ratios) - 1.0,
                statistics.mean(value - 1.0 for value in selected_ratios),
                max(value - 1.0 for value in selected_ratios),
            ))

    lines = [
        "# Tiling benchmark analysis",
        "",
        "## Data quality",
        "",
        f"- Rows: {len(rows)}; cases: {len(groups)}; candidates: {len(all_keys)}.",
        f"- Passed: {len(rows) - len(failed)}/{len(rows)}; duplicates: {duplicate_count}; incomplete cases: {incomplete}.",
        f"- Candidate keys: {', '.join(str(key) for key in all_keys)}.",
        f"- Repeats: {sorted({int(row['repeats']) for row in rows})}; warmups: {sorted({int(row['warmups']) for row in rows})}.",
        f"- Complete timing metrics: {', '.join(metrics)}; primary metric: {args.metric}.",
        "",
        "## Candidate summary",
        "",
        "| Key | Name | Wins | <=0.5% | <=1% | <=3% | <=5% | Geomean regret | Mean regret | Worst regret |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in all_keys:
        values = regrets[key]
        lines.append(
            f"| {key} | {names[key]} | {wins[key]} | {near_best[0.005][key]} | "
            f"{near_best[0.01][key]} | {near_best[0.03][key]} | {near_best[0.05][key]} | "
            f"{fmt_percent(geometric_mean([1.0 + value for value in values]) - 1.0)} | "
            f"{fmt_percent(statistics.mean(values))} | {fmt_percent(max(values))} |"
        )

    lines.extend([
        "",
        "## Closeness and timing stability",
        "",
        f"- Best-vs-runner-up gap <=0.5%: {sum(gap <= 0.005 for gap in runner_gaps)}/{len(runner_gaps)}.",
        f"- Best-vs-runner-up gap <=1%: {sum(gap <= 0.01 for gap in runner_gaps)}/{len(runner_gaps)}.",
        f"- Best-vs-runner-up gap <=3%: {sum(gap <= 0.03 for gap in runner_gaps)}/{len(runner_gaps)}.",
        f"- Gap quantiles P50/P75/P90: {fmt_percent(quantile(runner_gaps, 0.5))} / "
        f"{fmt_percent(quantile(runner_gaps, 0.75))} / {fmt_percent(quantile(runner_gaps, 0.9))}.",
    ])
    if "mean_call_ms" in metrics and "p50_call_ms" in metrics:
        disagreement = sum(
            metric_winners["mean_call_ms"][case] != metric_winners["p50_call_ms"][case]
            for case in groups
        )
        lines.append(f"- Mean and P50 select different winners: {disagreement}/{len(groups)} cases.")
    if "p95_call_ms" in metrics and "p50_call_ms" in metrics:
        tail_ratios = [float(row["p95_call_ms"]) / float(row["p50_call_ms"]) for row in rows]
        lines.append(
            f"- P95/P50 ratio P50/P90/max: {quantile(tail_ratios, 0.5):.3f}x / "
            f"{quantile(tail_ratios, 0.9):.3f}x / {max(tail_ratios):.3f}x."
        )

    lines.extend([
        "",
        f"## Closest cases by {args.metric}",
        "",
        "| Case | Shape | Dtype/layout | Best | Runner-up | Gap | Best ms | Runner ms |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ])
    for gap, case, best_key, second_key, best_time, second_time in sorted(closest)[:args.detail_limit]:
        case_name, b, m, n, k, dtype, tx1, tx2 = case
        lines.append(
            f"| {case_name} | {b}x{m}x{n}x{k} | {dtype}, tx={int(tx1)}{int(tx2)} | "
            f"{best_key} | {second_key} | {fmt_percent(gap)} | {best_time:.6f} | {second_time:.6f} |"
        )

    lines.extend([
        "",
        "## Policy-tree complexity sweep (in-sample)",
        "",
        "| Depth | Min leaf | Nodes | Leaves | Selected keys | Geomean regret | Mean regret | Worst regret |",
        "|---:|---:|---:|---:|---|---:|---:|---:|",
    ])
    for depth, min_leaf, nodes, leaves, selected_keys, geo, mean, worst in tree_rows:
        lines.append(
            f"| {depth} | {min_leaf} | {nodes} | {leaves} | "
            f"{','.join(map(str, sorted(selected_keys)))} | {fmt_percent(geo)} | "
            f"{fmt_percent(mean)} | {fmt_percent(worst)} |"
        )

    report = "\n".join(lines) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8", newline="\n")
        print(f"Wrote {args.output}")
    else:
        print(report, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
