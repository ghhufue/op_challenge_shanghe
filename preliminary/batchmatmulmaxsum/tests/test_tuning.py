from __future__ import annotations

from pathlib import Path
import sys
import unittest


OP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OP_ROOT))

from scripts.generate_tiling_catalog import OUTPUT, render  # noqa: E402
from tuning.catalog import load_catalog  # noqa: E402
from tuning.enumerate import enumerate_plans  # noqa: E402
from tuning.fit_policy import Leaf, Sample, fit_tree, render_policy  # noqa: E402
from tuning.model import Hardware, Problem  # noqa: E402


class CatalogTests(unittest.TestCase):
    def test_catalog_has_reference_bm_and_bmn_paths(self):
        configs = load_catalog()
        implemented = [item for item in configs if item.implemented]
        self.assertEqual([item.key for item in implemented], [0, 100, 101, 200, 201])
        self.assertEqual(implemented[0].path, "reference")
        self.assertEqual(implemented[1].path, "bm")
        self.assertEqual(implemented[2].path, "bm")
        self.assertEqual(implemented[3].path, "bmn")
        self.assertEqual(implemented[4].path, "bmn")

    def test_generated_header_is_current(self):
        self.assertEqual(OUTPUT.read_text(encoding="utf-8"), render())


class EnumerationTests(unittest.TestCase):
    def test_reference_workspace_is_padded_atomic_output(self):
        problem = Problem(2, 17, 33, 40)
        plans = enumerate_plans(problem, implemented_only=True)
        reference = next(plan for plan in plans if plan.config.key == 0)
        self.assertEqual(reference.workspace_bytes, 8 * 4)
        self.assertEqual(reference.launch_blocks, 34)

    def test_bm_workspace_has_task_padded_row_max_and_per_core_stage(self):
        hardware = Hardware(aic=20, aiv=40)
        plans = enumerate_plans(
            Problem(3, 33, 129, 64), hardware, implemented_only=True,
        )
        bm = next(plan for plan in plans if plan.config.key == 100)
        self.assertEqual(bm.task_count, 9)
        self.assertEqual(bm.launch_blocks, 9)
        self.assertEqual(bm.workspace_bytes, 1024 + 9 * 16 * 128 * 4 + 8 * 4)

    def test_bm32_workspace_uses_its_compile_time_tile(self):
        hardware = Hardware(aic=20, aiv=40)
        plans = enumerate_plans(
            Problem(3, 33, 129, 64), hardware, implemented_only=True,
        )
        bm = next(plan for plan in plans if plan.config.key == 101)
        self.assertEqual(bm.task_count, 6)
        self.assertEqual(bm.launch_blocks, 6)
        self.assertEqual(bm.workspace_bytes, 1024 + 6 * 32 * 128 * 4 + 8 * 4)

    def test_bmn_workspace_has_partial_max_stage_and_atomic_output(self):
        hardware = Hardware(aic=20, aiv=40)
        plans = enumerate_plans(
            Problem(1, 17, 1025, 64), hardware, implemented_only=True,
        )
        p2 = next(plan for plan in plans if plan.config.key == 200)
        self.assertEqual(p2.split_n, 2)
        self.assertEqual(p2.task_count, 4)
        self.assertEqual(p2.launch_blocks, 4)
        self.assertEqual(p2.workspace_bytes, 512 + 4 * 16 * 256 * 4 + 8 * 4)

        p4 = next(plan for plan in plans if plan.config.key == 201)
        self.assertEqual(p4.split_n, 4)
        self.assertEqual(p4.task_count, 8)
        self.assertEqual(p4.launch_blocks, 8)
        self.assertEqual(p4.workspace_bytes, 512 + 8 * 16 * 256 * 4 + 8 * 4)

    def test_n_split_does_not_create_empty_partition(self):
        plans = enumerate_plans(Problem(1, 1, 1, 32))
        for plan in plans:
            self.assertLessEqual(plan.split_n, plan.config.split_n)
            self.assertEqual(plan.split_n, 1)

    def test_bmn_round_robin_tile_assignment_is_complete_and_balanced(self):
        for n_tiles in range(1, 20):
            for configured_split in [2, 4]:
                split_n = min(configured_split, n_tiles)
                rounds = (n_tiles + split_n - 1) // split_n
                groups = [
                    [round_index * split_n + group
                     for round_index in range(rounds)
                     if round_index * split_n + group < n_tiles]
                    for group in range(split_n)
                ]
                assigned = sorted(tile for group in groups for tile in group)
                self.assertEqual(assigned, list(range(n_tiles)))
                sizes = [len(group) for group in groups]
                self.assertLessEqual(max(sizes) - min(sizes), 1)

    def test_capacity_filter_removes_optimized_tiles(self):
        tiny = Hardware(ub_bytes=1024, l1_bytes=1024, l0a_bytes=1024,
                        l0b_bytes=1024, l0c_bytes=1024)
        plans = enumerate_plans(Problem(1, 128, 128, 128), tiny)
        self.assertEqual([plan.config.key for plan in plans], [0])

    def test_invalid_problem_is_rejected(self):
        with self.assertRaises(ValueError):
            enumerate_plans(Problem(1, 1, 1, 33))


class PolicyTests(unittest.TestCase):
    def test_single_measured_key_produces_fallback(self):
        sample = Sample(
            {"B": 1, "M": 16, "N": 16, "K": 32, "work": 8192,
             "bm_rows": 16, "dtype": 1, "tx1": 0, "tx2": 0},
            {0: 1.0},
        )
        tree = fit_tree([sample], {0}, max_depth=3, min_samples_leaf=1)
        self.assertIsInstance(tree, Leaf)
        self.assertEqual(render_policy(tree, {0: "VECTOR_REFERENCE"}),
                         "return TilingKey::VECTOR_REFERENCE;\n")

    def test_tree_finds_shape_threshold(self):
        common = {"B": 1, "N": 128, "K": 64, "dtype": 1, "tx1": 0, "tx2": 0}
        small = Sample(dict(common, M=16, work=131072, bm_rows=16), {0: 1.0, 100: 2.0})
        large = Sample(dict(common, M=64, work=524288, bm_rows=64), {0: 2.0, 100: 1.0})
        tree = fit_tree([small, large], {0, 100}, max_depth=1, min_samples_leaf=1)
        policy = render_policy(tree, {0: "VECTOR_REFERENCE", 100: "BM_16X128X64"})
        self.assertIn("shape.m <= 16ULL", policy)
        self.assertIn("TilingKey::BM_16X128X64", policy)


if __name__ == "__main__":
    unittest.main()
