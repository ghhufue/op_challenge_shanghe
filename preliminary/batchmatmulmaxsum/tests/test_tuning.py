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
    def test_catalog_has_ordered_single_core_iterations(self):
        configs = load_catalog()
        implemented = [item for item in configs if item.implemented]
        self.assertEqual([item.key for item in implemented],
                         [0, 100, 110, 120, 121, 130])
        self.assertEqual([item.path for item in implemented],
                         ["reference", "auto_fused", "auto_fused",
                          "auto_fused", "auto_fused", "auto_fused"])
        self.assertEqual(implemented[1].name, "auto_matmul_fused")
        self.assertEqual(
            [(item.schedule, item.reduction, item.ub_input_buffers)
             for item in implemented[1:]],
            [("sync", "scalar", 1), ("sync", "vector", 1),
             ("async", "vector", 1), ("async", "vector", 2),
             ("async", "vector", 2)],
        )
        self.assertTrue(all(item.split_n == 1 for item in implemented[:-1]))
        self.assertEqual(implemented[-1].split_n, 2)

    def test_generated_header_is_current(self):
        self.assertEqual(OUTPUT.read_text(encoding="utf-8"), render())


class EnumerationTests(unittest.TestCase):
    def test_reference_workspace_is_padded_atomic_output(self):
        plans = enumerate_plans(Problem(2, 17, 33, 40),
                                implemented_only=True)
        reference = next(plan for plan in plans if plan.config.key == 0)
        self.assertEqual(reference.workspace_bytes, 8 * 4)
        self.assertEqual(reference.launch_blocks, 34)

    def test_auto_baseline_uses_m_shards_and_atomic_output(self):
        hardware = Hardware(aic=20, aiv=40,
                            system_workspace_bytes=4096)
        plans = enumerate_plans(
            Problem(3, 65, 257, 128), hardware, implemented_only=True,
        )
        auto = next(plan for plan in plans if plan.config.key == 100)
        self.assertEqual(auto.task_count, 6)
        self.assertEqual(auto.launch_blocks, 6)
        self.assertEqual(auto.split_n, 1)
        self.assertEqual(auto.single_core_m, 32)
        self.assertEqual(auto.single_core_n, 257)
        self.assertEqual(auto.workspace_bytes, 4608)

        more_m = next(
            plan for plan in enumerate_plans(
                Problem(3, 4096, 257, 128), hardware,
                implemented_only=True,
            )
            if plan.config.key == 100
        )
        self.assertEqual(more_m.workspace_bytes, auto.workspace_bytes)

    def test_auto_baseline_caps_launches_at_available_aic(self):
        plans = enumerate_plans(
            Problem(64, 129, 128, 64),
            Hardware(aic=20, aiv=40), implemented_only=True,
        )
        auto = next(plan for plan in plans if plan.config.key == 100)
        self.assertEqual(auto.task_count, 64 * 3)
        self.assertEqual(auto.launch_blocks, 20)

    def test_async_candidates_reserve_independent_aiv_workspaces(self):
        hardware = Hardware(aic=20, aiv=40,
                            system_workspace_bytes=4096)
        plans = enumerate_plans(
            Problem(3, 65, 257, 128), hardware, implemented_only=True,
        )
        by_key = {plan.config.key: plan for plan in plans}
        self.assertEqual(by_key[100].workspace_bytes, 4608)
        self.assertEqual(by_key[110].workspace_bytes, 4608)
        self.assertEqual(by_key[120].workspace_bytes, 594432)
        self.assertEqual(by_key[121].workspace_bytes, 594432)
        self.assertGreater(by_key[121].memory.ub_used,
                           by_key[120].memory.ub_used)

    def test_split_n_fills_idle_aic_and_reserves_partial_maxima(self):
        hardware = Hardware(aic=20, aiv=40,
                            system_workspace_bytes=4096)
        plans = enumerate_plans(
            Problem(1, 33, 513, 256), hardware, implemented_only=True,
        )
        split = next(plan for plan in plans if plan.config.key == 130)
        self.assertEqual(split.split_n, 5)
        self.assertEqual(split.task_count, 5)
        self.assertEqual(split.launch_blocks, 5)
        self.assertEqual(split.single_core_n, 113)
        self.assertEqual(split.workspace_bytes, 170496)

        already_full = next(
            plan for plan in enumerate_plans(
                Problem(64, 3, 5, 40), hardware, implemented_only=True,
            )
            if plan.config.key == 130
        )
        self.assertEqual(already_full.split_n, 1)
        self.assertEqual(already_full.task_count, 64)
        self.assertEqual(already_full.launch_blocks, 20)

    def test_capacity_filter_keeps_only_reference_on_tiny_hardware(self):
        tiny = Hardware(ub_bytes=1024, l1_bytes=1024,
                        l0a_bytes=1024, l0b_bytes=1024,
                        l0c_bytes=1024)
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

    def test_tree_can_compare_reference_with_auto_baseline(self):
        common = {"B": 1, "N": 128, "K": 64,
                  "dtype": 1, "tx1": 0, "tx2": 0}
        small = Sample(
            dict(common, M=16, work=131072, bm_rows=16),
            {0: 1.0, 100: 2.0},
        )
        large = Sample(
            dict(common, M=64, work=524288, bm_rows=64),
            {0: 2.0, 100: 1.0},
        )
        tree = fit_tree([small, large], {0, 100},
                        max_depth=1, min_samples_leaf=1)
        policy = render_policy(
            tree, {0: "VECTOR_REFERENCE", 100: "AUTO_MATMUL_FUSED"},
        )
        self.assertIn("shape.m <= 16ULL", policy)
        self.assertIn("TilingKey::AUTO_MATMUL_FUSED", policy)

    def test_tree_can_select_sync_and_async_candidates_by_n_and_k(self):
        common = {"B": 1, "M": 64, "work": 1,
                  "bm_rows": 64, "dtype": 1, "tx1": 0, "tx2": 0}
        samples = [
            Sample(dict(common, N=128, K=64),
                   {110: 1.0, 120: 2.0, 121: 3.0}),
            Sample(dict(common, N=512, K=64),
                   {110: 2.0, 120: 1.0, 121: 2.0}),
            Sample(dict(common, N=512, K=1024),
                   {110: 3.0, 120: 2.0, 121: 1.0}),
        ]
        tree = fit_tree(samples, {110, 120, 121},
                        max_depth=2, min_samples_leaf=1)
        policy = render_policy(
            tree,
            {110: "AUTO_MATMUL_FUSED_VECTOR",
             120: "AUTO_MATMUL_FUSED_ASYNC",
             121: "AUTO_MATMUL_FUSED_ASYNC_DB"},
        )
        self.assertIn("shape.n", policy)
        self.assertIn("shape.k", policy)
        self.assertIn("TilingKey::AUTO_MATMUL_FUSED_VECTOR", policy)
        self.assertIn("TilingKey::AUTO_MATMUL_FUSED_ASYNC", policy)
        self.assertIn("TilingKey::AUTO_MATMUL_FUSED_ASYNC_DB", policy)


if __name__ == "__main__":
    unittest.main()
