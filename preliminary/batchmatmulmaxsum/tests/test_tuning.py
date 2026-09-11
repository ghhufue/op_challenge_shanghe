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
        self.assertEqual(
            [item.key for item in implemented],
            [0, 100, 101, 102, 103, 200, 201],
        )
        self.assertEqual(implemented[0].path, "reference")
        self.assertTrue(all(item.path == "bm" for item in implemented[1:5]))
        self.assertTrue(all(item.path == "bmn" for item in implemented[5:]))

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

    def test_larger_bm_workspace_uses_each_compile_time_tile(self):
        hardware = Hardware(aic=20, aiv=40)
        plans = enumerate_plans(
            Problem(3, 65, 257, 128), hardware, implemented_only=True,
        )

        bm_n256 = next(plan for plan in plans if plan.config.key == 102)
        self.assertEqual(bm_n256.task_count, 9)
        self.assertEqual(bm_n256.launch_blocks, 9)
        self.assertEqual(
            bm_n256.workspace_bytes,
            1536 + 9 * 32 * 256 * 4 + 8 * 4,
        )

        bm_m64_k128 = next(plan for plan in plans if plan.config.key == 103)
        self.assertEqual(bm_m64_k128.task_count, 6)
        self.assertEqual(bm_m64_k128.launch_blocks, 6)
        self.assertEqual(
            bm_m64_k128.workspace_bytes,
            1536 + 6 * 64 * 128 * 4 + 8 * 4,
        )

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


class SubmissionSourceTests(unittest.TestCase):
    def test_dispatch_returns_without_internal_stream_synchronization(self):
        dispatch = (OP_ROOT / "kernels" / "kernel_dispatch.asc").read_text(
            encoding="utf-8",
        )
        self.assertNotIn("aclrtSynchronizeStream(", dispatch)

    def test_execution_resources_are_cached_per_stream_and_plan(self):
        dispatch = (OP_ROOT / "kernels" / "kernel_dispatch.asc").read_text(
            encoding="utf-8",
        )
        for member in (
            "aclrtStream stream;",
            "TilingKey tilingKey;",
            "Shape shape;",
            "uint64_t workspaceBytes;",
            "GM_ADDR cubeTiling;",
        ):
            self.assertIn(member, dispatch)
        self.assertNotIn("aclrtGetDevice(", dispatch)

    def test_cube_tiling_is_cached_in_gm_with_one_synchronous_upload(self):
        tiling = (OP_ROOT / "host" / "bm_tiling.h").read_text(encoding="utf-8")
        mixed = (OP_ROOT / "kernels" / "mixed.asc").read_text(encoding="utf-8")
        self.assertNotIn("aclrtMallocHost", tiling)
        self.assertNotIn("aclrtMemcpyAsync", tiling)
        self.assertIn("aclrtMemcpy(", tiling)
        self.assertIn("GM_ADDR cubeTilingAddress", mixed)
        self.assertIn("LoadMixedCubeTiling(cubeTilingAddress, cubeTiling);", mixed)
        self.assertIn("KERNEL_TYPE_MIX_AIC_1_2", mixed)

    def test_optimized_dispatch_uses_one_mixed_kernel_launch(self):
        mixed = (OP_ROOT / "kernels" / "mixed.asc").read_text(encoding="utf-8")
        dispatch = (OP_ROOT / "kernels" / "kernel_dispatch.asc").read_text(
            encoding="utf-8",
        )
        self.assertEqual(mixed.count("<<<"), 1)
        self.assertNotIn("<<<", dispatch)
        self.assertIn("LaunchBatchMatmulMaxSumMixedByTranspose", dispatch)
        self.assertNotIn("LaunchBmMatmul", dispatch)
        self.assertNotIn("UpdateBmRowMax", dispatch)
        self.assertNotIn("ReduceBmnPartialMax", dispatch)

    def test_mixed_kernel_matches_official_cross_core_protocol(self):
        mixed = (OP_ROOT / "kernels" / "mixed.asc").read_text(encoding="utf-8")
        self.assertIn("constexpr uint64_t kMixedReadyFlag = 5;", mixed)
        self.assertIn("constexpr uint64_t kMixedReusableFlagBase = 6;", mixed)
        self.assertIn("CrossCoreWaitFlag(kMixedReadyFlag);", mixed)
        self.assertIn(
            "CrossCoreSetFlag<kMixedSyncMode, PIPE_FIX>(kMixedReadyFlag);",
            mixed,
        )
        self.assertIn("SyncAll<false>();", mixed)
        self.assertIn(
            "matmul.template IterateAll<false>(stage[stageOffset], 0, true);",
            mixed,
        )
        self.assertIn("segment * RowsPerLane * 2 + row", mixed)
        self.assertLess(
            mixed.index("rowMax.SetValue(row, best);"),
            mixed.index("CrossCoreSetFlag<kMixedSyncMode, PIPE_MTE3>"),
        )

    def test_optimized_plan_uses_physical_cube_core_count(self):
        entry = (OP_ROOT / "kernel.asc").read_text(encoding="utf-8")
        self.assertIn("bmms::QueryCubeCoreCount()", entry)
        self.assertIn("bmms::MakePlan(shape, computeCoreNum, key)", entry)

    def test_mixed_scheduler_covers_tail_rows_without_overlap_or_deadlock(self):
        cases = (
            (1, 1, 3, 16, 128, 20),
            (3, 33, 257, 16, 256, 20),
            (64, 129, 8192, 64, 128, 20),
        )
        for batches, rows, columns, tile_m, tile_n, available_cores in cases:
            m_groups = (rows + tile_m - 1) // tile_m
            n_tiles = (columns + tile_n - 1) // tile_n
            bm_task_count = batches * m_groups
            n_groups = min(
                n_tiles,
                max(1, (available_cores + bm_task_count - 1) // bm_task_count),
            )
            task_count = bm_task_count * n_groups
            cube_blocks = min(task_count, available_cores)
            covered_values = set()
            for logical_core in range(cube_blocks):
                iterations = 0
                for task in range(logical_core, task_count, cube_blocks):
                    bm_task, n_group = divmod(task, n_groups)
                    batch, m_group = divmod(bm_task, m_groups)
                    row_base = m_group * tile_m
                    actual_m = min(tile_m, rows - row_base)
                    rows_per_lane = tile_m // 2
                    for n_tile in range(n_group, n_tiles, n_groups):
                        for lane in range(2):
                            lane_begin = lane * rows_per_lane
                            lane_end = min(actual_m, lane_begin + rows_per_lane)
                            for row in range(lane_begin, lane_end):
                                value = (batch, row_base + row, n_tile)
                                self.assertNotIn(value, covered_values)
                                covered_values.add(value)
                        iterations += 1

                for ping_pong in range(2):
                    waits = sum(
                        sequence >= 2 and sequence % 2 == ping_pong
                        for sequence in range(iterations)
                    )
                    releases = sum(
                        sequence + 2 < iterations and sequence % 2 == ping_pong
                        for sequence in range(iterations)
                    )
                    self.assertEqual(waits, releases)

            self.assertEqual(covered_values, {
                (batch, row, n_tile)
                for batch in range(batches)
                for row in range(rows)
                for n_tile in range(n_tiles)
            })

            vector_blocks = cube_blocks * 2
            reduced_rows = set()
            for worker in range(vector_blocks):
                for lane_task in range(worker, bm_task_count * 2, vector_blocks):
                    bm_task, lane = divmod(lane_task, 2)
                    batch, m_group = divmod(bm_task, m_groups)
                    row_base = m_group * tile_m
                    actual_m = min(tile_m, rows - row_base)
                    lane_begin = lane * (tile_m // 2)
                    lane_end = min(actual_m, lane_begin + tile_m // 2)
                    for row in range(lane_begin, lane_end):
                        reduced_rows.add((batch, row_base + row))
            self.assertEqual(reduced_rows, {
                (batch, row) for batch in range(batches) for row in range(rows)
            })

    def test_submission_policy_uses_optimized_paths(self):
        policy = (OP_ROOT / "tiling" / "submission_policy.h").read_text(
            encoding="utf-8",
        )
        self.assertIn("TilingKey::BM_", policy)
        self.assertIn("TilingKey::BMN_", policy)
        self.assertNotIn("return TilingKey::VECTOR_REFERENCE;", policy)


if __name__ == "__main__":
    unittest.main()
