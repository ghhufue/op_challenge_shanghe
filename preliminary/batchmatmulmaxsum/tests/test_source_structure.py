from __future__ import annotations

from pathlib import Path
import unittest


OP_ROOT = Path(__file__).resolve().parents[1]


class AutoFusedSourceTests(unittest.TestCase):
    def test_old_bm_and_bmn_kernel_files_are_removed(self):
        for name in ("bm.asc", "bmn.asc", "mixed.asc"):
            self.assertFalse((OP_ROOT / "kernels" / name).exists())
        self.assertFalse((OP_ROOT / "host" / "bm_tiling.h").exists())

    def test_auto_kernel_uses_official_high_level_api(self):
        source = (OP_ROOT / "kernels" / "auto_matmul_fused.asc").read_text(
            encoding="utf-8",
        )
        for token in (
            "__mix__(1, 2)",
            "matmul::Matmul<",
            "TPosition::VECIN",
            "REGIST_MATMUL_OBJ",
            "Iterate<true>",
            "GetTensorC<true>",
            "Iterate<false>",
            "GetTensorC<false>",
            "matmulObj.SetWorkspace(clientWorkspace)",
            "Duplicate(rowMax",
            "Max(rowMax, rowMax",
            "WholeReduceSum(segmentMax, rowMax",
            "TQue<TPosition::VECIN, CBufferCount>",
            "cQueue.EnQue(cLocal)",
            "cQueue.template DeQue<float>()",
            "matmulObj.End()",
        ):
            self.assertIn(token, source)
        for forbidden in (
            "MatmulImpl",
            "CrossCoreSetFlag",
            "CrossCoreWaitFlag",
            "stageOffset",
            "nGroups",
        ):
            self.assertNotIn(forbidden, source)
        self.assertIn("SyncAll<false>", source)
        self.assertLess(
            source.index("REGIST_MATMUL_OBJ"),
            source.index("kernel.Init(x1"),
        )

    def test_host_tiling_targets_vecin_and_full_n_ownership(self):
        source = (OP_ROOT / "host" / "fused_tiling.h").read_text(
            encoding="utf-8",
        )
        for token in (
            "MultiCoreMatmulTiling",
            "SetDim(1)",
            "TPosition::VECIN",
            "MatrixTraverse::FIRSTM",
            "std::min<uint32_t>(TileN, alignedN)",
            "result.singleCoreN",
            "GetLibApiWorkSpaceSize",
        ):
            self.assertIn(token, source)

    def test_vector_reduction_uses_the_compact_runtime_tile_width(self):
        source = (OP_ROOT / "kernels" / "auto_matmul_fused.asc").read_text(
            encoding="utf-8",
        )
        self.assertIn("cubeTiling.baseN", source)
        self.assertIn("nTile * baseN", source)
        self.assertIn("actualN / FloatBlockElements", source)
        self.assertIn("actualN % FloatBlockElements", source)
        self.assertIn("row * actualN", source)
        self.assertNotIn("baseN / 8", source)

    def test_dispatch_keeps_workspace_cached_and_asynchronous(self):
        source = (OP_ROOT / "kernels" / "kernel_dispatch.asc").read_text(
            encoding="utf-8",
        )
        for token in (
            "aclrtStream stream;",
            "TilingKey tilingKey;",
            "Shape shape;",
            "GM_ADDR workspace;",
            "TCubeTiling cubeTiling;",
            "GetOrCreateExecutionResources",
            "LaunchBatchMatmulMaxSumAutoFusedByTranspose",
            "aclrtMemsetAsync(auto fused atomic output)",
            "aclrtMemcpyAsync(auto fused atomic output to y)",
            "AUTO_MATMUL_FUSED_VECTOR",
            "AUTO_MATMUL_FUSED_ASYNC",
            "AUTO_MATMUL_FUSED_ASYNC_DB",
            "AUTO_MATMUL_FUSED_SPLIT_N",
        ):
            self.assertIn(token, source)
        self.assertNotIn("aclrtSynchronizeStream(", source)
        self.assertNotIn("LaunchReduceAutoFusedPartialScores", source)

    def test_auto_path_finalizes_split_n_inside_the_mixed_kernel(self):
        source = (OP_ROOT / "kernels" / "auto_matmul_fused.asc").read_text(
            encoding="utf-8",
        )
        split_source = (
            OP_ROOT / "kernels" / "auto_matmul_fused_split_n.asc"
        ).read_text(encoding="utf-8")
        dispatch = (OP_ROOT / "kernels" / "kernel_dispatch.asc").read_text(
            encoding="utf-8",
        )
        self.assertEqual(source.count("<<<"), 2)
        self.assertEqual(source.count("AtomicAddBatchSums("), 1)
        self.assertIn("localSums.GetValue(batchIndex) + laneSum", source)
        self.assertIn("ReduceAllNShardsAndSumM<TileM>", source)
        self.assertIn("partialMax.GetValue", split_source)
        self.assertIn("AtomicAddBatchSums(", split_source)
        self.assertNotIn("FinalizeAutoFusedSplitN", source + split_source)
        self.assertNotIn("LaunchFinalizeAutoFusedSplitN", dispatch)
        self.assertFalse((OP_ROOT / "kernels" / "final_reduce.asc").exists())

    def test_split_n_reduction_tasks_own_every_row_once(self):
        tile_m = 64
        rows_per_lane = tile_m // 2
        for batches, rows, workers in ((1, 33, 10), (3, 65, 40), (4, 129, 8)):
            m_groups = (rows + tile_m - 1) // tile_m
            task_count = batches * m_groups * 2
            covered = set()
            for worker in range(workers):
                for task in range(worker, task_count, workers):
                    lane = task % 2
                    m_task = task // 2
                    batch, m_group = divmod(m_task, m_groups)
                    row_base = m_group * tile_m
                    actual_m = min(tile_m, rows - row_base)
                    lane_begin = lane * rows_per_lane
                    lane_rows = max(0, min(rows_per_lane, actual_m - lane_begin))
                    for row in range(lane_rows):
                        value = (batch, row_base + lane_begin + row)
                        self.assertNotIn(value, covered)
                        covered.add(value)
            self.assertEqual(
                covered,
                {(batch, row) for batch in range(batches)
                 for row in range(rows)},
            )

    def test_each_row_is_owned_once_without_n_partitioning(self):
        cases = ((1, 1, 20), (3, 65, 20), (64, 129, 20))
        tile_m = 64
        rows_per_lane = tile_m // 2
        for batches, rows, available_aic in cases:
            m_groups = (rows + tile_m - 1) // tile_m
            tasks = batches * m_groups
            cube_blocks = min(tasks, available_aic)
            covered = set()
            for logical_core in range(cube_blocks):
                for task in range(logical_core, tasks, cube_blocks):
                    batch, m_group = divmod(task, m_groups)
                    row_base = m_group * tile_m
                    actual_m = min(tile_m, rows - row_base)
                    for lane in range(2):
                        begin = lane * rows_per_lane
                        end = min(actual_m, begin + rows_per_lane)
                        for local_row in range(begin, end):
                            value = (batch, row_base + local_row)
                            self.assertNotIn(value, covered)
                            covered.add(value)
            self.assertEqual(
                covered,
                {(batch, row) for batch in range(batches)
                 for row in range(rows)},
            )

    def test_entry_and_bundler_use_only_new_fragments(self):
        entry = (OP_ROOT / "kernel.asc").read_text(encoding="utf-8")
        bundler = (OP_ROOT / "scripts" / "bundle_submission.py").read_text(
            encoding="utf-8",
        )
        for source in (entry, bundler):
            self.assertIn("auto_matmul_fused.asc", source)
            self.assertNotIn("final_reduce.asc", source)
            self.assertNotIn('"kernels/bmn.asc"', source)
            self.assertNotIn('"kernels/mixed.asc"', source)
        self.assertIn("auto_matmul_fused_split_n.asc", bundler)

    def test_submission_policy_routes_large_and_underfilled_work(self):
        source = (OP_ROOT / "tiling" / "submission_policy.h").read_text(
            encoding="utf-8",
        )
        self.assertIn("TilingKey::VECTOR_REFERENCE", source)
        self.assertIn("TilingKey::AUTO_MATMUL_FUSED", source)
        self.assertNotIn("TilingKey::AUTO_MATMUL_FUSED_VECTOR", source)
        self.assertIn("TilingKey::AUTO_MATMUL_FUSED_ASYNC_DB", source)
        self.assertIn("TilingKey::AUTO_MATMUL_FUSED_SPLIT_N", source)
        self.assertIn("300000ULL", source)
        self.assertIn("workPerMTask", source)
        self.assertIn("1048576ULL", source)


if __name__ == "__main__":
    unittest.main()
