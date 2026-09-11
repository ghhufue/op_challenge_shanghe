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
            "cQueue.EnQue(cLocal)",
            "cQueue.DeQue<float>()",
            "matmulObj.End()",
        ):
            self.assertIn(token, source)
        for forbidden in (
            "MatmulImpl",
            "CrossCoreSetFlag",
            "CrossCoreWaitFlag",
            "SyncAll",
            "stageOffset",
            "nGroups",
        ):
            self.assertNotIn(forbidden, source)
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

    def test_vector_reduction_uses_the_runtime_matmul_base_n(self):
        source = (OP_ROOT / "kernels" / "auto_matmul_fused.asc").read_text(
            encoding="utf-8",
        )
        self.assertIn("cubeTiling.baseN", source)
        self.assertIn("nTile * baseN", source)
        self.assertIn("baseN / 8", source)

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
            "LaunchReduceAutoFusedPartialScores",
        ):
            self.assertIn(token, source)
        self.assertNotIn("aclrtSynchronizeStream(", source)

    def test_final_reduce_is_separate_and_writes_each_batch(self):
        source = (OP_ROOT / "kernels" / "final_reduce.asc").read_text(
            encoding="utf-8",
        )
        self.assertIn("__global__ __vector__", source)
        self.assertIn("output.SetValue(batch, total);", source)
        self.assertIn("task * kAivPerAic + lane", source)

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
            self.assertIn("final_reduce.asc", source)
            self.assertNotIn('"kernels/bmn.asc"', source)
            self.assertNotIn('"kernels/mixed.asc"', source)

    def test_submission_policy_keeps_reference_and_fixed_baseline(self):
        source = (OP_ROOT / "tiling" / "submission_policy.h").read_text(
            encoding="utf-8",
        )
        self.assertIn("TilingKey::VECTOR_REFERENCE", source)
        self.assertIn("TilingKey::AUTO_MATMUL_FUSED", source)


if __name__ == "__main__":
    unittest.main()
