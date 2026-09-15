template<uint32_t TileM, uint32_t TileN, uint32_t TileK>
inline void ExecuteAutoFusedSplitNPlan(
        const Plan& plan, GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
        int32_t inputDtype, aclrtStream stream,
        bool transposeX1, bool transposeX2) {
    const Shape shape{plan.tiling.b, plan.tiling.m,
                      plan.tiling.n, plan.tiling.k};
    const CachedExecutionResources resources =
        GetAutoFusedExecutionResources<TileM, TileN, TileK>(
            plan, stream, inputDtype, transposeX1, transposeX2);
    GM_ADDR matmulCache =
        resources.workspace + plan.tiling.matmulCacheOffsetBytes;
    GM_ADDR partialMax =
        resources.workspace + plan.tiling.partialMaxOffsetBytes;
    GM_ADDR syncWorkspace =
        resources.workspace + plan.tiling.syncWorkspaceOffsetBytes;
    GM_ADDR atomicOutput =
        resources.workspace + plan.tiling.atomicOutputOffsetBytes;
    const size_t atomicOutputBytes = static_cast<size_t>(
        AlignUpU64(shape.b, kAtomicAlignmentFloats) * sizeof(float));
    CheckAcl(aclrtMemsetAsync(
                 atomicOutput, atomicOutputBytes, 0,
                 atomicOutputBytes, stream),
             "aclrtMemsetAsync(split-N atomic output)");
    const size_t syncWorkspaceBytes =
        static_cast<size_t>(plan.tiling.syncWorkspaceBytes);
    CheckAcl(aclrtMemsetAsync(
                 syncWorkspace, syncWorkspaceBytes, 0,
                 syncWorkspaceBytes, stream),
             "aclrtMemsetAsync(split-N soft-sync workspace)");

    if (inputDtype == 1) {
        LaunchBatchMatmulMaxSumAutoFusedSplitNByTranspose<
            half, TileM, TileN, TileK>(
                x1, x2, atomicOutput, resources.workspace,
                matmulCache, plan.tiling.matmulCacheStrideBytes,
                partialMax, syncWorkspace, plan.tiling.splitN,
                resources.cubeTiling, shape,
                plan.tiling.launchBlocks, stream,
                transposeX1, transposeX2);
    } else {
        LaunchBatchMatmulMaxSumAutoFusedSplitNByTranspose<
            bfloat16_t, TileM, TileN, TileK>(
                x1, x2, atomicOutput, resources.workspace,
                matmulCache, plan.tiling.matmulCacheStrideBytes,
                partialMax, syncWorkspace, plan.tiling.splitN,
                resources.cubeTiling, shape,
                plan.tiling.launchBlocks, stream,
                transposeX1, transposeX2);
    }
    const size_t outputBytes = static_cast<size_t>(shape.b) * sizeof(float);
    CheckAcl(aclrtMemcpyAsync(
                 y, outputBytes, atomicOutput, outputBytes,
                 ACL_MEMCPY_DEVICE_TO_DEVICE, stream),
             "aclrtMemcpyAsync(split-N output to y)");
}
