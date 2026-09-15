template<bool VectorizedReduction, bool Async, uint32_t CBufferCount,
         uint32_t TileM, uint32_t TileN, uint32_t TileK>
inline void ExecuteAutoFusedPlan(
        const Plan& plan, GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
        int32_t inputDtype, aclrtStream stream,
        bool transposeX1, bool transposeX2) {
    const Shape shape{plan.tiling.b, plan.tiling.m,
                      plan.tiling.n, plan.tiling.k};
    const CachedExecutionResources resources =
        GetAutoFusedExecutionResources<TileM, TileN, TileK>(
            plan, stream, inputDtype, transposeX1, transposeX2);
    GM_ADDR atomicOutput =
        resources.workspace + plan.tiling.atomicOutputOffsetBytes;
    GM_ADDR matmulCache =
        resources.workspace + plan.tiling.matmulCacheOffsetBytes;
    const size_t atomicOutputBytes = static_cast<size_t>(
        AlignUpU64(shape.b, kAtomicAlignmentFloats) * sizeof(float));
    CheckAcl(aclrtMemsetAsync(
                 atomicOutput, atomicOutputBytes, 0,
                 atomicOutputBytes, stream),
             "aclrtMemsetAsync(auto fused atomic output)");

    if (inputDtype == 1) {
        LaunchBatchMatmulMaxSumAutoFusedByTranspose<
            half, VectorizedReduction, Async, CBufferCount,
            TileM, TileN, TileK>(
                x1, x2, atomicOutput, resources.workspace,
                matmulCache, plan.tiling.matmulCacheStrideBytes,
                resources.cubeTiling, shape,
                plan.tiling.launchBlocks, stream,
                transposeX1, transposeX2);
    } else {
        LaunchBatchMatmulMaxSumAutoFusedByTranspose<
            bfloat16_t, VectorizedReduction, Async, CBufferCount,
            TileM, TileN, TileK>(
                x1, x2, atomicOutput, resources.workspace,
                matmulCache, plan.tiling.matmulCacheStrideBytes,
                resources.cubeTiling, shape,
                plan.tiling.launchBlocks, stream,
                transposeX1, transposeX2);
    }
    const size_t outputBytes = static_cast<size_t>(shape.b) * sizeof(float);
    CheckAcl(aclrtMemcpyAsync(
                 y, outputBytes, atomicOutput, outputBytes,
                 ACL_MEMCPY_DEVICE_TO_DEVICE, stream),
             "aclrtMemcpyAsync(auto fused atomic output to y)");
}
