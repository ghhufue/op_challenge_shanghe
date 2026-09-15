template<class T, bool TX1, bool TX2,
         bool VectorizedReduction, bool Async, uint32_t CBufferCount,
         uint32_t TileM, uint32_t TileN, uint32_t TileK,
         bool SplitN = false>
__global__ __mix__(1, 2) void BatchMatmulMaxSumAutoFused(
        GM_ADDR x1, GM_ADDR x2, GM_ADDR atomicOutputAddress,
        __kfc_workspace__ GM_ADDR workspaceAddress,
        GM_ADDR matmulCacheAddress, uint64_t matmulCacheStrideBytes,
        GM_ADDR partialMaxAddress, GM_ADDR syncWorkspaceAddress,
        uint32_t splitN,
        TCubeTiling cubeTiling, Shape shape, uint32_t cubeBlocks) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    TPipe pipe;
    AutoMatmulFusedKernel<T, TX1, TX2, VectorizedReduction, Async,
                          CBufferCount, TileM, TileN, TileK, SplitN> kernel;
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), kernel.matmulObj,
                      &cubeTiling);
    kernel.Init(x1, x2, atomicOutputAddress, matmulCacheAddress,
                matmulCacheStrideBytes, partialMaxAddress,
                syncWorkspaceAddress, splitN,
                cubeTiling, shape, cubeBlocks, &pipe);
    kernel.Process();
}

template<class T, bool TX1, bool TX2,
         bool VectorizedReduction, bool Async, uint32_t CBufferCount,
         uint32_t TileM, uint32_t TileN, uint32_t TileK>
inline void LaunchBatchMatmulMaxSumAutoFused(
        GM_ADDR x1, GM_ADDR x2, GM_ADDR atomicOutput, GM_ADDR workspace,
        GM_ADDR matmulCache, uint64_t matmulCacheStrideBytes,
        const TCubeTiling& cubeTiling, Shape shape,
        uint32_t cubeBlocks, aclrtStream stream) {
    BatchMatmulMaxSumAutoFused<T, TX1, TX2, VectorizedReduction, Async,
                               CBufferCount, TileM, TileN, TileK>
        <<<cubeBlocks, nullptr, stream>>>(
            x1, x2, atomicOutput, workspace, matmulCache,
            matmulCacheStrideBytes, nullptr, nullptr, 1,
            cubeTiling, shape, cubeBlocks);
}

template<class T, bool VectorizedReduction, bool Async,
         uint32_t CBufferCount,
         uint32_t TileM, uint32_t TileN, uint32_t TileK>
inline void LaunchBatchMatmulMaxSumAutoFusedByTranspose(
        GM_ADDR x1, GM_ADDR x2, GM_ADDR atomicOutput, GM_ADDR workspace,
        GM_ADDR matmulCache, uint64_t matmulCacheStrideBytes,
        const TCubeTiling& cubeTiling, Shape shape,
        uint32_t cubeBlocks, aclrtStream stream,
        bool transposeX1, bool transposeX2) {
    if (transposeX1 && transposeX2) {
        LaunchBatchMatmulMaxSumAutoFused<T, true, true,
                                         VectorizedReduction, Async,
                                         CBufferCount, TileM, TileN, TileK>(
            x1, x2, atomicOutput, workspace, matmulCache,
            matmulCacheStrideBytes, cubeTiling, shape,
            cubeBlocks, stream);
    } else if (transposeX1) {
        LaunchBatchMatmulMaxSumAutoFused<T, true, false,
                                         VectorizedReduction, Async,
                                         CBufferCount, TileM, TileN, TileK>(
            x1, x2, atomicOutput, workspace, matmulCache,
            matmulCacheStrideBytes, cubeTiling, shape,
            cubeBlocks, stream);
    } else if (transposeX2) {
        LaunchBatchMatmulMaxSumAutoFused<T, false, true,
                                         VectorizedReduction, Async,
                                         CBufferCount, TileM, TileN, TileK>(
            x1, x2, atomicOutput, workspace, matmulCache,
            matmulCacheStrideBytes, cubeTiling, shape,
            cubeBlocks, stream);
    } else {
        LaunchBatchMatmulMaxSumAutoFused<T, false, false,
                                         VectorizedReduction, Async,
                                         CBufferCount, TileM, TileN, TileK>(
            x1, x2, atomicOutput, workspace, matmulCache,
            matmulCacheStrideBytes, cubeTiling, shape,
            cubeBlocks, stream);
    }
}

template<class T, bool TX1, bool TX2,
         uint32_t TileM, uint32_t TileN, uint32_t TileK>
inline void LaunchBatchMatmulMaxSumAutoFusedSplitN(
        GM_ADDR x1, GM_ADDR x2, GM_ADDR atomicOutput, GM_ADDR workspace,
        GM_ADDR matmulCache, uint64_t matmulCacheStrideBytes,
        GM_ADDR partialMax, GM_ADDR syncWorkspace, uint32_t splitN,
        const TCubeTiling& cubeTiling, Shape shape,
        uint32_t cubeBlocks, aclrtStream stream) {
    BatchMatmulMaxSumAutoFused<T, TX1, TX2, true, true, 2,
                               TileM, TileN, TileK, true>
        <<<cubeBlocks, nullptr, stream>>>(
            x1, x2, atomicOutput, workspace, matmulCache,
            matmulCacheStrideBytes, partialMax, syncWorkspace, splitN,
            cubeTiling, shape, cubeBlocks);
}

template<class T, uint32_t TileM, uint32_t TileN, uint32_t TileK>
inline void LaunchBatchMatmulMaxSumAutoFusedSplitNByTranspose(
        GM_ADDR x1, GM_ADDR x2, GM_ADDR atomicOutput, GM_ADDR workspace,
        GM_ADDR matmulCache, uint64_t matmulCacheStrideBytes,
        GM_ADDR partialMax, GM_ADDR syncWorkspace, uint32_t splitN,
        const TCubeTiling& cubeTiling, Shape shape,
        uint32_t cubeBlocks, aclrtStream stream,
        bool transposeX1, bool transposeX2) {
    if (transposeX1 && transposeX2) {
        LaunchBatchMatmulMaxSumAutoFusedSplitN<
            T, true, true, TileM, TileN, TileK>(
                x1, x2, atomicOutput, workspace, matmulCache,
                matmulCacheStrideBytes, partialMax, syncWorkspace, splitN,
                cubeTiling, shape, cubeBlocks, stream);
    } else if (transposeX1) {
        LaunchBatchMatmulMaxSumAutoFusedSplitN<
            T, true, false, TileM, TileN, TileK>(
                x1, x2, atomicOutput, workspace, matmulCache,
                matmulCacheStrideBytes, partialMax, syncWorkspace, splitN,
                cubeTiling, shape, cubeBlocks, stream);
    } else if (transposeX2) {
        LaunchBatchMatmulMaxSumAutoFusedSplitN<
            T, false, true, TileM, TileN, TileK>(
                x1, x2, atomicOutput, workspace, matmulCache,
                matmulCacheStrideBytes, partialMax, syncWorkspace, splitN,
                cubeTiling, shape, cubeBlocks, stream);
    } else {
        LaunchBatchMatmulMaxSumAutoFusedSplitN<
            T, false, false, TileM, TileN, TileK>(
                x1, x2, atomicOutput, workspace, matmulCache,
                matmulCacheStrideBytes, partialMax, syncWorkspace, splitN,
                cubeTiling, shape, cubeBlocks, stream);
    }
}

}  // namespace bmms
