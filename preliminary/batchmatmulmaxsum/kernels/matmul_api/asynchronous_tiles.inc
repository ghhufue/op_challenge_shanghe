    __aicore__ inline void ProcessAsyncTiles(
            LocalTensor<float>& rowMax, LocalTensor<float>& segmentMax,
            uint32_t laneRows, uint32_t rawBlock, uint32_t nExtent) {
        const uint32_t baseN = static_cast<uint32_t>(cubeTiling.baseN);
        const uint32_t tileCount = static_cast<uint32_t>(
            (nExtent + baseN - 1) / baseN);
        GlobalTensor<float> clientWorkspace;
        clientWorkspace.SetGlobalBuffer(
            reinterpret_cast<__gm__ float*>(
                matmulCacheAddress + rawBlock * matmulCacheStrideBytes),
            matmulCacheStrideBytes / sizeof(float));
        matmulObj.SetWorkspace(clientWorkspace);
        matmulObj.template Iterate<false>();

        if constexpr (CBufferCount == 1) {
            for (uint32_t nTile = 0; nTile < tileCount; ++nTile) {
                LocalTensor<float> cLocal = cQueue.template AllocTensor<float>();
                matmulObj.template GetTensorC<false>(cLocal, 0, true);
                cQueue.EnQue(cLocal);
                cLocal = cQueue.template DeQue<float>();
                ReduceTile(cLocal, rowMax, segmentMax, laneRows, nTile,
                           nExtent);
                cQueue.FreeTensor(cLocal);
            }
        } else {
            LocalTensor<float> first = cQueue.template AllocTensor<float>();
            matmulObj.template GetTensorC<false>(first, 0, true);
            cQueue.EnQue(first);

            for (uint32_t nTile = 1; nTile < tileCount; ++nTile) {
                LocalTensor<float> next = cQueue.template AllocTensor<float>();
                matmulObj.template GetTensorC<false>(next, 0, true);
                cQueue.EnQue(next);

                LocalTensor<float> previous = cQueue.template DeQue<float>();
                ReduceTile(previous, rowMax, segmentMax, laneRows,
                           nTile - 1, nExtent);
                cQueue.FreeTensor(previous);
            }

            LocalTensor<float> last = cQueue.template DeQue<float>();
            ReduceTile(last, rowMax, segmentMax, laneRows, tileCount - 1,
                       nExtent);
            cQueue.FreeTensor(last);
        }
    }
