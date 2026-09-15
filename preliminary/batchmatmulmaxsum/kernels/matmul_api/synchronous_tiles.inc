    __aicore__ inline void ProcessSynchronousTiles(
            LocalTensor<float>& rowMax, LocalTensor<float>& segmentMax,
            uint32_t laneRows, uint32_t nExtent) {
        uint32_t nTile = 0;
        while (matmulObj.template Iterate<true>()) {
            LocalTensor<float> cLocal = cQueue.template AllocTensor<float>();
            matmulObj.template GetTensorC<true>(cLocal, 0, true);
            cQueue.EnQue(cLocal);
            cLocal = cQueue.template DeQue<float>();
            ReduceTile(cLocal, rowMax, segmentMax, laneRows, nTile,
                       nExtent);
            cQueue.FreeTensor(cLocal);
            ++nTile;
        }
    }
