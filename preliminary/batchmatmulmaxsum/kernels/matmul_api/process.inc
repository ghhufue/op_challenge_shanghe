
    __aicore__ inline void Process() {
        const uint32_t rawBlock = GetBlockIdx();
        const uint32_t taskRatio = static_cast<uint32_t>(GetTaskRation());
        const uint32_t logicalCore = rawBlock / taskRatio;
        const uint32_t vectorLane = rawBlock - logicalCore * taskRatio;
        const uint64_t mGroups =
            (static_cast<uint64_t>(shape.m) + TileM - 1) / TileM;
        const uint64_t mTaskCount = static_cast<uint64_t>(shape.b) * mGroups;
        const uint64_t taskCount = SplitN
            ? mTaskCount * splitN : mTaskCount;

        LocalTensor<float> rowMax = rowMaxBuffer.Get<float>();
        LocalTensor<float> segmentMax = segmentMaxBuffer.Get<float>();
        LocalTensor<float> localSums = atomicBuffer.Get<float>();
        const uint32_t atomicElements = AtomicOutputElements(shape.b);
        for (uint32_t index = 0; index < atomicElements; ++index) {
            localSums.SetValue(index, 0.0f);
        }

        // The original matrix strides and bias state are invariant across
        // every (batch, M-tile) task owned by this client.  SetOrgShape emits
        // a KFC configuration message, so keep it out of the hot task loop.
        matmulObj.SetOrgShape(
            static_cast<int>(shape.m), static_cast<int>(shape.n),
            static_cast<int>(shape.k), static_cast<int>(shape.k),
            static_cast<int>(shape.n));
        matmulObj.DisableBias();

        for (uint64_t task = logicalCore; task < taskCount;
             task += cubeBlocks) {
            const uint64_t mTask = SplitN ? task / splitN : task;
            const uint32_t nShard = SplitN
                ? static_cast<uint32_t>(task - mTask * splitN) : 0;
            const uint64_t batch = mTask / mGroups;
            const uint64_t mGroup = mTask - batch * mGroups;
            const uint64_t rowBase = mGroup * TileM;
            const uint32_t actualM = static_cast<uint32_t>(
                shape.m - static_cast<int64_t>(rowBase) < TileM
                    ? shape.m - static_cast<int64_t>(rowBase) : TileM);
            const uint32_t laneRowBase = vectorLane * RowsPerLane;
            const uint32_t laneRows = actualM > laneRowBase
                ? (actualM - laneRowBase < RowsPerLane
                       ? actualM - laneRowBase : RowsPerLane)
                : 0;

            if (laneRows == 0) {
                continue;
            }
            InitRowMax(rowMax);

            const uint64_t globalRowBase = rowBase + laneRowBase;
            const uint64_t lhsBatch =
                batch * static_cast<uint64_t>(shape.m) * shape.k;
            const uint64_t rhsBatch =
                batch * static_cast<uint64_t>(shape.n) * shape.k;
            const uint64_t nBase = SplitN
                ? (static_cast<uint64_t>(shape.n) * nShard /
                   splitN / 16) * 16
                : 0;
            const uint64_t nEnd = SplitN
                ? (nShard + 1 == splitN
                       ? static_cast<uint64_t>(shape.n)
                       : (static_cast<uint64_t>(shape.n) * (nShard + 1) /
                          splitN / 16) * 16)
                : static_cast<uint64_t>(shape.n);
            const uint32_t actualShardN = SplitN
                ? static_cast<uint32_t>(nEnd - nBase)
                : static_cast<uint32_t>(shape.n);
            const uint64_t lhsOffset = lhsBatch +
                (TX1 ? globalRowBase
                     : globalRowBase * static_cast<uint64_t>(shape.k));
            const uint64_t rhsOffset = rhsBatch +
                (TX2 ? nBase * static_cast<uint64_t>(shape.k) : nBase);

            matmulObj.SetTensorA(lhs[lhsOffset], TX1);
            matmulObj.SetTensorB(rhs[rhsOffset], TX2);
            matmulObj.SetTail(static_cast<int>(laneRows),
                              static_cast<int>(actualShardN),
                              static_cast<int>(shape.k));

            if constexpr (Async) {
                ProcessAsyncTiles(rowMax, segmentMax, laneRows, rawBlock,
                                  actualShardN);
            } else {
                ProcessSynchronousTiles(rowMax, segmentMax, laneRows,
                                        actualShardN);
            }

            if constexpr (SplitN) {
                StorePartialMax(rowMax, task, vectorLane);
            } else {
                const float laneSum = ReduceLaneSum(
                    rowMax, segmentMax, laneRows);
                const uint32_t batchIndex = static_cast<uint32_t>(batch);
                localSums.SetValue(
                    batchIndex, localSums.GetValue(batchIndex) + laneSum);
            }
        }
        if constexpr (!SplitN) {
            AtomicAddBatchSums(atomicOutput, localSums, atomicElements);
        }
        matmulObj.End();
        if constexpr (SplitN) {
            const uint32_t workerCount = cubeBlocks * taskRatio;
            LocalTensor<int32_t> syncLocal = syncBuffer.Get<int32_t>();
            // false selects physical mixed-kernel block IDs, so both AIV
            // subblocks participate independently in this software barrier.
            SyncAll<false>(syncWorkspace, syncLocal, workerCount);
            ReduceAllNShardsAndSumM<TileM>(
                partialMax, atomicOutput, localSums, shape, splitN,
                rawBlock, workerCount);
        }
    }

private:
