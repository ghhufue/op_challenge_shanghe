
    __aicore__ inline void Init(
            GM_ADDR x1, GM_ADDR x2, GM_ADDR atomicOutputAddress,
            GM_ADDR matmulCacheAddress, uint64_t matmulCacheStrideBytes,
            GM_ADDR partialMaxAddress, GM_ADDR syncWorkspaceAddress,
            uint32_t splitN,
            const TCubeTiling& cubeTiling, Shape shape,
            uint32_t cubeBlocks, TPipe* pipe) {
        static_assert(TileM % kAivPerAic == 0,
                      "M shard must split evenly across paired AIV cores");
        static_assert(TileN % 8 == 0,
                      "N tile stride must be a whole number of data blocks");
        static_assert(CBufferCount == 1 || CBufferCount == 2,
                      "C queue supports one or two UB buffers");
        static_assert(Async || CBufferCount == 1,
                      "UB double buffering requires asynchronous Matmul");
        this->shape = shape;
        this->cubeTiling = cubeTiling;
        this->cubeBlocks = cubeBlocks;
        this->matmulCacheAddress = matmulCacheAddress;
        this->matmulCacheStrideBytes = matmulCacheStrideBytes;
        this->splitN = splitN;

        lhs.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x1));
        rhs.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x2));
        atomicOutput.SetGlobalBuffer(
            reinterpret_cast<__gm__ float*>(atomicOutputAddress));
        if constexpr (SplitN) {
            partialMax.SetGlobalBuffer(
                reinterpret_cast<__gm__ float*>(partialMaxAddress));
            syncWorkspace.SetGlobalBuffer(
                reinterpret_cast<__gm__ int32_t*>(syncWorkspaceAddress));
        }

        pipe->InitBuffer(cQueue, CBufferCount,
                         RowsPerLane * TileN * sizeof(float));
        pipe->InitBuffer(rowMaxBuffer, RowsPerLane * sizeof(float));
        pipe->InitBuffer(segmentMaxBuffer,
                         RowsPerLane * MaxSegments * FloatBlockElements *
                             sizeof(float));
        pipe->InitBuffer(atomicBuffer,
                         kMaxOutputBatches * sizeof(float));
        if constexpr (SplitN) {
            pipe->InitBuffer(syncBuffer,
                             cubeBlocks * kAivPerAic * kSoftSyncSlotBytes);
        }
    }
