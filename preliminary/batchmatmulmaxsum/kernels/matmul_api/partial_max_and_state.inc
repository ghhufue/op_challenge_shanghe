    __aicore__ inline void StorePartialMax(
            LocalTensor<float>& rowMax, uint64_t task,
            uint32_t vectorLane) {
        // Row maxima may have last been produced by either Vector or Scalar
        // instructions (the compact, unaligned-N path uses both).  Join both
        // producers before MTE3 consumes the full padded lane record.
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        SetFlag<HardEvent::S_MTE3>(0);
        WaitFlag<HardEvent::S_MTE3>(0);
        const uint64_t record =
            task * kAivPerAic + static_cast<uint64_t>(vectorLane);
        DataCopy(partialMax[record * RowsPerLane], rowMax, RowsPerLane);
        SetFlag<HardEvent::MTE3_S>(0);
        WaitFlag<HardEvent::MTE3_S>(0);
    }

    Shape shape{};
    TCubeTiling cubeTiling{};
    uint32_t cubeBlocks = 0;
    GlobalTensor<T> lhs;
    GlobalTensor<T> rhs;
    GlobalTensor<float> atomicOutput;
    GlobalTensor<float> partialMax;
    GlobalTensor<int32_t> syncWorkspace;
    GM_ADDR matmulCacheAddress = nullptr;
    uint64_t matmulCacheStrideBytes = 0;
    uint32_t splitN = 1;
    TQue<TPosition::VECIN, CBufferCount> cQueue;
    TBuf<TPosition::VECCALC> rowMaxBuffer;
    TBuf<TPosition::VECCALC> segmentMaxBuffer;
    TBuf<TPosition::VECCALC> atomicBuffer;
    TBuf<TPosition::VECCALC> syncBuffer;
};
