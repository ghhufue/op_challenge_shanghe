    __aicore__ inline float ReduceLaneSum(
            LocalTensor<float>& rowMax, LocalTensor<float>& segmentMax,
            uint32_t laneRows) {
        if constexpr (VectorizedReduction) {
            WholeReduceSum(segmentMax, rowMax,
                           static_cast<int32_t>(laneRows), 1, 1, 1, 8);
            SetFlag<HardEvent::V_S>(0);
            WaitFlag<HardEvent::V_S>(0);
            return segmentMax.GetValue(0);
        }
        float laneSum = 0.0f;
        for (uint32_t row = 0; row < laneRows; ++row) {
            laneSum += rowMax.GetValue(row);
        }
        return laneSum;
    }
