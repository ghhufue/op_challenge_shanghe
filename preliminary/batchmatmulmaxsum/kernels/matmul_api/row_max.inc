    __aicore__ inline void InitRowMax(LocalTensor<float>& rowMax) {
        if constexpr (VectorizedReduction) {
            Duplicate(rowMax, -__builtin_inff(),
                      static_cast<int32_t>(RowsPerLane));
        } else {
            for (uint32_t row = 0; row < RowsPerLane; ++row) {
                rowMax.SetValue(row, -__builtin_inff());
            }
        }
    }
