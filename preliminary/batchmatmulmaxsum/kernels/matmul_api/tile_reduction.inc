    __aicore__ inline void ReduceTile(
            const LocalTensor<float>& tile,
            LocalTensor<float>& rowMax,
            LocalTensor<float>& segmentMax,
            uint32_t laneRows, uint32_t nTile, uint32_t nExtent) {
        const uint32_t baseN = static_cast<uint32_t>(cubeTiling.baseN);
        const uint32_t columnBase = nTile * baseN;
        const uint32_t actualN = static_cast<uint32_t>(
            nExtent - columnBase < baseN
                ? nExtent - columnBase : baseN);
        const uint32_t activeSegments =
            (actualN + FloatRepeat - 1) / FloatRepeat;

        if (actualN % FloatBlockElements != 0) {
            // A compact row can begin at an unaligned element offset.  Peel
            // at most one data block so every WholeReduceMax source and
            // destination is 32-byte aligned, then reduce the row remainder
            // in vector-sized segments.
            if constexpr (VectorizedReduction) {
                SetFlag<HardEvent::V_S>(0);
                WaitFlag<HardEvent::V_S>(0);
            }
            for (uint32_t row = 0; row < laneRows; ++row) {
                const uint32_t rowStart = row * actualN;
                const uint32_t misalignment =
                    rowStart % FloatBlockElements;
                uint32_t prefixCount = misalignment == 0
                    ? 0 : FloatBlockElements - misalignment;
                prefixCount = prefixCount < actualN ? prefixCount : actualN;

                float best = rowMax.GetValue(row);
                for (uint32_t column = 0; column < prefixCount; ++column) {
                    const float value = tile.GetValue(rowStart + column);
                    best = value > best ? value : best;
                }
                rowMax.SetValue(row, best);

                const uint32_t vectorCount = actualN - prefixCount;
                const uint32_t vectorSegments =
                    (vectorCount + FloatRepeat - 1) / FloatRepeat;
                for (uint32_t segment = 0; segment < vectorSegments;
                     ++segment) {
                    const uint32_t segmentStart =
                        rowStart + prefixCount + segment * FloatRepeat;
                    const uint32_t segmentCount =
                        vectorCount - segment * FloatRepeat < FloatRepeat
                            ? vectorCount - segment * FloatRepeat
                            : FloatRepeat;
                    WholeReduceMax(
                        segmentMax[row * MaxSegments * FloatBlockElements +
                                   segment * FloatBlockElements],
                        tile[segmentStart],
                        static_cast<int32_t>(segmentCount), 1, 1, 1, 1,
                        ReduceOrder::ORDER_ONLY_VALUE);
                }
            }

            SetFlag<HardEvent::V_S>(0);
            WaitFlag<HardEvent::V_S>(0);
            for (uint32_t row = 0; row < laneRows; ++row) {
                const uint32_t rowStart = row * actualN;
                const uint32_t misalignment =
                    rowStart % FloatBlockElements;
                uint32_t prefixCount = misalignment == 0
                    ? 0 : FloatBlockElements - misalignment;
                prefixCount = prefixCount < actualN ? prefixCount : actualN;
                const uint32_t vectorCount = actualN - prefixCount;
                const uint32_t vectorSegments =
                    (vectorCount + FloatRepeat - 1) / FloatRepeat;
                float best = rowMax.GetValue(row);
                for (uint32_t segment = 0; segment < vectorSegments;
                     ++segment) {
                    const float value = segmentMax.GetValue(
                        row * MaxSegments * FloatBlockElements +
                        segment * FloatBlockElements);
                    best = value > best ? value : best;
                }
                rowMax.SetValue(row, best);
            }
            if constexpr (VectorizedReduction) {
                SetFlag<HardEvent::S_V>(0);
                WaitFlag<HardEvent::S_V>(0);
            }
            return;
        }

        for (uint32_t segment = 0; segment < activeSegments; ++segment) {
            const uint32_t segmentStart = segment * FloatRepeat;
            const uint32_t segmentCount = actualN - segmentStart < FloatRepeat
                ? actualN - segmentStart : FloatRepeat;
            // Sequential GetTensorC writes every N tile compactly.  Its row
            // stride is the current tile width, not the aligned Matmul baseN.
            WholeReduceMax(
                segmentMax[segment * RowsPerLane * FloatBlockElements],
                tile[segmentStart], static_cast<int32_t>(segmentCount),
                static_cast<int32_t>(laneRows), 1, 1,
                static_cast<int32_t>(actualN / FloatBlockElements),
                ReduceOrder::ORDER_ONLY_VALUE);
        }
        if constexpr (VectorizedReduction) {
            for (uint32_t segment = 0; segment < activeSegments; ++segment) {
                Max(rowMax, rowMax,
                    segmentMax[segment * RowsPerLane * FloatBlockElements],
                    static_cast<int32_t>(laneRows));
            }
        } else {
            SetFlag<HardEvent::V_S>(0);
            WaitFlag<HardEvent::V_S>(0);
            for (uint32_t row = 0; row < laneRows; ++row) {
                float best = rowMax.GetValue(row);
                for (uint32_t segment = 0; segment < activeSegments; ++segment) {
                    const float value = segmentMax.GetValue(
                        segment * RowsPerLane * FloatBlockElements + row);
                    best = value > best ? value : best;
                }
                rowMax.SetValue(row, best);
            }
        }
    }
