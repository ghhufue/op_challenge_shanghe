#pragma once

#include "tiling_catalog.h"

namespace bmms {

inline TilingKey SelectSubmissionTilingKey(
        const Shape& shape, int64_t availableCoreNum, int32_t inputDtype,
        bool transposeX1, bool transposeX2) {
    (void)inputDtype;
    (void)transposeX1;
    (void)transposeX2;
    const uint64_t work = static_cast<uint64_t>(shape.b) * shape.m *
                          shape.n * shape.k;
    if (work <= 65536ULL) {
        return TilingKey::VECTOR_REFERENCE;
    }

    const uint64_t mGroups =
        (static_cast<uint64_t>(shape.m) + 63ULL) / 64ULL;
    const uint64_t mTasks = static_cast<uint64_t>(shape.b) * mGroups;
    const uint64_t workPerMTask =
        static_cast<uint64_t>(shape.n) * shape.k;
    if (workPerMTask >= 1048576ULL && shape.n >= 512 &&
        mTasks * 2ULL <= static_cast<uint64_t>(availableCoreNum)) {
        return TilingKey::AUTO_MATMUL_FUSED_SPLIT_N;
    }
    if (work >= 300000ULL) {
        return TilingKey::AUTO_MATMUL_FUSED_ASYNC_DB;
    }
    return TilingKey::AUTO_MATMUL_FUSED;
}

}  // namespace bmms
