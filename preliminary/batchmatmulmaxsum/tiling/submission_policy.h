#pragma once

#include "tiling_catalog.h"

namespace bmms {

// Keep the first policy deliberately simple. Forced-key benchmarking remains
// available, and measured results can replace this threshold later.
inline TilingKey SelectSubmissionTilingKey(
        const Shape& shape, int64_t availableCoreNum, int32_t inputDtype,
        bool transposeX1, bool transposeX2) {
    (void)shape;
    (void)availableCoreNum;
    (void)inputDtype;
    (void)transposeX1;
    (void)transposeX2;
    return TilingKey::AUTO_MATMUL_FUSED;
}

}  // namespace bmms
