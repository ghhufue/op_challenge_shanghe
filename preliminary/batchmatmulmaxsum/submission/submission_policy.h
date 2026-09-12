#pragma once


namespace bmms {

// Keep the first policy deliberately simple. Forced-key benchmarking remains
// available, and measured results can replace this threshold later.
inline TilingKey SelectSubmissionTilingKey(
        const Shape& shape, int64_t availableCoreNum, int32_t inputDtype,
        bool transposeX1, bool transposeX2) {
    (void)availableCoreNum;
    (void)inputDtype;
    (void)transposeX1;
    (void)transposeX2;
    const uint64_t work = static_cast<uint64_t>(shape.b) * shape.m *
                          shape.n * shape.k;
    return work <= 65536ULL
        ? TilingKey::VECTOR_REFERENCE
        : TilingKey::AUTO_MATMUL_FUSED;
}

}  // namespace bmms
