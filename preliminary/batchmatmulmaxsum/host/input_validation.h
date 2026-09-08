#pragma once

#include <cstdint>
#include <stdexcept>
#include "acl/acl.h"
#include "tiling/tiling_types.h"

namespace bmms {

inline Shape ValidateInputs(GM_ADDR x1, const TensorGroupInfo& infoX1,
                            GM_ADDR x2, const TensorGroupInfo& infoX2,
                            GM_ADDR y, const TensorGroupInfo& infoY,
                            bool transposeX1, bool transposeX2) {
    if (x1 == nullptr || x2 == nullptr || y == nullptr) {
        throw std::invalid_argument("Input and output device addresses must be non-null");
    }
    if (infoX1.numTensors != 1 || infoX2.numTensors != 1 || infoY.numTensors != 1 ||
        infoX1.tensors == nullptr || infoX2.tensors == nullptr || infoY.tensors == nullptr) {
        throw std::invalid_argument("Each tensor group must contain exactly one tensor");
    }

    const auto& lhs = infoX1.tensors[0];
    const auto& rhs = infoX2.tensors[0];
    const auto& output = infoY.tensors[0];
    if (lhs.shape == nullptr || rhs.shape == nullptr || output.shape == nullptr ||
        lhs.numDims != 3 || rhs.numDims != 3 || output.numDims != 1) {
        throw std::invalid_argument("Expected rank-3 inputs and a rank-1 output");
    }
    if ((lhs.dtype != 1 && lhs.dtype != 2) || rhs.dtype != lhs.dtype || output.dtype != 0) {
        throw std::invalid_argument("Expected matching FP16/BF16 inputs and FP32 output");
    }

    Shape shape{
        lhs.shape[0],
        transposeX1 ? lhs.shape[2] : lhs.shape[1],
        transposeX2 ? rhs.shape[1] : rhs.shape[2],
        transposeX1 ? lhs.shape[1] : lhs.shape[2]
    };
    const int64_t rhsK = transposeX2 ? rhs.shape[2] : rhs.shape[1];
    if (rhs.shape[0] != shape.b || rhsK != shape.k || output.shape[0] != shape.b) {
        throw std::invalid_argument("Input/output shapes are inconsistent");
    }
    if (shape.b < 1 || shape.b > 64 || shape.m < 1 || shape.m > 8192 ||
        shape.n < 1 || shape.n > 8192 || shape.k < 32 || shape.k > 8192 ||
        shape.k % 8 != 0 || shape.b * shape.m * shape.k > (1LL << 26) ||
        shape.b * shape.n * shape.k > (1LL << 26)) {
        throw std::invalid_argument("Shape is outside the supported problem range");
    }
    return shape;
}

}  // namespace bmms
