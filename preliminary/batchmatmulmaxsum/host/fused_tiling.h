#pragma once

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_types.h"

namespace bmms {

inline platform_ascendc::PlatformAscendC* GetAscendPlatform() {
    static platform_ascendc::PlatformAscendC* platform = []() {
        const char* socName = aclrtGetSocName();
        if (socName == nullptr) {
            throw std::runtime_error("aclrtGetSocName returned null");
        }
        auto* instance =
            platform_ascendc::PlatformAscendCManager::GetInstance(socName);
        if (instance == nullptr) {
            throw std::runtime_error(
                "Failed to initialize AscendC platform information");
        }
        return instance;
    }();
    return platform;
}

inline uint64_t QueryMatmulSystemWorkspaceBytes() {
    static const uint64_t bytes = static_cast<uint64_t>(
        GetAscendPlatform()->GetLibApiWorkSpaceSize());
    return bytes;
}

template<uint32_t TileM, uint32_t TileN, uint32_t TileK>
inline AscendC::tiling::TCubeTiling MakeAutoFusedCubeTiling(
        const Shape& shape, int32_t inputDtype,
        bool transposeX1, bool transposeX2) {
    static_assert(TileM % kAivPerAic == 0,
                  "M shard must split evenly across the paired AIV cores");
    constexpr uint32_t rowsPerLane = TileM / kAivPerAic;
    const uint32_t alignedN = static_cast<uint32_t>(
        ((shape.n + 15) / 16) * 16);
    const uint32_t baseN = std::min<uint32_t>(TileN, alignedN);
    const matmul_tiling::DataType dataType = inputDtype == 1
        ? matmul_tiling::DataType::DT_FLOAT16
        : matmul_tiling::DataType::DT_BF16;

    matmul_tiling::MultiCoreMatmulTiling tiling(*GetAscendPlatform());
    // Each AIV client submits an independent rowsPerLane x N Matmul to the
    // paired AIC. SetDim(1) prevents the library tiler from splitting N a
    // second time between the two clients.
    if (tiling.SetDim(1) != 0 ||
        tiling.SetAType(matmul_tiling::TPosition::GM,
                        matmul_tiling::CubeFormat::ND,
                        dataType, transposeX1) != 0 ||
        tiling.SetBType(matmul_tiling::TPosition::GM,
                        matmul_tiling::CubeFormat::ND,
                        dataType, transposeX2) != 0 ||
        tiling.SetCType(matmul_tiling::TPosition::VECIN,
                        matmul_tiling::CubeFormat::ND,
                        matmul_tiling::DataType::DT_FLOAT) != 0 ||
        tiling.SetBiasType(matmul_tiling::TPosition::GM,
                           matmul_tiling::CubeFormat::ND,
                           matmul_tiling::DataType::DT_FLOAT) != 0 ||
        tiling.SetOrgShape(static_cast<int32_t>(shape.m),
                           static_cast<int32_t>(shape.n),
                           static_cast<int32_t>(shape.k)) != 0 ||
        tiling.SetShape(static_cast<int32_t>(rowsPerLane),
                        static_cast<int32_t>(shape.n),
                        static_cast<int32_t>(shape.k)) != 0 ||
        tiling.EnableBias(false) != 0 ||
        tiling.SetTraverse(matmul_tiling::MatrixTraverse::FIRSTM) != 0 ||
        tiling.SetFixSplit(static_cast<int32_t>(rowsPerLane),
                           static_cast<int32_t>(baseN),
                           static_cast<int32_t>(TileK)) != 0 ||
        tiling.SetBufferSpace(-1, -1, -1) != 0) {
        throw std::runtime_error("Failed to configure automatic fused Matmul tiling");
    }

    AscendC::tiling::TCubeTiling result{};
    if (tiling.GetTiling(result) != 0) {
        throw std::runtime_error("Failed to generate automatic fused Matmul tiling");
    }
    if (result.baseM != static_cast<int32_t>(rowsPerLane) ||
        result.baseN < 1 || result.baseN > static_cast<int32_t>(TileN) ||
        result.baseN % 8 != 0 || result.usedCoreNum != 1 ||
        result.singleCoreM != static_cast<int32_t>(rowsPerLane) ||
        result.singleCoreN < static_cast<int32_t>(shape.n)) {
        throw std::runtime_error(
            "Automatic Matmul tiling did not preserve full-N row ownership");
    }
    return result;
}

}  // namespace bmms
