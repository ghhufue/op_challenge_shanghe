#pragma once

#include <cstdint>
#include <stdexcept>
#include <vector>
#include "lib/matmul/matmul_tiling.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_types.h"

namespace bmms {

inline optiling::TCubeTiling MakeBm16x128x64CubeTiling(
        const Shape& shape, int32_t inputDtype, bool transposeX1, bool transposeX2) {
    const char* socName = aclrtGetSocName();
    if (socName == nullptr) {
        throw std::runtime_error("aclrtGetSocName returned null");
    }
    auto* platform = platform_ascendc::PlatformAscendCManager::GetInstance(socName);
    if (platform == nullptr) {
        throw std::runtime_error("Failed to initialize AscendC platform information");
    }

    const matmul_tiling::DataType dataType = inputDtype == 1
        ? matmul_tiling::DataType::DT_FLOAT16
        : matmul_tiling::DataType::DT_BF16;
    matmul_tiling::MatmulApiTiling tiling(*platform);
    if (tiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                        dataType, transposeX1) != 0 ||
        tiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                        dataType, transposeX2) != 0 ||
        tiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                        matmul_tiling::DataType::DT_FLOAT) != 0 ||
        tiling.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                           matmul_tiling::DataType::DT_FLOAT) != 0 ||
        tiling.SetShape(16, 128, static_cast<int32_t>(shape.k)) != 0 ||
        tiling.SetOrgShape(static_cast<int32_t>(shape.m), static_cast<int32_t>(shape.n),
                           static_cast<int32_t>(shape.k)) != 0 ||
        tiling.SetFixSplit(16, 128, 64) != 0 ||
        tiling.EnableBias(false) != 0 ||
        tiling.SetBufferSpace(-1, -1, -1) != 0) {
        throw std::runtime_error("Failed to configure BM_16X128X64 Matmul tiling");
    }

    optiling::TCubeTiling result;
    if (tiling.GetTiling(result) != 0) {
        throw std::runtime_error("Failed to generate BM_16X128X64 Matmul tiling");
    }
    return result;
}

inline DeviceAllocation UploadCubeTiling(optiling::TCubeTiling& tiling) {
    const uint64_t bytes = static_cast<uint64_t>(tiling.GetDataSize());
    std::vector<uint8_t> hostData(static_cast<size_t>(bytes));
    tiling.SaveToBuffer(hostData.data(), hostData.size());
    DeviceAllocation deviceData(bytes);
    CheckAcl(aclrtMemcpy(deviceData.Address(), static_cast<size_t>(bytes),
                         hostData.data(), hostData.size(), ACL_MEMCPY_HOST_TO_DEVICE),
             "aclrtMemcpy(Matmul tiling H2D)");
    return deviceData;
}

}  // namespace bmms
