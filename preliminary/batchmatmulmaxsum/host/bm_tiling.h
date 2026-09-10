#pragma once

#include <cstdint>
#include <stdexcept>
#include "lib/matmul/matmul_tiling.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_types.h"

namespace bmms {

template<uint32_t TileM, uint32_t TileN, uint32_t TileK>
inline AscendC::tiling::TCubeTiling MakeBmCubeTiling(
        const Shape& shape, int32_t inputDtype, bool transposeX1, bool transposeX2) {
    static_assert(TileM > 0 && TileN > 0 && TileK > 0, "BM tile sizes must be positive");
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
        tiling.SetShape(static_cast<int32_t>(TileM), static_cast<int32_t>(TileN),
                        static_cast<int32_t>(shape.k)) != 0 ||
        tiling.SetOrgShape(static_cast<int32_t>(shape.m), static_cast<int32_t>(shape.n),
                           static_cast<int32_t>(shape.k)) != 0 ||
        tiling.SetFixSplit(static_cast<int32_t>(TileM), static_cast<int32_t>(TileN),
                           static_cast<int32_t>(TileK)) != 0 ||
        tiling.EnableBias(false) != 0 ||
        tiling.SetBufferSpace(-1, -1, -1) != 0) {
        throw std::runtime_error("Failed to configure BM Matmul tiling");
    }

    AscendC::tiling::TCubeTiling result;
    if (tiling.GetTiling(result) != 0) {
        throw std::runtime_error("Failed to generate BM Matmul tiling");
    }
    return result;
}

inline GM_ADDR UploadPersistentCubeTiling(
        const AscendC::tiling::TCubeTiling& tiling) {
    constexpr size_t bytes = sizeof(AscendC::tiling::TCubeTiling);
    void* deviceAddress = nullptr;
    CheckAcl(aclrtMalloc(&deviceAddress, bytes, ACL_MEM_MALLOC_HUGE_FIRST),
             "aclrtMalloc(persistent Matmul tiling)");
    const aclError copyStatus = aclrtMemcpy(
        deviceAddress, bytes, &tiling, bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (copyStatus != ACL_SUCCESS) {
        aclrtFree(deviceAddress);
        CheckAcl(copyStatus, "aclrtMemcpy(Matmul tiling H2D)");
    }
    return static_cast<uint8_t*>(deviceAddress);
}

}  // namespace bmms
