#pragma once

#include <cstdint>
#include <stdexcept>
#include "lib/matmul/matmul_tiling.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_types.h"

namespace bmms {

template<uint32_t TileM, uint32_t TileN, uint32_t TileK>
inline optiling::TCubeTiling MakeBmCubeTiling(
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

    optiling::TCubeTiling result;
    if (tiling.GetTiling(result) != 0) {
        throw std::runtime_error("Failed to generate BM Matmul tiling");
    }
    return result;
}

struct PersistentCubeTiling {
    GM_ADDR deviceAddress;
    void* hostAddress;
    uint64_t bytes;
};

// Cached execution resources own both allocations for the process lifetime.
// Pinned Host memory keeps the source valid until the stream reaches the
// asynchronous H2D copy; the Device buffer then remains valid for every
// cached Matmul launch.
inline PersistentCubeTiling UploadPersistentCubeTiling(
        optiling::TCubeTiling& tiling, aclrtStream stream) {
    const uint64_t bytes = static_cast<uint64_t>(tiling.GetDataSize());
    if (bytes == 0) {
        throw std::runtime_error("Matmul tiling data must not be empty");
    }

    void* hostAddress = nullptr;
    CheckAcl(aclrtMallocHost(&hostAddress, static_cast<size_t>(bytes)),
             "aclrtMallocHost(persistent Matmul tiling)");
    tiling.SaveToBuffer(static_cast<uint8_t*>(hostAddress), static_cast<size_t>(bytes));

    void* deviceAddress = nullptr;
    const aclError mallocStatus = aclrtMalloc(
        &deviceAddress, static_cast<size_t>(bytes), ACL_MEM_MALLOC_HUGE_FIRST);
    if (mallocStatus != ACL_SUCCESS) {
        aclrtFreeHost(hostAddress);
        CheckAcl(mallocStatus, "aclrtMalloc(persistent Matmul tiling)");
    }

    const aclError copyStatus = aclrtMemcpyAsync(
        deviceAddress, static_cast<size_t>(bytes), hostAddress,
        static_cast<size_t>(bytes), ACL_MEMCPY_HOST_TO_DEVICE, stream);
    if (copyStatus != ACL_SUCCESS) {
        aclrtFree(deviceAddress);
        aclrtFreeHost(hostAddress);
        CheckAcl(copyStatus, "aclrtMemcpyAsync(Matmul tiling H2D)");
    }
    return PersistentCubeTiling{
        static_cast<uint8_t*>(deviceAddress), hostAddress, bytes};
}

}  // namespace bmms
