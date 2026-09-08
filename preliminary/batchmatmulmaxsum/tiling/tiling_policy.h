#pragma once

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include "tiling_catalog.h"

namespace bmms {

inline uint64_t CeilDivU64(uint64_t value, uint64_t divisor) {
    return (value + divisor - 1) / divisor;
}

inline TilingKey SelectTilingKey(const Shape& shape, int64_t availableCoreNum,
                                 int32_t inputDtype, bool transposeX1, bool transposeX2) {
    if (availableCoreNum < 1) {
        throw std::invalid_argument("No compute cores are available");
    }
    (void)shape;
    (void)inputDtype;
    (void)transposeX1;
    (void)transposeX2;
#include "generated_policy.inc"
}

inline Plan MakePlan(const Shape& shape, int64_t availableCoreNum, TilingKey key) {
    if (availableCoreNum < 1) {
        throw std::invalid_argument("No compute cores are available");
    }
    const StaticTilingConfig* config = FindTilingConfig(key);
    if (config == nullptr) {
        throw std::invalid_argument("Unknown tiling key");
    }
    if (!config->implemented) {
        throw std::invalid_argument(std::string("Tiling key is not implemented: ") + config->name);
    }

    TilingData data{};
    data.b = shape.b;
    data.m = shape.m;
    data.n = shape.n;
    data.k = shape.k;
    data.tileM = config->tileM;
    data.tileN = config->tileN;
    data.tileK = config->tileK;
    data.vecM = config->vecM;
    data.vecN = config->vecN;
    data.splitN = config->splitN;

    if (config->path == KernelPath::REFERENCE) {
        const uint64_t taskCount = static_cast<uint64_t>(shape.b) * shape.m;
        data.splitM = static_cast<uint32_t>(std::min<int64_t>(shape.m, availableCoreNum));
        data.launchBlocks = static_cast<uint32_t>(std::min<uint64_t>(taskCount, availableCoreNum));
        data.workspaceBytes = taskCount * sizeof(float);
    } else {
        const uint64_t mGroups = CeilDivU64(shape.m, config->tileM);
        const uint64_t nGroups = CeilDivU64(shape.n, config->tileN);
        data.splitN = static_cast<uint32_t>(std::min<uint64_t>(config->splitN, nGroups));
        const uint64_t denominator = static_cast<uint64_t>(shape.b) * data.splitN;
        const uint64_t perBatchN = std::max<uint64_t>(1, availableCoreNum / denominator);
        data.splitM = static_cast<uint32_t>(std::min<uint64_t>(mGroups, perBatchN));
        const uint64_t tasks = static_cast<uint64_t>(shape.b) * data.splitM * data.splitN;
        data.launchBlocks = static_cast<uint32_t>(std::min<uint64_t>(tasks, availableCoreNum));
        const uint64_t singleCoreM = CeilDivU64(shape.m, data.splitM);
        if (data.splitN == 1) {
            data.workspaceBytes = static_cast<uint64_t>(shape.b) * data.splitM * sizeof(float);
        } else {
            data.workspaceBytes = static_cast<uint64_t>(shape.b) * data.splitM * data.splitN *
                                  singleCoreM * sizeof(float);
            data.workspaceBytes += static_cast<uint64_t>(shape.b) * data.splitM * sizeof(float);
        }
    }
    return Plan{key, config->path, config->name, data};
}

inline Plan SelectPlan(const Shape& shape, int64_t availableCoreNum,
                       int32_t inputDtype, bool transposeX1, bool transposeX2) {
    return MakePlan(shape, availableCoreNum,
                    SelectTilingKey(shape, availableCoreNum, inputDtype,
                                    transposeX1, transposeX2));
}

}  // namespace bmms
