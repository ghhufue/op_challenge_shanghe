#pragma once

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include "tiling_catalog.h"
#include "submission_policy.h"
#include "host/fused_tiling.h"

namespace bmms {

inline uint64_t CeilDivU64(uint64_t value, uint64_t divisor) {
    return (value + divisor - 1) / divisor;
}

inline uint64_t AlignUpU64(uint64_t value, uint64_t alignment) {
    return CeilDivU64(value, alignment) * alignment;
}

inline TilingKey SelectTilingKey(const Shape& shape, int64_t availableCoreNum,
                                 int32_t inputDtype, bool transposeX1, bool transposeX2) {
    if (availableCoreNum < 1) {
        throw std::invalid_argument("No compute cores are available");
    }
    const TilingKey key = SelectSubmissionTilingKey(
        shape, availableCoreNum, inputDtype, transposeX1, transposeX2);
    const StaticTilingConfig* config = FindTilingConfig(key);
    if (config == nullptr) {
        throw std::invalid_argument("Submission policy selected an unknown tiling key");
    }
    if (!config->implemented) {
        throw std::invalid_argument(
            std::string("Submission policy selected an unimplemented tiling key: ") +
            config->name);
    }
    return key;
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
    data.systemWorkspaceBytes = 0;
    data.partialScoreOffsetBytes = 0;

    if (config->path == KernelPath::REFERENCE) {
        const uint64_t taskCount = static_cast<uint64_t>(shape.b) * shape.m;
        data.splitM = static_cast<uint32_t>(std::min<int64_t>(shape.m, availableCoreNum));
        data.launchBlocks = static_cast<uint32_t>(std::min<uint64_t>(taskCount, availableCoreNum));
        data.workspaceBytes = AlignUpU64(shape.b, kAtomicAlignmentFloats) * sizeof(float);
    } else if (config->path == KernelPath::AUTO_FUSED) {
        const uint64_t mGroups = CeilDivU64(shape.m, config->tileM);
        const uint64_t tasks = static_cast<uint64_t>(shape.b) * mGroups;
        data.splitM = static_cast<uint32_t>(std::min<uint64_t>(mGroups, availableCoreNum));
        data.launchBlocks = static_cast<uint32_t>(std::min<uint64_t>(tasks, availableCoreNum));
        data.systemWorkspaceBytes = QueryMatmulSystemWorkspaceBytes();
        data.partialScoreOffsetBytes = AlignUpU64(data.systemWorkspaceBytes, 512);
        const uint64_t partialBytes = tasks * kAivPerAic * sizeof(float);
        data.workspaceBytes = AlignUpU64(
            data.partialScoreOffsetBytes + partialBytes, 512);
    } else {
        throw std::invalid_argument("Unknown kernel path");
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
