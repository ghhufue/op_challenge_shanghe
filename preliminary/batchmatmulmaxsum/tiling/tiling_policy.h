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
    data.nShardColumns = static_cast<uint32_t>(shape.n);
    data.systemWorkspaceBytes = 0;
    data.matmulCacheOffsetBytes = 0;
    data.matmulCacheStrideBytes = 0;
    data.partialMaxOffsetBytes = 0;
    data.syncWorkspaceOffsetBytes = 0;
    data.syncWorkspaceBytes = 0;
    data.atomicOutputOffsetBytes = 0;

    if (config->path == KernelPath::REFERENCE) {
        const uint64_t taskCount = static_cast<uint64_t>(shape.b) * shape.m;
        data.splitM = static_cast<uint32_t>(std::min<int64_t>(shape.m, availableCoreNum));
        data.launchBlocks = static_cast<uint32_t>(std::min<uint64_t>(taskCount, availableCoreNum));
        data.workspaceBytes = AlignUpU64(shape.b, kAtomicAlignmentFloats) * sizeof(float);
    } else if (config->path == KernelPath::AUTO_FUSED) {
        const uint64_t mGroups = CeilDivU64(shape.m, config->tileM);
        const uint64_t mTasks = static_cast<uint64_t>(shape.b) * mGroups;
        const bool splitNPath = config->splitN > 1;
        if (splitNPath) {
            const uint64_t nTileCount = CeilDivU64(shape.n, config->tileN);
            const uint64_t desiredNShards = std::max<uint64_t>(
                1, static_cast<uint64_t>(availableCoreNum) / mTasks);
            data.splitN = static_cast<uint32_t>(std::max<uint64_t>(
                1, std::min<uint64_t>(nTileCount, desiredNShards)));
            data.nShardColumns = 0;
            for (uint32_t shard = 0; shard < data.splitN; ++shard) {
                const uint64_t begin =
                    (static_cast<uint64_t>(shape.n) * shard /
                     data.splitN / 16) * 16;
                const uint64_t end = shard + 1 == data.splitN
                    ? static_cast<uint64_t>(shape.n)
                    : (static_cast<uint64_t>(shape.n) * (shard + 1) /
                       data.splitN / 16) * 16;
                data.nShardColumns = std::max<uint32_t>(
                    data.nShardColumns,
                    static_cast<uint32_t>(end - begin));
            }
        } else {
            data.splitN = 1;
        }
        const uint64_t tasks = mTasks * data.splitN;
        data.splitM = static_cast<uint32_t>(std::min<uint64_t>(mGroups, availableCoreNum));
        data.launchBlocks = static_cast<uint32_t>(std::min<uint64_t>(tasks, availableCoreNum));
        data.systemWorkspaceBytes = QueryMatmulSystemWorkspaceBytes();
        uint64_t workspaceCursor = AlignUpU64(data.systemWorkspaceBytes, 512);
        if (config->schedule == MatmulSchedule::ASYNC) {
            const uint64_t paddedN = AlignUpU64(
                data.nShardColumns, config->tileN);
            data.matmulCacheOffsetBytes = workspaceCursor;
            data.matmulCacheStrideBytes = AlignUpU64(
                static_cast<uint64_t>(config->vecM) * paddedN * sizeof(float), 512);
            workspaceCursor += data.matmulCacheStrideBytes * data.launchBlocks * kAivPerAic;
        }
        if (splitNPath) {
            data.partialMaxOffsetBytes = AlignUpU64(workspaceCursor, 512);
            const uint64_t partialMaxBytes =
                tasks * kAivPerAic * config->vecM * sizeof(float);
            workspaceCursor = data.partialMaxOffsetBytes + partialMaxBytes;
            data.syncWorkspaceOffsetBytes = AlignUpU64(workspaceCursor, 512);
            data.syncWorkspaceBytes = static_cast<uint64_t>(
                data.launchBlocks) * kAivPerAic * kSoftSyncSlotBytes;
            workspaceCursor = data.syncWorkspaceOffsetBytes +
                data.syncWorkspaceBytes;
        }
        data.atomicOutputOffsetBytes = AlignUpU64(workspaceCursor, 512);
        const uint64_t atomicOutputBytes =
            AlignUpU64(shape.b, kAtomicAlignmentFloats) * sizeof(float);
        data.workspaceBytes = AlignUpU64(
            data.atomicOutputOffsetBytes + atomicOutputBytes, 512);
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
