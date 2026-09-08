#pragma once

#include <cstdint>
#include "tiling_catalog_generated.h"

namespace bmms {

struct Shape {
    int64_t b;
    int64_t m;
    int64_t n;
    int64_t k;
};

struct TilingData {
    int64_t b;
    int64_t m;
    int64_t n;
    int64_t k;
    uint32_t tileM;
    uint32_t tileN;
    uint32_t tileK;
    uint32_t vecM;
    uint32_t vecN;
    uint32_t splitM;
    uint32_t splitN;
    uint32_t launchBlocks;
    uint64_t workspaceBytes;
};

struct Plan {
    TilingKey key;
    KernelPath path;
    const char* name;
    TilingData tiling;
};

}  // namespace bmms
