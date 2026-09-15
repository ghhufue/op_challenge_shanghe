#pragma once

namespace bmms {

// The competition ABI has no workspace parameter. Cache one allocation and
// one host-side Matmul tiling value per stream and exact execution plan so
// run_kernel stays asynchronous after the first use.
struct ExecutionResourceKey {
    aclrtStream stream;
    TilingKey tilingKey;
    Shape shape;
    int32_t inputDtype;
    bool transposeX1;
    bool transposeX2;
    uint32_t splitN;
    uint32_t nShardColumns;
    uint32_t launchBlocks;
    uint64_t workspaceBytes;
};

struct CachedExecutionResources {
    ExecutionResourceKey key;
    GM_ADDR workspace;
    uint64_t workspaceBytes;
    AscendC::tiling::TCubeTiling cubeTiling;
};

inline bool SameExecutionResourceKey(
        const ExecutionResourceKey& lhs, const ExecutionResourceKey& rhs) {
    return lhs.stream == rhs.stream && lhs.tilingKey == rhs.tilingKey &&
           lhs.shape.b == rhs.shape.b && lhs.shape.m == rhs.shape.m &&
           lhs.shape.n == rhs.shape.n && lhs.shape.k == rhs.shape.k &&
           lhs.inputDtype == rhs.inputDtype &&
           lhs.transposeX1 == rhs.transposeX1 &&
           lhs.transposeX2 == rhs.transposeX2 &&
           lhs.splitN == rhs.splitN &&
           lhs.nShardColumns == rhs.nShardColumns &&
           lhs.launchBlocks == rhs.launchBlocks &&
           lhs.workspaceBytes == rhs.workspaceBytes;
}
inline std::vector<CachedExecutionResources>& ExecutionResourceCache() {
    static std::vector<CachedExecutionResources> cache;
    return cache;
}

inline std::atomic_flag& ExecutionResourceCacheLock() {
    static std::atomic_flag lock = ATOMIC_FLAG_INIT;
    return lock;
}

class ExecutionResourceCacheGuard {
public:
    ExecutionResourceCacheGuard() {
        while (ExecutionResourceCacheLock().test_and_set(
                   std::memory_order_acquire)) {
        }
    }

    ~ExecutionResourceCacheGuard() {
        ExecutionResourceCacheLock().clear(std::memory_order_release);
    }

    ExecutionResourceCacheGuard(const ExecutionResourceCacheGuard&) = delete;
    ExecutionResourceCacheGuard& operator=(
        const ExecutionResourceCacheGuard&) = delete;
};

inline GM_ADDR AllocatePersistentDeviceBuffer(
        uint64_t bytes, const char* operation) {
    if (bytes == 0) {
        throw std::invalid_argument(
            "Persistent device buffer must not be empty");
    }
    void* address = nullptr;
    CheckAcl(aclrtMalloc(&address, static_cast<size_t>(bytes),
                         ACL_MEM_MALLOC_HUGE_FIRST), operation);
    return static_cast<uint8_t*>(address);
}

inline ExecutionResourceKey MakeExecutionResourceKey(
        const Plan& plan, aclrtStream stream, int32_t inputDtype,
        bool transposeX1, bool transposeX2) {
    return ExecutionResourceKey{
        stream,
        plan.key,
        Shape{plan.tiling.b, plan.tiling.m,
              plan.tiling.n, plan.tiling.k},
        inputDtype,
        transposeX1,
        transposeX2,
        plan.tiling.splitN,
        plan.tiling.nShardColumns,
        plan.tiling.launchBlocks,
        plan.tiling.workspaceBytes};
}

template<class Factory>
inline CachedExecutionResources GetOrCreateExecutionResources(
        const ExecutionResourceKey& key, Factory factory) {
    ExecutionResourceCacheGuard guard;
    for (const auto& resources : ExecutionResourceCache()) {
        if (SameExecutionResourceKey(resources.key, key)) {
            return resources;
        }
    }
    CachedExecutionResources resources = factory();
    ExecutionResourceCache().push_back(resources);
    return resources;
}

inline CachedExecutionResources GetReferenceExecutionResources(
        const Plan& plan, aclrtStream stream, int32_t inputDtype,
        bool transposeX1, bool transposeX2) {
    const ExecutionResourceKey key = MakeExecutionResourceKey(
        plan, stream, inputDtype, transposeX1, transposeX2);
    return GetOrCreateExecutionResources(key, [&]() {
        return CachedExecutionResources{
            key,
            AllocatePersistentDeviceBuffer(
                plan.tiling.workspaceBytes,
                "aclrtMalloc(reference workspace cache)"),
            plan.tiling.workspaceBytes,
            AscendC::tiling::TCubeTiling{}};
    });
}

template<uint32_t TileM, uint32_t TileN, uint32_t TileK>
inline CachedExecutionResources GetAutoFusedExecutionResources(
        const Plan& plan, aclrtStream stream, int32_t inputDtype,
        bool transposeX1, bool transposeX2) {
    const ExecutionResourceKey key = MakeExecutionResourceKey(
        plan, stream, inputDtype, transposeX1, transposeX2);
    return GetOrCreateExecutionResources(key, [&]() {
        const Shape shape{plan.tiling.b, plan.tiling.m,
                          plan.tiling.n, plan.tiling.k};
        return CachedExecutionResources{
            key,
            AllocatePersistentDeviceBuffer(
                plan.tiling.workspaceBytes,
                "aclrtMalloc(auto fused workspace cache)"),
            plan.tiling.workspaceBytes,
            MakeAutoFusedCubeTiling<TileM, TileN, TileK>(
                shape, inputDtype, transposeX1, transposeX2,
                plan.tiling.nShardColumns)};
    });
}
