#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include "acl/acl.h"

namespace bmms {

inline void CheckAcl(aclError status, const char* operation) {
    if (status != ACL_SUCCESS) {
        throw std::runtime_error(std::string(operation) + ": " + std::to_string(status));
    }
}

inline int64_t QueryCoreCount(aclrtDevAttr attribute, const char* name) {
    int32_t deviceId = 0;
    CheckAcl(aclrtGetDevice(&deviceId), "aclrtGetDevice");
    int64_t count = 0;
    CheckAcl(aclrtGetDeviceInfo(deviceId, attribute, &count), name);
    if (count < 1) {
        throw std::runtime_error(std::string("No ") + name + " are available");
    }
    return count;
}

inline int64_t QueryCubeCoreCount() {
    return QueryCoreCount(ACL_DEV_ATTR_CUBE_CORE_NUM, "Cube core count");
}

inline int64_t QueryVectorCoreCount() {
    return QueryCoreCount(ACL_DEV_ATTR_VECTOR_CORE_NUM, "Vector core count");
}

class DeviceAllocation {
public:
    explicit DeviceAllocation(uint64_t bytes) : bytes_(bytes) {
        if (bytes_ == 0) {
            throw std::invalid_argument("Device allocation size must be positive");
        }
        CheckAcl(aclrtMalloc(&address_, static_cast<size_t>(bytes_), ACL_MEM_MALLOC_HUGE_FIRST),
                 "aclrtMalloc");
    }

    ~DeviceAllocation() {
        if (address_ != nullptr) {
            aclrtFree(address_);
        }
    }

    DeviceAllocation(const DeviceAllocation&) = delete;
    DeviceAllocation& operator=(const DeviceAllocation&) = delete;

    DeviceAllocation(DeviceAllocation&& other) noexcept
        : address_(other.address_), bytes_(other.bytes_) {
        other.address_ = nullptr;
        other.bytes_ = 0;
    }

    DeviceAllocation& operator=(DeviceAllocation&&) = delete;

    GM_ADDR Address() const {
        return static_cast<uint8_t*>(address_);
    }

    uint64_t Bytes() const {
        return bytes_;
    }

private:
    void* address_ = nullptr;
    uint64_t bytes_ = 0;
};

}  // namespace bmms
