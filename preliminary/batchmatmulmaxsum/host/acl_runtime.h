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

}  // namespace bmms
