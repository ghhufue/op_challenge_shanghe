#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include "tiling_types.h"

namespace bmms {

inline const StaticTilingConfig* FindTilingConfig(TilingKey key) {
    for (size_t index = 0; index < kTilingConfigCount; ++index) {
        if (kTilingConfigs[index].key == key) {
            return &kTilingConfigs[index];
        }
    }
    return nullptr;
}

inline TilingKey ParseTilingKey(int64_t value) {
    if (value < 0 || value > UINT32_MAX) {
        throw std::invalid_argument("Tiling key is outside uint32 range");
    }
    const TilingKey key = static_cast<TilingKey>(static_cast<uint32_t>(value));
    if (FindTilingConfig(key) == nullptr) {
        throw std::invalid_argument("Unknown tiling key: " + std::to_string(value));
    }
    return key;
}

}  // namespace bmms
