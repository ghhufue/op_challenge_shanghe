#pragma once

#include "lib/matmul_intf.h"
#include "split_n_reduce.asc"

namespace bmms {

using namespace AscendC;

template<class T, bool TX1, bool TX2,
         bool VectorizedReduction, bool Async, uint32_t CBufferCount,
         uint32_t TileM, uint32_t TileN, uint32_t TileK,
         bool SplitN = false>
class AutoMatmulFusedKernel {
public:
    static constexpr uint32_t RowsPerLane = TileM / kAivPerAic;
    static constexpr uint32_t FloatRepeat = 64;
    static constexpr uint32_t FloatBlockElements = 8;
    static constexpr uint32_t MaxSegments =
        (TileN + FloatRepeat - 1) / FloatRepeat;

    using AType = MatmulType<TPosition::GM, CubeFormat::ND, T, TX1>;
    using BType = MatmulType<TPosition::GM, CubeFormat::ND, T, TX2>;
    using CType = MatmulType<TPosition::VECIN, CubeFormat::ND, float>;
    using BiasType = MatmulType<TPosition::GM, CubeFormat::ND, float>;

    matmul::Matmul<AType, BType, CType, BiasType> matmulObj;
