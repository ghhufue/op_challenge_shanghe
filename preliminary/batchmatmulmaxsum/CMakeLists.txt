cmake_minimum_required(VERSION 3.16)

set(CMAKE_ASC_ARCHITECTURES "dav-2201" CACHE STRING "Target Ascend architecture")
set(CMAKE_ASC_RUN_MODE "npu" CACHE STRING "Run on Ascend NPU" FORCE)
find_package(ASC REQUIRED)

project(batch_matmul_max_sum_custom LANGUAGES ASC CXX)

set(CMAKE_CXX_STANDARD 14)

if(DEFINED NPU_ARCH)
    set(SOC_ARCH ${NPU_ARCH})
else()
    set(SOC_ARCH "${CMAKE_ASC_ARCHITECTURES}")
endif()

add_executable(batch_matmul_max_sum_custom main.asc)

target_link_libraries(batch_matmul_max_sum_custom PRIVATE
    tiling_api
    register
    platform
    unified_dlog
    dl
    m
    graph_base
)

target_include_directories(batch_matmul_max_sum_custom PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_options(batch_matmul_max_sum_custom PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${SOC_ARCH}>
)
