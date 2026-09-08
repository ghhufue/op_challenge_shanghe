cmake_minimum_required(VERSION 3.16)

set(CMAKE_ASC_ARCHITECTURES "dav-2201" CACHE STRING "Target Ascend architecture")
set(CMAKE_ASC_RUN_MODE "npu" CACHE STRING "Run on Ascend NPU" FORCE)
find_package(ASC REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter)

project(batch_matmul_max_sum_custom LANGUAGES ASC CXX)

set(CMAKE_CXX_STANDARD 14)

if(DEFINED NPU_ARCH)
    set(SOC_ARCH ${NPU_ARCH})
else()
    set(SOC_ARCH "${CMAKE_ASC_ARCHITECTURES}")
endif()

add_executable(batch_matmul_max_sum_custom main.asc)

set(TILING_CATALOG_JSON ${CMAKE_CURRENT_SOURCE_DIR}/configs/tiling_candidates.json)
set(TILING_CATALOG_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/tiling/tiling_catalog_generated.h)
add_custom_command(
    OUTPUT ${TILING_CATALOG_HEADER}
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/generate_tiling_catalog.py
    DEPENDS
        ${TILING_CATALOG_JSON}
        ${CMAKE_CURRENT_SOURCE_DIR}/scripts/generate_tiling_catalog.py
    COMMENT "Generating tiling candidate catalog"
    VERBATIM
)
add_custom_target(generate_tiling_catalog DEPENDS ${TILING_CATALOG_HEADER})
add_dependencies(batch_matmul_max_sum_custom generate_tiling_catalog)

# The competition includes kernel.asc directly and does not define this flag.
# The local executable exposes a forced-key entry for offline tuning only.
target_compile_definitions(batch_matmul_max_sum_custom PRIVATE BMMS_ENABLE_TUNING=1)

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
