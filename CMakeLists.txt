cmake_minimum_required(VERSION 3.18)
project(SPH LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(USE_CUDA "Build with CUDA support" ON)

if(USE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 20)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        add_compile_definitions(USE_CUDA)
        # LTO is problematic with nvcc
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)
    else()
        message(WARNING "CUDA compiler not found - building without CUDA")
        set(USE_CUDA OFF)
    endif()
endif()

find_package(pybind11 CONFIG REQUIRED)
find_package(TBB       CONFIG REQUIRED)

# --- ソースとターゲット -------------------------------------------------
set(SPH_SOURCES
    src/sph/core/world.cpp
)

if(USE_CUDA)
    list(APPEND SPH_SOURCES src/sph/core/kernels_cuda.cu)
endif()

add_library(sph STATIC ${SPH_SOURCES})           # ← ここ一回でOK
set_target_properties(sph PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(sph PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
)

target_link_libraries(sph PUBLIC
    TBB::tbb
)

add_subdirectory(bindings)
add_subdirectory(tests)
