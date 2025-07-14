cmake_minimum_required(VERSION 3.27)
project(SPH LANGUAGES CXX)
enable_testing()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(USE_CUDA "Build with CUDA support" ON)
if(USE_CUDA)
    enable_language(CUDA)
endif()
option(SPH_ENABLE_HASH2D "Enable GPU 2-D spatial hashing" ${USE_CUDA})
option(DEBUG_GPU "Enable additional CUDA checks" OFF)

if(USE_CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    add_compile_definitions(USE_CUDA)
    if(DEBUG_GPU)
        add_compile_definitions(DEBUG_GPU)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G -lineinfo -O0")
    else()
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo -O3 --use_fast_math")
    endif()
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)  # LTOは nvcc が苦手なのでOFF
endif()
if(USE_CUDA)
    # Target Ada Lovelace GPUs such as the RTX 4090 while still supporting
    # other compute capability 8.x devices. The architectures are listed in
    # ascending order so CMake generates code for each supported target.
    set(CMAKE_CUDA_ARCHITECTURES 80 86 89)

    add_library(sph_gpu STATIC
        src/sph/gpu/hash_grid_2d.cu
        src/sph/gpu/neighbor_search_2d.cu)

    target_link_libraries(sph_gpu PRIVATE cudadevrt)
    target_include_directories(sph_gpu PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>)
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
