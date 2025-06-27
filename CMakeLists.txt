cmake_minimum_required(VERSION 3.18)
project(SPH LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_testing()

option(USE_CUDA "Build with CUDA support" ON)

if(USE_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    add_compile_definitions(USE_CUDA)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)
endif()

set(SPH_SOURCES src/sph/core/world.cpp)
if(USE_CUDA)
    list(APPEND SPH_SOURCES src/sph/core/kernels_cuda.cu)
endif()
add_library(sph STATIC ${SPH_SOURCES})
set_target_properties(sph PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(sph PUBLIC src)
find_package(TBB REQUIRED)
target_link_libraries(sph PUBLIC TBB::tbb)



add_subdirectory(bindings)
add_subdirectory(tests)
