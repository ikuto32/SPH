cmake_minimum_required(VERSION 3.18)
project(SPH LANGUAGES CXX)

option(USE_CUDA "Build with CUDA support" ON)

if(USE_CUDA)
    enable_language(CUDA)
    add_compile_definitions(USE_CUDA)
endif()

add_library(sph STATIC src/sph/core/world.cpp)
target_include_directories(sph PUBLIC src)

if(USE_CUDA)
    set_source_files_properties(src/sph/core/world.cpp PROPERTIES LANGUAGE CUDA)
endif()

add_subdirectory(bindings)
