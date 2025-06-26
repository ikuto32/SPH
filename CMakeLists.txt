cmake_minimum_required(VERSION 3.18)
project(SPH LANGUAGES CXX)

enable_testing()

option(USE_CUDA "Build with CUDA support" ON)

if(USE_CUDA)
    enable_language(CUDA)
    add_compile_definitions(USE_CUDA)
endif()

add_library(sph STATIC src/sph/core/world.cpp)
set_target_properties(sph PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(sph PUBLIC src)
find_package(TBB REQUIRED)
target_link_libraries(sph PUBLIC TBB::tbb)

if(USE_CUDA)
    set_source_files_properties(src/sph/core/world.cpp PROPERTIES LANGUAGE CUDA)
endif()

add_subdirectory(bindings)
add_subdirectory(tests)
