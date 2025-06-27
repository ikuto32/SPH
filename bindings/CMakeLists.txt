find_package(pybind11 REQUIRED)

pybind11_add_module(_sph pybind_module.cpp)

target_link_libraries(_sph PRIVATE sph)

set_target_properties(_sph PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

if(USE_CUDA)
    set_target_properties(_sph PROPERTIES
        INTERPROCEDURAL_OPTIMIZATION OFF
    )
endif()

if(USE_CUDA)
    set_target_properties(_sph PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

