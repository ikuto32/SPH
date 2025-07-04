find_package(Python3 COMPONENTS Interpreter REQUIRED)

add_executable(test_calc test_calc.cpp)
target_link_libraries(test_calc PRIVATE sph)
add_test(NAME cpp_test_calc COMMAND test_calc)

add_executable(test_kernel_compare test_kernel_compare.cpp)
target_link_libraries(test_kernel_compare PRIVATE sph)
add_test(NAME cpp_test_kernel_compare COMMAND test_kernel_compare)

if(SPH_ENABLE_HASH2D)
    add_executable(test_grid2d test_grid2d.cu)
    target_link_libraries(test_grid2d PRIVATE sph_gpu)
    add_test(NAME cuda_test_grid2d COMMAND test_grid2d)
endif()

add_test(NAME python_tests
          COMMAND ${Python3_EXECUTABLE} -m pytest ${CMAKE_CURRENT_SOURCE_DIR}
)
set_tests_properties(python_tests PROPERTIES
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}"
)
