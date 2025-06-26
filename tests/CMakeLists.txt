find_package(Python3 COMPONENTS Interpreter REQUIRED)

add_executable(test_calc test_calc.cpp)
target_link_libraries(test_calc PRIVATE sph)
add_test(NAME cpp_test_calc COMMAND test_calc)

add_executable(test_kernel_compare test_kernel_compare.cpp)
target_link_libraries(test_kernel_compare PRIVATE sph)
add_test(NAME cpp_test_kernel_compare COMMAND test_kernel_compare)

add_test(NAME python_tests
          COMMAND ${Python3_EXECUTABLE} -m pytest ${CMAKE_CURRENT_SOURCE_DIR}
)
set_tests_properties(python_tests PROPERTIES
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}"
)
