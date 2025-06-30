find_package(Python3 COMPONENTS Interpreter QUIET)
find_package(pybind11 CONFIG QUIET)

if(Python3_Interpreter_FOUND AND pybind11_FOUND)
    find_package(Python3 COMPONENTS NumPy QUIET)
endif()

add_executable(test_calc test_calc.cpp)
target_link_libraries(test_calc PRIVATE sph)
add_test(NAME cpp_test_calc COMMAND test_calc)

add_executable(test_kernel_compare test_kernel_compare.cpp)
target_link_libraries(test_kernel_compare PRIVATE sph)
add_test(NAME cpp_test_kernel_compare COMMAND test_kernel_compare)

add_executable(test_profile test_profile.cpp)
target_link_libraries(test_profile PRIVATE sph)
add_test(NAME cpp_test_profile COMMAND test_profile)

if(USE_CUDA AND SPH_ENABLE_HASH2D)
    add_executable(test_grid2d test_grid2d.cu)
    set_target_properties(test_grid2d PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(test_grid2d PRIVATE sph)
    add_test(NAME cpp_test_grid2d COMMAND test_grid2d)
endif()

if(Python3_Interpreter_FOUND AND pybind11_FOUND AND Python3_NumPy_FOUND)
    add_test(NAME python_tests
              COMMAND ${Python3_EXECUTABLE} -m pytest ${CMAKE_CURRENT_SOURCE_DIR}
    )
    set_tests_properties(python_tests PROPERTIES
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}"
    )
else()
    message(WARNING "Python bindings tests skipped: missing Python interpreter, pybind11 or NumPy")
endif()
