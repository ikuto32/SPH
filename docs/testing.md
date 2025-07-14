# Test Suite

This document describes the repository's automated tests. Both C++ and Python tests are executed via CTest.

## Running the tests
After building the project, run:

```console
ctest --output-on-failure
```

## C++ tests
- `test_calc.cpp` checks the smoothing kernel implementation.
- `test_kernel_compare.cpp` compares CPU and CUDA kernel results.
- `test_grid2d.cu` exercises the GPU spatial hashing path when enabled.
- `test_device_query.cpp` prints information about the detected CUDA device.

## Python tests
`tests/test_bindings.py` verifies the `PyWorld` bindings and simulation behaviour:
- world creation and property getters
- stepping updates particle positions
- interaction force methods
- neighbour and spatial hash queries
- custom simulation parameters
- particle count handling
- initial velocities are zero
- gravity influences velocity
- a particle's own index is returned when querying its position
- neighbour results are contained in the spatial hash candidates
- larger smoothing radii yield at least as many neighbours

## Extending the suite
Add new tests alongside the existing ones and re-run `ctest`.
