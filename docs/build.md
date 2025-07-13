# Building and Running SPH

This repository uses CMake to build the native library and Python bindings. Both CPU-only and CUDA accelerated builds are supported.

## Requirements
- CMake 3.27+
- C++20 compiler (GCC/Clang)
- TBB
- Python 3 with `pybind11`
- Optional: CUDA toolkit for GPU acceleration

## CPU build
```console
mkdir build
cd build
cmake -DUSE_CUDA=OFF ..
cmake --build . --target _sph
```

## CUDA build
```console
mkdir build
cd build
cmake -DUSE_CUDA=ON ..
cmake --build . --target _sph
```

## Running the tests
```console
ctest --output-on-failure
```

## Benchmarks
The benchmark script steps the simulation for increasing particle counts:
```console
PYTHONPATH=build python bench/run_dambreak.py
```

## Python usage
Run the Pygame example after building the extension:
```console
PYTHONPATH=build python examples/run_gui.py
```

### Changing the particle count
Specify `num_particles` when constructing `PyWorld`:
```python
from _sph import PyWorld
w = PyWorld(num_particles=5000)
```
