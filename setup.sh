#!/bin/sh
set -e

# Install required Python packages
python3 -m pip install --user pybind11

# Create build directory and build the Python extension
export CMAKE_PREFIX_PATH="$(python3 -m pybind11 --cmakedir):${CMAKE_PREFIX_PATH}"
mkdir -p build
cd build
cmake -DUSE_CUDA=OFF ..   # set to ON when the CUDA toolkit is available
cmake --build . --target _sph
