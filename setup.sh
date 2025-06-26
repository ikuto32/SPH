#!/bin/sh
set -e

# Install required Python packages
python3 -m pip install --user pybind11

# Create build directory and build the Python extension
mkdir -p build
cd build
cmake -DUSE_CUDA=OFF ..   # set to ON when the CUDA toolkit is available
cmake --build . --target _sph
