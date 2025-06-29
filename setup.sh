#!/bin/sh
set -e

# Install required Python packages
python3 -m pip install --user pybind11

# Create build directory and build the Python extension
mkdir -p build
cd build
PYBIND11_DIR=$(pybind11-config --cmakedir)
cmake -DUSE_CUDA=OFF -Dpybind11_DIR="$PYBIND11_DIR" ..   # set to ON when the CUDA toolkit is available
cmake --build . --target _sph
