# SPH

This repository contains a Windows example project implementing a simple
Smoothed Particle Hydrodynamics (SPH) simulation.

## Building the library

The project is distributed as a Visual Studio solution.  Open
`WindowsProject_optimization_SPH.sln` with Visual Studio 2022 (or newer)
and build either the `Debug` or `Release` configuration for `x64`.
The resulting binaries are written to `x64/Debug` or `x64/Release`.

## Building the Python module

Python bindings can be built with `pybind11`.  After installing
`pybind11` and a working C++ toolchain run the following command in the
repository root:

```console
python setup.py build_ext --inplace
```

This will create a `sph` module that can be imported from Python.

## Running the example GUI

After building the solution an executable named
`WindowsProject_optimization_SPH.exe` will be produced inside the build
output directory.  Launching this executable starts the example GUI
showing the SPH simulation.

