# SPH

This repository contains a simple Smoothed Particle Hydrodynamics (SPH) simulation.

## Building the library (Visual Studio)

The original project is distributed as a Visual Studio solution.  Open
`WindowsProject_optimization_SPH.sln` with Visual Studio 2022 (or newer)
and build either the `Debug` or `Release` configuration for `x64`.
The resulting binaries are written to `x64/Debug` or `x64/Release`.

## Building with CMake

The library and Python bindings can also be built using CMake.  Before
invoking CMake run the setup script in the repository root:

```console
./setup.sh
```

After the script completes you can use the usual workflow:

```console
mkdir build
cd build
cmake -DUSE_CUDA=OFF ..   # set to ON when the CUDA toolkit is available
cmake --build . --target _sph
```

This will produce a Python extension named `_sph` inside the build
directory.

## Using the Python module

The bindings expose a `PyWorld` class:

```python
from _sph import PyWorld
w = PyWorld()
w.step(1/60.0)
positions = w.get_positions()
```

## Running the example GUI

After building the Visual Studio solution an executable named
`WindowsProject_optimization_SPH.exe` will be produced inside the build
output directory.  Launching this executable starts the example GUI
showing the SPH simulation.

## Running the Python demo

After building the `_sph` Python extension a small demo can be launched
using Pygame:

```console
python examples/run_gui.py
```

## Visualizing particle velocities

After building the Python extension you can generate a scatter plot of the
particle positions with colors representing their speed:

```console
PYTHONPATH=build/bindings python examples/plot_snapshot.py
```

Running the script produces an image named `snapshot.png` in the
`examples` directory.
