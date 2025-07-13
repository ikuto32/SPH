# SPH

For details on optional CUDA neighbour-search support see [docs/GPU_Spatial_Hash_2D.md](docs/GPU_Spatial_Hash_2D.md).

This repository contains a simple Smoothed Particle Hydrodynamics (SPH) simulation.

## Build instructions

### CMake

The project is built with CMake. Both CPU and CUDA builds are supported.

1. Execute the setup script to install dependencies and perform an initial
   CPU build:

   ```console
   ./setup.sh
   ```

2. To build manually, use the following commands. Pass `-DUSE_CUDA=ON` to
   enable GPU acceleration:

   ```console
   mkdir build
   cd build
   cmake -DUSE_CUDA=OFF ..
   cmake --build . --target _sph
   ```

   The resulting `_sph` Python extension is created inside the `build`
   directory.

## Using the Python module

The bindings expose a `PyWorld` class:

```python
from _sph import PyWorld
w = PyWorld(num_particles=500)  # particle count can be changed
w.step(1/60.0)
positions = w.get_positions()
```

## Running the example GUI

After building the `_sph` Python extension a small demo can be launched
using Pygame:

```console
PYTHONPATH=build python examples/run_gui.py
```

## Visualizing particle velocities

After building the Python extension you can generate a scatter plot of the
particle positions with colors representing their speed:

```console
PYTHONPATH=build/bindings python examples/plot_snapshot.py
```

Running the script produces an image named `snapshot.png` in the
`examples` directory.

## Benchmark

A benchmark script reports the time for a single simulation step while
increasing the particle count:

```console
PYTHONPATH=build python bench/run_dambreak.py
```

See [docs/benchmarks.md](docs/benchmarks.md) for more details.
