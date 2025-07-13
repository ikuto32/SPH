# Benchmarking the SPH Simulation

This repository contains a simple benchmark that measures the time for a single simulation step while scaling the particle count. After building the Python extension, run the following command from the repository root:

```console
PYTHONPATH=build python bench/run_dambreak.py
```

The script iterates over increasing particle counts and prints the step time in milliseconds.
