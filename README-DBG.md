# GPU Debugging Quick Start

This document outlines the environment setup and runtime tools for debugging the CUDA portions of the SPH project.

## Environment

Set the following environment variable to force synchronous kernel launches while debugging:

```bash
export CUDA_LAUNCH_BLOCKING=1
```

Enable additional error checking by configuring the build with
`-DDEBUG_GPU=ON` when invoking CMake. This passes `-g -G -lineinfo -O0`
to `nvcc` and activates the debug macros.

## Compute Sanitizer

Use NVIDIA Compute Sanitizer to catch memory and synchronization errors:

```bash
compute-sanitizer --tool memcheck   ./app_debug
compute-sanitizer --tool synccheck ./app_debug
```

## cuda-gdb Quick Start

```bash
cuda-gdb ./app_debug
(cuda-gdb) break myKernel
(cuda-gdb) run
(cuda-gdb) cuda kernel launch stop
```

The first launched kernel will print `kernel alive` once when built with
`DEBUG_GPU` enabled, verifying that the device code executed.

## Device capability test

Run the `cuda_device_query` test to report the detected GPU and its compute
capability:

```bash
cmake --build build --target test_device_query
cd build && ctest -R cuda_device_query --output-on-failure
```
