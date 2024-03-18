# pygs
Gaussian Splatting

## Requirements
- submodules
```bash
$ git submodule update --init --recursive
```

- conda
```bash
$ conda create -n pygs python=3.10
$ conda activate pygs
$ conda install conda-forge::cmake
$ conda install nvidia/label/cuda-12.2.2::cuda-toolkit
```

## Build
```bash
$ cmake . -B build
$ cmake --build build --config Release -j
```

## Run
```bash
$ ./build/pygs_base
```

## Notes
- Directly updating to vulkan-cuda mapped mempry in kernel is slower than running cuda kernel and then memcpy (3.2ms vs. 1ms for 1600x900 rgba32 image)
