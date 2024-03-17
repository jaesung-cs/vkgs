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
$ conda install nvidia::cuda-toolkit
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
