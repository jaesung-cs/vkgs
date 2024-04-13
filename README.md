# pygs
Gaussian Splatting

## Desktop Viewer

![](/media/screenshot.png)

Like other web based viewer, it uses traditional graphics pipeline, drawing splats projected in 2D screen space.

One of benefits of using graphics pipeline rather than compute pipeline is that splats can be drawn together with other objects.
The screenshot shows splats rendered with render pass, with depth buffer and 4x MSAA.


### Requirements
- `VulkanSDK>=1.3`
  - Download from https://vulkan.lunarg.com/ and follow install instruction.
- `cmake>=3.24`
- submodules
```bash
$ git submodule update --init --recursive
```

### Dependencies
- VulkanMemoryAllocator
- glm
- glfw
- imgui
- [VkRadixSort](https://github.com/MircoWerner/VkRadixSort): modified to support indirect dispatch, copied shader code to `src/pygs/engine/vulkan/shader/multi_radixsort.h`.
- [argparse](https://github.com/p-ranav/argparse): copied header file to project.


### Build
```bash
$ cmake . -B build
$ cmake --build build --config Release -j
```

### Run
```bash
$ ./build/pygs_base
$ ./build/pygs_base -i <ply_filepath>
```
Drag and drop .ply file (pre-)trained from official gaussian splatting.


## Python and CUDA (WIP)

1. To pass data torch cuda to renderer
1. To benchmark cub sort performance


### Requirements

- conda
```bash
$ conda create -n pygs python=3.10
$ conda activate pygs
$ conda install conda-forge::cmake
$ conda install nvidia/label/cuda-12.2.2::cuda-toolkit  # or any other version
```


### Notes
- Directly updating to vulkan-cuda mapped mempry in kernel is slower than memcpy (3.2ms vs. 1ms for 1600x900 rgba32 image)
