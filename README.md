# pygs
Gaussian Splatting

## Desktop Viewer

- with VkRadixSort:

  ![](/media/screenshot.jpg)

  ~100FPS with ~2.4M visible splats. Radix sort is bottleneck.

- With my radix sort implementation:

  ![](/media/screenshot-fast.jpg)

  ~200FPS with 3M visible splats. Much faster sort speed. Rendering is bottleneck.

Like other web based viewer, it uses traditional graphics pipeline, drawing splats projected in 2D screen space.

One of benefits of using graphics pipeline rather than compute pipeline is that splats can be drawn together with other objects.
The screenshot shows splats rendered with render pass, with depth buffer and 4x MSAA.


### Requirements
- `VulkanSDK>=1.3`
  - Download from https://vulkan.lunarg.com/ and follow install instruction.
  - Requires several features available in `1.3`.
- `cmake>=3.24`
  - `Vulkan::shaderc_combined` new in version `3.24`.


### Dependencies
- submodules
  ```bash
  $ git submodule update --init --recursive
  ```
  - VulkanMemoryAllocator
  - glm
  - glfw
  - imgui
  - argparse
  - [vulkan_radix_sort](https://github.com/jaesung-cs/vulkan_radix_sort): my implementation of state-of-the-art radix sort algorithm, Onesweep.


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
Drag and drop pretrained .ply file from official gaussian splatting.


## Python and CUDA (WIP)

1. To pass data torch cuda to renderer
1. To benchmark cub sort performance


### Requirements (WIP)

- conda
```bash
$ conda create -n pygs python=3.10
$ conda activate pygs
$ conda install conda-forge::cmake
$ conda install nvidia/label/cuda-12.2.2::cuda-toolkit  # or any other version
```


### Notes
- Directly updating to vulkan-cuda mapped mempry in kernel is slower than memcpy (3.2ms vs. 1ms for 1600x900 rgba32 image)
