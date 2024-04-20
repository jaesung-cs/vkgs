# pygs
Gaussian Splatting

## Desktop Viewer

![](/media/screenshot-fast.jpg)

~200FPS with 3M visible splats, with NVIDIA GeForce RTX 4090.

Like other web based viewer, it uses traditional graphics pipeline, drawing splats projected in 2D screen space.

One of benefits of using graphics pipeline rather than compute pipeline is that splats can be drawn together with other objects and graphics pipeline features such as MSAA.
The screenshot shows splats rendered with render pass, with depth buffer and 4x MSAA. Splats are drawn over grid and axis.

Tested only on my desktop PC with NVIDIA GeForce RTX 4090, on Windows and Linux. Let me know if there is any issue building or running the viewer.


### Rendering Algorithm Details
1. (COMPUTE) rank: cull splats outside view frustum, create key-value pairs to sort, based on view space depth.
1. (COMPUTE) sort: perform 32bit key-value radix sort.
1. (COMPUTE) inverse: create inverse index map from splat order from sorted index. This is for sequential memory access pattern in the next step.
1. (COMPUTE) projection: calculate 3D-to-2D gaussian splat projection, and color using spherical harmonics.
1. (GRAPHICS) rendering: simply draw 2D guassian quads.

Rendering is bottleneck.
The number of visible points is the main factor.
Drawing 2~5 millions points with the simplest point shader (even with depth test/write on and color blend off) already costs 2~3ms.
Without reducing the number of points to draw (e.g. deleting less important splts during training, using octree, early stopping in tile-based rendering), it seems hard to make more dramatic improvements in rendering speed.


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
  - [vulkan_radix_sort](https://github.com/jaesung-cs/vulkan_radix_sort): my Vulkan/GLSL implementation of [Onesweep](https://research.nvidia.com/publication/2022-06_onesweep-faster-least-significant-digit-radix-sort-gpus), state-of-the-art radix sort algorithm.


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

Left drag to rotate.

Right drag to translate.

Left+Right drag to zoom in/out.

WASD, Space to move.


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
