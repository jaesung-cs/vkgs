# vkgs

Gaussian splatting viewer written in Vulkan.

The main goal of this project is to maximize rendering speed.

For more details, refer to [details](DETAILS.md).


## Notes

Currently I am **not actively maintaining** this project, but I am still open to suggestions or discussions, and will be happy to answer any questions or make quick fixes. Please open github issues about anything. Job offer is also welcome ;)

Consider using [vk_gaussian_splatting](https://github.com/nvpro-samples/vk_gaussian_splatting), another Vulkan 3DGS renderer by NVidia with richer functionalities.
They mentioned my project (which is a great honor for me), and obviously their codes are far more well-organized.

Also, check out [vulkan_radix_sort](https://github.com/jaesung-cs/vulkan_radix_sort).
It has a way better user-friendly interface that can be easily used by other projects.
It is adopted by the NVidia sample, implying that the sort algorithm is quite stable.

Please check out [pygs](https://github.com/jaesung-cs/pygs) for more organized c++ codes and experimental version of python binding.


## Desktop Viewer

![](/media/screenshot-fast2.jpg)

Viewer works with pre-trained vanilla 3DGS models as input.


### Feature Highlights

- Fast rendering speed
  - 350+ FPS on 1600x900, high-end GPU (NVidia GeForce RTX 4090)
  - 50+ FPS on 1600x900, high-end MacOS laptop (Apple M2 Pro)
  - 1-1.5x speed compared to SIBR viewer, but difference becomes bigger when scene is zoomed out,
    - because the number of tiles increases, and
    - more splats overlap in a single tile, so sequential blending operation takes more time
- Using graphics pipeline
  - Draw gaussian splats over other opaque objects, interacting with depth buffer
- 100% GPU tasks
  - No CPU-GPU synchronization for single frame: while GPU is working on frame i, CPU prepares a commands buffer and submits for frame i+1. No synchronization for frame i to get number of visible splats.
  - Indirect sort & draw: sorting and rendering only visible points
  - My vulkan radix sort implementation


### Requirements
- `VulkanSDK>=1.3.296.0`
  - `slangc` (or `slangc.exe` for Windows) is included in or above `1.3.296.0`.
  - Required by `vulkan_radix_sort` submodule to compile `slang` shader files.
  - Download the latest version from https://vulkan.lunarg.com/ and follow install instruction.
- `cmake>=3.15`


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
  - [vulkan_radix_sort](https://github.com/jaesung-cs/vulkan_radix_sort): my Vulkan/GLSL implementation of reduce-then-scan radix sort.


### Build
```bash
$ cmake . -B build
$ cmake --build build --config Release -j
```


### Run
```bash
$ ./build/vkgs_viewer  # or ./build/Release/vkgs_viewer
$ ./build/vkgs_viewer -i <ply_filepath>
```
Drag and drop pretrained .ply file from [official gaussian splatting](https://github.com/graphdeco-inria/gaussian-splatting), Pre-trained Models (14 GB).

- Left drag to rotate.

- Right drag, or Ctrl + left drag to translate.

- Left+Right drag to zoom in/out.

- WASD, Space to move.

- Wheel to zoom in/out.

- Ctrl+wheel to change FOV.


## pygs: Python Binding (WIP)

GUI is created in an off thread.
According to GLFW documentation, the user should create window in main thread.
However, managing windows off-thread seems working in Windows and Linux somehow.

Unfortunately, MacOS doesn't allow this.
MacOS’s UI frameworks can only be called from the main thread.
Here's a related [thread](https://forums.developer.apple.com/forums/thread/659010) by Apple staff.


### Requirements

- Windows or Linux (Doesn't work for MacOS.)
- conda: cmake, pybind11, cuda-toolkit (cuda WIP, not necessary yet)
```bash
$ conda create -n pygs python=3.10
$ conda activate pygs
$ conda install conda-forge::cmake
$ conda install conda-forge::pybind11
$ conda install nvidia/label/cuda-12.2.2::cuda-toolkit  # or any other version
```


### Build

The python package dynamically links to c++ shared library file.

So, first build the shared library first, then install python package.

```bash
$ cmake . -B build
$ cmake --build build --config Release -j
$ pip install -e binding/python
```


### Test

```bash
$ python
>>> import pygs
>>> pygs.show()
>>> pygs.load("./models/bicycle_30000.ply")  # asynchronously load model to viewer
>>> pygs.load("./models/garden_30000.ply")
>>> pygs.close()
```
