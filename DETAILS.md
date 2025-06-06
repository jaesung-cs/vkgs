# Details

## Rendering Algorithm Details

Like other web based viewer, it uses traditional graphics pipeline, drawing splats projected in 2D screen space.

One of benefits of using graphics pipeline rather than compute pipeline is that splats can be drawn together with other objects and graphics pipeline features such as MSAA.

1. (COMPUTE) rank
    - Cull splats outside view frustum, create key-value pairs to sort, based on view space depth.
1. (COMPUTE) sort
    - Perform 32bit key-value radix sort.
    - Indirect dispatch, sorting only visible points. Not a big deal, sort time is negligible compared to projection/rendering step.
1. (COMPUTE) inverse
    - Create inverse index map from splat order from sorted index.
    - This is for sequential memory access pattern in the next step.
1. (COMPUTE) projection
    - Calculate 3D-to-2D gaussian splat projection, and color using spherical harmonics.
    - Using F16 Spherical Harmonics increased rendering speed.
1. (GRAPHICS) rendering
    - Simply draw 2D guassian quads.
    - Speed up with indirect rendering, issuing only visible splats to draw command, reducing the number of shader invocations.

Projection and rendering steps are bottlenecks.

Current Onesweep radix sort implementation doesn't seem to work on MacOS.

https://raphlinus.github.io/gpu/2021/11/17/prefix-sum-portable.html

So I've implemented reduce-then-scan radix sort. No big performance difference even on NVidia GPU.


## Experiments

- Order Independent Transparency (OIT) doesn't work. I've tried Weighted Blended OIT (WBOIT). There are many nearly-opaque splats overlapped in a pixel, thus colors are blended in unsatisfactory manner. More importantly, OIT is slow.

- Rendering guassian splats with 4x MSAA is slow. Turning MSAA off gives about 2x rendering time boost.

- I've tried 4x MSAA and depth resolve for opaque objects in the first subpass and gaussian splat rendering with no MSAA in the second subpass, where 4x MSAA color/depth images are resolved to 1x MSAA images. Multisample colors are blended with background color into a pixel.

    ![](/media/depth_resolve.jpg)

- Directly updating to vulkan-cuda mapped memory in kernel is slower than memcpy (3.2ms vs. 1ms for 1600x900 rgba32 image). Regardlessly, it is better to manipulate swapchain image only in Vulkan. 1ms of copy cost is too much.

- Rendering triangle list is 0-3% faster than geometry shader. Also, geometry shader is not available in MacOS Metal/MoltenVK. Rendering triangle list is better choice.

- Using SH F16 storage increases speed by 20% on NVidia Geforce RTX 4090, 10% on Apple M2 Pro.


## Performance Test

- Added SH F16 storage feature: ~20% speed boost on NVidia GeFroce RTX 4090, ~10% speed boost on macbook.

- Tested geometry shader: 0-3% speed decrease with geometry shader.

- FPS may vary depending on splat scale, splat distribution, etc.

- NVidia GeForce RTX 4090, Windows
  - `bicycle.ply` (total 6,131,954 points)

    | View | Visible splats | Screen | MSAA | FPS |
    |--|--|--|--|--|
    | view 1 | 1M | 1280x720 | NO | 620 |
    | view 1 | 1M | 1280x720 | 2x | 480 |
    | view 1 | 1M | 1280x720 | 4x | 390 |
    | view 1 | 1M | 1600x900 | NO | 560 |
    | view 1 | 1M | 1600x900 | 2x | 430 |
    | view 1 | 1M | 1600x900 | 4x | 340 |
    | view 2 | 2M | 1280x720 | NO | 470 |
    | view 2 | 2M | 1280x720 | 2x | 400 |
    | view 2 | 2M | 1280x720 | 4x | 330 |
    | view 2 | 2M | 1600x900 | NO | 460 |
    | view 2 | 2M | 1600x900 | 2x | 400 |
    | view 2 | 2M | 1600x900 | 4x | 330 |

  - `garden.ply` (total 5,834,734 points)

    | View | Visible splats | Screen | MSAA | FPS |
    |--|--|--|--|--|
    | view 1 | 1.5M | 1280x720 | NO | 530 |
    | view 1 | 1.5M | 1600x900 | NO | 500 |
    | view 2 |   2M | 1280x720 | NO | 470 |
    | view 2 |   2M | 1600x900 | NO | 430 |
    | view 3 |   3M | 1280x720 | NO | 370 |
    | view 3 |   3M | 1600x900 | NO | 340 |

  - No MSAA gives huge FPS boost, without any quality loss. MSAA only affects opaque objects other than splats, such axes and grid.
  - View number is different from camera index in model json. I just randomly posed camera.
  - Small models such as `bonsai.ply`: 800~1000 FPS.
  - Rendering quads are slightly (0-3%) faster than rendering with geometry shader.

- Apple M2 Pro
  - MacOS is not my main target environment, but to just give some brief idea about rendering speed:
  - `bicycle.ply` (total 6,131,954 points)

    | View | Visible splats | Screen | MSAA | FPS |
    |--|--|--|--|--|
    | view 1 | 1M | 1280x720  | NO | 84 |
    | view 1 | 1M | 1600x900  | NO | 76 |
    | view 1 | 1M | 3200x1800 | NO | 51 |
    | view 2 | 2M | 1280x720  | NO | 54 |
    | view 2 | 2M | 1600x900  | NO | 52 |
    | view 2 | 2M | 3200x1800 | NO | 40 |

    - About 2x performance reported by [UnityGaussianSplatting](https://github.com/aras-p/UnityGaussianSplatting), 46FPS at 1200x800 with Apple M1 Max (note that my laptop is M2 Pro.)

  - `garden.ply` (total 5,834,734 points)

    | View | Visible splats | Screen | MSAA | FPS |
    |--|--|--|--|--|
    | view 1 | 1.5M | 1280x720  | NO | 78 |
    | view 1 | 1.5M | 1600x900  | NO | 73 |
    | view 1 | 1.5M | 3200x1800 | NO | 48 |
    | view 2 |   2M | 1280x720  | NO | 62 |
    | view 2 |   2M | 1600x900  | NO | 59 |
    | view 2 |   2M | 3200x1800 | NO | 43 |
    | view 3 |   3M | 1280x720  | NO | 44 |
    | view 3 |   3M | 1600x900  | NO | 43 |
    | view 3 |   3M | 3200x1800 | NO | 34 |

  - `bonsai.ply`: 120FPS at 1600x900. 100FPS at 3200x1800.
  - Geometry shader is not available. (`VkPhysicalDeviceFeatures::geometryShader = false`)


## References

- https://github.com/aras-p/UnityGaussianSplatting : Performance report, probably similar rendering pipeline
- https://github.com/shg8/3DGS.cpp : Vulkan viewer, but tile-based rendering with compute shader.
- https://github.com/dendenxu/fast-gaussian-rasterization : Very similar rendering approach. They used geometry shader, while I used storage buffer in vertex shader to save memory.