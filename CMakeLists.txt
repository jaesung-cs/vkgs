cmake_minimum_required(VERSION 3.24)

project(pygs LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED True)

find_package(Vulkan REQUIRED COMPONENTS shaderc_combined)
add_subdirectory(third_party/VulkanMemoryAllocator EXCLUDE_FROM_ALL)
add_subdirectory(third_party/glfw EXCLUDE_FROM_ALL)
add_subdirectory(third_party/glm EXCLUDE_FROM_ALL)

# pygs
add_library(pygs STATIC
  src/pygs/window/window.cc
  src/pygs/scene/camera.cc
  src/pygs/scene/splats.cc
  src/pygs/engine/engine.cc
  src/pygs/engine/vulkan/attachment.cc
  src/pygs/engine/vulkan/buffer.cc
  src/pygs/engine/vulkan/compute_pipeline.cc
  src/pygs/engine/vulkan/context.cc
  src/pygs/engine/vulkan/cuda_image.cu
  src/pygs/engine/vulkan/cuda_semaphore.cu
  src/pygs/engine/vulkan/descriptor_layout.cc
  src/pygs/engine/vulkan/descriptor.cc
  src/pygs/engine/vulkan/framebuffer.cc
  src/pygs/engine/vulkan/graphics_pipeline.cc
  src/pygs/engine/vulkan/pipeline_layout.cc
  src/pygs/engine/vulkan/render_pass.cc
  src/pygs/engine/vulkan/shader_module.cc
  src/pygs/engine/vulkan/swapchain.cc
  src/pygs/engine/vulkan/uniform_buffer.cc
  src/pygs/engine/vulkan/vma_impl.cc
)

target_include_directories(pygs
  PUBLIC include
  PRIVATE src
)

target_link_libraries(pygs
  PUBLIC glm
  PRIVATE Vulkan::Vulkan Vulkan::shaderc_combined VulkanMemoryAllocator glfw
)

# executable
add_executable(pygs_base examples/pygs_base.cc)
target_link_libraries(pygs_base PRIVATE pygs)
