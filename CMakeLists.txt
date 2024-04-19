cmake_minimum_required(VERSION 3.24)

project(pygs LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(Vulkan REQUIRED COMPONENTS shaderc_combined)
add_subdirectory(third_party/VulkanMemoryAllocator EXCLUDE_FROM_ALL)
add_subdirectory(third_party/vulkan_radix_sort EXCLUDE_FROM_ALL)
add_subdirectory(third_party/glfw EXCLUDE_FROM_ALL)
add_subdirectory(third_party/glm EXCLUDE_FROM_ALL)
add_subdirectory(third_party/argparse EXCLUDE_FROM_ALL)

# pygs
add_library(pygs STATIC
  src/pygs/scene/camera.cc
  src/pygs/scene/splats.cc
  src/pygs/engine/engine.cc
  src/pygs/engine/splat_load_thread.cc
  src/pygs/engine/vulkan/attachment.cc
  src/pygs/engine/vulkan/buffer.cc
  src/pygs/engine/vulkan/compute_pipeline.cc
  src/pygs/engine/vulkan/context.cc
  src/pygs/engine/vulkan/cpu_buffer.cc
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
  # imgui
  third_party/imgui/backends/imgui_impl_glfw.cpp
  third_party/imgui/backends/imgui_impl_vulkan.cpp
  third_party/imgui/imgui.cpp
  third_party/imgui/imgui_draw.cpp
  third_party/imgui/imgui_demo.cpp
  third_party/imgui/imgui_tables.cpp
  third_party/imgui/imgui_widgets.cpp
)

target_include_directories(pygs
  PUBLIC include
  PRIVATE
    src
    third_party/imgui
    third_party/imgui/backends
)

target_link_libraries(pygs
  PUBLIC glm
  PRIVATE Vulkan::Vulkan Vulkan::shaderc_combined VulkanMemoryAllocator vk_radix_sort glfw
)

# executable
add_executable(pygs_base examples/pygs_base.cc)
target_link_libraries(pygs_base PRIVATE pygs argparse)
