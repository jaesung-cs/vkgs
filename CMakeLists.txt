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
  src/pygs/engine/engine.cc
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
