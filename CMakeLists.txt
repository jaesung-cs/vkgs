cmake_minimum_required(VERSION 3.15)

project(vkgs LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(Vulkan REQUIRED)
add_subdirectory(third_party/VulkanMemoryAllocator EXCLUDE_FROM_ALL)
add_subdirectory(third_party/vulkan_radix_sort EXCLUDE_FROM_ALL)
add_subdirectory(third_party/glfw EXCLUDE_FROM_ALL)
add_subdirectory(third_party/glm EXCLUDE_FROM_ALL)
add_subdirectory(third_party/argparse EXCLUDE_FROM_ALL)

set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

# shaders
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/generated)

# add_shader(TARGET SHADER OUTPUT DEFINE...)
function(add_shader)
  list(POP_FRONT ARGV TARGET SHADER OUTPUT)
  list(TRANSFORM ARGV PREPEND "-D" OUTPUT_VARIABLE DEFINES)

  get_filename_component(SHADER ${SHADER} ABSOLUTE)

  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/src/generated/${OUTPUT}.h
    COMMAND
      ${Vulkan_GLSLANG_VALIDATOR_EXECUTABLE}
      --target-env spirv1.5
      -V
      --vn ${OUTPUT}
      -o ${CMAKE_CURRENT_SOURCE_DIR}/src/generated/${OUTPUT}.h
      ${DEFINES}
      ${SHADER}
    DEPENDS ${SHADER}
    COMMENT "Compiling ${CMAKE_CURRENT_SOURCE_DIR}/src/generated/${OUTPUT}.h"
  )

  add_custom_target(${OUTPUT} DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/generated/${OUTPUT}.h)
  add_dependencies(${TARGET} ${OUTPUT})
endfunction()

# vkgs
add_library(vkgs SHARED
  src/vkgs/scene/camera.cc
  src/vkgs/engine/engine.cc
  src/vkgs/engine/splat_load_thread.cc
  src/vkgs/engine/vulkan/attachment.cc
  src/vkgs/engine/vulkan/buffer.cc
  src/vkgs/engine/vulkan/compute_pipeline.cc
  src/vkgs/engine/vulkan/context.cc
  src/vkgs/engine/vulkan/cpu_buffer.cc
  src/vkgs/engine/vulkan/descriptor_layout.cc
  src/vkgs/engine/vulkan/descriptor.cc
  src/vkgs/engine/vulkan/framebuffer.cc
  src/vkgs/engine/vulkan/graphics_pipeline.cc
  src/vkgs/engine/vulkan/pipeline_layout.cc
  src/vkgs/engine/vulkan/render_pass.cc
  src/vkgs/engine/vulkan/swapchain.cc
  src/vkgs/engine/vulkan/uniform_buffer.cc
  src/vkgs/engine/vulkan/vma_impl.cc
  # imgui
  third_party/imgui/backends/imgui_impl_glfw.cpp
  third_party/imgui/backends/imgui_impl_vulkan.cpp
  third_party/imgui/imgui.cpp
  third_party/imgui/imgui_draw.cpp
  third_party/imgui/imgui_demo.cpp
  third_party/imgui/imgui_tables.cpp
  third_party/imgui/imgui_widgets.cpp
)

target_include_directories(vkgs
  PUBLIC include
  PRIVATE
    src
    third_party/imgui
    third_party/imgui/backends
)

target_link_libraries(vkgs
  PUBLIC glm
  PRIVATE Vulkan::Vulkan VulkanMemoryAllocator vk_radix_sort glfw
)

add_shader(vkgs src/shader/color.vert color_vert)
add_shader(vkgs src/shader/color.frag color_frag)
add_shader(vkgs src/shader/inverse_index.comp inverse_index_comp)
add_shader(vkgs src/shader/parse_ply.comp parse_ply_comp)
add_shader(vkgs src/shader/projection.comp projection_comp)
add_shader(vkgs src/shader/rank.comp rank_comp)
add_shader(vkgs src/shader/splat.vert splat_vert)
add_shader(vkgs src/shader/splat.frag splat_frag)

# executable
add_executable(vkgs_viewer examples/vkgs_viewer.cc)
target_link_libraries(vkgs_viewer
  PUBLIC vkgs
  PRIVATE argparse
)
