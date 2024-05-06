cmake_minimum_required(VERSION 3.24)

project(pygs LANGUAGES C CXX)

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

# pygs
add_library(pygs SHARED
  src/pygs/scene/camera.cc
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
  PRIVATE Vulkan::Vulkan VulkanMemoryAllocator vk_radix_sort glfw
)

add_shader(pygs src/shader/color.vert color_vert)
add_shader(pygs src/shader/color.frag color_frag)
add_shader(pygs src/shader/inverse_index.comp inverse_index_comp)
add_shader(pygs src/shader/parse_ply.comp parse_ply_comp)
add_shader(pygs src/shader/projection.comp projection_comp)
add_shader(pygs src/shader/rank.comp rank_comp)
add_shader(pygs src/shader/splat.vert splat_vert)
add_shader(pygs src/shader/splat.frag splat_frag)

# executable
add_executable(pygs_base examples/pygs_base.cc)
target_link_libraries(pygs_base
  PUBLIC pygs
  PRIVATE argparse
)
