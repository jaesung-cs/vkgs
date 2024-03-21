#ifndef PYGS_ENGINE_VULKAN_SHADER_POINT_H
#define PYGS_ENGINE_VULKAN_SHADER_POINT_H

namespace pygs {
namespace vk {
namespace shader {

const char* point_vert = R"shader(
#version 450 core

layout (location = 0) in vec3 position;
layout (location = 1) in vec4 color;

layout (set = 0, binding = 0) uniform Camera {
  mat4 projection;
  mat4 view;
};

layout (location = 0) out vec4 out_color;

void main() {
  gl_Position = projection * view * vec4(position, 1.f);
  out_color = color;

  // Hard-coded point size
  gl_PointSize = 5.f;
}
)shader";

const char* point_frag = R"shader(
#version 450 core

layout (location = 0) in vec4 color;

layout (location = 0) out vec4 out_color;

void main() {
  out_color = color;
}
)shader";

}  // namespace shader
}  // namespace vk
}  // namespace pygs

#endif  // PYGS_ENGINE_VULKAN_SHADER_POINT_H
