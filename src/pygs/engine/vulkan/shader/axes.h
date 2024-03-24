#ifndef PYGS_ENGINE_VULKAN_SHADER_AXES_H
#define PYGS_ENGINE_VULKAN_SHADER_AXES_H

namespace pygs {
namespace vk {
namespace shader {

const char* axes_vert = R"shader(
#version 460

layout (location = 0) in vec3 position;
layout (location = 1) in vec3 color;
layout (location = 2) in mat4 transform;

layout (set = 0, binding = 0) uniform Camera {
  mat4 projection;
  mat4 view;
};

layout (location = 0) out vec3 out_color;

void main() {
  gl_Position = projection * view * transform * vec4(position, 1.f);
  out_color = color;
}
)shader";

const char* axes_frag = R"shader(
#version 460

layout (location = 0) in vec3 color;

layout (location = 0) out vec4 out_color;

void main() {
  out_color = vec4(color, 1.f);
}
)shader";

}  // namespace shader
}  // namespace vk
}  // namespace pygs

#endif  // PYGS_ENGINE_VULKAN_SHADER_AXES_H
