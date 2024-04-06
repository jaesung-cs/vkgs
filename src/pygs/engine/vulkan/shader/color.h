#ifndef PYGS_ENGINE_VULKAN_SHADER_COLOR_H
#define PYGS_ENGINE_VULKAN_SHADER_COLOR_H

namespace pygs {
namespace vk {
namespace shader {

const char* color_vert = R"shader(
#version 460

layout (location = 0) in vec3 position;
layout (location = 1) in vec4 color;

layout (set = 0, binding = 0) uniform Camera {
  mat4 projection;
  mat4 view;
  vec3 camera_position;
  float pad0;
  uvec2 screen_size;  // (width, height)
};

layout (push_constant, std430) uniform PushConstants {
  mat4 model;
};

layout (location = 0) out vec4 out_color;

void main() {
  gl_Position = projection * view * model * vec4(position, 1.f);
  out_color = color;
}
)shader";

const char* color_frag = R"shader(
#version 460

layout (location = 0) in vec4 color;

layout (location = 0) out vec4 out_color;

void main() {
  // premultiplied alpha
  out_color = vec4(color.rgb * color.a, color.a);
}
)shader";

}  // namespace shader
}  // namespace vk
}  // namespace pygs

#endif  // PYGS_ENGINE_VULKAN_SHADER_COLOR_H
