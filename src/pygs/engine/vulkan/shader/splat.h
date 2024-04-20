#ifndef PYGS_ENGINE_VULKAN_SHADER_SPLAT_H
#define PYGS_ENGINE_VULKAN_SHADER_SPLAT_H

namespace pygs {
namespace vk {
namespace shader {

const char* splat_vert = R"shader(
#version 460

// vertex
layout (location = 0) in vec2 position;

// instance
layout (location = 1) in vec3 ndc_position;
layout (location = 2) in vec3 scale_rot;  // (s0, s1, theta)
layout (location = 3) in vec4 color;

layout (location = 0) out vec4 out_color;
layout (location = 1) out vec2 out_position;

void main() {
  float theta = scale_rot.z;
  mat2 rot = mat2(cos(theta), sin(theta), -sin(theta), cos(theta));
  mat2 scale = mat2(scale_rot.x, 0.f, 0.f, scale_rot.y);

  float confidence_radius = 4.f;

  gl_Position = vec4(ndc_position + vec3(rot * scale * position * confidence_radius, 0.f), 1.f);
  out_color = color;
  out_position = position * confidence_radius;
}
)shader";

const char* splat_frag = R"shader(
#version 460

layout (location = 0) in vec4 color;
layout (location = 1) in vec2 position;

layout (location = 0) out vec4 out_color;

void main() {
  float gaussian_alpha = exp(-0.5f * dot(position, position));
  float alpha = color.a * gaussian_alpha;
  // premultiplied alpha
  out_color = vec4(color.rgb * alpha, alpha);
}
)shader";

}  // namespace shader
}  // namespace vk
}  // namespace pygs

#endif  // PYGS_ENGINE_VULKAN_SHADER_SPLAT_H
