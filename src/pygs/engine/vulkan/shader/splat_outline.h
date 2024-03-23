#ifndef PYGS_ENGINE_VULKAN_SHADER_SPLAT_OUTLINE_H
#define PYGS_ENGINE_VULKAN_SHADER_SPLAT_OUTLINE_H

namespace pygs {
namespace vk {
namespace shader {

const char* splat_outline_vert = R"shader(
#version 450 core

// vertex
layout (location = 0) in vec2 position;
layout (location = 1) in vec4 color;

// instance
layout (location = 2) in vec3 cov2d;
layout (location = 3) in vec3 projected_position;

layout (location = 0) out vec4 out_color;

void main() {
  // eigendecomposition
  // [a c] = [x y]
  // [c b]   [y z]
  float a = cov2d.x;
  float b = cov2d.z;
  float c = cov2d.y;
  float D = sqrt((a - b) * (a - b) + 4.f * c * c);
  float s0 = sqrt(0.5f * (a + b + D));
  float s1 = sqrt(0.5f * (a + b - D));
  // decompose to R^T S^2 R
  float sin2t = 2.f * c / D;
  float cos2t = (a - b) / D;
  float theta = atan(sin2t, cos2t) / 2.f;
  mat2 rot = mat2(cos(theta), sin(theta), -sin(theta), cos(theta));
  mat2 scale = mat2(s0, 0.f, 0.f, s1);

  gl_Position = vec4(projected_position + vec3(rot * scale * position, 0.f), 1.f);
  out_color = color;
}
)shader";

const char* splat_outline_frag = R"shader(
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

#endif  // PYGS_ENGINE_VULKAN_SHADER_SPLAT_OUTLINE_H
