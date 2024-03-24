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
layout (location = 2) in vec3 cov2d;
layout (location = 3) in vec4 color;

layout (location = 0) out vec4 out_color;
layout (location = 1) out vec2 out_position;

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

  float confidence_radius = 4.f;

  gl_Position = vec4(ndc_position + vec3(rot * scale * position * confidence_radius, 0.f), 1.f);
  //gl_Position = vec4(ndc_position + vec3(position * 0.01, 0.f), 1.f);
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
  // TODO: premultiplied alpha
  float gaussian_alpha = exp(-0.5f * length(position));
  out_color = vec4(color.rgb, color.a * gaussian_alpha);
}
)shader";

}  // namespace shader
}  // namespace vk
}  // namespace pygs

#endif  // PYGS_ENGINE_VULKAN_SHADER_SPLAT_H
