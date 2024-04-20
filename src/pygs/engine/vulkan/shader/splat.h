#ifndef PYGS_ENGINE_VULKAN_SHADER_SPLAT_H
#define PYGS_ENGINE_VULKAN_SHADER_SPLAT_H

namespace pygs {
namespace vk {
namespace shader {

const char* splat_vert = R"shader(
#version 460

layout (std430, set = 1, binding = 1) readonly buffer Instances {
  float instances[];  // (N, 10). 3 for ndc position, 3 for scale rot, 4 for color
};

layout (location = 0) out vec4 out_color;
layout (location = 1) out vec2 out_position;

void main() {
  // index [0,1,2,2,1,3], 4 vertices for a splat.
  int index = gl_VertexIndex / 4;
  vec3 ndc_position = vec3(instances[index * 10 + 0], instances[index * 10 + 1], instances[index * 10 + 2]);
  vec3 scale_rot = vec3(instances[index * 10 + 3], instances[index * 10 + 4], instances[index * 10 + 5]);
  vec4 color = vec4(instances[index * 10 + 6], instances[index * 10 + 7], instances[index * 10 + 8], instances[index * 10 + 9]);

  // quad positions (-1, -1), (-1, 1), (1, -1), (1, 1), ccw in screen space.
  int vert_index = gl_VertexIndex % 4;
  vec2 position = vec2(vert_index / 2, vert_index % 2) * 2.f - 1.f;

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
