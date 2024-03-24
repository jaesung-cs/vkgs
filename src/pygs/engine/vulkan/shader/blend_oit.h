#ifndef PYGS_ENGINE_VULKAN_SHADER_BLEND_OIT_H
#define PYGS_ENGINE_VULKAN_SHADER_BLEND_OIT_H

namespace pygs {
namespace vk {
namespace shader {

const char* blend_oit_vert = R"shader(
#version 460

layout (location = 0) in vec2 position;

void main() {
  // screen triangle
  gl_Position = vec4(position, 0.f, 1.f);
}
)shader";

const char* blend_oit_frag = R"shader(
#version 460

layout (input_attachment_index = 0, set = 2, binding = 0) uniform subpassInputMS input_color;  // sum (Cw, aw)
layout (input_attachment_index = 1, set = 2, binding = 1) uniform subpassInputMS input_alpha;  // prod (1-alpha)

layout (location = 0) out vec4 out_color;

void main() {
  // (sum Cw / sum aw) * (1 - prod (1-a)) + C0 * prod (1-a)
  // src color = (sum Cw / sum aw)
  // src alpha = prod (1-a)
  // src factor = ONE_MINUS_SRC_ALPHA
  // dst color = C0
  // dst alpha = any
  // dst factor = SRC_ALPHA
  vec4 color = subpassLoad(input_color, gl_SampleID);
  float prod_alpha = subpassLoad(input_alpha, gl_SampleID).r;

  out_color = vec4(color.rgb / color.a, 1.f - prod_alpha);
  // out_color = vec4(color.rgb / color.a, 0.5f);
}
)shader";

}  // namespace shader
}  // namespace vk
}  // namespace pygs

#endif  // PYGS_ENGINE_VULKAN_SHADER_BLEND_OIT_H
